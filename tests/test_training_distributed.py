"""Actual two-process CPU/Gloo SFT and no-padding streaming shard invariants.

The smoke test needs local TCP rendezvous permission, but no GPU, network model
download, or cloud service. It does not assert CUDA/NCCL compatibility.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from jev_dataops.training_distributed import _rank_batches


def test_streamed_rank_shards_have_equal_training_steps_and_no_eval_duplicates(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps({"id": i, "text": str(i)}) + "\n" for i in range(11)))
    train = [list(_rank_batches(path, rank, 3, 2, limit=6)) for rank in range(3)]
    assert [len(batches) for batches in train] == [1, 1, 1]
    assigned = [[row["id"] for batch in batches for row in batch] for batches in train]
    assert assigned == [[0, 3], [1, 4], [2, 5]]
    evaluation = [[row["id"] for batch in _rank_batches(path, rank, 3, 2) for row in batch] for rank in range(3)]
    assert sorted(value for shard in evaluation for value in shard) == list(range(11))
    assert list(_rank_batches(path, 1, 2, 2, limit=1)) == []


def _tiny_bundle(tmp_path):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.normalizers import Replace
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from jev_dataops.training_launch import write_launch_files

    torch.manual_seed(17)
    vocabulary = {word: index for index, word in enumerate(
        ["[UNK]", "[EOS]", "user", "assistant", ":", "Question", "Answer", "extra"] + [str(i) for i in range(12)])}
    token_backend = Tokenizer(WordLevel(vocab=vocabulary, unk_token="[UNK]"))
    token_backend.pre_tokenizer = Whitespace()
    token_backend.normalizer = Replace("EMPTY", "")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=token_backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[EOS]")
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    base = tmp_path / "tiny local model"
    tokenizer.save_pretrained(base)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocabulary), n_positions=64, n_ctx=64, n_embd=16,
                                     n_layer=1, n_head=2, bos_token_id=1, eos_token_id=1, pad_token_id=1))
    model.save_pretrained(base, safe_serialization=True)
    bundle = tmp_path / "bundle"
    (bundle / "data").mkdir(parents=True)
    counts = {"train": 7, "validation": 1, "test": 3}
    for split, count in counts.items():
        rows = [{"id": f"{split}-{i}", "_split_group": f"{split}-{i}",
                 "prompt": f"Question {i}", "response": "Answer " + "extra " * i + str(i)} for i in range(count)]
        # Rank 0 has zero supervised tokens for the first (and only) batch in
        # each epoch. It must still participate in backward/all collectives.
        if split == "train":
            rows[0] = {"id": "train-0", "_split_group": "train-0", "text": "EMPTY"}
            rows[2] = {"id": "train-2", "_split_group": "train-2", "text": "EMPTY"}
        (bundle / f"data/{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {"schema_version": 1, "status": "prepared", "target": "sft", "base_model": str(base),
                "data": {split: f"data/{split}.jsonl" for split in counts}, "split_counts": counts,
                "group_counts": counts, "config": {"n_gpus": 2, "batch_size": 4, "epochs": 2, "max_steps": 2,
                "max_seq_length": 32, "seed": 42, "learning_rate": 0.002, "loss_mask": "answer"}}
    write_launch_files(bundle, manifest)
    manifest["checksums"] = {path.relative_to(bundle).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in bundle.rglob("*") if path.is_file()}
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    return bundle, manifest, tokenizer


def test_two_cpu_workers_train_real_lora_and_match_serial_heldout_metrics(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("This PyTorch build does not provide Gloo.")
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from safetensors.torch import load_file

    bundle, manifest, tokenizer = _tiny_bundle(tmp_path)
    output = tmp_path / "results"
    command = [sys.executable, "-I", "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--local-addr=127.0.0.1",
               "--nproc-per-node=2", "--max-restarts=0", "--no-python", sys.executable, "-I", "-m",
               "jev_dataops.training_distributed", "--bundle", str(bundle), "--output", str(output), "--backend", "gloo"]
    environment = {**os.environ, "JEV_MODEL_LOCAL_ONLY": "1", "JEV_TRAIN_DEVICE": "cpu",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1"}
    result = subprocess.run(command, cwd=bundle, env=environment, capture_output=True, text=True, timeout=180, shell=False)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "training_report.json").read_text())
    assert report["steps"] == 2 and report["training_record_visits"] == 8
    assert report["distributed"]["world_size"] == 2
    assert report["distributed"]["backend"] == "gloo" and report["gpu_validated"] is False
    assert report["distributed"]["global_batch_size"] == 4
    assert report["distributed"]["per_rank_batch_size"] == 2
    assert report["distributed"]["dropped_training_rows_per_epoch"] == 3
    assert report["distributed"]["evaluation_padding_duplicates"] == 0
    assert report["evaluation_rows"] == {"validation": 1, "test": 3}
    assert report["split_counts"] == manifest["split_counts"]
    assert report["loss_mask"] == "answer" and report["masking"]["rows_with_masked_prompt"] == 4
    assert all(math.isfinite(item["loss"]) for item in report["loss_history"])
    assert len(list(output.glob("**/adapter_model.safetensors"))) == 1
    weights = load_file(output / "model/adapter_model.safetensors")
    assert any(bool(torch.count_nonzero(value)) for key, value in weights.items() if "lora_B" in key)
    for split in manifest["data"]:
        original = [json.loads(line)["_split_group"] for line in (bundle / manifest["data"][split]).read_text().splitlines()]
        saved = [json.loads(line)["_split_group"] for line in (output / f"{split}.jsonl").read_text().splitlines()]
        assert saved == original

    # Independently evaluate each complete split in a single process, using the
    # known fixture template and standard HF mean loss, then weight by tokens.
    model = AutoModelForCausalLM.from_pretrained(manifest["base_model"], local_files_only=True, use_safetensors=True)

    def serial(split):
        model.eval()
        weighted, total = 0.0, 0
        with torch.no_grad():
            for line in (bundle / manifest["data"][split]).read_text().splitlines():
                row = json.loads(line)
                turns = [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["response"]}]
                text = tokenizer.apply_chat_template(turns, tokenize=False)
                prefix = tokenizer.apply_chat_template(turns[:1], tokenize=False, add_generation_prompt=True)
                ids = tokenizer(text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
                cut = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
                inputs = torch.tensor([ids])
                labels = inputs.clone()
                labels[:, :cut] = -100
                tokens = int((labels[:, 1:] != -100).sum())
                loss = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), labels=labels).loss.item()
                weighted += loss * tokens
                total += tokens
        return weighted / total, total

    for split in ("validation", "test"):
        loss, tokens = serial(split)
        assert loss == pytest.approx(report["baseline_loss" if split == "test" else "baseline_validation_loss"], abs=1e-6)
        assert tokens == report["evaluation_tokens"][split]
    model = PeftModel.from_pretrained(model, output / "model", is_trainable=False)
    for split in ("validation", "test"):
        loss, tokens = serial(split)
        assert loss == pytest.approx(report["trained_loss" if split == "test" else "trained_validation_loss"], abs=1e-6)
        assert tokens == report["evaluation_tokens"][split]
