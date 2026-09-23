"""Bundle integrity, safe dispatch, algorithm isolation, and actual CPU SFT."""

import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from jev_dataops import training_launch as launch
from jev_dataops.rewards import MAX_REWARD_CHARACTERS, compute_score


def bundle_fixture(tmp_path, target="sft", reference="ground_truth"):
    bundle = tmp_path / "bundle with spaces"
    (bundle / "data").mkdir(parents=True)
    suffix = "jsonl" if target == "sft" else "parquet"
    data = {}
    for split in launch.SPLITS:
        data[split] = f"data/{split}.{suffix}"
        path = bundle / data[split]
        if target == "sft":
            path.write_text("".join(json.dumps({"id": f"{split}-{i}", "text": f"Useful {split} example {i}",
                                                "_split_group": f"{split}-{i}"}) + "\n" for i in range(4)))
        else:
            # Dispatch validation treats data as opaque, checksum-protected files;
            # actual Parquet schema is covered by exporter integration tests.
            path.write_bytes(b"PAR1" + split.encode())
    manifest = {"schema_version": 1, "status": "prepared", "target": target,
                "base_model": "operator/example-instruct", "data": data,
                "split_counts": {split: 4 for split in launch.SPLITS},
                "group_counts": {split: 4 for split in launch.SPLITS},
                "config": {"max_steps": 2, "learning_rate": 1e-6},
                "reward": {"mode": "exact_match", "field": reference} if target != "sft" else None}
    launch.write_launch_files(bundle, manifest)
    save_manifest(bundle, manifest)
    return bundle, manifest


def save_manifest(bundle, manifest):
    manifest["checksums"] = {p.relative_to(bundle).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in bundle.rglob("*") if p.is_file() and p.name != "manifest.json"}
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("target", ["sft", "verl-grpo", "verl-ppo"])
def test_dry_run_is_portable_and_does_not_spawn_or_create_outputs(tmp_path, monkeypatch, target):
    bundle, manifest = bundle_fixture(tmp_path, target)
    moved = tmp_path / "relocated bundle"
    shutil.move(bundle, moved)

    def forbidden(*args, **kwargs):
        pytest.fail("Dry run must not invoke a subprocess or probe a GPU/model.")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    result = launch.launch_training(moved)
    assert result["status"] == "prepared"
    assert result["dry_run"] is True
    assert result["shell"] is False
    assert result["gpu_validated"] is False
    assert not (moved / "results").exists()
    assert result["cwd"] == str(moved)
    assert not any("data/test.parquet" in part for part in result["command"])
    if target != "sft":
        overrides = dict(arg.split("=", 1) for arg in result["command"][4:])
        assert json.loads(overrides["data.val_files"]) == str(moved / manifest["data"]["validation"])
        assert json.loads(overrides["reward.custom_reward_function.path"]) == str(moved / "rewards.py")
        assert json.loads(overrides["actor_rollout_ref.actor.optim.lr"]) == 1e-6
        assert json.loads(overrides["algorithm.adv_estimator"]) == ("grpo" if target.endswith("grpo") else "gae")
        assert json.loads(overrides["critic.enable"]) == target.endswith("ppo")
        assert json.loads(overrides["actor_rollout_ref.rollout.n"]) == (4 if target.endswith("grpo") else 1)
        assert overrides["trainer.val_before_train"] == "true"
        assert overrides["trainer.total_training_steps"] == "1"
        assert overrides["trainer.test_freq"] == "1"
        assert overrides["trainer.save_freq"] == "1"


def test_checksum_and_reward_code_tampering_fail_closed(tmp_path):
    bundle, manifest = bundle_fixture(tmp_path, "verl-grpo", reference="verified_answer")
    launch.launch_training(bundle)
    (bundle / "data/train.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        launch.launch_training(bundle)
    save_manifest(bundle, manifest)
    (bundle / "rewards.py").write_text("raise RuntimeError('arbitrary bundle code')\n")
    save_manifest(bundle, manifest)
    with pytest.raises(ValueError, match="trusted exact-match"):
        launch.launch_training(bundle)


@pytest.mark.parametrize("path", ["../outside.jsonl", "/tmp/outside.jsonl", "data/../outside.jsonl", "C:/outside.jsonl", "data/${oc.env:HOME}.jsonl"])
def test_traversal_and_hydra_interpolation_are_rejected(tmp_path, path):
    bundle, manifest = bundle_fixture(tmp_path)
    manifest["data"]["train"] = path
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        launch.launch_training(bundle)


def test_symlinks_cannot_escape_even_with_matching_checksum(tmp_path):
    bundle, manifest = bundle_fixture(tmp_path)
    path = bundle / manifest["data"]["train"]
    outside = tmp_path / "outside.jsonl"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="Symlinked"):
        launch.launch_training(bundle)


def test_fresh_output_and_grpo_group_size_are_required(tmp_path):
    bundle, manifest = bundle_fixture(tmp_path, "verl-grpo")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        launch.launch_training(bundle, output=existing)
    manifest["config"]["rollout_n"] = 1
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="GRPO"):
        launch.launch_training(bundle)


def test_execution_is_foreground_and_records_failure_without_success_claim(tmp_path, monkeypatch):
    bundle, _ = bundle_fixture(tmp_path, "verl-ppo")
    calls = []
    monkeypatch.setattr(launch, "_probe", lambda *args, **kwargs: {"resolved_base_model": "/operator/local model", "verl": "0.9.1"})

    class Process:
        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))

        def wait(self):
            return 3

    monkeypatch.setattr(subprocess, "Popen", Process)
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-inherit")
    result = launch.launch_training(bundle, dry_run=False)
    assert result["status"] == "failed"
    assert result["returncode"] == 3
    assert result["gpu_validated"] is False
    command, options = calls[0]
    assert isinstance(command, list) and command[:4] == [sys.executable, "-I", "-m", "verl.trainer.main_ppo"]
    assert options["shell"] is False
    assert options["start_new_session"] is True
    assert options["env"]["RAY_ADDRESS"] == "local"
    assert "OPENROUTER_API_KEY" not in options["env"]
    assert options["env"]["VERL_FILE_LOGGER_PATH"] == str(bundle / "results/metrics.jsonl")
    assert (bundle / "results/launch_started.json").is_file()
    assert json.loads((bundle / "results/launch_report.json").read_text())["status"] == "failed"


def test_exact_reward_is_reference_only_bounded_and_never_executes_code():
    assert compute_score("anything", " cafe\u0301\n42 ", "café 42", {"score": 0}) == {"score": 1.0, "acc": 1.0}
    for answer, reference in [("", ""), ("Answer", "answer"), ("42.", "42"), ("42", None),
                              ("42", {"code": "42"}), ("x" * (MAX_REWARD_CHARACTERS + 1), "x"),
                              ("__import__('os').system('false')", "42")]:
        assert compute_score("jev_dataops", answer, reference, {"reward": 100})["score"] == 0


def test_sft_preserves_fixed_groups_and_actual_cpu_lora(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from jev_dataops.training_export import export_training

    vocabulary = {word: index for index, word in enumerate(
        ["[UNK]", "[EOS]", "user", "assistant", ":", "Question", "Answer"] + [str(i) for i in range(24)])}
    backend = Tokenizer(WordLevel(vocab=vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[EOS]")
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    base = tmp_path / "local tiny model"
    tokenizer.save_pretrained(base)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocabulary), n_positions=64, n_ctx=64, n_embd=16,
                                     n_layer=1, n_head=2, bos_token_id=1, eos_token_id=1, pad_token_id=1))
    model.save_pretrained(base, safe_serialization=True)
    source = tmp_path / "keep.jsonl"
    source.write_text("".join(json.dumps({"id": str(i), "prompt": f"Question {i}", "response": f"Answer {i}",
                                         "group_id": f"group-{i}"}) + "\n" for i in range(24)))
    bundle = tmp_path / "sft bundle"
    manifest = export_training(source, bundle, target="sft", base_model=str(base),
                               config={"max_steps": 1, "max_seq_length": 32, "batch_size": 2})
    monkeypatch.setenv("JEV_MODEL_LOCAL_ONLY", "1")
    monkeypatch.setenv("JEV_TRAIN_DEVICE", "cpu")
    result = launch.launch_training(bundle, dry_run=False, python=sys.executable)
    log = (bundle / "results/training.log").read_text()
    assert result["status"] == "completed", log
    report = json.loads((bundle / "results/training_report.json").read_text())
    assert report["steps"] == 1
    assert report["loss_mask"] == "answer"
    assert report["masking"]["rows_with_masked_prompt"] > 0
    assert report["split_counts"] == manifest["split_counts"]
    assert report["evaluation_split"] == "test"
    assert math.isfinite(report["baseline_loss"]) and math.isfinite(report["trained_loss"])
    for split in launch.SPLITS:
        original = [json.loads(line)["_split_group"] for line in (bundle / manifest["data"][split]).read_text().splitlines()]
        trained = [json.loads(line)["_split_group"] for line in (bundle / f"results/{split}.jsonl").read_text().splitlines()]
        assert original == trained
    assert (bundle / "results/model/adapter_model.safetensors").is_file()
