import json
import math
from pathlib import Path

import pytest

from jev_dataops.training import TrainingCancelled, _text, train_and_evaluate, validate_training_config


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def records(count=30):
    return [{"id": str(i), "group_id": f"group-{i // 2}",
             "text": f"Example {i}: reliable data makes a useful language model. 数据样本。"} for i in range(count)]


def test_demo_trains_real_model_and_independently_reproduces_metrics(tmp_path):
    source = write_jsonl(tmp_path / "keep.jsonl", records())
    output = tmp_path / "run"
    events = []
    report = train_and_evaluate(source, output, {"trainer": "demo", "max_steps": 3}, progress=events.append)
    checkpoint = json.loads((output / "model" / "byte_bigram.json").read_text())
    test_rows = read_jsonl(output / "test.jsonl")
    total_loss, n_tokens = 0, 0
    for row in test_rows:
        previous = checkpoint["bos"]
        for token in list(row["text"].encode("utf-8")[:256]) + [checkpoint["eos"]]:
            probability = ((checkpoint["counts"][previous][token] + 0.5) /
                           (checkpoint["context_totals"][previous] + 0.5 * 257))
            total_loss -= math.log(probability)
            n_tokens += 1
            previous = token
    assert report["trainer"] == "demo"
    assert report["is_llm"] is False
    assert report["steps"] == 3
    assert sum(checkpoint["context_totals"]) > 0
    assert report["baseline_loss"] == pytest.approx(math.log(257))
    assert report["trained_loss"] == pytest.approx(total_loss / n_tokens)
    assert report["trained_perplexity"] == pytest.approx(math.exp(total_loss / n_tokens))
    assert report["delta_loss"] == pytest.approx(report["trained_loss"] - report["baseline_loss"])
    assert sum(report["split_counts"].values()) == 30
    assert all(report["split_counts"].values())
    assert json.loads((output / "training_report.json").read_text())["status"] == "completed"
    assert (output / "model_report.json").is_file()
    assert not (output / "split_index.sqlite3").exists()
    assert events[-1]["stage"] == "completed"


def test_groups_and_cross_group_duplicate_components_never_leak(tmp_path):
    rows = records(40)
    # Bridge two declared groups through exact text and then a conversation ID.
    rows[2]["text"] = rows[0]["text"]
    rows[3]["conversation_id"] = "shared-conversation"
    rows[4]["conversation_id"] = "shared-conversation"
    rows[5]["text"] = "  " + rows[0]["text"].replace(" ", "  ") + "  "
    source = write_jsonl(tmp_path / "keep.jsonl", rows)
    report = train_and_evaluate(source, tmp_path / "run", {})
    where, content_where, group_where, conversation_where = {}, {}, {}, {}
    for split in ("train", "validation", "test"):
        for row in read_jsonl(tmp_path / "run" / f"{split}.jsonl"):
            where[row["id"]] = split
            text = " ".join(row["text"].split())
            assert content_where.setdefault(text, split) == split
            assert group_where.setdefault(row["group_id"], split) == split
            if "conversation_id" in row:
                assert conversation_where.setdefault(row["conversation_id"], split) == split
    assert len({where[str(i)] for i in range(6)}) == 1
    assert report["independent_groups"] == 18


def test_seeded_component_assignment_is_independent_of_input_order(tmp_path):
    rows = records(60)
    source_a = write_jsonl(tmp_path / "a.jsonl", rows)
    source_b = write_jsonl(tmp_path / "b.jsonl", list(reversed(rows)))
    assignments = []
    for source, output in ((source_a, tmp_path / "a"), (source_b, tmp_path / "b")):
        train_and_evaluate(source, output, {"seed": 23})
        assignments.append({row["id"]: split for split in ("train", "validation", "test")
                            for row in read_jsonl(output / f"{split}.jsonl")})
    assert assignments[0] == assignments[1]


@pytest.mark.parametrize("rows", [[], records(8), [{"text": "identical content", "group_id": str(i)} for i in range(20)]])
def test_insufficient_independent_groups_fail_without_success_report(tmp_path, rows):
    source = write_jsonl(tmp_path / "keep.jsonl", rows)
    with pytest.raises(ValueError, match="No retained|At least 6"):
        train_and_evaluate(source, tmp_path / "run", {})
    assert not (tmp_path / "run" / "training_report.json").exists()
    assert not (tmp_path / "run" / "split_index.sqlite3").exists()


@pytest.mark.parametrize("config", [
    {"trainer": "shell"}, {"epochs": 0}, {"epochs": 1.5}, {"batch_size": 1000},
    {"max_steps": -1}, {"max_seq_length": 999999}, {"seed": True},
    {"learning_rate": float("nan")}, {"learning_rate": "0.1"}, {"test_fraction": 0.9},
])
def test_compute_config_validation(config):
    with pytest.raises(ValueError):
        validate_training_config(config)


def test_model_allowlist_is_enforced(monkeypatch):
    monkeypatch.setenv("JEV_BASE_MODEL", "operator/allowed-model")
    assert validate_training_config({"trainer": "huggingface"})["base_model"] == "operator/allowed-model"
    with pytest.raises(ValueError, match="JEV_BASE_MODEL"):
        validate_training_config({"trainer": "huggingface", "base_model": "user/arbitrary-model"})


def test_cancellation_during_splitting_never_reports_success(tmp_path):
    source = write_jsonl(tmp_path / "keep.jsonl", records())
    checks = 0

    def cancelled():
        nonlocal checks
        checks += 1
        return checks > 10

    with pytest.raises(TrainingCancelled):
        train_and_evaluate(source, tmp_path / "run", {}, cancelled=cancelled)
    assert not (tmp_path / "run" / "training_report.json").exists()


def test_text_formats_and_heldout_splits(tmp_path):
    rows = [{"messages": [{"role": "user", "content": f"Question {i}"},
                          {"role": "assistant", "content": f"Answer {i}"}]} for i in range(10)]
    rows.extend({"instruction": f"Explain item {i}", "input": "A sample", "output": f"Result {i}"}
                for i in range(10))
    rows.extend({"messages": json.dumps([{"role": "user", "content": f"CSV question {i}"},
                                         {"role": "assistant", "content": f"CSV answer {i}"}])}
                for i in range(10))
    source = write_jsonl(tmp_path / "keep.jsonl", rows)
    result = train_and_evaluate(source, tmp_path / "run", {})
    assert sum(result["split_counts"].values()) == 30
    assert result["evaluation_tokens"]["test"] > 0


def test_training_uses_the_same_content_precedence_as_screening():
    row = {"text": "unscored distraction", "prompt": "another distraction", "response": "irrelevant",
           "messages": json.dumps([{"role": "assistant", "content": "Scored message"}])}
    assert _text(row) == "assistant: Scored message"
    assert _text({"text": "Selected text", "instruction": "ignored", "output": "ignored"}) == "Selected text"
    assert _text({"instruction": "Selected instruction", "output": "Selected output",
                  "prompt": "ignored", "response": "ignored"}) == "Selected instruction\nSelected output"


def test_existing_output_is_not_overwritten(tmp_path):
    source = write_jsonl(tmp_path / "keep.jsonl", records())
    output = tmp_path / "run"
    train_and_evaluate(source, output, {})
    original = (output / "training_report.json").read_bytes()
    with pytest.raises(ValueError, match="already exist"):
        train_and_evaluate(source, output, {})
    assert (output / "training_report.json").read_bytes() == original


def synthetic_base(tmp_path, chat_template=None):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    base = tmp_path / "synthetic-gpt2"
    words = ["[UNK]", "[EOS]", "Example", ":", "reliable", "data", "makes", "a", "useful", "language", "model", ".",
             "user", "assistant", "Question", "Answer", "<turn>", "</turn>"] + [str(i) for i in range(30)]
    vocabulary = {word: index for index, word in enumerate(words)}
    backend = Tokenizer(WordLevel(vocab=vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[EOS]")
    # Set explicitly in both cases: some transformers versions keep the last template on the class.
    tokenizer.chat_template = chat_template
    tokenizer.save_pretrained(base)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocabulary), n_positions=64, n_ctx=64,
                                     n_embd=16, n_layer=1, n_head=2, bos_token_id=1,
                                     eos_token_id=1, pad_token_id=1))
    model.save_pretrained(base, safe_serialization=True)
    return base


def huggingface_environment(monkeypatch, base):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    monkeypatch.setenv("JEV_BASE_MODEL", str(base))
    monkeypatch.setenv("JEV_MODEL_LOCAL_ONLY", "1")
    monkeypatch.setenv("JEV_TRAIN_DEVICE", "cpu")


def test_huggingface_real_lora_with_local_synthetic_fixture(tmp_path, monkeypatch):
    """Exercise actual forward/backward/evaluation, with no downloads or pretrained-quality claim."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    huggingface_environment(monkeypatch, synthetic_base(tmp_path))
    source = write_jsonl(tmp_path / "keep.jsonl", records(20))
    result = train_and_evaluate(source, tmp_path / "run", {
        "trainer": "huggingface", "max_steps": 1, "max_seq_length": 32, "batch_size": 2,
    })
    assert result["is_llm"] is True
    assert result["mode"] == "llm_lora"
    assert result["steps"] == 1
    assert result["training_record_visits"] == 2
    assert 0 < result["trainable_parameters"] < result["total_parameters"]
    assert result["device"] == "cpu"
    assert result["dtype"] == "float32"
    assert math.isfinite(result["baseline_loss"])
    assert math.isfinite(result["trained_loss"])
    assert result["evaluation_tokens"]["test"] > 0
    # Bare passages have no prompt, so nothing was masked and the report says so.
    assert result["masking"]["rows_with_masked_prompt"] == 0
    assert any("effectively full-text" in note for note in result["limitations"])
    assert result["lora"] == {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": "all-linear"}
    assert result["loss_history"][0]["lr"] > 0
    assert (tmp_path / "run" / "model" / "adapter_model.safetensors").is_file()
    assert (tmp_path / "run" / "model" / "tokenizer.json").is_file()


def conversations(count=20):
    return [{"id": str(i), "messages": [{"role": "user", "content": f"Question {i} {i + 1} {i + 2}"},
                                        {"role": "assistant", "content": f"Answer {i}"}]} for i in range(count)]


def test_answer_only_loss_masks_the_prompt_without_a_chat_template(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    huggingface_environment(monkeypatch, synthetic_base(tmp_path))
    source = write_jsonl(tmp_path / "keep.jsonl", conversations())
    result = train_and_evaluate(source, tmp_path / "run", {
        "trainer": "huggingface", "max_steps": 2, "max_seq_length": 32, "batch_size": 2, "lora_r": 4, "lora_alpha": 8,
    })
    assert result["chat_template"] is False
    assert result["loss_mask"] == "answer"
    assert result["masking"]["rows_with_masked_prompt"] == 4
    # "user : Question i j k assistant :" is masked; "Answer i [EOS]" is learned.
    assert result["masking"]["prompt_tokens_masked"] == 4 * 8
    assert result["masking"]["tokens_learned"] == 4 * 3
    assert result["metric_unit"].endswith("per answer token")
    assert result["lora"]["r"] == 4
    # The held-out count is answer tokens only: "Answer", "i" and EOS for each row.
    test_rows = read_jsonl(tmp_path / "run" / "test.jsonl")
    assert result["evaluation_tokens"]["test"] == 3 * len(test_rows)


def test_chat_template_is_used_when_the_tokenizer_has_one(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    template = ("{% for m in messages %}<turn> {{ m['role'] }} : {{ m['content'] }} </turn> {% endfor %}"
                "{% if add_generation_prompt %}<turn> assistant : {% endif %}")
    huggingface_environment(monkeypatch, synthetic_base(tmp_path, template))
    source = write_jsonl(tmp_path / "keep.jsonl", conversations())
    result = train_and_evaluate(source, tmp_path / "run", {
        "trainer": "huggingface", "max_steps": 1, "max_seq_length": 32, "batch_size": 2,
    })
    assert result["chat_template"] is True
    assert result["masking"]["rows_with_masked_prompt"] == 2
    # "<turn> user : Question i j k </turn> <turn> assistant :" is the prompt; the
    # whitespace pre-tokenizer cuts each marker into three pieces, so 17 tokens.
    assert result["masking"]["prompt_tokens_masked"] == 2 * 17
    # "Answer i </turn> [EOS]" is learned: 2 + 3 + 1 tokens.
    assert result["masking"]["tokens_learned"] == 2 * 6


def test_full_loss_mask_learns_every_token(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    huggingface_environment(monkeypatch, synthetic_base(tmp_path))
    source = write_jsonl(tmp_path / "keep.jsonl", conversations())
    result = train_and_evaluate(source, tmp_path / "run", {
        "trainer": "huggingface", "max_steps": 1, "max_seq_length": 32, "batch_size": 2, "loss_mask": "full",
    })
    assert result["masking"]["rows_with_masked_prompt"] == 0
    assert result["masking"]["tokens_learned"] == 2 * 11
    assert result["limitations"][0].startswith("Full-text")


@pytest.mark.parametrize("config", [{"lora_r": 0}, {"lora_alpha": 5000}, {"loss_mask": "prompt"}])
def test_lora_config_validation(config):
    with pytest.raises(ValueError):
        validate_training_config(config)
