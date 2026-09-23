"""Streaming exports for local SFT and operator-managed verl training hosts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

from .screening import normalize_record
from .training import (_conversation, _digest, _integer, _json_file, _prepare_splits,
                       _rows, validate_training_config)

TARGETS = ("sft", "verl-grpo", "verl-ppo")
DEFAULT_CHAT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prompt_group(row: dict[str, Any]) -> list[str]:
    turns = _conversation(normalize_record(row))
    if not turns or turns[-1]["role"] != "assistant" or len(turns) < 2:
        return []
    prompt = [{"role": turn["role"], "content": " ".join(turn["content"].split())}
              for turn in turns[:-1]]
    return ["prompt:" + _digest(json.dumps(prompt, ensure_ascii=False, sort_keys=True))]


def _reward(row: dict[str, Any], field: str) -> str:
    # Never infer a verifier target from JEV scores or a free-form response.
    from .rewards import MAX_REWARD_CHARACTERS
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Every RL record needs a nonempty string in the explicit reward field {field!r}.")
    if len(value) > MAX_REWARD_CHARACTERS:
        raise ValueError(f"RL reference answers must not exceed {MAX_REWARD_CHARACTERS:,} characters.")
    return value


def _rl_prompt(state: dict[str, Any]) -> list[dict[str, str]]:
    turns = _conversation(state)
    if not turns or len(turns) < 2 or turns[-1]["role"] != "assistant":
        raise ValueError("RL export requires a prompt and final assistant response; bare text is SFT-only.")
    prompt = turns[:-1]
    if not any(turn["role"] == "user" and turn["content"].strip() for turn in prompt):
        raise ValueError("RL prompts must contain a nonempty user turn.")
    if prompt[-1]["role"] not in {"user", "tool"}:
        raise ValueError("RL prompts must end with a user or tool turn before the reference answer.")
    return prompt


def _sft_row(row: dict[str, Any]) -> dict[str, Any]:
    state = row["_state"]
    turns = _conversation(state)
    value = {"messages": turns} if turns is not None else {"text": state["text"]}
    value["_split_group"] = row["_split_group"]
    if row.get("id") is not None:
        value["id"] = str(row["id"])
    return value


def export_training(input_path: Path, output_dir: Path, *, target: str,
                    base_model: str = DEFAULT_CHAT_MODEL, reward_field: str | None = None,
                    config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Prepare a relocatable bundle; do not download a model or start training.

    Export uses bounded JSONL reads, a disk-backed split index, and Parquet row
    groups of 256 records. The final directory appears only after a full export.
    """
    source, destination = Path(input_path).resolve(), Path(output_dir).resolve()
    if target not in TARGETS:
        raise ValueError("target must be sft, verl-grpo, or verl-ppo.")
    if not source.is_file():
        raise ValueError("The retained JSONL input does not exist.")
    if destination.exists():
        raise ValueError("Training bundle output already exists; choose a new directory.")
    if not isinstance(base_model, str) or not base_model.strip() or any(c in base_model for c in "\r\n\x00"):
        raise ValueError("base_model must be a model ID or local model path.")
    local_model = Path(base_model).expanduser()
    if local_model.is_dir():
        base_model = str(local_model.resolve())
    elif base_model.startswith(("./", "../", "~")):
        raise ValueError("The explicitly relative local base_model directory does not exist.")
    options = dict(config or {})
    options.setdefault("learning_rate", 0.000001 if target != "sft" else 0.0002)
    is_rl = target != "sft"
    batch_size = _integer(options, "batch_size", 4, 1, 4096 if is_rl else 32)
    # The workbench caps its local mini-batches at 32. verl's global prompt
    # batch can be larger and is validated separately from that local runner.
    validated = validate_training_config({**options, "trainer": "demo", "batch_size": min(batch_size, 32)})
    validated["batch_size"] = batch_size
    validated.pop("trainer")
    validated.update({
        "n_gpus": _integer(options, "n_gpus", 1, 1, 64),
        "rollout_n": _integer(options, "rollout_n", 4, 1, 64),
        "max_prompt_length": _integer(options, "max_prompt_length", 512, 16, 32768),
        "max_response_length": _integer(options, "max_response_length", 512, 16, 32768),
    })
    if is_rl and (not isinstance(reward_field, str) or not reward_field.strip()):
        raise ValueError("RL export requires --reward-field naming an explicit, verified answer field.")
    if is_rl and reward_field in {"text", "_state", "_split_group"}:
        raise ValueError("The reward field is reserved for training preparation; copy verified answers to ground_truth or another metadata field.")
    if target == "verl-grpo" and validated["rollout_n"] < 2:
        raise ValueError("GRPO requires at least 2 rollout responses per prompt.")
    if validated["batch_size"] % validated["n_gpus"]:
        raise ValueError("Global batch_size must be divisible by n_gpus; use at least one record per GPU.")
    if is_rl:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ValueError("Parquet export requires: pip install 'jev-dataops[export]'.") from exc
    screening_report = source.parent / "data_report.json"
    if screening_report.is_file():
        with screening_report.open(encoding="utf-8") as stream:
            screening = json.load(stream)
        if not isinstance(screening, dict) or not screening.get("complete") or not screening.get("training_ready", True):
            raise ValueError("Screening is incomplete or not training-ready; resolve the screening report first.")
    # Fail before splitting if any reference is missing or any prompt is unsuitable.
    if is_rl:
        for row in _rows(source):
            _rl_prompt(normalize_record(row))
            _reward(row, reward_field)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".jev-export-", dir=destination.parent) as temporary:
        root = Path(temporary)
        bundle = root / "bundle"
        bundle.mkdir()
        splits = root / "splits"
        splits.mkdir()
        info = _prepare_splits(source, splits, validated, None, None, extra_group_keys=_prompt_group)
        if (is_rl or validated["n_gpus"] > 1) and info["split_counts"]["train"] < validated["batch_size"]:
            raise ValueError("Training split is smaller than global batch_size; lower --batch-size or add data.")
        info["strategy"] += " plus normalized prompt identity"
        data = bundle / "data"
        data.mkdir()
        paths = {}
        for split in ("train", "validation", "test"):
            suffix = "parquet" if is_rl else "jsonl"
            paths[split] = f"data/{split}.{suffix}"
            path = bundle / paths[split]
            if not is_rl:
                with path.open("w", encoding="utf-8") as stream:
                    for row in _rows(splits / f"{split}.jsonl"):
                        stream.write(json.dumps(_sft_row(row), ensure_ascii=False, allow_nan=False) + "\n")
                continue
            schema = pa.schema([
                ("data_source", pa.string()),
                ("prompt", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
                ("ability", pa.string()),
                ("reward_model", pa.struct([("style", pa.string()), ("ground_truth", pa.string())])),
                ("extra_info", pa.struct([("split", pa.string()), ("index", pa.int64()),
                                           ("id", pa.string()), ("split_group", pa.string())])),
            ])
            with pq.ParquetWriter(path, schema, compression="snappy") as writer:
                batch = []
                for index, row in enumerate(_rows(splits / f"{split}.jsonl")):
                    batch.append({"data_source": "jev_dataops", "prompt": _rl_prompt(row["_state"]),
                                  "ability": "verified_answer", "reward_model": {
                                      "style": "rule", "ground_truth": _reward(row, reward_field)},
                                  "extra_info": {"split": split, "index": index,
                                      "id": str(row.get("id", index)), "split_group": row["_split_group"]}})
                    if len(batch) >= 256:
                        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                        batch.clear()
                if batch:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
        manifest = {
            "schema_version": 1, "status": "prepared", "target": target,
            "base_model": base_model, "data": paths, **info, "config": validated,
            "source": {"name": source.name, "sha256": file_sha256(source)},
            "reward": {"mode": "exact_match", "field": reward_field} if is_rl else None,
            "limitations": ["Export prepares data and launch configuration; no model has been trained.",
                            "Validation is for training monitoring; test data must remain reserved for final evaluation.",
                            "Exact prompt/content and declared groups are isolated; semantic leakage still requires review."],
        }
        if screening_report.is_file():
            manifest["source"]["screening_report_sha256"] = file_sha256(screening_report)
        if is_rl:
            manifest["limitations"].append("Exact-match reward is suitable only for verified, closed-answer tasks, not open-ended domain quality.")
        from .training_launch import write_launch_files
        write_launch_files(bundle, manifest)
        manifest["checksums"] = {path.relative_to(bundle).as_posix(): file_sha256(path)
                                 for path in sorted(bundle.rglob("*")) if path.is_file()}
        _json_file(bundle / "manifest.json", manifest)
        bundle.rename(destination)
    return manifest
