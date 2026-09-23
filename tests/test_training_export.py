import json
from pathlib import Path
import subprocess
import sys

import pytest

from jev_dataops.training_export import export_training, file_sha256


def source(tmp_path, rows=None):
    path = tmp_path / "keep.jsonl"
    rows = rows if rows is not None else [
        {"id": f"example-{i}", "group_id": f"question-{i}",
         "instruction": f"Reply with the integer only. What is {i} + 7?", "output": str(i + 7),
         "ground_truth": str(i + 7)} for i in range(24)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def read_sft(bundle):
    return {split: [json.loads(line) for line in (bundle / "data" / f"{split}.jsonl").read_text().splitlines()]
            for split in ("train", "validation", "test")}


def test_sft_normalized_conversations_and_checksums(tmp_path):
    bundle = tmp_path / "bundle"
    manifest = export_training(source(tmp_path), bundle, target="sft")
    assert manifest["status"] == "prepared"
    assert manifest["split_counts"] == {"train": 18, "validation": 3, "test": 3}
    assert manifest["config"]["learning_rate"] == 0.0002
    rows = [row for split in read_sft(bundle).values() for row in split]
    assert all(row["messages"][-1]["role"] == "assistant" for row in rows)
    assert all("ground_truth" not in row for row in rows)
    for relative, digest in manifest["checksums"].items():
        assert file_sha256(bundle / relative) == digest
    assert not (bundle / "model_report.json").exists()
    assert not list(tmp_path.glob(".jev-export-*"))


def test_prompt_variants_and_transitive_groups_never_leak(tmp_path):
    path = source(tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows += [{**rows[0], "id": "variant", "output": "another answer", "group_id": "different"},
             {**rows[1], "id": "linked", "group_id": "different"}]
    source(tmp_path, rows)
    bundle = tmp_path / "bundle"
    export_training(path, bundle, target="sft")
    locations = {row["id"]: split for split, values in read_sft(bundle).items() for row in values}
    assert locations["example-0"] == locations["variant"] == locations["linked"] == locations["example-1"]
    groups = [{row["_split_group"] for row in values} for values in read_sft(bundle).values()]
    assert all(not groups[a] & groups[b] for a in range(3) for b in range(a))


@pytest.mark.parametrize("target", ["verl-grpo", "verl-ppo"])
def test_verl_parquet_has_prompt_only_and_explicit_reference(tmp_path, target):
    pq = pytest.importorskip("pyarrow.parquet")
    bundle = tmp_path / "bundle"
    manifest = export_training(source(tmp_path), bundle, target=target, reward_field="ground_truth")
    assert manifest["config"]["learning_rate"] == 0.000001
    for split, relative in manifest["data"].items():
        rows = pq.read_table(bundle / relative).to_pylist()
        assert len(rows) == manifest["split_counts"][split]
        for row in rows:
            assert row["prompt"][-1]["role"] == "user"
            assert len(row["prompt"]) == 1
            assert row["reward_model"]["style"] == "rule"
            assert isinstance(row["reward_model"]["ground_truth"], str)
            assert row["extra_info"]["split"] == split
    assert (bundle / "rewards.py").is_file()


def test_rl_missing_verifier_never_silently_uses_response(tmp_path):
    path = source(tmp_path)
    with pytest.raises(ValueError, match="reward-field"):
        export_training(path, tmp_path / "bundle", target="verl-grpo")
    pytest.importorskip("pyarrow")
    with pytest.raises(ValueError, match="nonempty string"):
        export_training(path, tmp_path / "bundle", target="verl-ppo", reward_field="missing")
    assert not (tmp_path / "bundle").exists()
    with pytest.raises(ValueError, match="reserved"):
        export_training(path, tmp_path / "bundle", target="verl-grpo", reward_field="text")


def test_existing_output_and_incomplete_screening_rejected(tmp_path):
    path = source(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "user-file.txt").write_text("preserve")
    with pytest.raises(ValueError, match="already exists"):
        export_training(path, output, target="sft")
    assert (output / "user-file.txt").read_text() == "preserve"
    (tmp_path / "data_report.json").write_text(json.dumps({"complete": True, "training_ready": False}))
    with pytest.raises(ValueError, match="not training-ready"):
        export_training(path, tmp_path / "new", target="sft")


def test_split_failure_leaves_no_partial_bundle(tmp_path):
    path = source(tmp_path, [{"text": f"Useful sample {i}.", "group_id": "one"} for i in range(24)])
    with pytest.raises(ValueError, match="6 independent"):
        export_training(path, tmp_path / "bundle", target="sft")
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".jev-export-*"))


def test_export_and_default_dry_run_cli(tmp_path):
    path = source(tmp_path)
    bundle = tmp_path / "bundle"
    export = subprocess.run([sys.executable, "-m", "jev_dataops", "export-training", "--input", str(path),
                             "--output", str(bundle), "--target", "sft"], capture_output=True, text=True)
    assert export.returncode == 0, export.stderr
    assert json.loads(export.stdout)["status"] == "prepared"
    dry = subprocess.run([sys.executable, "-m", "jev_dataops", "launch-training", "--bundle", str(bundle)],
                         capture_output=True, text=True)
    assert dry.returncode == 0, dry.stderr
    assert not (bundle / "results").exists()


def test_local_model_path_survives_bundle_working_directory(tmp_path, monkeypatch):
    (tmp_path / "model").mkdir()
    path = source(tmp_path)
    monkeypatch.chdir(tmp_path)
    manifest = export_training(path, tmp_path / "bundle", target="sft", base_model="./model")
    assert manifest["base_model"] == str(tmp_path / "model")
    with pytest.raises(ValueError, match="does not exist"):
        export_training(path, tmp_path / "missing-model-bundle", target="sft", base_model="./missing-model")


def test_sft_and_rl_reuse_the_same_split_assignment(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    path = source(tmp_path)
    sft, rl = tmp_path / "sft", tmp_path / "rl"
    export_training(path, sft, target="sft")
    manifest = export_training(path, rl, target="verl-grpo", reward_field="ground_truth")
    for split, rows in read_sft(sft).items():
        rl_rows = pq.read_table(rl / manifest["data"][split]).to_pylist()
        assert {row["id"] for row in rows} == {row["extra_info"]["id"] for row in rl_rows}


@pytest.mark.parametrize("target", ["sft", "verl-grpo", "verl-ppo"])
def test_gpu_count_does_not_duplicate_or_resplit_uploaded_data(tmp_path, target):
    if target != "sft":
        pytest.importorskip("pyarrow")
    path = source(tmp_path)
    exports = []
    for count in (1, 2, 4, 8):
        bundle = tmp_path / f"{target}-{count}"
        manifest = export_training(path, bundle, target=target,
                                   reward_field="ground_truth" if target != "sft" else None,
                                   config={"n_gpus": count, "batch_size": 8})
        assert manifest["config"]["n_gpus"] == count
        assert len(list((bundle / "data").iterdir())) == 3
        exports.append({split: file_sha256(bundle / relative) for split, relative in manifest["data"].items()})
    assert all(item == exports[0] for item in exports)


def test_multi_gpu_requires_complete_global_batch(tmp_path):
    path = source(tmp_path)
    with pytest.raises(ValueError, match="divisible"):
        export_training(path, tmp_path / "invalid-batch", target="sft", config={"n_gpus": 8, "batch_size": 4})
    with pytest.raises(ValueError, match="smaller than global batch_size"):
        export_training(path, tmp_path / "too-few-rows", target="sft", config={"n_gpus": 8, "batch_size": 32})
    assert not (tmp_path / "too-few-rows").exists()


def test_verl_accepts_global_batch_larger_than_local_sft_limit(tmp_path):
    pytest.importorskip("pyarrow")
    rows = [{"instruction": f"What is {i} plus 1?", "output": str(i + 1), "ground_truth": str(i + 1)} for i in range(128)]
    manifest = export_training(source(tmp_path, rows), tmp_path / "large-batch", target="verl-grpo",
                               reward_field="ground_truth", config={"n_gpus": 8, "batch_size": 64})
    assert manifest["config"]["batch_size"] == 64
    assert manifest["split_counts"]["train"] >= 64
