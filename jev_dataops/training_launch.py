"""Validated, synchronous launch of prepared SFT / pinned verl training bundles.

Dry runs verify local artifacts and construct argument vectors without importing a
GPU library, downloading a model, executing bundle code, or creating output files.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
from typing import Any


VERL_VERSION = "0.9.1"
TARGETS = {"sft", "verl-grpo", "verl-ppo"}
SPLITS = ("train", "validation", "test")
MANIFEST_LIMIT = 1024 * 1024
_PROBE_MARKER = "JEV_PREFLIGHT_JSON="


def _safe_string(value: Any, name: str, limit: int = 4096) -> str:
    if (not isinstance(value, str) or not value.strip() or len(value) > limit
            or any(char in value for char in ("\x00", "\n", "\r", "${"))):
        raise ValueError(f"{name} must be a nonempty literal string without control characters or interpolation.")
    return value


def _integer(config: dict, key: str, default: int, low: int, high: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{key} must be an integer between {low} and {high}.")
    return value


def _config(manifest: dict) -> dict:
    source = manifest.get("config", {})
    if not isinstance(source, dict):
        raise ValueError("manifest.config must be an object.")
    config = {
        "seed": _integer(source, "seed", 42, 0, 2**31 - 1),
        "epochs": _integer(source, "epochs", 1, 1, 20),
        "max_steps": _integer(source, "max_steps", 20, 1, 10000),
        "batch_size": _integer(source, "batch_size", 4, 1, 32 if manifest["target"] == "sft" else 4096),
        "max_seq_length": _integer(source, "max_seq_length", 1024, 16, 4096),
        "n_gpus": _integer(source, "n_gpus", 1, 1, 64),
        "rollout_n": _integer(source, "rollout_n", 4, 1, 64),
        "max_prompt_length": _integer(source, "max_prompt_length", 512, 16, 32768),
        "max_response_length": _integer(source, "max_response_length", 512, 16, 32768),
        "lora_r": _integer(source, "lora_r", 8, 1, 256),
        "lora_alpha": _integer(source, "lora_alpha", 16, 1, 1024),
        "loss_mask": source.get("loss_mask", "answer"),
    }
    rate = source.get("learning_rate", 0.0002 if manifest["target"] == "sft" else 0.000001)
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 1e-7 <= rate <= 0.01:
        raise ValueError("learning_rate must be a finite number between 1e-7 and 0.01.")
    config["learning_rate"] = float(rate)
    if config["loss_mask"] not in ("answer", "full"):
        raise ValueError("loss_mask must be answer or full.")
    if config["batch_size"] % config["n_gpus"]:
        raise ValueError("Global batch_size must be divisible by n_gpus.")
    if manifest["target"] != "sft" or config["n_gpus"] > 1:
        if manifest["split_counts"]["train"] < config["batch_size"]:
            raise ValueError("Distributed training requires at least batch_size training rows (drop_last=True).")
    if manifest["target"] != "sft":
        if manifest["target"] == "verl-grpo" and config["rollout_n"] < 2:
            raise ValueError("GRPO requires rollout_n >= 2 to compare responses within a prompt group.")
    return config


def _relative_path(value: Any) -> PurePosixPath:
    value = _safe_string(value, "bundle relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or "\\" in value or ":" in value or any(part in (".", "..", "") for part in value.split("/")):
        raise ValueError("Bundle paths must be relative and cannot contain traversal or drive prefixes.")
    return path


def _contained_file(directory: Path, value: Any) -> Path:
    relative = _relative_path(value)
    path = directory.joinpath(*relative.parts)
    for candidate in (path, *path.parents):
        if candidate == directory:
            break
        if candidate.is_symlink():
            raise ValueError("Symlinked bundle artifacts are not allowed.")
    if directory not in path.resolve().parents or not path.is_file():
        raise ValueError(f"Bundle artifact is missing or outside the bundle: {relative}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_metadata(manifest: Any) -> dict:
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("A version-1 training bundle manifest is required.")
    if manifest.get("status") != "prepared" or manifest.get("target") not in TARGETS:
        raise ValueError("Manifest must describe a prepared sft, verl-grpo, or verl-ppo bundle.")
    _safe_string(manifest.get("base_model"), "base_model")
    paths = manifest.get("data")
    counts = manifest.get("split_counts")
    if not isinstance(paths, dict) or set(paths) != set(SPLITS):
        raise ValueError("Manifest must identify train, validation, and test data separately.")
    if not isinstance(counts, dict) or any(type(counts.get(split)) is not int or counts[split] < 1 for split in SPLITS):
        raise ValueError("All split counts must be positive integers.")
    expected_suffix = ".jsonl" if manifest["target"] == "sft" else ".parquet"
    normalized_paths = [_relative_path(paths[split]).as_posix() for split in SPLITS]
    if len(set(normalized_paths)) != 3 or any(not path.endswith(expected_suffix) for path in normalized_paths):
        raise ValueError(f"Three distinct {expected_suffix} split files are required.")
    if manifest["target"] != "sft":
        reward = manifest.get("reward")
        if not isinstance(reward, dict) or reward.get("mode") != "exact_match":
            raise ValueError("RL bundles require the built-in exact_match reward over an explicit reference field.")
        _safe_string(reward.get("field"), "reward source field", 200)
    _config(manifest)
    return manifest


def load_bundle(directory: Path) -> dict:
    """Read bounded metadata and verify every enumerated checksum before any launch."""
    directory = Path(directory).resolve()
    path = _contained_file(directory, "manifest.json")
    with path.open("rb") as stream:
        raw = stream.read(MANIFEST_LIMIT + 1)
    if len(raw) > MANIFEST_LIMIT:
        raise ValueError("Manifest exceeds the 1 MiB limit.")
    try:
        manifest = _validate_metadata(json.loads(raw))
    except (UnicodeError, RecursionError, TypeError) as exc:
        raise ValueError("Invalid training bundle manifest.") from exc
    checksums = manifest.get("checksums")
    required = set(manifest["data"].values()) | {"launcher.json"}
    if manifest["target"] != "sft":
        required.add("rewards.py")
    if not isinstance(checksums, dict) or not required.issubset(checksums):
        raise ValueError("Manifest checksums must include all data and launch artifacts.")
    for relative, expected in checksums.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", expected):
            raise ValueError("Every artifact checksum must be a SHA-256 hex string.")
        path = _contained_file(directory, relative)
        if _sha256(path) != expected.lower():
            raise ValueError(f"Bundle checksum mismatch: {relative}")
    if manifest["target"] != "sft" and (directory / "rewards.py").read_bytes() != _reward_source():
        raise ValueError("Bundle reward code differs from the installed trusted exact-match implementation; re-export it.")
    return manifest


def _reward_source() -> bytes:
    return Path(__file__).with_name("rewards.py").read_bytes()


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _hydra(key: str, value: Any) -> str:
    if isinstance(value, str):
        _safe_string(value, key)
    # Quotes must reach Hydra itself, not be consumed by a shell. OmegaConf
    # interpolation is rejected separately, even inside quoted strings.
    return key + "=" + json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _rl_overrides(manifest: dict, output: str, python: str, model: str | None = None,
                  bundle: Path | None = None) -> list[str]:
    config = _config(manifest)
    grpo = manifest["target"] == "verl-grpo"
    length = config["max_prompt_length"] + config["max_response_length"]
    # verl's drop_last loader may exhaust the epoch before the requested step
    # cap. Make its final-step checkpoint/validation branch reachable either way.
    effective_steps = min(config["max_steps"], config["epochs"] * (manifest["split_counts"]["train"] // config["batch_size"]))
    settings = {
        "algorithm.adv_estimator": "grpo" if grpo else "gae",
        "algorithm.use_kl_in_reward": False,
        "data.train_files": str(bundle / manifest["data"]["train"]) if bundle else manifest["data"]["train"],
        "data.val_files": str(bundle / manifest["data"]["validation"]) if bundle else manifest["data"]["validation"],
        "data.train_batch_size": config["batch_size"],
        "data.val_batch_size": config["batch_size"],
        "data.max_prompt_length": config["max_prompt_length"],
        "data.max_response_length": config["max_response_length"],
        "data.truncation": "error",
        "data.filter_overlong_prompts": False,
        "data.trust_remote_code": False,
        "data.seed": config["seed"],
        "data.dataloader_num_workers": 0,
        "actor_rollout_ref.model.path": model or manifest["base_model"],
        "actor_rollout_ref.model.trust_remote_code": False,
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.optim.lr": config["learning_rate"],
        "actor_rollout_ref.actor.ppo_mini_batch_size": config["batch_size"],
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.use_dynamic_bsz": False,
        "actor_rollout_ref.actor.use_kl_loss": True,
        "actor_rollout_ref.actor.kl_loss_coef": 0.001,
        "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
        "actor_rollout_ref.actor.fsdp_config.seed": config["seed"],
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.4,
        "actor_rollout_ref.rollout.n": config["rollout_n"] if grpo else 1,
        "actor_rollout_ref.rollout.seed": config["seed"],
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.enforce_eager": True,
        "actor_rollout_ref.rollout.load_format": "safetensors",
        "actor_rollout_ref.rollout.max_model_len": length,
        "actor_rollout_ref.rollout.max_num_batched_tokens": max(8192, length),
        "actor_rollout_ref.rollout.val_kwargs.do_sample": False,
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 0.0,
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
        "critic.enable": not grpo,
        "reward.custom_reward_function.path": str(bundle / "rewards.py") if bundle else "rewards.py",
        "reward.custom_reward_function.name": "compute_score",
        "reward.reward_manager.name": "naive",
        "reward.num_workers": 1,
        "trainer.use_v1": True,
        "trainer.nnodes": 1,
        "trainer.n_gpus_per_node": config["n_gpus"],
        "trainer.total_epochs": config["epochs"],
        "trainer.total_training_steps": effective_steps,
        "trainer.logger": ["console", "file"],
        "trainer.project_name": "jev-dataops",
        "trainer.experiment_name": manifest["target"],
        "trainer.default_local_dir": output,
        "trainer.default_hdfs_dir": None,
        "trainer.resume_mode": "disable",
        "trainer.val_before_train": True,
        "trainer.test_freq": effective_steps,
        "trainer.save_freq": effective_steps,
        "trainer.max_actor_ckpt_to_keep": 1,
        "trainer.max_critic_ckpt_to_keep": 1,
        "ray_kwargs.ray_init.runtime_env.py_executable": shlex.join([python, "-I"]),
        "+ray_kwargs.ray_init.address": "local",
        "+ray_kwargs.ray_init.include_dashboard": False,
        "hydra.job.chdir": False,
    }
    if not grpo:
        settings.update({
            "critic.model.path": model or manifest["base_model"],
            "critic.model.trust_remote_code": False,
            "critic.optim.lr": config["learning_rate"],
            "critic.ppo_micro_batch_size_per_gpu": 1,
            "critic.forward_micro_batch_size_per_gpu": 1,
            "critic.fsdp.seed": config["seed"],
        })
    return [_hydra(key, value) for key, value in settings.items()]


def write_launch_files(directory: Path, manifest: dict) -> dict[str, str]:
    """Write portable declarative launch artifacts; caller subsequently hashes them."""
    directory = Path(directory).resolve()
    _validate_metadata(manifest)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"config": "launcher.json"}
    if manifest["target"] != "sft":
        paths["reward"] = "rewards.py"
    if any((directory / relative).exists() for relative in paths.values()):
        raise ValueError("Launch artifacts already exist; export into a fresh directory.")
    distributed_sft = manifest["target"] == "sft" and _config(manifest)["n_gpus"] > 1
    recipe = {
        "schema_version": 1, "status": "prepared", "target": manifest["target"],
        "entrypoint": ("jev_dataops.training_distributed" if distributed_sft else "jev_dataops.training_launch") if manifest["target"] == "sft" else "verl.trainer.main_ppo",
        "config": _config(manifest), "data": manifest["data"],
        "execution": "Use jev-dataops launch-training; default is dry-run. Explicit execution stays synchronous on your own host.",
        "evaluation": "SFT compares fixed held-out test losses; verl periodically evaluates only validation, never test.",
        "gpu_validation": "No GPU execution was performed when preparing this bundle.",
        "distribution": "single-node DDP; batch_size is global" if distributed_sft else "single-process SFT" if manifest["target"] == "sft" else "single-node verl FSDP; batch_size is global",
    }
    if manifest["target"] != "sft":
        recipe.update({"verl_version": VERL_VERSION,
                       "overrides": _rl_overrides(manifest, "results", "python"),
                       "environment": "Official verl v0.9.1 frozen uv environment: Linux/Python 3.12, FSDP + vLLM.",
                       "reward": manifest["reward"]})
        (directory / "rewards.py").write_bytes(_reward_source())
    _write_json(directory / "launcher.json", recipe)
    return paths


_ENV_PROBE = r'''
import importlib.metadata, importlib.util, json, os, platform, sys
target, expected, n_gpus = sys.argv[1], sys.argv[2], int(sys.argv[3])
modules = ("torch", "transformers", "peft", "jev_dataops") if target == "sft" else ("torch", "transformers", "verl", "vllm")
missing = [name for name in modules if importlib.util.find_spec(name) is None]
if missing:
    raise RuntimeError("Missing training dependencies: " + ", ".join(missing))
result = {"python": sys.executable, "python_version": platform.python_version(), "platform": sys.platform}
if target == "sft" and n_gpus > 1:
    import torch
    import torch.distributed as dist
    if sys.platform != "linux":
        raise RuntimeError("Multi-GPU SFT requires Linux with CUDA/NCCL.")
    if os.environ.get("JEV_TRAIN_DEVICE", "auto") not in ("auto", "cuda"):
        raise RuntimeError("Multi-GPU SFT requires JEV_TRAIN_DEVICE=auto or cuda.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < n_gpus:
        raise RuntimeError("Requested CUDA devices are unavailable for multi-GPU SFT.")
    if not dist.is_available() or not dist.is_nccl_available():
        raise RuntimeError("Multi-GPU SFT requires the PyTorch NCCL backend.")
    result.update(cuda_devices=torch.cuda.device_count(), torch=torch.__version__, backend="nccl", world_size=n_gpus)
if target != "sft":
    if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("The pinned verl adapter requires Linux and Python 3.12.")
    version = importlib.metadata.version("verl")
    if version != expected:
        raise RuntimeError("Expected verl " + expected + "; found " + version)
    for package, required in (("transformers", "5.9.0"), ("vllm", "0.24.0")):
        actual = importlib.metadata.version(package)
        if actual != required:
            raise RuntimeError("Expected " + package + " " + required + "; found " + actual)
    import torch
    if torch.__version__.split("+")[0] != "2.11.0":
        raise RuntimeError("Expected the official torch 2.11.0 verl environment.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < n_gpus:
        raise RuntimeError("Requested CUDA devices are unavailable.")
    result.update(verl=version, cuda_devices=torch.cuda.device_count(), torch=torch.__version__)
print("JEV_PREFLIGHT_JSON=" + json.dumps(result))
'''


_MODEL_PROBE = r'''
import json, os, pathlib, sys
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
model = sys.argv[1]
path = pathlib.Path(model).expanduser()
if not path.is_dir():
    path = pathlib.Path(snapshot_download(model, local_files_only=os.environ.get("JEV_MODEL_LOCAL_ONLY", "0") == "1",
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.jinja"]))
path = path.resolve()
if not any(path.glob("*.safetensors")):
    raise RuntimeError("The RL base model must provide local SafeTensors weights.")
tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=False, local_files_only=True)
if not tokenizer.chat_template:
    raise RuntimeError("verl chat prompts require a base model tokenizer with a chat_template; select a compatible instruct model.")
print("JEV_PREFLIGHT_JSON=" + json.dumps({"resolved_base_model": str(path)}))
'''


def _interpreter(value: str | None) -> str:
    requested = _safe_string(value or sys.executable, "python interpreter")
    found = shutil.which(requested)
    if found is None or not Path(found).is_file():
        raise ValueError("The selected Python interpreter does not exist or is not executable.")
    # Preserve a venv symlink: resolving it would silently launch the system Python.
    return os.path.abspath(found)


def _probe(python: str, source: str, args: list[str], cwd: Path, env: dict, timeout: int = 120) -> dict:
    result = subprocess.run([python, "-I", "-c", source, *args], cwd=cwd, env=env,
                            capture_output=True, text=True, check=False, timeout=timeout, shell=False)
    if result.returncode:
        detail = (result.stderr or result.stdout)[-3000:]
        raise RuntimeError("Training preflight failed: " + detail)
    line = next((line for line in reversed(result.stdout.splitlines()) if line.startswith(_PROBE_MARKER)), None)
    if line is None:
        raise RuntimeError("Training preflight did not return a valid environment report.")
    return json.loads(line[len(_PROBE_MARKER):])


def launch_training(bundle: Path, *, dry_run: bool = True, python: str | None = None,
                    output: Path | None = None) -> dict:
    """Validate a bundle; only explicit dry_run=False starts a local foreground job."""
    directory = Path(bundle).resolve()
    manifest = load_bundle(directory)
    interpreter = _interpreter(python)
    destination = Path(output).expanduser().resolve() if output is not None else directory / "results"
    _safe_string(str(destination), "output directory")
    if destination.exists():
        raise ValueError("Training output already exists; choose a fresh output directory.")
    config = _config(manifest)
    distributed_sft = manifest["target"] == "sft" and config["n_gpus"] > 1
    if distributed_sft:
        command = [interpreter, "-I", "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--local-addr=127.0.0.1",
                   f"--nproc-per-node={config['n_gpus']}", "--max-restarts=0", "--no-python",
                   interpreter, "-I", "-m", "jev_dataops.training_distributed",
                   "--bundle", str(directory), "--output", str(destination), "--backend", "nccl"]
    elif manifest["target"] == "sft":
        command = [interpreter, "-I", "-m", "jev_dataops.training_launch", "_sft",
                   "--bundle", str(directory), "--output", str(destination)]
    else:
        command = [interpreter, "-I", "-m", "verl.trainer.main_ppo",
                   *_rl_overrides(manifest, str(destination / "checkpoints"), interpreter, bundle=directory)]
    plan = {
        "schema_version": 1, "status": "prepared", "dry_run": bool(dry_run),
        "target": manifest["target"], "bundle": str(directory), "output": str(destination),
        "command": command, "cwd": str(directory), "shell": False,
        "n_gpus": config["n_gpus"], "global_batch_size": config["batch_size"],
        "distribution": "single_node_ddp" if distributed_sft else "single_process" if manifest["target"] == "sft" else "single_node_verl",
        "test_split_used_by_training": False,
        "evaluation": "held-out baseline/final test NLL" if manifest["target"] == "sft" else "baseline and periodic validation reward; final test remains reserved",
        "gpu_validated": False,
    }
    if dry_run:
        return plan
    env = os.environ.copy()
    env["RAY_ADDRESS"] = "local"
    # No hosted tracking or provider credentials are needed by these local jobs.
    for name in ("OPENROUTER_API_KEY", "TYPESAFE_API_KEY", "JEV_API_TOKEN", "WANDB_API_KEY"):
        env.pop(name, None)
    env["WANDB_MODE"] = "disabled"
    env["VERL_FILE_LOGGER_PATH"] = str(destination / "metrics.jsonl")
    preflight = _probe(interpreter, _ENV_PROBE, [manifest["target"], VERL_VERSION, str(config["n_gpus"])], directory, env)
    if manifest["target"] != "sft":
        preflight.update(_probe(interpreter, _MODEL_PROBE, [manifest["base_model"]], directory, env, timeout=3600))
        command = [interpreter, "-I", "-m", "verl.trainer.main_ppo",
                   *_rl_overrides(manifest, str(destination / "checkpoints"), interpreter, preflight["resolved_base_model"], directory)]
        plan["command"] = command
    # Recheck checksums after potentially long model/environment preparation.
    load_bundle(directory)
    destination.mkdir(parents=True, exist_ok=False)
    plan.update(status="launched", dry_run=False, preflight=preflight,
                started_at=datetime.now(timezone.utc).isoformat(), log="training.log")
    _write_json(destination / "launch_started.json", plan)
    status, returncode = "failed", None
    try:
        with (destination / "training.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       shell=False, start_new_session=True)
            try:
                returncode = process.wait()
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=20)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                status = "cancelled"
                raise
        status = "completed" if returncode == 0 else "failed"
        if returncode == 0 and manifest["target"] == "sft" and not (destination / "training_report.json").is_file():
            status = "failed"
        result = {**plan, "status": status, "returncode": returncode,
                  "finished_at": datetime.now(timezone.utc).isoformat(),
                  "gpu_validated": (manifest["target"] != "sft" or distributed_sft) and status == "completed",
                  "result_scope": "Local subprocess exit status; consult actual training reports/checkpoints and logs. No test reward is fabricated."}
        _write_json(destination / "launch_report.json", result)
        return result
    except BaseException:
        if not (destination / "launch_report.json").exists():
            _write_json(destination / "launch_report.json", {**plan, "status": status, "returncode": returncode,
                        "finished_at": datetime.now(timezone.utc).isoformat()})
        raise


def prepare_sft_bundle(bundle: Path, output: Path) -> tuple[dict, dict, dict]:
    """Normalize fixed splits once, with a disk-backed cross-split group check."""
    from .screening import normalize_record
    from .training import _rows, _text, validate_training_config

    directory, destination = Path(bundle).resolve(), Path(output).resolve()
    manifest = load_bundle(directory)
    if manifest["target"] != "sft":
        raise ValueError("This entrypoint accepts only an SFT bundle.")
    destination.mkdir(parents=True, exist_ok=True)
    protected = [destination / name for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "model", "training_report.json", "model_report.json")]
    if any(path.exists() for path in protected):
        raise ValueError("SFT output artifacts already exist.")
    config = validate_training_config({**manifest.get("config", {}), "trainer": "demo"})
    config.update(trainer="huggingface", base_model=manifest["base_model"])
    config["n_gpus"] = _config(manifest)["n_gpus"]
    counts = {split: 0 for split in SPLITS}
    index = destination / "group_check.sqlite3"
    db = sqlite3.connect(index)
    try:
        db.execute("PRAGMA cache_size=-4096")
        db.execute("CREATE TABLE groups (identity TEXT PRIMARY KEY, split TEXT NOT NULL)")
        for split in SPLITS:
            with (destination / f"{split}.jsonl").open("x", encoding="utf-8") as stream:
                for row in _rows(_contained_file(directory, manifest["data"][split])):
                    group = _safe_string(row.get("_split_group"), "_split_group")
                    existing = db.execute("SELECT split FROM groups WHERE identity=?", (group,)).fetchone()
                    if existing and existing[0] != split:
                        raise ValueError("An exported group appears in multiple splits.")
                    db.execute("INSERT OR IGNORE INTO groups VALUES (?,?)", (group, split))
                    normalized = {"_state": normalize_record(row), "text": _text(row), "_split_group": group}
                    if "id" in row:
                        normalized["id"] = row["id"]
                    stream.write(json.dumps(normalized, ensure_ascii=False, allow_nan=False) + "\n")
                    counts[split] += 1
            if counts[split] != manifest["split_counts"][split]:
                raise ValueError(f"Manifest record count disagrees with the {split} split.")
        db.commit()
    finally:
        db.close()
        index.unlink(missing_ok=True)
    return manifest, config, counts


def write_sft_report(bundle: Path, output: Path, manifest: dict, config: dict, counts: dict, report: dict) -> dict:
    """Write final metrics only after at least one real optimizer update."""
    from .training import _json_file, _perplexity

    directory, destination = Path(bundle).resolve(), Path(output).resolve()
    if report.get("steps", 0) < 1:
        raise ValueError("No trainable answer tokens produced an optimizer step; shorten prompts or increase max_seq_length.")
    report.update({
        "schema_version": 1, "status": "completed", "target": "sft", "config": config,
        "split_counts": counts, "group_counts": manifest.get("group_counts"), "split_strategy": "fixed exported bundle; no resplitting",
        "baseline_perplexity": _perplexity(report["baseline_loss"]),
        "trained_perplexity": _perplexity(report["trained_loss"]),
        "delta_loss": report["trained_loss"] - report["baseline_loss"],
        "delta_loss_direction": "negative means lower held-out loss; improvement is not guaranteed",
        "evaluation_split": "test", "bundle_manifest_sha256": _sha256(directory / "manifest.json"),
        "artifacts": {"model": "model", "training_report": "training_report.json", "model_report": "model_report.json"},
    })
    _json_file(destination / "training_report.json", report)
    _json_file(destination / "model_report.json", {key: value for key, value in report.items() if key != "loss_history"})
    return report


def run_sft_bundle(bundle: Path, output: Path) -> dict:
    """Internal single-process entrypoint: use fixed splits without repartitioning."""
    from .training import _huggingface

    if _config(load_bundle(Path(bundle)))["n_gpus"] != 1:
        raise ValueError("Multi-GPU SFT must use the distributed launcher.")
    manifest, config, counts = prepare_sft_bundle(bundle, output)
    report = _huggingface(Path(output), config, progress=lambda event: print(json.dumps(event), flush=True), cancelled=None)
    return write_sft_report(bundle, output, manifest, config, counts, report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Internal prepared-bundle training process")
    parser.add_argument("command", choices=["_sft"])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_sft_bundle(args.bundle, args.output)


if __name__ == "__main__":
    main()
