"""Streaming, leakage-aware train/evaluate runners. Demo metrics are not LLM metrics."""

from __future__ import annotations

from array import array
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
from typing import Any, Callable, Iterator


DEFAULT_BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
MAX_LINE_BYTES = 8 * 1024 * 1024
MAX_TEXT_CHARS = 1_000_000
Progress = Callable[[dict[str, Any]], None] | None
Cancelled = Callable[[], bool] | None


class TrainingCancelled(RuntimeError):
    """A cooperative cancellation was requested."""


def _check_cancelled(cancelled: Cancelled) -> None:
    if cancelled and cancelled():
        raise TrainingCancelled("Training cancelled.")


def _emit(progress: Progress, stage: str, **values: Any) -> None:
    if progress:
        progress({"stage": stage, **values})


def _integer(config: dict, name: str, default: int, low: int, high: int) -> int:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}.")
    return value


def _real(config: dict, name: str, default: float, low: float, high: float) -> float:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric.")
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}.")
    return value


def validate_training_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate user-controlled compute limits and the operator's model allowlist."""
    trainer = config.get("trainer", "demo")
    if trainer not in ("demo", "huggingface"):
        raise ValueError("trainer must be demo or huggingface.")
    normalized = {
        "trainer": trainer,
        "epochs": _integer(config, "epochs", 1, 1, 20),
        "max_steps": _integer(config, "max_steps", 20, 1, 10000),
        "batch_size": _integer(config, "batch_size", 4, 1, 32),
        "seed": _integer(config, "seed", 42, 0, 2**31 - 1),
        "max_seq_length": _integer(config, "max_seq_length", 256, 16, 4096),
        "learning_rate": _real(config, "learning_rate", 0.0002, 1e-7, 0.01),
        "validation_fraction": _real(config, "validation_fraction", 0.15, 0.05, 0.4),
        "test_fraction": _real(config, "test_fraction", 0.15, 0.05, 0.4),
        "lora_r": _integer(config, "lora_r", 8, 1, 256),
        "lora_alpha": _integer(config, "lora_alpha", 16, 1, 1024),
        "loss_mask": config.get("loss_mask", "answer"),
    }
    if normalized["loss_mask"] not in ("answer", "full"):
        raise ValueError("loss_mask must be answer or full.")
    if normalized["validation_fraction"] + normalized["test_fraction"] > 0.8:
        raise ValueError("Validation and test fractions must leave at least 20% for training.")
    if trainer == "huggingface":
        allowed = os.environ.get("JEV_BASE_MODEL", DEFAULT_BASE_MODEL)
        requested = config.get("base_model") or allowed
        if requested != allowed:
            raise ValueError("base_model must match the operator-configured JEV_BASE_MODEL.")
        normalized["base_model"] = allowed
    return normalized


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as stream:
        line_number = 0
        while True:
            line = stream.readline(MAX_LINE_BYTES + 1)
            if not line:
                break
            line_number += 1
            if len(line) > MAX_LINE_BYTES:
                raise ValueError(f"JSONL line {line_number} exceeds the 8 MiB limit.")
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}.") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {line_number} must be an object.")
            yield row


def _conversation(state: dict[str, Any]) -> list[dict[str, str]] | None:
    """The record as chat turns, or None for a bare passage that has no prompt."""
    if "messages" in state:
        return state["messages"]
    if "instruction" in state:
        prompt = state["instruction"] + ("\n\n" + state["input"] if state.get("input", "").strip() else "")
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": state["output"]}]
    if "prompt" in state:
        return [{"role": "user", "content": state["prompt"]}, {"role": "assistant", "content": state["response"]}]
    return None


def _text(row: dict[str, Any]) -> str:
    # Share schema and precedence with screening: users must train on the content
    # that was actually scored, even when a source row contains multiple layouts.
    from .screening import normalize_record, state_text

    state = normalize_record(row)
    if "messages" in state:
        result = "\n".join(f"{message['role']}: {message['content']}"
                           for message in state["messages"]).strip()
    else:
        result = state_text(state).strip()
    if not result:
        raise ValueError("Every retained record must contain nonempty text or text messages.")
    if len(result) > MAX_TEXT_CHARS:
        raise ValueError("A training record exceeds the 1,000,000 character limit.")
    return result


def _render(tokenizer: Any, row: dict[str, Any]) -> tuple[str, str | None, bool]:
    """(full_text, prompt_text, templated) for one split row.

    With a chat template the model's own turn markers are used and the prompt is
    everything up to the assistant's final answer; without one, the plain
    `role: content` text of the split file is used with the same cut. A bare
    passage has no prompt. `templated` says whether special tokens are already in
    the text (a template writes its own) or the tokenizer must add them.
    """
    from .screening import normalize_record

    # The split row carries the content it was scored on under `_state`; the `text`
    # the split writer added would otherwise shadow an instruction/output pair.
    state = row.get("_state") or normalize_record(row)
    turns = _conversation(state)
    if turns is None or turns[-1]["role"] != "assistant":
        return row["text"], None, False
    if getattr(tokenizer, "chat_template", None):
        try:
            full = tokenizer.apply_chat_template(turns, tokenize=False)
            prompt = tokenizer.apply_chat_template(turns[:-1], tokenize=False, add_generation_prompt=True)
            if isinstance(full, str) and isinstance(prompt, str) and full.startswith(prompt):
                return full, prompt, True
        except Exception:  # noqa: BLE001 - a template that cannot render these turns falls back to plain text.
            pass
    full, answer = row["text"], turns[-1]["content"].strip()
    if answer and full.endswith(answer) and len(full) > len(answer):
        return full, full[:len(full) - len(answer)], False
    return full, None, False


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _root(db: sqlite3.Connection, key: str) -> str:
    # Path halving keeps both memory usage and future traversals bounded.
    while True:
        parent = db.execute("SELECT parent FROM nodes WHERE key=?", (key,)).fetchone()[0]
        if parent == key:
            return key
        grandparent = db.execute("SELECT parent FROM nodes WHERE key=?", (parent,)).fetchone()[0]
        if parent != grandparent:
            db.execute("UPDATE nodes SET parent=? WHERE key=?", (grandparent, key))
        key = parent


def _join(db: sqlite3.Connection, left: str, right: str) -> None:
    left, right = _root(db, left), _root(db, right)
    if left != right:
        # A stable representative makes the split independent of input ordering.
        small, large = sorted((left, right))
        db.execute("UPDATE nodes SET parent=? WHERE key=?", (small, large))


def _prepare_splits(source: Path, destination: Path, config: dict, progress: Progress,
                    cancelled: Cancelled,
                    extra_group_keys: Callable[[dict[str, Any]], list[str]] | None = None) -> dict[str, Any]:
    index_path = destination / "split_index.sqlite3"
    if index_path.exists():
        raise ValueError("Output directory already contains a training split index; use a new run directory.")
    db = sqlite3.connect(index_path)
    try:
        db.execute("PRAGMA cache_size=-8192")
        db.execute("PRAGMA temp_store=FILE")
        # The index is scratch space deleted at the end of the split; nothing in it
        # needs to survive a crash, so skip the journal and the fsyncs.
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA synchronous=OFF")
        db.executescript("""
            CREATE TABLE nodes (key TEXT PRIMARY KEY, parent TEXT NOT NULL);
            CREATE TABLE records (id INTEGER PRIMARY KEY, node TEXT NOT NULL, payload TEXT NOT NULL,
                                  component TEXT);
            CREATE TABLE components (key TEXT PRIMARY KEY, ordering TEXT NOT NULL, split TEXT);
        """)
        count = 0
        from .screening import normalize_record

        for row in _rows(source):
            _check_cancelled(cancelled)
            text = _text(row)
            state = normalize_record(row)
            # Exact content (with whitespace normalized) joins even distinct declared groups.
            keys = ["text:" + _digest(" ".join(text.split()))]
            if extra_group_keys:
                keys.extend(extra_group_keys(row))
            for field in ("group_id", "conversation_id"):
                if row.get(field) is not None:
                    group = row[field]
                    if not isinstance(group, (str, int)) or isinstance(group, bool):
                        raise ValueError(f"{field} must be a string or integer.")
                    if str(group).strip():
                        keys.append(field + ":" + _digest(str(group)))
            for key in keys:
                db.execute("INSERT OR IGNORE INTO nodes VALUES (?,?)", (key, key))
            for key in keys[1:]:
                _join(db, keys[0], key)
            output = dict(row)
            output["text"] = text
            output["_state"] = state
            db.execute("INSERT INTO records(node,payload) VALUES (?,?)",
                       (keys[0], json.dumps(output, ensure_ascii=False, allow_nan=False)))
            count += 1
            if count % 1000 == 0:
                db.commit()
                _emit(progress, "splitting", records=count)
        if not count:
            raise ValueError("No retained records are available for training.")
        db.commit()
        for record_id, node in db.execute("SELECT id,node FROM records ORDER BY id"):
            _check_cancelled(cancelled)
            root = _root(db, node)
            ordering = _digest(f"{config['seed']}:{root}")
            db.execute("UPDATE records SET component=? WHERE id=?", (root, record_id))
            db.execute("INSERT OR IGNORE INTO components(key,ordering) VALUES (?,?)", (root, ordering))
        group_count = db.execute("SELECT COUNT(*) FROM components").fetchone()[0]
        if group_count < 6:
            raise ValueError(f"At least 6 independent content/conversation groups are required; found {group_count}.")
        val_groups = max(1, int(group_count * config["validation_fraction"]))
        test_groups = max(1, int(group_count * config["test_fraction"]))
        for index, (key,) in enumerate(db.execute("SELECT key FROM components ORDER BY ordering,key")):
            split = "validation" if index < val_groups else "test" if index < val_groups + test_groups else "train"
            db.execute("UPDATE components SET split=? WHERE key=?", (split, key))
        db.commit()
        db.execute("CREATE INDEX records_component ON records(component)")
        counts = {"train": 0, "validation": 0, "test": 0}
        with ExitStack() as stack:
            streams = {name: stack.enter_context((destination / f"{name}.jsonl").open("w", encoding="utf-8"))
                       for name in counts}
            for payload, component, split in db.execute(
                "SELECT r.payload,r.component,c.split FROM records r JOIN components c ON r.component=c.key ORDER BY r.id"
            ):
                _check_cancelled(cancelled)
                row = json.loads(payload)
                row["_split_group"] = component
                streams[split].write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                counts[split] += 1
        if not all(counts.values()):
            raise ValueError("Every evaluation split must contain at least one record.")
        info = {
            "split_counts": counts,
            "group_counts": {"train": group_count - val_groups - test_groups,
                             "validation": val_groups, "test": test_groups},
            "independent_groups": group_count,
            "strategy": "seeded content/group/conversation connected components",
            "split_limitations": "Exact normalized duplicates and supplied groups are isolated; semantic duplicates require upstream checks.",
        }
        _emit(progress, "splitting", records=count, **info)
        return info
    finally:
        db.close()
        # The durable JSONL splits carry the group identifiers; discard duplicated raw data in the temporary index.
        for suffix in ("", "-journal", "-wal", "-shm"):
            Path(str(index_path) + suffix).unlink(missing_ok=True)


def _json_file(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _perplexity(loss: float) -> float | None:
    # JSON has no portable representation for Infinity. Preserve the NLL and disclose overflow.
    return math.exp(loss) if loss < 700 else None


def _batches(path: Path, size: int, cancelled: Cancelled) -> Iterator[list[dict]]:
    batch = []
    for row in _rows(path):
        _check_cancelled(cancelled)
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _demo(directory: Path, config: dict, progress: Progress, cancelled: Cancelled) -> dict:
    # UTF-8 bytes plus EOS, with a separate BOS context: fixed 258 x 257 memory.
    vocabulary, bos, eos, alpha = 257, 257, 256, 0.5
    counts = [array("Q", [0]) * vocabulary for _ in range(258)]
    totals = [0] * 258

    def tokens(row: dict) -> list[int]:
        # The demo sequence limit is UTF-8 bytes, not tokenizer tokens.
        return list(row["text"].encode("utf-8")[:config["max_seq_length"]]) + [eos]

    def evaluate(path: Path) -> tuple[float, int]:
        total_loss, token_count = 0.0, 0
        for row in _rows(path):
            _check_cancelled(cancelled)
            previous = bos
            for token in tokens(row):
                total_loss -= math.log((counts[previous][token] + alpha) / (totals[previous] + alpha * vocabulary))
                token_count += 1
                previous = token
        if not token_count:
            raise ValueError("Evaluation split contains no usable tokens.")
        return total_loss / token_count, token_count

    baseline_validation, validation_tokens = evaluate(directory / "validation.jsonl")
    baseline_test, test_tokens = evaluate(directory / "test.jsonl")
    losses, step, seen = [], 0, 0
    for epoch in range(config["epochs"]):
        for batch in _batches(directory / "train.jsonl", config["batch_size"], cancelled):
            if step >= config["max_steps"]:
                break
            loss, n = 0.0, 0
            for row in batch:
                previous = bos
                for token in tokens(row):
                    loss -= math.log((counts[previous][token] + alpha) / (totals[previous] + alpha * vocabulary))
                    counts[previous][token] += 1
                    totals[previous] += 1
                    previous = token
                    n += 1
                seen += 1
            step += 1
            losses.append({"step": step, "epoch": epoch + 1, "loss": loss / n})
            _emit(progress, "training", step=step, max_steps=config["max_steps"], loss=loss / n)
        if step >= config["max_steps"]:
            break
    _emit(progress, "evaluating", message="Evaluating the trained byte-bigram model on held-out records.")
    validation_loss, _ = evaluate(directory / "validation.jsonl")
    test_loss, _ = evaluate(directory / "test.jsonl")
    model_dir = directory / "model"
    model_dir.mkdir(exist_ok=True)
    _json_file(model_dir / "byte_bigram.json", {
        "model_type": "utf8_byte_bigram", "vocabulary_size": vocabulary, "bos": bos,
        "eos": eos, "smoothing_alpha": alpha, "counts": [list(row) for row in counts],
        "context_totals": totals, "max_seq_bytes": config["max_seq_length"],
    })
    return {
        "mode": "demo", "trainer": "demo", "model_type": "utf8_byte_bigram",
        "is_llm": False, "baseline_loss": baseline_test, "trained_loss": test_loss,
        "baseline_validation_loss": baseline_validation, "trained_validation_loss": validation_loss,
        "evaluation_tokens": {"validation": validation_tokens, "test": test_tokens},
        "steps": step, "training_record_visits": seen, "loss_history": losses,
        "metric_unit": "natural-log NLL per UTF-8 byte/EOS", "device": "cpu",
        "limitations": ["Demo trains a byte-bigram count model, not an LLM or a learned JEV scorer.",
                        "Its uniform baseline and byte metrics cannot be compared with LLM token metrics.",
                        "learning_rate is unused by count-based demo training."],
    }


def _huggingface(directory: Path, config: dict, progress: Progress, cancelled: Cancelled) -> dict:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("Hugging Face training requires the optional training dependencies: pip install '.[train]'.") from exc

    _check_cancelled(cancelled)
    set_seed(config["seed"])
    base_model = config["base_model"]
    # Network access is restricted to loading this operator-configured model; no user-selected Python code.
    local_only = os.environ.get("JEV_MODEL_LOCAL_ONLY", "0") == "1"
    _emit(progress, "loading_model", base_model=base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False, local_files_only=local_only)
    if tokenizer.eos_token_id is None:
        raise ValueError("The configured causal tokenizer must define an EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    configured_device = os.environ.get("JEV_TRAIN_DEVICE", "auto")
    if configured_device not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("JEV_TRAIN_DEVICE must be auto, cpu, cuda, or mps.")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if configured_device == "auto" else configured_device
    # The frozen base is held in bfloat16 where the device does it natively; the LoRA
    # weights stay in float32 (PEFT's default), so the optimiser step is unaffected.
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    loading = {"trust_remote_code": False, "use_safetensors": True, "local_files_only": local_only}
    # transformers renamed torch_dtype to dtype in 4.56; older versions accept the new
    # spelling without complaint and silently ignore it, so pick by version.
    import transformers
    release = tuple(int(part) for part in transformers.__version__.split(".")[:2])
    loading["dtype" if release >= (4, 56) else "torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(base_model, **loading)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model_limit = getattr(model.config, "max_position_embeddings", None)
    if isinstance(model_limit, int) and config["max_seq_length"] > model_limit:
        raise ValueError(f"max_seq_length exceeds the model limit of {model_limit}.")
    model.to(device)
    uses_template = bool(getattr(tokenizer, "chat_template", None))
    masked_rows, prompt_tokens, learned_tokens = 0, 0, 0

    def encode(rows: list[dict], count: bool = False) -> dict:
        nonlocal masked_rows, prompt_tokens, learned_tokens
        rendered = [_render(tokenizer, row) for row in rows]
        limit = config["max_seq_length"] - 1
        ids, prompt_lengths = [], []
        for full, prompt, templated in rendered:
            sequence = tokenizer(full, truncation=True, max_length=limit, add_special_tokens=not templated)["input_ids"]
            ids.append(sequence + [tokenizer.eos_token_id])
            cut = 0
            if config["loss_mask"] == "answer" and prompt is not None:
                prompt_ids = tokenizer(prompt, add_special_tokens=not templated)["input_ids"]
                # The prompt's tokens must be a prefix of the full sequence for the mask to be
                # exact; when a tokenizer merges across the boundary, learn the whole row instead.
                if sequence[:len(prompt_ids)] == prompt_ids and len(prompt_ids) < len(sequence):
                    cut = len(prompt_ids)
            prompt_lengths.append(cut)
        # Padding and EOS can share an ID; mask via attention_mask so genuine EOS is still learned.
        batch = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for index, cut in enumerate(prompt_lengths):
            if cut:
                labels[index, :cut] = -100
        if count:
            masked_rows += sum(1 for cut in prompt_lengths if cut)
            prompt_tokens += sum(prompt_lengths)
            learned_tokens += int((labels != -100).sum().item())
        batch["labels"] = labels
        return {key: value.to(device) for key, value in batch.items()}

    def evaluate(path: Path) -> tuple[float, int]:
        model.eval()
        total_loss, token_count = 0.0, 0
        with torch.no_grad():
            for rows in _batches(path, config["batch_size"], cancelled):
                batch = encode(rows)
                predicted = int((batch["labels"][:, 1:] != -100).sum().item())
                if not predicted:
                    continue
                loss = float(model(**batch).loss.detach().cpu())
                if not math.isfinite(loss):
                    raise RuntimeError("Model evaluation returned a non-finite loss.")
                total_loss += loss * predicted
                token_count += predicted
        if not token_count:
            raise ValueError("Evaluation split has no predictable tokens.")
        return total_loss / token_count, token_count

    _emit(progress, "evaluating_baseline", message="Measuring the base model before adapter training.")
    baseline_validation, validation_tokens = evaluate(directory / "validation.jsonl")
    baseline_test, test_tokens = evaluate(directory / "test.jsonl")
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, target_modules="all-linear", r=config["lora_r"],
        lora_alpha=config["lora_alpha"], lora_dropout=0.05, bias="none",
    ))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"], weight_decay=0.0)
    # Linear warm-up over the first tenth of the run, then linear decay to zero: the
    # usual shape for a short adapter run, and it stops the last steps from
    # overshooting on a small batch.
    total_steps = config["max_steps"]
    warmup = max(1, total_steps // 10)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))
    step, seen, losses = 0, 0, []
    for epoch in range(config["epochs"]):
        model.train()
        for rows in _batches(directory / "train.jsonl", config["batch_size"], cancelled):
            if step >= config["max_steps"]:
                break
            batch = encode(rows, count=True)
            if not int((batch["labels"][:, 1:] != -100).sum().item()):
                continue  # every token of this batch is prompt or padding; nothing to learn from
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Training returned a non-finite loss; no successful model report was written.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            learning_rate = optimizer.param_groups[0]["lr"]
            optimizer.step()
            schedule.step()
            step += 1
            seen += len(rows)
            loss_value = float(loss.detach().cpu())
            losses.append({"step": step, "epoch": epoch + 1, "loss": loss_value, "lr": learning_rate})
            _emit(progress, "training", step=step, max_steps=config["max_steps"], loss=loss_value)
        if step >= config["max_steps"]:
            break
    _emit(progress, "evaluating", message="Evaluating the LoRA adapter on the same held-out records.")
    validation_loss, _ = evaluate(directory / "validation.jsonl")
    test_loss, _ = evaluate(directory / "test.jsonl")
    _check_cancelled(cancelled)
    model_dir = directory / "model"
    model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    answer_only = config["loss_mask"] == "answer"
    limitations = [
        "Held-out loss does not establish task accuracy, fairness, factuality, or production readiness.",
        "Single-process streaming training; one local model replica must fit in memory.",
        "Sequence tails beyond max_seq_length are truncated; row order is preserved within each split.",
    ]
    if answer_only:
        limitations.insert(0, "Answer-only loss: prompt tokens are masked for rows with a prompt; bare passages are learned in full.")
        if masked_rows == 0 and seen:
            limitations.insert(1, "No row had a maskable prompt, so the run was effectively full-text SFT.")
    else:
        limitations.insert(0, "Full-text causal SFT; prompt tokens are included in the loss.")
    return {
        "mode": "llm_lora", "trainer": "huggingface", "model_type": "causal_lm_lora", "is_llm": True,
        "base_model": base_model, "base_model_revision": getattr(model.config, "_commit_hash", None),
        "baseline_loss": baseline_test, "trained_loss": test_loss,
        "baseline_validation_loss": baseline_validation, "trained_validation_loss": validation_loss,
        "evaluation_tokens": {"validation": validation_tokens, "test": test_tokens},
        "steps": step, "training_record_visits": seen, "loss_history": losses,
        "metric_unit": ("natural-log NLL per answer token" if answer_only else "natural-log NLL per causal tokenizer token"),
        "device": device, "dtype": str(dtype).replace("torch.", ""),
        "chat_template": uses_template, "loss_mask": config["loss_mask"],
        "masking": {"rows_with_masked_prompt": masked_rows, "prompt_tokens_masked": prompt_tokens, "tokens_learned": learned_tokens},
        "lora": {"r": config["lora_r"], "alpha": config["lora_alpha"], "dropout": 0.05, "target_modules": "all-linear"},
        "schedule": {"warmup_steps": warmup, "decay": "linear", "peak_learning_rate": config["learning_rate"]},
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "limitations": limitations,
    }


def train_and_evaluate(keep_path: Path, output_dir: Path, config: dict,
                       progress: Progress = None, cancelled: Cancelled = None) -> dict:
    """Train on retained JSONL and compare baseline/tuned loss on identical held-out splits.

    Uses an existing empty output directory or creates one. Raises rather than claiming
    success on invalid data, missing dependencies, cancellation, or model failures.
    """
    config = validate_training_config(config)
    source, directory = Path(keep_path), Path(output_dir)
    if not source.is_file():
        raise ValueError("Retained JSONL input does not exist.")
    directory.mkdir(parents=True, exist_ok=True)
    owned_paths = [directory / name for name in ("train.jsonl", "validation.jsonl", "test.jsonl",
                                               "training_report.json", "model_report.json", "model")]
    if any(path.exists() for path in owned_paths):
        raise ValueError("Training output artifacts already exist; use a new run directory.")
    _check_cancelled(cancelled)
    _emit(progress, "splitting", message="Building leakage-aware train, validation, and test splits.")
    split_info = _prepare_splits(source, directory, config, progress, cancelled)
    result = (_demo if config["trainer"] == "demo" else _huggingface)(directory, config, progress, cancelled)
    _check_cancelled(cancelled)
    result.update(split_info)
    result.update({
        "schema_version": 1, "status": "completed", "config": config,
        "baseline_perplexity": _perplexity(result["baseline_loss"]),
        "trained_perplexity": _perplexity(result["trained_loss"]),
        "delta_loss": result["trained_loss"] - result["baseline_loss"],
        "delta_loss_direction": "negative means lower held-out loss; improvement is not guaranteed",
        "evaluation_split": "test",
        "artifacts": {"train": "train.jsonl", "validation": "validation.jsonl", "test": "test.jsonl",
                      "model": "model", "training_report": "training_report.json", "model_report": "model_report.json"},
    })
    _json_file(directory / "training_report.json", result)
    _json_file(directory / "model_report.json", {key: value for key, value in result.items() if key != "loss_history"})
    _emit(progress, "completed", trainer=result["trainer"], baseline_loss=result["baseline_loss"],
          trained_loss=result["trained_loss"], delta_loss=result["delta_loss"])
    return result
