"""Streaming dataset screening with bounded workers and SQLite-backed dedupe/cache."""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
import csv
import json
import math
from pathlib import Path
import re
import sqlite3
import threading

from .jev import JevAPIError, JevClient, MODELS, digest, validate_response

ENGINE_VERSION = "1"
MAX_ROW_BYTES = 1024 * 1024
STATE_FIELDS = frozenset({"text", "messages", "instruction", "input", "output", "prompt", "response"})


def normalize_record(record):
    """Return a content-only state; arbitrary metadata never enters API payloads.

    Supported layouts: text; messages with string role/content; instruction with
    optional input and output; prompt with response. Original rows stay intact in
    partition files. CSV cells in a messages column may contain a JSON array.
    """
    if not isinstance(record, dict):
        raise ValueError("record_must_be_object")
    if "messages" in record and record["messages"] not in (None, ""):
        messages = record["messages"]
        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except (ValueError, TypeError):
                raise ValueError("messages_must_be_json_array") from None
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages_must_be_nonempty_array")
        clean = []
        for message in messages:
            if (not isinstance(message, dict) or not isinstance(message.get("role"), str)
                    or message["role"] not in {"system", "user", "assistant", "tool"}
                    or not isinstance(message.get("content"), str)):
                raise ValueError("messages_require_string_role_and_content")
            clean.append({"role": message["role"], "content": message["content"]})
        state = {"messages": clean}
    elif isinstance(record.get("text"), str) and record["text"].strip():
        state = {"text": record["text"]}
    elif "instruction" in record or "output" in record:
        state = {key: record[key] for key in ("instruction", "input", "output") if key in record and record[key] is not None}
        if not all(isinstance(value, str) for value in state.values()):
            raise ValueError("instruction_input_output_must_be_strings")
        if not state.get("instruction", "").strip() or not state.get("output", "").strip():
            raise ValueError("instruction_and_output_required")
    elif "prompt" in record or "response" in record:
        state = {key: record[key] for key in ("prompt", "response") if key in record}
        if not all(isinstance(value, str) for value in state.values()):
            raise ValueError("prompt_response_must_be_strings")
        if not state.get("prompt", "").strip() or not state.get("response", "").strip():
            raise ValueError("prompt_and_response_required")
    else:
        raise ValueError("missing_supported_content")
    if not state_text(state).strip():
        raise ValueError("empty_content")
    # Reject unpaired JSON surrogate escapes before UTF-8 serialization/hash.
    try:
        state_text(state).encode("utf-8")
    except UnicodeError:
        raise ValueError("content_is_not_valid_utf8") from None
    return state


def state_text(state):
    if "messages" in state:
        return "\n".join(message["content"] for message in state["messages"])
    return "\n".join(state[key] for key in ("text", "instruction", "input", "output", "prompt", "response") if key in state)


def _invalid(line, reason):
    # Do not retain oversized content or invalid binary bytes in memory or logs.
    return {"_invalid_input": {"line": line, "reason": reason}}


def _jsonl_rows(path):
    with path.open("rb") as source:
        line_no = 0
        while True:
            raw = source.readline(MAX_ROW_BYTES + 1)
            if not raw:
                break
            line_no += 1
            if len(raw) > MAX_ROW_BYTES:
                while raw and not raw.endswith(b"\n"):
                    raw = source.readline(MAX_ROW_BYTES + 1)
                yield line_no, _invalid(line_no, "row_exceeds_1_mib"), "row_exceeds_1_mib", False
                continue
            if not raw.strip():
                continue
            try:
                text = raw.decode("utf-8-sig" if line_no == 1 else "utf-8")
            except UnicodeError:
                yield line_no, _invalid(line_no, "invalid_utf8"), "invalid_utf8", False
                continue
            try:
                record = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("non_finite_json")))
                # Metadata is preserved only if it can be represented in valid UTF-8.
                json.dumps(record, ensure_ascii=False, allow_nan=False).encode("utf-8")
            except (ValueError, UnicodeError, RecursionError):
                yield line_no, _invalid(line_no, "invalid_json"), "invalid_json", False
                continue
            if not isinstance(record, dict):
                yield line_no, {"_invalid_value": record}, "record_must_be_object", False
            else:
                yield line_no, record, None, False


class _BoundedCSVLines:
    def __init__(self, source):
        self.source = source
        self.line = 0
        self.row_bytes = 0

    def __iter__(self):
        return self

    def __next__(self):
        raw = self.source.readline(MAX_ROW_BYTES - self.row_bytes + 1)
        if not raw:
            raise StopIteration
        self.line += 1
        self.row_bytes += len(raw)
        if self.row_bytes > MAX_ROW_BYTES:
            raise ValueError("csv_row_exceeds_1_mib")
        try:
            return raw.decode("utf-8-sig" if self.line == 1 else "utf-8")
        except UnicodeError:
            raise ValueError("invalid_utf8") from None


def _csv_rows(path):
    csv.field_size_limit(MAX_ROW_BYTES)
    with path.open("rb") as source:
        lines = _BoundedCSVLines(source)
        reader = csv.reader(lines, strict=True)
        try:
            header = next(reader)
            if not header or any(not cell for cell in header) or len(set(header)) != len(header):
                raise ValueError("csv_requires_unique_nonempty_headers")
            while True:
                lines.row_bytes = 0
                start_line = lines.line + 1
                try:
                    values = next(reader)
                except StopIteration:
                    break
                if not values:
                    continue
                if len(values) != len(header):
                    yield start_line, {"_invalid_csv_values": values}, "csv_column_count_mismatch", False
                else:
                    yield start_line, dict(zip(header, values)), None, False
        except StopIteration:
            return
        except (ValueError, csv.Error) as exc:
            # CSV parsing cannot safely resume in the middle of a quoted record.
            reason = str(exc) if isinstance(exc, ValueError) else "invalid_csv_syntax"
            yield max(1, lines.line), _invalid(lines.line, reason), reason, True


def iter_records(input_path):
    path = Path(input_path)
    suffix = path.suffix.lower()
    if suffix not in {".jsonl", ".ndjson", ".csv"}:
        raise ValueError("Upload UTF-8 .jsonl, .ndjson or .csv data")
    yield from (_csv_rows(path) if suffix == ".csv" else _jsonl_rows(path))


def _configuration(config):
    if not isinstance(config, dict):
        raise ValueError("config must be an object")
    result = {"provider": "demo", "confidence": 0.85, "concurrency": 4,
              "max_requests": 1000, "rubric": "general", "min_chars": 8,
              "max_chars": 32000, "timeout": 20, "attempts": 3}
    result.update({key: config[key] for key in result if key in config})
    if not isinstance(result["provider"], str) or result["provider"] not in {"demo", "openrouter", "typesafe"}:
        raise ValueError("provider must be demo, openrouter or typesafe")
    if not isinstance(result["rubric"], str) or result["rubric"] not in {"general", "finance", "code"}:
        raise ValueError("rubric must be general, finance or code")
    for name, low, high in (("concurrency", 1, 32), ("max_requests", 1, 1000000), ("min_chars", 1, MAX_ROW_BYTES), ("max_chars", 1, MAX_ROW_BYTES), ("attempts", 1, 3)):
        if type(result[name]) is not int or not low <= result[name] <= high:
            raise ValueError(f"{name} must be an integer between {low} and {high}")
    for name, low, high in (("confidence", 0, 1), ("timeout", 1, 60)):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high or not math.isfinite(value):
            raise ValueError(f"{name} must be between {low} and {high}")
    if result["min_chars"] > result["max_chars"]:
        raise ValueError("min_chars must not exceed max_chars")
    if result["provider"] != "demo":
        model = config.get("model", MODELS[result["provider"]])
        if not isinstance(model, str) or not model.strip() or len(model) > 200:
            raise ValueError("model must be a nonempty string of at most 200 characters")
        result["model"] = model
    return result


def _local_result(decision, reason, **extra):
    return {"decision": decision, "reason": reason, "dimensions": {}, **extra}


def _demo(state):
    content = state_text(state)
    if re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", content):
        return _local_result("review", "demo_email_pattern")
    return _local_result("keep", "demo_rules_passed")


def screen_dataset(input_path: Path, output_dir: Path, config: dict, progress=None, cancelled=None) -> dict:
    """Screen a file, replaying a validated disk cache on subsequent runs.

    Memory is bounded by twice the worker count times the 1 MiB row limit.
    Reusing output_dir rebuilds the partitions and the run-local dedupe set while
    retaining cache.sqlite3. Callers must serialize runs sharing an output_dir.
    A status other than complete must prevent automatic training.
    """
    cfg = _configuration(config)
    path, destination = Path(input_path), Path(output_dir)
    if path.suffix.lower() not in {".jsonl", ".ndjson", ".csv"}:
        raise ValueError("Upload UTF-8 .jsonl, .ndjson or .csv data")
    if not path.is_file():
        raise ValueError("input file does not exist")
    destination.mkdir(parents=True, exist_ok=True)
    if path.resolve().parent == destination.resolve() and path.name in {"keep.jsonl", "review.jsonl", "reject.jsonl", "audit.jsonl"}:
        raise ValueError("input path would be overwritten by output partitions")
    rubric = json.loads((Path(__file__).parent / "rubrics" / f"{cfg['rubric']}.json").read_text(encoding="utf-8"))
    semantic = {key: value for key, value in cfg.items() if key not in {"concurrency", "max_requests", "timeout", "attempts"}}
    config_hash = digest({"engine_version": ENGINE_VERSION, "config": semantic, "rubric": rubric})
    stop_event = threading.Event()
    user_cancelled = cancelled or (lambda: False)
    stopped = lambda: stop_event.is_set() or bool(user_cancelled())
    client = JevClient(cfg["provider"], cfg["max_requests"], cfg["timeout"], cfg["attempts"]) if cfg["provider"] != "demo" else None
    counts = {name: 0 for name in ("total", "keep", "review", "reject", "duplicates")}
    report = {"status": "running", "complete": False, "mode": "demo_rule_based" if client is None else "jev_api", "provider": cfg["provider"],
              "counts": counts, "processed": 0, "errors": [], "error_count": 0, "config": cfg, "config_hash": config_hash,
              "engine_version": ENGINE_VERSION, "dimensions": {}, "api_requests": 0, "cache_hits": 0,
              "input_exhausted": False, "artifacts": {name: f"{name}.jsonl" for name in ("keep", "review", "reject", "audit")}}
    if client is None:
        report["notice"] = "Demo performs local length, exact-duplicate and email-pattern checks only; no Jev model or semantic quality evaluation is used."
    fatal = False

    def evaluate(state):
        if stopped():
            return _local_result("review", "cancelled", error="cancelled")
        if client is None:
            return _demo(state)
        try:
            response = client({"model": cfg["model"], "state": state, "questions": rubric["questions"]}, cancelled=stopped)
            return validate_response(response, rubric, cfg["confidence"])
        except ValueError:
            return _local_result("review", "invalid_response", error="invalid_response")
        except JevAPIError as exc:
            row_failure = exc.category in {"invalid_json", "response_too_large", "invalid_request", "request_too_large"}
            return _local_result("review", exc.category, error=exc.category, fatal=not row_failure and exc.category != "cancelled")

    with ExitStack() as stack:
        db = sqlite3.connect(destination / "cache.sqlite3")
        stack.callback(db.close)
        db.execute("PRAGMA cache_size=-4096")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("CREATE TABLE IF NOT EXISTS cache (config_hash TEXT NOT NULL, state_hash TEXT NOT NULL, result TEXT NOT NULL, PRIMARY KEY(config_hash,state_hash))")
        db.execute("CREATE TABLE IF NOT EXISTS seen (state_hash TEXT PRIMARY KEY)")
        db.execute("DELETE FROM seen")
        db.commit()
        files = {name: stack.enter_context((destination / f"{name}.jsonl").open("w", encoding="utf-8")) for name in ("keep", "review", "reject", "audit")}
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=cfg["concurrency"], thread_name_prefix="jev"))
        rows = iter(iter_records(path))
        stack.callback(rows.close)
        pending = deque()
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < cfg["concurrency"] * 2 and not stopped():
                try:
                    line, record, input_error, input_fatal = next(rows)
                except StopIteration:
                    exhausted = True
                    report["input_exhausted"] = True
                    break
                counts["total"] += 1
                state_hash = None
                cache_hit = False
                if input_error:
                    result = _local_result("review", input_error, error=input_error, fatal=input_fatal)
                else:
                    try:
                        state = normalize_record(record)
                        state_hash = digest(state)
                        size = len(state_text(state))
                    except (ValueError, UnicodeError, RecursionError) as exc:
                        result = _local_result("review", "invalid_content", error=str(exc)[:120])
                    else:
                        inserted = db.execute("INSERT OR IGNORE INTO seen VALUES (?)", (state_hash,)).rowcount
                        if not inserted:
                            counts["duplicates"] += 1
                            result = _local_result("reject", "exact_duplicate")
                        elif not cfg["min_chars"] <= size <= cfg["max_chars"]:
                            result = _local_result("reject", "content_length_outside_bounds")
                        else:
                            cached = db.execute("SELECT result FROM cache WHERE config_hash=? AND state_hash=?", (config_hash, state_hash)).fetchone()
                            if cached:
                                result = json.loads(cached[0])
                                cache_hit = True
                                report["cache_hits"] += 1
                            else:
                                result = pool.submit(evaluate, state)
                pending.append((line, record, state_hash, result, cache_hit))
                if input_fatal:
                    exhausted = True
                    break
            if not pending:
                break
            line, record, state_hash, result, cache_hit = pending.popleft()
            if isinstance(result, Future):
                result = result.result()
                if "error" not in result:
                    db.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?)", (config_hash, state_hash, json.dumps(result, allow_nan=False)))
            if result.get("fatal"):
                fatal = True
                stop_event.set()
            if result.get("error"):
                report["error_count"] += 1
                if len(report["errors"]) < 20:
                    report["errors"].append({"line": line, "error": result["error"]})
            decision = result["decision"]
            counts[decision] += 1
            report["processed"] += 1
            for name, dimension in result["dimensions"].items():
                aggregate = report["dimensions"].setdefault(name, {"keep": 0, "review": 0, "reject": 0, "evaluated": 0, "confidence_sum": 0})
                aggregate[dimension["decision"]] += 1
                aggregate["evaluated"] += 1
                aggregate["confidence_sum"] += dimension.get("confidence", 0)
            files[decision].write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            audit = {"line": line, "state_hash": state_hash, "config_hash": config_hash, "mode": report["mode"], "cache_hit": cache_hit, **result}
            files["audit"].write(json.dumps(audit, ensure_ascii=False, allow_nan=False) + "\n")
            # Commit each completed result so retrying a failed run can reuse it.
            db.commit()
            report["api_requests"] = client.requests if client else 0
            if progress:
                progress({"processed": report["processed"], "counts": dict(counts), "api_requests": report["api_requests"], "cache_hits": report["cache_hits"], "mode": report["mode"]})
        db.commit()
    report["status"] = "incomplete" if fatal else "cancelled" if user_cancelled() else "complete" if report["input_exhausted"] else "incomplete"
    report["complete"] = report["status"] == "complete"
    report["api_requests"] = client.requests if client else 0
    for dimension in report["dimensions"].values():
        dimension["mean_confidence"] = dimension.pop("confidence_sum") / dimension["evaluated"] if dimension["evaluated"] else None
    report_path = destination / "data_report.json"
    temporary = destination / "data_report.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return report
