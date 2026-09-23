"""Declarative domain metrics over bounded JSONL streams, entirely offline.

Definitions choose only built-in operators. They cannot run Python, templates,
regular expressions, model requests, or external commands. A metric describes
its evidence source; imported counters and human scores are not re-verified here.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import unicodedata
from urllib.parse import urlsplit


MAX_LINE_BYTES = 1024 * 1024
MAX_PACK_BYTES = 1024 * 1024
PACK_DIRECTORY = Path(__file__).with_name("metric_packs")
EXAMPLE_DIRECTORY = Path(__file__).with_name("metric_examples")
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_FIELD_PATH = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*){0,15}\Z")
_TYPES = {"string", "number", "integer", "boolean", "string_array"}
_OPERATORS = {
    "exact_match": ({"prediction": "string", "reference": "string"}, set()),
    "token_f1": ({"prediction": "string", "reference": "string"}, set()),
    "numeric_tolerance": ({"prediction": "number", "reference": "number"}, {"absolute_tolerance", "relative_tolerance"}),
    "set_precision": ({"prediction": "string_array", "reference": "string_array"}, set()),
    "set_recall": ({"prediction": "string_array", "reference": "string_array"}, set()),
    "ratio": ({"numerator": "integer", "denominator": "integer"}, set()),
    "boolean": ({"value": "boolean"}, set()),
    "rubric_score": ({"value": "integer"}, {"min", "max"}),
    "json_valid": ({"value": "string"}, {"required_type"}),
}
_MISSING = object()
_INVALID_PATH = object()


def _string(value, name: str, limit: int = 8192, *, empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > limit or (not empty and not value.strip()) or "\x00" in value:
        raise ValueError(f"{name} must be a bounded {'possibly empty' if empty else 'nonempty'} string.")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError(f"{name} contains an invalid Unicode surrogate.") from exc
    return value


def _number(value) -> bool:
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _keys(value, required: set[str], allowed: set[str], name: str) -> None:
    if not isinstance(value, dict) or not required.issubset(value) or set(value) - allowed:
        raise ValueError(f"{name} has missing or unknown definition keys.")


def _reject_constant(value):
    raise ValueError("Non-finite JSON constants are not permitted.")


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON object keys are not permitted.")
        value[key] = item
    return value


def _validate_pack(pack: dict) -> dict:
    required = {"schema_version", "domain", "name", "version", "description", "field_schema", "metrics"}
    _keys(pack, required, required | {"limitations", "references"}, "Metric pack")
    if type(pack["schema_version"]) is not int or pack["schema_version"] != 1:
        raise ValueError("Metric pack schema_version must be 1.")
    if not isinstance(pack["domain"], str) or not _IDENTIFIER.fullmatch(pack["domain"]):
        raise ValueError("Metric pack domain must be a lowercase identifier of at most 64 characters.")
    for key in ("name", "version", "description"):
        _string(pack[key], f"pack.{key}", 8192 if key == "description" else 200)
    for key, limit in (("limitations", 64), ("references", 10)):
        values = pack.get(key, [])
        if not isinstance(values, list) or len(values) > limit:
            raise ValueError(f"pack.{key} must be a bounded list.")
        for value in values:
            _string(value, f"pack.{key}", 8192 if key == "limitations" else 2048)
            if key == "references":
                try:
                    url = urlsplit(value)
                    valid = url.scheme in {"http", "https"} and bool(url.hostname) and not any(char.isspace() for char in value)
                except ValueError:
                    valid = False
                if not valid:
                    raise ValueError("Pack references must be literal HTTP(S) URLs; they are never fetched.")
    schema = pack["field_schema"]
    if not isinstance(schema, dict) or not 1 <= len(schema) <= 256:
        raise ValueError("field_schema must declare between 1 and 256 dotted field paths.")
    for path, definition in schema.items():
        if len(path) > 256 or not _FIELD_PATH.fullmatch(path):
            raise ValueError("Field paths must be bounded dot-separated object keys, without array indexing.")
        _keys(definition, {"type", "description"}, {"type", "description"}, f"field_schema.{path}")
        if definition["type"] not in _TYPES:
            raise ValueError("Unknown field_schema type.")
        _string(definition["description"], f"field_schema.{path}.description")
    metrics = pack["metrics"]
    if not isinstance(metrics, list) or not 1 <= len(metrics) <= 64:
        raise ValueError("A metric pack must contain between 1 and 64 metrics.")
    ids = set()
    metric_keys = {"id", "name", "description", "operator", "fields", "parameters", "unit",
                   "direction", "aggregation", "evidence", "limitations"}
    for metric in metrics:
        _keys(metric, metric_keys, metric_keys | {"target"}, "Metric")
        identifier = metric["id"]
        if not isinstance(identifier, str) or not _IDENTIFIER.fullmatch(identifier) or identifier in ids:
            raise ValueError("Metric ids must be unique lowercase identifiers of at most 64 characters.")
        ids.add(identifier)
        for key in ("name", "description", "limitations"):
            _string(metric[key], f"metric.{key}", 200 if key == "name" else 8192)
        operator = metric["operator"]
        if not isinstance(operator, str) or operator not in _OPERATORS:
            raise ValueError("Unknown metric operator; only built-in offline operators are permitted.")
        roles, parameters = _OPERATORS[operator]
        roles = dict(roles)
        fields = metric["fields"]
        if operator == "numeric_tolerance" and isinstance(fields, dict) and {"prediction_unit", "reference_unit"} & fields.keys():
            roles.update(prediction_unit="string", reference_unit="string")
        _keys(fields, set(roles), set(roles), "Metric fields")
        for role, expected in roles.items():
            path = fields[role]
            if not isinstance(path, str) or path not in schema:
                raise ValueError("Every metric field must be declared in field_schema.")
            actual = schema[path]["type"]
            if actual != expected and not (expected == "number" and actual == "integer"):
                raise ValueError(f"Field {path} has an incompatible type for {operator}.{role}.")
        options = metric["parameters"]
        _keys(options, set(), parameters, "Operator parameters")
        if operator == "numeric_tolerance":
            for key in parameters:
                value = options.get(key, 0)
                if not _number(value) or value < 0:
                    raise ValueError("Numeric tolerances must be finite nonnegative numbers.")
        if operator == "rubric_score":
            low, high = options.get("min", 0), options.get("max", 4)
            if type(low) is not int or type(high) is not int or low >= high:
                raise ValueError("Rubric min and max must be integers with min < max.")
        if operator == "json_valid" and options.get("required_type", "object") not in ("any", "object", "array"):
            raise ValueError("JSON required_type must be any, object, or array.")
        if metric["unit"] != "fraction" or metric["direction"] != "higher_is_better" or metric["aggregation"] != "mean":
            raise ValueError("Metrics require fraction units, higher_is_better direction, and mean aggregation.")
        if metric["evidence"] not in ("computed", "external", "human"):
            raise ValueError("Metric evidence must be computed, external, or human.")
        target = metric.get("target")
        if target is not None:
            _keys(target, {"minimum", "minimum_coverage"}, {"minimum", "minimum_coverage"}, "Metric target")
            if any(not _number(value) or not 0 <= value <= 1 for value in target.values()):
                raise ValueError("Target minimum and minimum_coverage must be finite fractions in [0, 1].")
    return pack


def _read_pack(domain: str | None, path: Path | None) -> tuple[dict, bytes]:
    if (domain is None) == (path is None):
        raise ValueError("Choose exactly one built-in domain or custom metric-pack path.")
    if domain is not None:
        if not isinstance(domain, str) or not _IDENTIFIER.fullmatch(domain):
            raise ValueError("Unknown domain identifier.")
        source = PACK_DIRECTORY / f"{domain}.json"
    else:
        source = Path(path).expanduser()
    try:
        with source.open("rb") as stream:
            raw = stream.read(MAX_PACK_BYTES + 1)
    except OSError as exc:
        raise ValueError("Metric pack could not be read.") from exc
    if len(raw) > MAX_PACK_BYTES:
        raise ValueError("Metric packs must not exceed 1 MiB.")
    try:
        pack = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant, object_pairs_hook=_unique_object)
        _validate_pack(pack)
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise ValueError(f"Invalid metric pack: {exc}") from exc
    if domain is not None and pack["domain"] != domain:
        raise ValueError("The built-in filename and declared domain disagree.")
    return pack, raw


def load_metric_pack(domain: str | None = None, path: Path | None = None) -> dict:
    """Validate a data-only definition; neither references nor field text execute."""
    return _read_pack(domain, path)[0]


def list_metric_packs() -> list[dict]:
    return [{"domain": pack["domain"], "name": pack["name"], "version": pack["version"],
             "description": pack["description"], "metric_count": len(pack["metrics"]),
             "field_count": len(pack["field_schema"])}
            for pack in (load_metric_pack(domain=path.stem) for path in sorted(PACK_DIRECTORY.glob("*.json")))]


@contextmanager
def _atomic_output(output_dir: Path):
    destination = Path(output_dir).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("Metric output already exists; choose a fresh directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".jev-metrics-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "output"
        staging.mkdir()
        yield staging
        if destination.exists() or destination.is_symlink():
            raise ValueError("Metric output appeared during evaluation; refusing to overwrite it.")
        staging.rename(destination)


def init_metric_pack(domain: str, output_dir: Path) -> dict:
    """Copy a built-in definition and synthetic example to one fresh directory."""
    pack, raw = _read_pack(domain, None)
    try:
        with (EXAMPLE_DIRECTORY / f"{domain}.jsonl").open("rb") as stream:
            example = stream.read(MAX_PACK_BYTES + 1)
    except OSError as exc:
        raise ValueError("Built-in metric example could not be read.") from exc
    if len(example) > MAX_PACK_BYTES:
        raise ValueError("Built-in metric example exceeds the 1 MiB limit.")
    with _atomic_output(output_dir) as staging:
        (staging / "metrics.json").write_bytes(raw)
        (staging / "example.jsonl").write_bytes(example)
    return {"status": "initialized", "domain": pack["domain"], "version": pack["version"],
            "output": str(Path(output_dir).expanduser().absolute()), "pack_sha256": hashlib.sha256(raw).hexdigest(),
            "artifacts": {"metrics": "metrics.json", "example": "example.jsonl"}}


def _lookup(row: dict, path: str):
    value = row
    for key in path.split("."):
        if not isinstance(value, dict):
            return _INVALID_PATH
        if key not in value:
            return _MISSING
        value = value[key]
    return value


def _valid_type(value, kind: str) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "number":
        return _number(value)
    if kind == "integer":
        return type(value) is int
    if kind == "boolean":
        return type(value) is bool
    return isinstance(value, list) and all(isinstance(item, str) and bool(item.strip()) for item in value)


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split())


def _result(status: str, reason: str, score: float | None = None) -> dict:
    return {"status": status, "score": score, "reason": reason}


def _evaluate_metric(row: dict, metric: dict, schema: dict) -> dict:
    values = {role: _lookup(row, path) for role, path in metric["fields"].items()}
    missing = [role for role, value in values.items() if value is _MISSING]
    invalid = [role for role, value in values.items()
               if value is not _MISSING and (value is _INVALID_PATH or not _valid_type(value, schema[metric["fields"][role]]["type"]))]
    # A row is counted once per metric. Known invalid data takes precedence over
    # missing fields; reasons contain only bounded definition roles, never values.
    if invalid:
        return _result("invalid", "invalid_field_type:" + ",".join(sorted(invalid)))
    if missing:
        return _result("missing", "missing_field:" + ",".join(sorted(missing)))
    operator, parameters = metric["operator"], metric["parameters"]
    if operator in ("exact_match", "token_f1"):
        prediction, reference = _normalized(values["prediction"]), _normalized(values["reference"])
        if not reference:
            return _result("invalid", "empty_reference")
        if operator == "exact_match":
            score = float(prediction == reference)
        else:
            predicted, expected = Counter(prediction.split()), Counter(reference.split())
            overlap = sum((predicted & expected).values())
            score = 2 * overlap / (sum(predicted.values()) + sum(expected.values()))
    elif operator == "numeric_tolerance":
        if "prediction_unit" in values:
            if not values["prediction_unit"].strip() or not values["reference_unit"].strip():
                return _result("invalid", "empty_unit")
            if values["prediction_unit"] != values["reference_unit"]:
                return _result("evaluated", "unit_mismatch_no_conversion", 0.0)
        numbers = [values["prediction"], values["reference"], parameters.get("absolute_tolerance", 0), parameters.get("relative_tolerance", 0)]
        # Decimal avoids inf <= inf falsely passing when finite floats overflow
        # during subtraction or tolerance multiplication.
        with localcontext() as context:
            context.prec = max(40, 2 * max(len(str(number)) for number in numbers) + 20)
            prediction, reference, absolute, relative = map(lambda number: Decimal(str(number)), numbers)
            score = float(abs(prediction - reference) <= max(absolute, relative * abs(reference)))
    elif operator in ("set_precision", "set_recall"):
        predicted, reference = set(values["prediction"]), set(values["reference"])
        if operator == "set_recall" and not reference:
            return _result("not_applicable", "empty_reference_set")
        if not predicted and not reference:
            return _result("not_applicable", "both_sets_empty")
        denominator = len(predicted) if operator == "set_precision" else len(reference)
        score = len(predicted & reference) / denominator if denominator else 0.0
    elif operator == "ratio":
        numerator, denominator = values["numerator"], values["denominator"]
        if numerator < 0 or denominator < 0 or numerator > denominator:
            return _result("invalid", "counts_must_satisfy_0_le_numerator_le_denominator")
        if denominator == 0:
            return _result("not_applicable", "zero_denominator")
        score = numerator / denominator
    elif operator == "boolean":
        score = float(values["value"])
    elif operator == "rubric_score":
        low, high = parameters.get("min", 0), parameters.get("max", 4)
        if not low <= values["value"] <= high:
            return _result("invalid", "rubric_value_outside_defined_range")
        score = (values["value"] - low) / (high - low)
    else:  # json_valid is the only remaining validated operator.
        try:
            # Syntax and top-level container type are the only things checked.
            # Preserve numeric lexemes instead of materializing their values:
            # valid JSON can contain exponents or integers beyond runtime limits.
            parsed = json.loads(values["value"], parse_constant=_reject_constant,
                                parse_float=str, parse_int=str, object_pairs_hook=_unique_object)
            required = parameters.get("required_type", "object")
            score = float(required == "any" or (required == "object" and isinstance(parsed, dict))
                          or (required == "array" and isinstance(parsed, list)))
        except (ValueError, RecursionError):
            score = 0.0
    return _result("evaluated", "computed" if metric["evidence"] == "computed" else "supplied_evidence_not_independently_verified", float(score))


def _target_status(mean: float | None, coverage: float | None, target: dict | None) -> str:
    if target is None:
        return "not_configured"
    if mean is None or coverage is None:
        return "no_data"
    if coverage < target["minimum_coverage"]:
        return "insufficient_coverage"
    return "met" if mean >= target["minimum"] else "below_target"


def evaluate_metrics(input_path: Path, output_dir: Path, *, domain: str | None = None,
                     pack_path: Path | None = None) -> dict:
    """Evaluate declared fields once; publish artifacts only after full success."""
    pack, raw = _read_pack(domain, pack_path)  # Validate all operators before creating output.
    source = Path(input_path).expanduser()
    if not source.is_file():
        raise ValueError("Metric input must be an existing JSONL file.")
    accumulators = {metric["id"]: {"evaluated": 0, "missing": 0, "invalid": 0, "not_applicable": 0,
                                   "sum": 0.0, "compensation": 0.0} for metric in pack["metrics"]}
    digest, rows, blank_lines = hashlib.sha256(), 0, 0
    with _atomic_output(output_dir) as staging:
        with source.open("rb") as stream, (staging / "scores.jsonl").open("w", encoding="utf-8") as results:
            line_number = 0
            while True:
                line = stream.readline(MAX_LINE_BYTES + 1)
                if not line:
                    break
                line_number += 1
                if len(line) > MAX_LINE_BYTES:
                    raise ValueError(f"JSONL line {line_number} exceeds the 1 MiB limit.")
                digest.update(line)
                if not line.strip():
                    blank_lines += 1
                    continue
                try:
                    row = json.loads(line.decode("utf-8"), parse_constant=_reject_constant,
                                     object_pairs_hook=_unique_object)
                    if not isinstance(row, dict):
                        raise ValueError("Each record must be a JSON object.")
                except (ValueError, RecursionError, UnicodeError) as exc:
                    raise ValueError(f"Invalid JSONL object at line {line_number}.") from exc
                rows += 1
                identifier = row.get("id")
                if type(identifier) is int:
                    identifier = str(identifier)
                if (not isinstance(identifier, str) or len(identifier) > 200
                        or any(ord(char) < 32 or 0xD800 <= ord(char) <= 0xDFFF for char in identifier)):
                    identifier = None
                item = {"line": line_number, "id": identifier, "metrics": {}}
                for metric in pack["metrics"]:
                    result = _evaluate_metric(row, metric, pack["field_schema"])
                    item["metrics"][metric["id"]] = result
                    total = accumulators[metric["id"]]
                    total[result["status"]] += 1
                    if result["status"] == "evaluated":
                        # Compensated summation keeps aggregation stable without
                        # retaining per-row scores in memory.
                        corrected = result["score"] - total["compensation"]
                        updated = total["sum"] + corrected
                        total["compensation"] = (updated - total["sum"]) - corrected
                        total["sum"] = updated
                results.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
        metrics = {}
        for metric in pack["metrics"]:
            totals = accumulators[metric["id"]]
            eligible = rows - totals["not_applicable"]
            mean = totals["sum"] / totals["evaluated"] if totals["evaluated"] else None
            coverage = totals["evaluated"] / eligible if eligible else None
            metrics[metric["id"]] = {
                "name": metric["name"], "operator": metric["operator"], "evidence": metric["evidence"],
                "unit": metric["unit"], "direction": metric["direction"], "aggregation": metric["aggregation"],
                "counts": {**{key: totals[key] for key in ("evaluated", "missing", "invalid", "not_applicable")}, "eligible": eligible},
                "mean": mean, "coverage": coverage, "target": metric.get("target"),
                "target_status": _target_status(mean, coverage, metric.get("target")), "limitations": metric["limitations"],
            }
        report = {"schema_version": 1, "status": "completed", "mode": "offline_domain_metrics",
                  "pack": {"domain": pack["domain"], "name": pack["name"], "version": pack["version"], "sha256": hashlib.sha256(raw).hexdigest()},
                  "input": {"sha256": digest.hexdigest(), "rows": rows, "blank_lines": blank_lines},
                  "metrics": metrics, "limitations": pack.get("limitations", []),
                  "evaluation_scope": "Declared offline operators only. External counters and human scores are supplied evidence. No overall grade or automatic RL reward is produced.",
                  "artifacts": {"metrics": "metrics.json", "results": "scores.jsonl", "report": "report.json"}}
        (staging / "metrics.json").write_bytes(raw)
        (staging / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
