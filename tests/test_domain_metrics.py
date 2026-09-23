"""Offline operator semantics, missingness, reproducibility and atomic outputs."""

import copy
import hashlib
import json
import subprocess

import pytest

from jev_dataops import domain_metrics as metrics


def definition(operator="exact_match", *, parameters=None, target=None):
    roles = {"exact_match": {"prediction": "string", "reference": "string"},
             "token_f1": {"prediction": "string", "reference": "string"},
             "numeric_tolerance": {"prediction": "number", "reference": "number"},
             "set_precision": {"prediction": "string_array", "reference": "string_array"},
             "set_recall": {"prediction": "string_array", "reference": "string_array"},
             "ratio": {"numerator": "integer", "denominator": "integer"},
             "boolean": {"value": "boolean"}, "rubric_score": {"value": "integer"},
             "json_valid": {"value": "string"}}[operator]
    return {"schema_version": 1, "domain": "custom", "name": "Offline fixture", "version": "1.2.3",
            "description": "Explicit test evidence.",
            "field_schema": {key: {"type": kind, "description": "Declared test field."} for key, kind in roles.items()},
            "metrics": [{"id": "check", "name": "Test metric", "description": "Tests one declared operator.",
                         "operator": operator, "fields": {key: key for key in roles}, "parameters": parameters or {},
                         "unit": "fraction", "direction": "higher_is_better", "aggregation": "mean",
                         "evidence": "computed", "limitations": "A synthetic test, not a quality verdict.", "target": target}]}


def run(tmp_path, pack, rows):
    pack_path = tmp_path / "definition.json"
    pack_path.write_text(json.dumps(pack), encoding="utf-8")
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "result"
    report = metrics.evaluate_metrics(source, output, pack_path=pack_path)
    scores = [json.loads(line) for line in (output / "scores.jsonl").read_text().splitlines()]
    return report, scores


@pytest.mark.parametrize("operator,parameters,row,expected", [
    ("exact_match", {}, {"prediction": " cafe\u0301\n42 ", "reference": "café 42"}, 1),
    ("exact_match", {}, {"prediction": "Answer", "reference": "answer"}, 0),
    ("exact_match", {}, {"prediction": "", "reference": "answer"}, 0),
    ("token_f1", {}, {"prediction": "x x y", "reference": "x y y y"}, 4 / 7),
    ("token_f1", {}, {"prediction": "", "reference": "x"}, 0),
    ("numeric_tolerance", {"absolute_tolerance": 0.1}, {"prediction": 1.1, "reference": 1}, 1),
    ("numeric_tolerance", {"absolute_tolerance": 0.1}, {"prediction": 1.1001, "reference": 1}, 0),
    ("numeric_tolerance", {"relative_tolerance": 0.1}, {"prediction": -11, "reference": -10}, 1),
    ("numeric_tolerance", {"relative_tolerance": 1}, {"prediction": 1, "reference": 0}, 0),
    ("numeric_tolerance", {"relative_tolerance": 1.5}, {"prediction": 1e308, "reference": -1e308}, 0),
    ("set_precision", {}, {"prediction": ["a", "a", "b"], "reference": ["a", "a", "c"]}, 0.5),
    ("set_precision", {}, {"prediction": [], "reference": ["a"]}, 0),
    ("set_precision", {}, {"prediction": ["a"], "reference": []}, 0),
    ("set_recall", {}, {"prediction": ["a", "a", "b"], "reference": ["a", "c", "d"]}, 1 / 3),
    ("ratio", {}, {"numerator": 1, "denominator": 4}, 0.25),
    ("boolean", {}, {"value": True}, 1),
    ("boolean", {}, {"value": False}, 0),
    ("rubric_score", {}, {"value": 3}, 0.75),
    ("rubric_score", {"min": 1, "max": 5}, {"value": 3}, 0.5),
    ("json_valid", {}, {"value": '{"a": [1, true, null]}'}, 1),
    ("json_valid", {}, {"value": '[]'}, 0),
    ("json_valid", {"required_type": "array"}, {"value": '[]'}, 1),
    ("json_valid", {"required_type": "any"}, {"value": 'null'}, 1),
    ("json_valid", {}, {"value": '{"a":NaN}'}, 0),
    ("json_valid", {}, {"value": '{"a":Infinity}'}, 0),
    ("json_valid", {}, {"value": '{"a":1,"a":2}'}, 0),
    ("json_valid", {}, {"value": '{"x":1e9999999999999999999}'}, 1),
    ("json_valid", {}, {"value": '{"x":-1e-9999999999999999999}'}, 1),
    ("json_valid", {}, {"value": '{"x":' + '9' * 5000 + '}'}, 1),
    ("json_valid", {"required_type": "any"}, {"value": '1e9999999999999999999'}, 1),
    ("json_valid", {}, {"value": '```json\n{}\n```'}, 0),
])
def test_offline_operator_scores(tmp_path, operator, parameters, row, expected):
    report, scores = run(tmp_path, definition(operator, parameters=parameters), [row])
    result = scores[0]["metrics"]["check"]
    assert result["status"] == "evaluated"
    assert result["score"] == pytest.approx(expected)
    assert report["metrics"]["check"]["mean"] == pytest.approx(expected)
    assert report["metrics"]["check"]["coverage"] == 1


@pytest.mark.parametrize("operator,row,status", [
    ("exact_match", {"prediction": "anything", "reference": "   "}, "invalid"),
    ("exact_match", {"prediction": "x"}, "missing"),
    ("exact_match", {"prediction": None, "reference": "x"}, "invalid"),
    ("numeric_tolerance", {"prediction": True, "reference": 1}, "invalid"),
    ("set_precision", {"prediction": [], "reference": []}, "not_applicable"),
    ("set_recall", {"prediction": ["a"], "reference": []}, "not_applicable"),
    ("set_recall", {"prediction": "a", "reference": ["a"]}, "invalid"),
    ("set_recall", {"prediction": [""], "reference": ["a"]}, "invalid"),
    ("ratio", {"numerator": 0, "denominator": 0}, "not_applicable"),
    ("ratio", {"numerator": 1, "denominator": 0}, "invalid"),
    ("ratio", {"numerator": -1, "denominator": 2}, "invalid"),
    ("ratio", {"numerator": 1.0, "denominator": 2}, "invalid"),
    ("ratio", {"numerator": True, "denominator": 2}, "invalid"),
    ("rubric_score", {"value": 5}, "invalid"),
    ("rubric_score", {"value": -1}, "invalid"),
    ("rubric_score", {"value": 1.0}, "invalid"),
    ("boolean", {"value": 1}, "invalid"),
    ("json_valid", {"value": {}}, "invalid"),
])
def test_invalid_missing_and_inapplicable_values_never_become_zero_scores(tmp_path, operator, row, status):
    report, scores = run(tmp_path, definition(operator, target={"minimum": 0, "minimum_coverage": 0}), [row])
    result = scores[0]["metrics"]["check"]
    assert result["status"] == status and result["score"] is None
    aggregate = report["metrics"]["check"]
    assert aggregate["counts"][status] == 1
    assert aggregate["mean"] is None and aggregate["target_status"] == "no_data"
    assert aggregate["coverage"] == (None if status == "not_applicable" else 0)


def test_coverage_and_means_use_different_correct_denominators(tmp_path):
    pack = definition("ratio", target={"minimum": 0.4, "minimum_coverage": 0.5})
    report, _ = run(tmp_path, pack, [{"numerator": 1, "denominator": 2}, {"numerator": 0, "denominator": 0},
                                  {}, {"numerator": True, "denominator": 2}, {"numerator": -1, "denominator": 2}])
    result = report["metrics"]["check"]
    assert result["counts"] == {"evaluated": 1, "missing": 1, "invalid": 2, "not_applicable": 1, "eligible": 4}
    assert result["mean"] == 0.5 and result["coverage"] == 0.25
    assert result["target_status"] == "insufficient_coverage"
    assert "score" not in report and "grade" not in report and "target_status" not in report


@pytest.mark.parametrize("minimum,status", [(0.5, "met"), (0.6, "below_target")])
def test_optional_target_is_per_metric_only(tmp_path, minimum, status):
    report, _ = run(tmp_path, definition("ratio", target={"minimum": minimum, "minimum_coverage": 1}),
                    [{"numerator": 1, "denominator": 2}])
    assert report["metrics"]["check"]["target_status"] == status


def test_numeric_units_must_be_explicit_and_identical(tmp_path):
    pack = definition("numeric_tolerance", parameters={"absolute_tolerance": 1000})
    for role in ("prediction_unit", "reference_unit"):
        pack["field_schema"][role] = {"type": "string", "description": "Exact unit ID."}
        pack["metrics"][0]["fields"][role] = role
    report, scores = run(tmp_path, pack, [
        {"prediction": 1, "reference": 1000, "prediction_unit": "USD_thousand", "reference_unit": "USD"},
        {"prediction": 1, "reference": 1, "prediction_unit": "", "reference_unit": ""},
        {"prediction": 1, "reference": 1, "prediction_unit": "USD"},
    ])
    assert [item["metrics"]["check"]["status"] for item in scores] == ["evaluated", "invalid", "missing"]
    assert scores[0]["metrics"]["check"]["reason"] == "unit_mismatch_no_conversion"
    assert report["metrics"]["check"]["mean"] == 0


def test_nested_fields_and_unused_metadata_do_not_implicitly_require_metadata(tmp_path):
    pack = definition()
    pack["field_schema"] = {"response.text": {"type": "string", "description": "Prediction"},
                            "label.text": {"type": "string", "description": "Reference"},
                            "metadata.reviewer_id": {"type": "string", "description": "Optional provenance"}}
    pack["metrics"][0]["fields"] = {"prediction": "response.text", "reference": "label.text"}
    _, scores = run(tmp_path, pack, [{"response": {"text": "x"}, "label": {"text": "x"}},
                                   {"response": {}, "label": {"text": "x"}},
                                   {"response": "not an object", "label": {"text": "x"}}])
    assert [item["metrics"]["check"]["status"] for item in scores] == ["evaluated", "missing", "invalid"]


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(schema_version=True),
    lambda p: p.update(domain="../escape"),
    lambda p: p.update(references=["file:///etc/passwd"]),
    lambda p: p["metrics"][0].update(operator="eval_python"),
    lambda p: p["metrics"][0].update(parameters={"typo": 1}),
    lambda p: p["metrics"][0].update(fields={"prediction": "prediction", "reference": "undeclared"}),
    lambda p: p["field_schema"]["reference"].update(type="number"),
    lambda p: p["metrics"].append(copy.deepcopy(p["metrics"][0])),
    lambda p: p["metrics"][0].update(target={"minimum": True, "minimum_coverage": 1}),
    lambda p: p["metrics"][0].update(target={"minimum": 1.1, "minimum_coverage": 1}),
    lambda p: p["metrics"][0].update(direction="lower_is_better"),
])
def test_bad_definitions_fail_before_any_output_directory_is_created(tmp_path, mutation):
    pack = definition()
    mutation(pack)
    path = tmp_path / "pack.json"
    path.write_text(json.dumps(pack))
    source = tmp_path / "input.jsonl"
    source.write_text("{}\n")
    with pytest.raises(ValueError, match="Invalid metric pack"):
        metrics.evaluate_metrics(source, tmp_path / "new-parent/results", pack_path=path)
    assert not (tmp_path / "new-parent").exists()


@pytest.mark.parametrize("operator,parameters", [
    ("numeric_tolerance", {"absolute_tolerance": -1}),
    ("numeric_tolerance", {"relative_tolerance": True}),
    ("numeric_tolerance", {"relative_tolerance": float("inf")}),
    ("rubric_score", {"min": 4, "max": 4}),
    ("rubric_score", {"min": False}),
    ("json_valid", {"required_type": "python"}),
])
def test_bad_operator_parameters_fail_closed(tmp_path, operator, parameters):
    path = tmp_path / "pack.json"
    path.write_text(json.dumps(definition(operator, parameters=parameters)))
    with pytest.raises(ValueError):
        metrics.load_metric_pack(path=path)


@pytest.mark.parametrize("suffix", [b"{broken\n", b"[]\n", b'{"a":1,"a":2}\n', b'{"a":NaN}\n', b'{"a":Infinity}\n',
                                    b'x' * (metrics.MAX_LINE_BYTES + 1), '{"a":"x"}'.encode("utf-16")])
def test_late_input_errors_leave_no_partial_output(tmp_path, suffix):
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(json.dumps(definition()))
    source = tmp_path / "input.jsonl"
    source.write_bytes(b'{"prediction":"x","reference":"x"}\n' + suffix)
    with pytest.raises(ValueError, match="line 2"):
        metrics.evaluate_metrics(source, tmp_path / "result", pack_path=pack_path)
    assert not (tmp_path / "result").exists()
    assert not list(tmp_path.glob(".jev-metrics-*"))


def test_valid_json_numeric_overflow_is_an_invalid_field_not_an_infinite_score(tmp_path):
    pack = tmp_path / "pack.json"
    pack.write_text(json.dumps(definition("numeric_tolerance")))
    source = tmp_path / "input.jsonl"
    source.write_text('{"prediction":1e999,"reference":1}\n')
    report = metrics.evaluate_metrics(source, tmp_path / "result", pack_path=pack)
    assert report["metrics"]["check"]["counts"]["invalid"] == 1
    assert report["metrics"]["check"]["mean"] is None


def test_utf16_metric_pack_is_rejected_before_output(tmp_path):
    path = tmp_path / "pack.json"
    path.write_bytes(json.dumps(definition()).encode("utf-16"))
    with pytest.raises(ValueError, match="Invalid metric pack"):
        metrics.load_metric_pack(path=path)


def test_outputs_are_reproducible_private_bounded_and_never_execute_text(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Offline scoring must not spawn processes."))
    pack = definition("json_valid")
    secret = "PRIVATE_RESPONSE_DO_NOT_COPY"
    malicious = "__import__('pathlib').Path('should_not_exist').touch()"
    report, scores = run(tmp_path, pack, [{"id": "shared", "value": secret}, {"id": "shared", "value": malicious},
                                        {"id": "x" * 201, "value": "{}"}])
    result_text = (tmp_path / "result/scores.jsonl").read_text()
    assert secret not in result_text and malicious not in result_text
    assert [row["id"] for row in scores] == ["shared", "shared", None]
    assert [row["line"] for row in scores] == [1, 2, 3]
    assert report["input"]["sha256"] == hashlib.sha256((tmp_path / "input.jsonl").read_bytes()).hexdigest()
    assert report["pack"]["sha256"] == hashlib.sha256((tmp_path / "definition.json").read_bytes()).hexdigest()
    assert (tmp_path / "result/metrics.json").read_bytes() == (tmp_path / "definition.json").read_bytes()
    assert not (tmp_path / "should_not_exist").exists()
    assert set(report["artifacts"].values()) == {"report.json", "scores.jsonl", "metrics.json"}
    before = (tmp_path / "result/report.json").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        metrics.evaluate_metrics(tmp_path / "input.jsonl", tmp_path / "result", pack_path=tmp_path / "definition.json")
    assert (tmp_path / "result/report.json").read_bytes() == before


def test_empty_input_and_blanks_have_no_fictional_quality_score(tmp_path):
    path = tmp_path / "pack.json"
    path.write_text(json.dumps(definition()))
    source = tmp_path / "input.jsonl"
    source.write_bytes(b"\n  \n")
    report = metrics.evaluate_metrics(source, tmp_path / "result", pack_path=path)
    assert report["input"]["rows"] == 0 and report["input"]["blank_lines"] == 2
    assert report["metrics"]["check"]["mean"] is None and report["metrics"]["check"]["coverage"] is None
    assert report["input"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_metric_definition_size_and_selection_are_bounded(tmp_path):
    source = tmp_path / "large.json"
    source.write_bytes(b" " * (metrics.MAX_PACK_BYTES + 1))
    with pytest.raises(ValueError, match="1 MiB"):
        metrics.load_metric_pack(path=source)
    with pytest.raises(ValueError, match="exactly one"):
        metrics.load_metric_pack()
    with pytest.raises(ValueError, match="exactly one"):
        metrics.load_metric_pack(domain="general", path=source)


@pytest.mark.parametrize("domain", ["general", "finance", "code", "enterprise", "legal", "medical"])
def test_every_bundled_pack_initializes_and_evaluates_its_synthetic_example(tmp_path, domain):
    initialized = metrics.init_metric_pack(domain, tmp_path / "starter")
    assert initialized["status"] == "initialized"
    assert metrics.load_metric_pack(path=tmp_path / "starter/metrics.json") == metrics.load_metric_pack(domain)
    report = metrics.evaluate_metrics(tmp_path / "starter/example.jsonl", tmp_path / "result",
                                      pack_path=tmp_path / "starter/metrics.json")
    assert report["input"]["rows"] == 3
    assert report["pack"]["domain"] == domain and report["pack"]["version"]
    assert any(value["counts"]["missing"] for value in report["metrics"].values())
    assert all(value["target_status"] == "not_configured" for value in report["metrics"].values())
    scores = [json.loads(line) for line in (tmp_path / "result/scores.jsonl").read_text().splitlines()]
    assert all(result["status"] == "evaluated" and result["score"] == 1 for result in scores[0]["metrics"].values())
    with pytest.raises(ValueError, match="already exists"):
        metrics.init_metric_pack(domain, tmp_path / "starter")
    catalog = {pack["domain"]: pack for pack in metrics.list_metric_packs()}
    assert catalog[domain]["metric_count"] == len(report["metrics"])
