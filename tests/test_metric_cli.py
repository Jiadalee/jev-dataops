"""Exercise shipped packs through the installed CLI, including custom targets."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


DOMAINS = ("general", "finance", "code", "enterprise", "legal", "medical")


def command(*args):
    return subprocess.run([sys.executable, "-m", "jev_dataops", *map(str, args)],
                          text=True, capture_output=True, timeout=20)


def test_cli_catalog_exposes_all_six_shipped_domains():
    result = command("list-metrics")
    assert result.returncode == 0, result.stderr
    catalog = json.loads(result.stdout)
    assert {item["domain"] for item in catalog} == set(DOMAINS)
    assert all(item["metric_count"] >= 3 and item["field_count"] >= 3 for item in catalog)


@pytest.mark.parametrize("domain", DOMAINS)
def test_domain_starter_can_be_copied_customized_and_scored(tmp_path, domain):
    starter, output = tmp_path / "starter", tmp_path / "results"
    initialized = command("init-metrics", "--domain", domain, "--output", starter)
    assert initialized.returncode == 0, initialized.stderr
    pack_file = starter / "metrics.json"
    pack = json.loads(pack_file.read_text())
    assert pack["domain"] == domain
    assert all(metric["target"] is None for metric in pack["metrics"])
    # The fixture deliberately has a missing row. Even a low mean target must
    # not pass when its user-required coverage is incomplete.
    pack["metrics"][0]["target"] = {"minimum": 0.1, "minimum_coverage": 1.0}
    pack_file.write_text(json.dumps(pack), encoding="utf-8")
    scored = command("evaluate-metrics", "--input", starter / "example.jsonl",
                     "--pack", pack_file, "--output", output)
    assert scored.returncode == 0, scored.stderr
    report = json.loads(scored.stdout)
    assert report == json.loads((output / "report.json").read_text())
    assert report["input"]["rows"] == 3
    metric = report["metrics"][pack["metrics"][0]["id"]]
    assert metric["counts"]["evaluated"] == 2
    assert metric["counts"]["missing"] == 1
    assert metric["coverage"] == pytest.approx(2 / 3)
    assert metric["target_status"] == "insufficient_coverage"
    assert len((output / "scores.jsonl").read_text().splitlines()) == 3
    assert json.loads((output / "metrics.json").read_text()) == pack
    # A second run cannot erase the existing result set.
    previous = (output / "report.json").read_bytes()
    repeated = command("evaluate-metrics", "--input", starter / "example.jsonl",
                       "--domain", domain, "--output", output)
    assert repeated.returncode == 2
    assert (output / "report.json").read_bytes() == previous


def test_cli_requires_one_metric_definition_source(tmp_path):
    result = command("evaluate-metrics", "--input", tmp_path / "input.jsonl", "--output", tmp_path / "results")
    assert result.returncode == 2
    result = command("evaluate-metrics", "--input", tmp_path / "input.jsonl", "--output", tmp_path / "results",
                     "--domain", "finance", "--pack", tmp_path / "metrics.json")
    assert result.returncode == 2
    assert not (tmp_path / "results").exists()
