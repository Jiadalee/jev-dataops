import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jev_dataops.jev import JevAPIError, JevClient, validate_response
from jev_dataops.screening import MAX_ROW_BYTES, normalize_record, screen_dataset


def valid_response(questions, confidence=0.95):
    answers = {}
    for name, question in questions.items():
        choice = next(iter(question["criteria"]))
        answers[name] = {"type": "choice", "choice": choice, "confidence": confidence,
                         "probabilities": {key: 1 if key == choice else 0 for key in question["criteria"]}}
    return {"model": "test-jev", "answers": answers}


class FakeClient:
    calls = []
    behavior = None

    def __init__(self, provider, max_requests, timeout, attempts):
        self.requests = 0

    def __call__(self, payload, cancelled=None):
        self.requests += 1
        type(self).calls.append(payload)
        if type(self).behavior:
            return type(self).behavior(payload)
        return valid_response(payload["questions"])


class ScreeningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "dataset.jsonl"
        self.out = self.root / "result"
        FakeClient.calls = []
        FakeClient.behavior = None

    def write(self, rows):
        self.source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def live(self, **config):
        with patch("jev_dataops.screening.JevClient", FakeClient):
            return screen_dataset(self.source, self.out, {"provider": "openrouter", **config})

    def records(self, partition):
        return [json.loads(line) for line in (self.out / f"{partition}.jsonl").read_text().splitlines()]

    def test_normalizes_layouts_without_metadata_leak(self):
        for row, expected in [
            ({"text": "Training data", "secret_metadata": "local"}, {"text": "Training data"}),
            ({"messages": [{"role": "user", "content": "Hello there", "private": "local"}], "metadata": {"a": 1}}, {"messages": [{"role": "user", "content": "Hello there"}]}),
            ({"instruction": "Explain", "input": "Gravity", "output": "Mass attracts mass", "owner": "local"}, {"instruction": "Explain", "input": "Gravity", "output": "Mass attracts mass"}),
            ({"prompt": "Explain", "response": "A complete answer", "id": 1}, {"prompt": "Explain", "response": "A complete answer"}),
        ]:
            self.assertEqual(normalize_record(row), expected)

    def test_metadata_preserved_locally_and_not_sent(self):
        row = {"text": "A sufficiently useful training example.", "private_metadata": {"customer": "private"}}
        self.write([row])
        report = self.live()
        self.assertTrue(report["complete"])
        self.assertEqual(self.records("keep"), [row])
        self.assertEqual(FakeClient.calls[0]["state"], {"text": row["text"]})
        self.assertNotIn("private_metadata", json.dumps(FakeClient.calls))

    def test_malformed_message_roles_are_reviewed_without_aborting(self):
        roles = [[], {}, 123, True, None]
        malformed = [{"messages": [{"role": role, "content": "A useful training example"}]} for role in roles]
        self.write(malformed + [{"text": "A valid row after malformed message roles"}])
        report = screen_dataset(self.source, self.out, {})
        self.assertTrue(report["complete"])
        self.assertEqual(report["counts"]["review"], len(roles))
        self.assertEqual(report["counts"]["keep"], 1)
        self.assertEqual(self.records("review"), malformed)

    def test_cache_resume_does_not_turn_first_row_into_duplicate(self):
        self.write([{"text": "A useful training example", "id": 1}, {"text": "A useful training example", "id": 2}])
        first = self.live(concurrency=1)
        second = self.live(concurrency=2, max_requests=3)
        self.assertEqual(first["counts"], {"total": 2, "keep": 1, "review": 0, "reject": 1, "duplicates": 1})
        self.assertEqual(first["counts"], second["counts"])
        self.assertEqual(len(FakeClient.calls), 1)
        self.assertEqual(second["cache_hits"], 1)
        self.assertEqual(second["api_requests"], 0)
        self.assertEqual(self.records("keep")[0]["id"], 1)

    def test_semantic_config_change_invalidates_cache(self):
        self.write([{"text": "A useful training example"}])
        self.live(confidence=0.85)
        report = self.live(confidence=0.99)
        self.assertEqual(len(FakeClient.calls), 2)
        self.assertEqual(report["counts"]["review"], 1)

    def test_invalid_response_never_kept_or_cached(self):
        self.write([{"text": "A useful training example"}])
        FakeClient.behavior = lambda payload: {"model": "test", "answers": {}}
        first = self.live()
        self.assertEqual(first["counts"]["keep"], 0)
        self.assertEqual(first["counts"]["review"], 1)
        FakeClient.behavior = None
        second = self.live()
        self.assertEqual(second["counts"]["keep"], 1)
        self.assertEqual(len(FakeClient.calls), 2)

    def test_authentication_failure_marks_incomplete(self):
        self.write([{"text": f"Useful unique record number {i}"} for i in range(20)])
        def fail(payload):
            raise JevAPIError("authentication", 401)
        FakeClient.behavior = fail
        report = self.live(concurrency=1)
        self.assertFalse(report["complete"])
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["counts"]["keep"], 0)
        self.assertFalse(report["input_exhausted"])
        self.assertLessEqual(report["processed"], 2)

    def test_invalid_rows_and_utf8_are_reviewed(self):
        self.source.write_bytes(b'{broken}\n[1,2]\n{"text": "\xff"}\n{"text":"A valid long enough example"}\n')
        report = screen_dataset(self.source, self.out, {})
        self.assertEqual(report["counts"]["review"], 3)
        self.assertEqual(report["counts"]["keep"], 1)
        self.assertEqual(report["error_count"], 3)
        self.assertEqual(report["mode"], "demo_rule_based")

    def test_oversized_jsonl_row_recovers_next_row(self):
        with self.source.open("wb") as stream:
            stream.write(b'x' * (MAX_ROW_BYTES + 1) + b'\n')
            stream.write(b'{"text":"The next record remains usable"}\n')
        report = screen_dataset(self.source, self.out, {})
        self.assertEqual(report["counts"]["review"], 1)
        self.assertEqual(report["counts"]["keep"], 1)
        self.assertTrue(report["complete"])

    def test_csv_quoted_newlines_and_metadata(self):
        self.source = self.root / "dataset.csv"
        self.source.write_text('text,owner\n"A multiline\nuseful passage",local\n"Another useful passage",other\n', encoding="utf-8")
        report = self.live()
        self.assertEqual(report["counts"]["keep"], 2)
        self.assertEqual(self.records("keep")[0]["owner"], "local")
        self.assertNotIn("owner", json.dumps(FakeClient.calls))

    def test_corrupt_csv_is_incomplete(self):
        self.source = self.root / "dataset.csv"
        self.source.write_text('text,owner\n"unterminated\n', encoding="utf-8")
        report = screen_dataset(self.source, self.out, {})
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["counts"]["keep"], 0)

    def test_cancellation_does_not_report_complete(self):
        self.write([{"text": "A useful training example"}])
        report = screen_dataset(self.source, self.out, {}, cancelled=lambda: True)
        self.assertEqual(report["status"], "cancelled")
        self.assertFalse(report["complete"])
        self.assertEqual(report["processed"], 0)

    def test_demo_only_checks_rules_and_has_no_semantic_dimensions(self):
        self.write([{"text": "Just enough"}, {"text": "a"}, {"text": "Email example@example.com for details."}])
        report = screen_dataset(self.source, self.out, {})
        self.assertEqual(report["counts"], {"total": 3, "keep": 1, "review": 1, "reject": 1, "duplicates": 0})
        self.assertEqual(report["dimensions"], {})
        self.assertIn("no Jev", report["notice"])

    def test_nonfinite_or_boolean_configuration_rejected(self):
        self.write([])
        for config in ({"confidence": float("nan")}, {"confidence": True}, {"confidence": 10 ** 1000}, {"concurrency": True}, {"max_chars": 0}, {"rubric": "../../secret"}, {"rubric": []}, {"provider": {}}):
            with self.assertRaises(ValueError):
                screen_dataset(self.source, self.out, config)

    def test_provider_key_only_from_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
                JevClient("openrouter")

    def test_request_budget_is_enforced_before_network(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-not-secret"}):
            client = JevClient("openrouter", max_requests=0)
            with patch("jev_dataops.jev.request.build_opener") as opener:
                with self.assertRaisesRegex(JevAPIError, "request_budget_exhausted"):
                    client({"state": {"text": "demo"}})
                opener.assert_not_called()


class ResponseValidationTests(unittest.TestCase):
    def setUp(self):
        self.rubric = json.loads((Path(__file__).parents[1] / "jev_dataops/rubrics/general.json").read_text())
        self.valid = valid_response(self.rubric["questions"])

    def test_valid_and_low_confidence(self):
        self.assertEqual(validate_response(self.valid, self.rubric, 0.85)["decision"], "keep")
        self.assertEqual(validate_response(self.valid, self.rubric, 0.99)["decision"], "review")

    def test_invalid_distributions_types_and_missing_dimensions(self):
        mutations = [
            lambda r: r["answers"].pop("quality"),
            lambda r: r["answers"]["quality"].update(confidence=True),
            lambda r: r["answers"]["quality"].update(confidence=float("nan")),
            lambda r: r["answers"]["quality"].update(type="score"),
            lambda r: r["answers"]["quality"]["probabilities"].update(good=0.2),
            lambda r: r["answers"]["quality"].update(choice="bad"),
            lambda r: r["answers"]["quality"].update(choice=[]),
            lambda r: r["answers"]["quality"].update(probabilities=[]),
            lambda r: r["answers"]["quality"].update(confidence={}),
            lambda r: r.update(model="invalid-surrogate-\ud800"),
            lambda r: r.pop("model"),
        ]
        for mutate in mutations:
            response = copy.deepcopy(self.valid)
            mutate(response)
            with self.assertRaises(ValueError):
                validate_response(response, self.rubric, 0.85)


if __name__ == "__main__":
    unittest.main()
