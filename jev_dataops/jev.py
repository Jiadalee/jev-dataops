"""Bounded, environment-only clients for Jev's structured decisions API.

OpenRouter uses its alpha decisions endpoint; TypeSafe uses System One.
These are decision APIs, not OpenAI-compatible chat completions endpoints.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from urllib import error, request

ENDPOINTS = {
    "openrouter": "https://openrouter.ai/api/alpha/decisions",
    "typesafe": "https://api.typesafe.ai/v1/systemone",
}
MODELS = {"openrouter": "~typesafe/jev-latest", "typesafe": "jev-latest"}
ENV_KEYS = {"openrouter": "OPENROUTER_API_KEY", "typesafe": "TYPESAFE_API_KEY"}
MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


class JevAPIError(RuntimeError):
    """Sanitized exception: never includes credentials, payloads or HTTP bodies."""

    def __init__(self, category, status=None):
        self.category = category
        self.status = status
        super().__init__(category + (f" (HTTP {status})" if status else ""))


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _number(value, maximum=1):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid numeric response")
    if not math.isfinite(value) or not 0 <= value <= maximum:
        raise ValueError("numeric response outside range")
    return value


def validate_response(response, rubric, confidence):
    """Validate every dimension before admitting a row. Raises ValueError on error."""
    try:
        if not isinstance(response, dict) or not isinstance(response.get("model"), str) or not response["model"].strip():
            raise ValueError("missing resolved model")
        response["model"].encode("utf-8")
        answers = response["answers"]
        questions = rubric["questions"]
        if not isinstance(answers, dict) or set(answers) != set(questions):
            raise ValueError("unexpected answer dimensions")
        dimensions = {}
        for name, question in questions.items():
            answer, gate = answers[name], rubric["gates"][name]
            kind = question["type"]
            if not isinstance(answer, dict) or answer.get("type") != kind:
                raise ValueError("wrong answer type")
            details = {"type": kind}
            if kind != "noul":
                certainty = _number(answer["confidence"])
                probabilities = answer["probabilities"]
                options = set(question["criteria"]) if kind == "choice" else {str(i) for i in range(len(question["criteria"]))}
                if not isinstance(probabilities, dict) or set(probabilities) != options:
                    raise ValueError("unexpected probability options")
                if not math.isclose(sum(_number(p) for p in probabilities.values()), 1, abs_tol=1e-4):
                    raise ValueError("invalid probability distribution")
                details["confidence"] = certainty
            if kind == "choice":
                value = answer["choice"]
                if not isinstance(value, str) or value not in options or probabilities[value] < max(probabilities.values()):
                    raise ValueError("invalid selected choice")
                decision = next(key for key in ("keep", "review", "reject") if value in gate[key])
            elif kind in {"score", "noul"}:
                value = _number(answer[kind], len(question["criteria"]) - 1 if kind == "score" else 1)
                if kind == "score" and not math.isclose(value, sum(int(k) * p for k, p in probabilities.items()), abs_tol=1e-4):
                    raise ValueError("inconsistent expected score")
                decision = "keep" if value >= gate["keep_at_or_above"] else "reject" if value <= gate["reject_at_or_below"] else "review"
            else:
                raise ValueError("unsupported answer type")
            if kind != "noul" and certainty < confidence:
                decision = "review"
            details.update({"value": value, "decision": decision})
            dimensions[name] = details
        decision = "reject" if any(d["decision"] == "reject" for d in dimensions.values()) else "review" if any(d["decision"] == "review" for d in dimensions.values()) else "keep"
        return {"decision": decision, "dimensions": dimensions, "model": response["model"], "reason": "jev_decision"}
    except (KeyError, TypeError, AttributeError, StopIteration, OverflowError) as exc:
        raise ValueError("invalid Jev response") from exc


class JevClient:
    """A per-run client; request budget includes retries and is shared across workers."""

    def __init__(self, provider, max_requests=1000, timeout=20, attempts=3):
        if not isinstance(provider, str) or provider not in ENDPOINTS:
            raise ValueError("unsupported Jev provider")
        self.provider = provider
        self.api_key = os.environ.get(ENV_KEYS[provider], "").strip()
        if not self.api_key:
            raise ValueError(f"Set {ENV_KEYS[provider]} to use live Jev screening")
        self.max_requests = max_requests
        self.timeout = timeout
        self.attempts = attempts
        self.requests = 0
        self._lock = threading.Lock()

    def __call__(self, payload, cancelled=None):
        stopped = cancelled or (lambda: False)
        try:
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (ValueError, TypeError, UnicodeError):
            raise JevAPIError("invalid_request") from None
        if len(body) > MAX_REQUEST_BYTES:
            raise JevAPIError("request_too_large")
        for attempt in range(self.attempts):
            if stopped():
                raise JevAPIError("cancelled")
            with self._lock:
                if self.requests >= self.max_requests:
                    raise JevAPIError("request_budget_exhausted")
                self.requests += 1
            req = request.Request(ENDPOINTS[self.provider], data=body, headers={"Content-Type": "application/json"})
            req.add_unredirected_header("Authorization", "Bearer " + self.api_key)
            # A fresh opener has no mutable connection state shared between workers.
            opener = request.build_opener(_NoRedirect())
            retryable = False
            delay = min(2 ** attempt, 5)
            try:
                with opener.open(req, timeout=self.timeout) as response:
                    raw = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise JevAPIError("response_too_large")
                try:
                    return json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeError, RecursionError):
                    raise JevAPIError("invalid_json") from None
            except error.HTTPError as exc:
                status = exc.code
                retryable = status in {408, 429, 500, 502, 503, 504, 529}
                category = "authentication" if status in {401, 403} else "rate_limit" if status == 429 else "provider_http_error"
                failure = JevAPIError(category, status)
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                exc.close()
                if retry_after:
                    try:
                        advised = float(retry_after)
                        if not math.isfinite(advised) or advised > 5:
                            retryable = False
                        else:
                            delay = max(0, advised)
                    except ValueError:
                        # HTTP-date or malformed backoffs are left to a later run.
                        retryable = False
            except (error.URLError, TimeoutError, ConnectionError, OSError):
                failure = JevAPIError("network_error")
                retryable = True
            if not retryable or attempt + 1 == self.attempts:
                raise failure from None
            deadline = time.monotonic() + delay
            while time.monotonic() < deadline:
                if stopped():
                    raise JevAPIError("cancelled")
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        raise JevAPIError("provider_unavailable")
