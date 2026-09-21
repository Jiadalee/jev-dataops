"""Bounded, environment-only clients for Jev's structured decisions API.

OpenRouter uses its alpha decisions endpoint; TypeSafe uses System One.
These are decision APIs, not OpenAI-compatible chat completions endpoints.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import socket
import ssl
import threading
import time
from base64 import b64encode
from urllib.parse import unquote, urlsplit
from urllib.request import getproxies, proxy_bypass

ENDPOINTS = {
    "openrouter": "https://openrouter.ai/api/alpha/decisions",
    "typesafe": "https://api.typesafe.ai/v1/systemone",
}
MODELS = {"openrouter": "~typesafe/jev-latest", "typesafe": "jev-latest"}
ENV_KEYS = {"openrouter": "OPENROUTER_API_KEY", "typesafe": "TYPESAFE_API_KEY"}
MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
# Jev reports probabilities rounded to two decimals, so a well-formed
# distribution can legitimately sum to 0.99 or 1.01.
PROBABILITY_TOLERANCE = 0.015
# Gate defaults when a rubric gate does not say otherwise. `min_probability` is the
# probability mass the model must put on the chosen decision class before a row is
# kept; `max_reject_probability` is the mass on the reject class above which a row
# goes to review even when the argmax says keep.
DEFAULT_GATE = {"use_confidence": True, "min_probability": 0.5, "max_reject_probability": 1.0}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


class JevAPIError(RuntimeError):
    """Sanitized exception: never includes credentials, payloads or HTTP bodies."""

    def __init__(self, category, status=None):
        self.category = category
        self.status = status
        super().__init__(category + (f" (HTTP {status})" if status else ""))


def _number(value, maximum=1):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid numeric response")
    if not math.isfinite(value) or not 0 <= value <= maximum:
        raise ValueError("numeric response outside range")
    return value


def effective_thresholds(rubric, confidence):
    """The per-dimension gate settings a run will apply, for the report and the audit."""
    result = {}
    for name in rubric["questions"]:
        gate = rubric["gates"][name]
        settings = dict(DEFAULT_GATE)
        settings.update({key: gate[key] for key in DEFAULT_GATE if key in gate})
        settings["min_confidence"] = gate.get("min_confidence", confidence) if settings["use_confidence"] else None
        result[name] = settings
    return result


def _usage(response):
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return {}
    clean = {}
    for key in ("input_tokens", "output_tokens", "cost"):
        value = usage.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
            clean[key] = value
    return clean


def validate_response(response, rubric, confidence):
    """Validate every dimension before admitting a row. Raises ValueError on error.

    A choice dimension is kept only when the argmax falls in the gate's keep set,
    the mass on that keep set reaches `min_probability`, the mass on the reject
    set stays under `max_reject_probability`, and -- unless the gate turns it off --
    Jev's own confidence reaches `min_confidence` (the run's confidence by default).
    """
    thresholds = effective_thresholds(rubric, confidence)
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
            answer, gate, settings = answers[name], rubric["gates"][name], thresholds[name]
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
                if not math.isclose(sum(_number(p) for p in probabilities.values()), 1, abs_tol=PROBABILITY_TOLERANCE):
                    raise ValueError("invalid probability distribution")
                details["confidence"] = certainty
            if kind == "choice":
                value = answer["choice"]
                if not isinstance(value, str) or value not in options or probabilities[value] < max(probabilities.values()):
                    raise ValueError("invalid selected choice")
                decision = next(key for key in ("keep", "review", "reject") if value in gate[key])
                mass = {key: sum(probabilities[option] for option in gate[key] if option in probabilities) for key in ("keep", "review", "reject")}
                details["probability"] = probabilities[value]
                details["reject_probability"] = mass["reject"]
                if decision == "keep":
                    if mass["keep"] < settings["min_probability"]:
                        decision = "review"
                        details["gate"] = "keep_probability_below_threshold"
                    elif mass["reject"] > settings["max_reject_probability"]:
                        decision = "review"
                        details["gate"] = "reject_probability_above_threshold"
            elif kind in {"score", "noul"}:
                value = _number(answer[kind], len(question["criteria"]) - 1 if kind == "score" else 1)
                if kind == "score" and not math.isclose(value, sum(int(k) * p for k, p in probabilities.items()), abs_tol=PROBABILITY_TOLERANCE * 10):
                    raise ValueError("inconsistent expected score")
                decision = "keep" if value >= gate["keep_at_or_above"] else "reject" if value <= gate["reject_at_or_below"] else "review"
            else:
                raise ValueError("unsupported answer type")
            if kind != "noul" and settings["use_confidence"] and certainty < settings["min_confidence"] and decision == "keep":
                decision = "review"
                details["gate"] = "confidence_below_threshold"
            details.update({"value": value, "decision": decision})
            dimensions[name] = details
        decision = "reject" if any(d["decision"] == "reject" for d in dimensions.values()) else "review" if any(d["decision"] == "review" for d in dimensions.values()) else "keep"
        result = {"decision": decision, "dimensions": dimensions, "model": response["model"], "reason": "jev_decision"}
        usage = _usage(response)
        if usage:
            result["usage"] = usage
        if isinstance(response.get("id"), str) and len(response["id"]) <= 200:
            result["provider_id"] = response["id"]
        return result
    except (KeyError, TypeError, AttributeError, StopIteration, OverflowError) as exc:
        raise ValueError("invalid Jev response") from exc


class JevClient:
    """A per-run client; request budget includes retries and is shared across workers.

    Each worker thread keeps one TLS connection open across requests, so a run of
    a thousand rows pays for a handful of handshakes rather than a thousand.
    """

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
        self._local = threading.local()
        parts = urlsplit(ENDPOINTS[provider])
        self._host, self._path = parts.hostname, parts.path
        self._port = parts.port or 443
        self._context = ssl.create_default_context()
        self._proxy = self._proxy_for(self._host)

    @staticmethod
    def _proxy_for(host):
        """(proxy_host, proxy_port, tunnel_headers) from HTTPS_PROXY/NO_PROXY, or None.

        The same environment urllib honours, so a workstation behind a corporate
        proxy screens without extra configuration. The provider is reached through
        a CONNECT tunnel; TLS still terminates at the provider, never at the proxy.
        """
        if proxy_bypass(host):
            return None
        url = getproxies().get("https")
        if not url:
            return None
        parts = urlsplit(url if "://" in url else "http://" + url)
        if not parts.hostname:
            raise ValueError("HTTPS_PROXY is not a valid URL")
        headers = {}
        if parts.username is not None:
            credentials = unquote(parts.username) + ":" + unquote(parts.password or "")
            headers["Proxy-Authorization"] = "Basic " + b64encode(credentials.encode("utf-8")).decode("ascii")
        return parts.hostname, parts.port or 3128, headers

    def _connection(self, fresh=False):
        connection = getattr(self._local, "connection", None)
        if connection is None or fresh:
            if connection is not None:
                connection.close()
            if self._proxy:
                proxy_host, proxy_port, tunnel_headers = self._proxy
                connection = http.client.HTTPSConnection(proxy_host, proxy_port, timeout=self.timeout, context=self._context)
                connection.set_tunnel(self._host, self._port, headers=tunnel_headers)
            else:
                connection = http.client.HTTPSConnection(self._host, self._port, timeout=self.timeout, context=self._context)
            self._local.connection = connection
        return connection

    def close(self):
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None

    def _post(self, body):
        """One HTTP exchange; returns (status, headers, raw). Reconnects once on a stale socket."""
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key, "Content-Length": str(len(body))}
        for fresh in (False, True):
            connection = self._connection(fresh)
            try:
                connection.request("POST", self._path, body=body, headers=headers)
                response = connection.getresponse()
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                if response.getheader("Connection", "").lower() == "close":
                    self.close()
                return response.status, response.headers, raw
            except (http.client.RemoteDisconnected, http.client.CannotSendRequest, BrokenPipeError, ConnectionResetError):
                self.close()
                if fresh:
                    raise
        raise ConnectionError("unreachable")

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
            retryable = False
            delay = min(2 ** attempt, 5)
            try:
                status, headers, raw = self._post(body)
                if 200 <= status < 300:
                    if len(raw) > MAX_RESPONSE_BYTES:
                        raise JevAPIError("response_too_large")
                    try:
                        return json.loads(raw.decode("utf-8"))
                    except (ValueError, UnicodeError, RecursionError):
                        raise JevAPIError("invalid_json") from None
                # Redirects are not followed: the credential must only ever reach the configured host.
                retryable = status in {408, 429, 500, 502, 503, 504, 529}
                category = "authentication" if status in {401, 403} else "rate_limit" if status == 429 else "provider_http_error"
                failure = JevAPIError(category, status)
                retry_after = headers.get("Retry-After") if headers else None
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
            except (http.client.HTTPException, socket.timeout, TimeoutError, ConnectionError, OSError, ssl.SSLError):
                self.close()
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
