"""Small synthetic latency diagnostic; never connects to or controls the game.

One verified HTTPS connection is measured and reused. A server disconnect stops
the probe; there are no reconnects, redirects, or request retries. A reused
connection does not establish that the model or provider cache is warm.
"""
from __future__ import annotations

import http.client
import ssl
import time
from urllib.parse import urlsplit

from .demo import sample
from .jev import (ENDPOINT, OPENROUTER_ENDPOINT, MAX_RESPONSE_BYTES, HttpResponse,
                  JevClient, JevError, JevResponseError, JevTransportError)


MAX_SAMPLES = 4
TIMEOUT_SECONDS = 15.0
FRESHNESS_MS = 250.0


class _ConnectionTransport:
    def __init__(self, endpoint, factory, clock):
        parsed = urlsplit(endpoint)
        self.endpoint = endpoint
        self.path = parsed.path
        self.clock = clock
        # create_default_context enables certificate and hostname verification.
        self.connection = factory(parsed.hostname, timeout=TIMEOUT_SECONDS,
                                  context=ssl.create_default_context())
        # http.client otherwise silently opens another connection after a close.
        self.connection.auto_open = 0
        self.used = False
        self.requests = 0
        self.connect_ms = None
        self.closed_after_response = False

    def __call__(self, request, timeout):
        if request.full_url != self.endpoint or request.get_method() != "POST":
            raise JevTransportError("Timing probe rejected an unexpected endpoint or method.")
        if self.used and (self.closed_after_response or self.connection.sock is None):
            raise JevTransportError("Timing connection closed; no reconnect was attempted.")
        started = self.clock()
        deadline = started + timeout
        try:
            if not self.used:
                self.connection.connect()
                self.connect_ms = (self.clock() - started) * 1000
                self.used = True
            self._set_remaining_timeout(deadline)
            self.requests += 1
            self.connection.request("POST", self.path, body=request.data,
                                    headers=dict(request.header_items()))
            self._set_remaining_timeout(deadline)
            response = self.connection.getresponse()
            try:
                self.closed_after_response = response.will_close
                # No error bodies, redirect locations, or provider text enter reports.
                if response.status != 200:
                    return HttpResponse(response.status, b"")
                chunks = []
                size = 0
                while size <= MAX_RESPONSE_BYTES:
                    self._set_remaining_timeout(deadline)
                    chunk = response.read1(min(16_384, MAX_RESPONSE_BYTES + 1 - size))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    size += len(chunk)
                return HttpResponse(response.status, b"".join(chunks))
            finally:
                response.close()
        except http.client.HTTPException:
            # JevClient turns this into a safe, timed transport failure.
            raise OSError("HTTP transport failed") from None

    def _set_remaining_timeout(self, deadline):
        remaining = deadline - self.clock()
        if remaining <= 0:
            raise TimeoutError("Timing request exceeded its time allowance")
        if self.connection.sock is not None:
            self.connection.sock.settimeout(remaining)

    def close(self):
        self.connection.close()


def run_timing(api_key: str, provider: str = "typesafe", model: str | None = None,
               samples: int = 4, *, state=None, _connection_factory=None, _clock=None) -> dict:
    """Return a credential-free report for one to four sequential decisions.

    The key must be supplied explicitly; this function never reads environment
    credentials. Private factory/clock injection lets tests run without network.
    Latency includes request construction, connection if new, response reading,
    and response validation. The separate first connect time includes DNS/TCP/TLS.
    The 15-second socket timeout is also checked between body reads and after
    completion; it is not a hard interruption of operating-system DNS resolution.
    Reported token totals cover validated successful responses only.
    """
    if isinstance(samples, bool) or not isinstance(samples, int) or not 1 <= samples <= MAX_SAMPLES:
        raise ValueError("samples must be an integer from 1 to 4.")
    if provider not in ("typesafe", "openrouter"):
        raise ValueError("provider must be typesafe or openrouter.")
    clock = _clock or time.perf_counter
    report = {
        "mode": "timing-check", "provider": provider, "success": False,
        "synthetic_state": state is None, "replayed_state": state is not None, "game_controls_sent": False,
        "requested_samples": samples, "requests": 0, "completed_samples": 0,
        "request_timeout_seconds": TIMEOUT_SECONDS,
        "freshness_threshold_ms": FRESHNESS_MS, "initial_connect_ms": None,
        "within_250ms": 0, "reused_within_250ms": 0,
        "reported_input_tokens": 0, "reported_output_tokens": 0,
        "measurements": [],
    }
    transport = None
    try:
        # Empty rather than None prevents JevClient's optional environment lookup.
        client = JevClient(api_key=api_key if api_key is not None else "",
                           provider=provider, model=model, timeout=TIMEOUT_SECONDS,
                           transport=lambda request, timeout: transport(request, timeout),
                           clock=clock)
        endpoint = ENDPOINT if provider == "typesafe" else OPENROUTER_ENDPOINT
        transport = _ConnectionTransport(endpoint, _connection_factory or
                                          http.client.HTTPSConnection, clock)
        for index in range(samples):
            if index and (transport.closed_after_response or transport.connection.sock is None):
                raise JevTransportError("Timing connection closed; no reconnect was attempted.")
            started = clock()
            decision = client.decide(sample() if state is None else state)
            latency_ms = (clock() - started) * 1000
            within = latency_ms <= FRESHNESS_MS
            # Only fixed choices, validated counts, and a redacted model label
            # survive from the response. Never retain headers or raw bodies.
            resolved_model = decision.model.replace(api_key, "[redacted]")[:120]
            report["measurements"].append({
                "sample": index + 1, "connection": "new" if index == 0 else "reused",
                "latency_ms": round(latency_ms, 3), "model": resolved_model,
                "move": decision.move, "shoot": decision.shoot,
                "within_250ms": within,
                "reported_input_tokens": decision.usage["input_tokens"],
                "reported_output_tokens": decision.usage["output_tokens"],
                "connection_closed_after_response": transport.closed_after_response,
            })
            report["completed_samples"] += 1
            report["reported_input_tokens"] += decision.usage["input_tokens"]
            report["reported_output_tokens"] += decision.usage["output_tokens"]
            report["within_250ms"] += int(within)
            report["reused_within_250ms"] += int(index > 0 and within)
        report["success"] = True
    except JevError as exc:
        report["error"] = str(exc)
        report["error_type"] = type(exc).__name__
        if isinstance(exc, JevResponseError):
            report["validation_code"] = exc.validation_code
            report["question"] = exc.question
            report["diagnostics"] = exc.diagnostics
    finally:
        if transport is not None:
            report["requests"] = transport.requests
            report["initial_connect_ms"] = (round(transport.connect_ms, 3)
                                               if transport.connect_ms is not None else None)
            transport.close()
    return report
