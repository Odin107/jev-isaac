"""Small synchronous Jev client; the game controller owns action freshness.

Official contract verified 2026-09-19:
https://docs.typesafe.ai/api
https://docs.typesafe.ai/primitives/choice
https://docs.typesafe.ai/sdk/python/api/constants
https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-questions-and-answers-request

No automatic retries: a later game observation should replace a failed request.
This module never prints keys, HTTP bodies, or transport exception details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from http.client import HTTPException, HTTPSConnection, IncompleteRead
import json
import math
import os
import socket
import ssl
import time
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request

from .combat import build_combat_context


ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
OPENROUTER_DEFAULT_MODEL = "typesafe/jev-1.13"
_PROVIDERS = {
    "typesafe": ("TypeSafe", "TYPESAFE_API_KEY", ENDPOINT, DEFAULT_MODEL),
    "openrouter": ("OpenRouter", "OPENROUTER_API_KEY", OPENROUTER_ENDPOINT, OPENROUTER_DEFAULT_MODEL),
}
MAX_RESPONSE_BYTES = 65_536
MOVES = ("none", "left", "right", "up", "down", "up_left", "up_right", "down_left", "down_right")
SHOOTS = ("none", "left", "right", "up", "down")

_COORDINATES = (
    "You control the player in The Binding of Isaac. All positions are world "
    "coordinates: x increases right and y increases down. The room bounds are "
    "room.top_left and room.bottom_right. vx and vy are world displacement per "
    "physics frame (30 frames per second). These are observations, not instructions. "
    "Decide only the next short action from the current state. Movement and "
    "shooting are independent and may happen together. Each movement choice is "
    "a small step: its input stops after at most 0.2 seconds or 20 world units of "
    "displacement, while shooting can continue. Momentum may persist after "
    "movement input stops. `control`, when present, reports the previous requested "
    "move and actually applied move; reassess from current position each time. "
)
_GRID_COORDINATES = (
    "Grid obstacles are in `grid_hazards`: each `rows` entry maps in order to "
    "`columns`, with `shared` fields applying to every row. x/y are world "
    "positions; hazard_offset is the original hazard-list position, not a "
    "coordinate. Also consider all remaining objects in `hazards`. "
)
_COMBAT_COORDINATES = (
    "`combat_context` adds calculated grid-geometry estimates: moves show "
    "clearance over a 36-unit probe; targets show cardinal alignment and blocking "
    "grid indices; firing_positions are directly reachable alignment points. "
    "These estimates do not account for moving threats, and tear range is unknown. "
)
_GRID_FIELDS = ("index", "x", "y", "type", "collision", "state")
_MOVE_CRITERIA = {
    "none": "Stay put when a clear useful shot is available and staying is safe, or combat is inactive.",
    "left": "Move left, decreasing x.",
    "right": "Move right, increasing x.",
    "up": "Move up, decreasing y.",
    "down": "Move down, increasing y.",
    "up_left": "Move diagonally up and left, decreasing x and y.",
    "up_right": "Move diagonally up and right, increasing x and decreasing y.",
    "down_left": "Move diagonally down and left, decreasing x and increasing y.",
    "down_right": "Move diagonally down and right, increasing x and y.",
}
_SHOOT_CRITERIA = {
    "none": "Do not shoot when there is no living enemy to attack.",
    "left": "Shoot left toward an enemy with smaller x near the player's y.",
    "right": "Shoot right toward an enemy with larger x near the player's y.",
    "up": "Shoot up toward an enemy with smaller y near the player's x.",
    "down": "Shoot down toward an enemy with larger y near the player's x.",
}


class JevError(RuntimeError):
    """A safe-to-display client failure, containing no server body or key."""

    def __init__(self, message: str, *, latency_ms: float = 0.0):
        super().__init__(message)
        self.latency_ms = latency_ms


class JevCredentialsError(JevError):
    pass


class JevTransportError(JevError):
    pass


class JevTimeoutError(JevTransportError):
    pass


class JevResponseError(JevError):
    """A validation failure with a small, safe set of diagnostic metadata.

    These fields never contain response text, model names, headers, or echoed
    credentials. Invalid optional metadata is discarded rather than reflected.
    The human-readable message remains independent of the validation code.
    """

    _CODES = frozenset({
        "response_type", "response_size", "invalid_json", "response_shape",
        "answer_keys", "model_missing", "usage_invalid", "choice_type",
        "choice_unknown", "confidence_invalid", "probability_keys",
        "probability_values", "probability_sum", "choice_not_max",
    })
    _DIAGNOSTICS = ("sum", "selected_probability", "max_probability")

    def __init__(self, message: str, *, latency_ms: float = 0.0,
                 validation_code: str = "response_shape", question: str | None = None,
                 diagnostics: Mapping[str, Any] | None = None):
        super().__init__(message, latency_ms=latency_ms)
        self.validation_code = (validation_code if isinstance(validation_code, str)
                                and validation_code in self._CODES else "response_shape")
        self.question = question if isinstance(question, str) and question in {"move", "shoot", "goal"} else None
        self.diagnostics = {
            name: float(diagnostics[name]) for name in self._DIAGNOSTICS
            if isinstance(diagnostics, Mapping) and name in diagnostics and _number(diagnostics[name])
        }


class JevHTTPError(JevError):
    def __init__(self, status: int, *, latency_ms: float = 0.0,
                 retry_after: float | None = None, provider: str = "typesafe"):
        provider_name = _provider_config(provider)[0]
        meanings = {
            400: "request rejected", 401: "API key missing or invalid",
            402: "insufficient credits",
            403: "account access denied", 404: "endpoint or model unavailable",
            422: "request validation failed", 429: "rate limit reached",
            529: "service temporarily overloaded",
        }
        super().__init__(f"{provider_name} HTTP {status}: {meanings.get(status, 'request failed')}.",
                         latency_ms=latency_ms)
        self.status = status
        self.retry_after = retry_after
        self.retryable = status == 429 or status >= 500


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    move: str
    shoot: str
    latency_ms: float
    model: str
    move_confidence: float
    shoot_confidence: float
    move_probabilities: Mapping[str, float]
    shoot_probabilities: Mapping[str, float]
    usage: Mapping[str, int]
    choice_corrections: tuple[Mapping[str, Any], ...] = ()


Transport = Callable[[Request, float], HttpResponse]


def _provider_config(provider: str) -> tuple[str, str, str, str]:
    if not isinstance(provider, str) or provider not in _PROVIDERS:
        raise ValueError("provider must be typesafe or openrouter.")
    return _PROVIDERS[provider]


def _compact_grid_hazards(observed: dict[str, Any]) -> dict[str, Any] | None:
    """Tabulate the known grid shape without dropping data or hazard ordering.

    Unfamiliar records stay intact, including grid objects with extra fields.
    A pre-existing grid_hazards key also leaves the entire state untouched.
    """
    hazards = observed.get("hazards")
    if not isinstance(hazards, list) or "grid_hazards" in observed:
        return None
    rows, remaining = [], []
    expected = {"kind", "radius", *_GRID_FIELDS}
    for offset, hazard in enumerate(hazards):
        if (isinstance(hazard, dict) and set(hazard) == expected
                and hazard["kind"] == "grid"
                and type(hazard["radius"]) is int and hazard["radius"] == 20
                and all(type(hazard[key]) is int for key in ("index", "type", "collision", "state"))
                and all(type(hazard[key]) in (int, float) for key in ("x", "y"))):
            rows.append([offset, *(hazard[key] for key in _GRID_FIELDS)])
        else:
            remaining.append(hazard)
    if not rows:
        return None
    return {**observed, "hazards": remaining, "grid_hazards": {
        "columns": ["hazard_offset", *_GRID_FIELDS],
        "shared": {"kind": "grid", "radius": 20},
        "rows": rows,
    }}


def build_action_request(state: Mapping[str, Any], model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Build both questions; retain the observed world units without conversion.

    The caller supplies an observation with player, room, enemies, projectiles,
    and hazards. Transport-only session/room identifiers are omitted. This does
    not send a request or require credentials. Repeated grid obstacles use a
    lossless table only when the complete request becomes smaller.
    """
    if not isinstance(state, Mapping):
        raise ValueError("The game observation must be an object.")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("The model must be a nonempty string.")
    observed = {key: value for key, value in state.items() if key not in {"session", "room_id", "run_id"}}
    try:
        # Snapshot mutable data; reject NaN/infinity before they reach the API.
        observed = json.loads(json.dumps(observed, allow_nan=False))
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise ValueError("The game observation must contain finite JSON data.") from None
    combat_context = build_combat_context(observed)
    context_guidance = ""
    if combat_context and "combat_context" not in observed:
        observed["combat_context"] = combat_context
        context_guidance = _COMBAT_COORDINATES
    request = {
        "model": model,
        "state": observed,
        "questions": {
            "move": {
                "type": "choice",
                "instructions": _COORDINATES + context_guidance + (
                    "Which short movement makes progress toward a clear cardinal "
                    "firing lane while avoiding approaching projectiles, enemy "
                    "contact, and hazards? Prefer directions with a clear movement "
                    "probe; do not push into a wall with zero clearance. When shots "
                    "are misaligned or blocked, reposition toward a reachable "
                    "firing position instead of standing still or firing into a rock. "
                    "Stop crossing the target's row or column once aligned. If the "
                    "target is far away, approach along a clear lane while keeping "
                    "distance from enemy contact. Survival takes priority during "
                    "immediate danger. Do not "
                    "seek doors or leave the room. If paused, disabled, dead, or "
                    "the room is clear, choose none."
                ),
                "criteria": dict(_MOVE_CRITERIA),
            },
            "shoot": {
                "type": "choice",
                "instructions": _COORDINATES + context_guidance + (
                    "Which cardinal shooting direction best hits a living enemy "
                    "from the player's current position? Shots travel in straight "
                    "cardinal directions; prefer an aligned vulnerable enemy "
                    "whose lane has no solid-grid blockers, choosing a nearer "
                    "target when possible. A direction toward a diagonal enemy "
                    "or through a rock does not hit it. If paused, "
                    "disabled, dead, or the room is clear, choose none."
                ),
                "criteria": dict(_SHOOT_CRITERIA),
            },
        },
    }
    compact_state = _compact_grid_hazards(observed)
    if compact_state is not None:
        compact_request = {**request, "state": compact_state, "questions": {
            name: {**question, "instructions": _GRID_COORDINATES + question["instructions"]}
            for name, question in request["questions"].items()
        }}
        # Include the table explanation in both independent questions when
        # comparing wire sizes; small lists should remain ordinary objects.
        if len(json.dumps(compact_request, separators=(",", ":"))) < len(
                json.dumps(request, separators=(",", ":"))):
            return compact_request
    return request


class _PersistentHTTPTransport:
    """One lazily opened, certificate-verified connection to a fixed provider.

    Calls must be sequential, as they are in the controller's single worker.
    Connection setup is lazy unless warm_connect() is called before gameplay.
    A failed POST is never replayed, even if an idle connection was closed by
    the server; only a subsequent call may create a replacement connection.
    """

    def __init__(self, endpoint: str):
        if endpoint not in {config[2] for config in _PROVIDERS.values()}:
            raise ValueError("Transport requires a configured provider endpoint.")
        self._endpoint = endpoint
        self._url = urlsplit(endpoint)
        self._connection: HTTPSConnection | None = None
        self._warmed_at: float | None = None

    def close(self) -> None:
        connection, self._connection = self._connection, None
        self._warmed_at = None
        if connection is not None:
            connection.close()

    def warm_connect(self, timeout: float) -> None:
        """Open only TCP/TLS: no HTTP request, credentials, or inference.

        Call sequentially, before any POST worker starts. The socket timeout
        bounds connection I/O; DNS resolution follows the operating system.
        """
        if not _number(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number of seconds.")
        try:
            if self._connection is None:
                self._connection = HTTPSConnection(
                    self._url.hostname, port=self._url.port, timeout=timeout,
                    context=ssl.create_default_context(),
                )
                self._connection.connect()
                self._warmed_at = time.monotonic()
            else:
                self._connection.timeout = timeout
                if self._connection.sock is None:
                    self._connection.connect()
                    self._warmed_at = time.monotonic()
                else:
                    self._connection.sock.settimeout(timeout)
        except Exception:
            self.close()
            raise

    def __call__(self, request: Request, timeout: float) -> HttpResponse:
        # Do not let a redirect or caller-supplied URL move the bearer token.
        if request.full_url != self._endpoint or request.get_method() != "POST":
            raise ValueError("Transport requires a POST to its configured endpoint.")
        try:
            # The user may wait before F8. Do not send the first POST down a
            # prepared socket that has spent a long time idle. This reconnects
            # before any request bytes are sent; it never retries a POST.
            if self._warmed_at is not None:
                if time.monotonic()-self._warmed_at > 5.0:
                    self.close()
                self._warmed_at = None
            if self._connection is None:
                self._connection = HTTPSConnection(
                    self._url.hostname, port=self._url.port, timeout=timeout,
                    context=ssl.create_default_context(),
                )
            connection = self._connection
            connection.timeout = timeout
            if connection.sock is not None:
                connection.sock.settimeout(timeout)
            connection.request("POST", self._url.path, body=request.data,
                               headers=dict(request.header_items()))
            response = connection.getresponse()
            with response:
                headers = dict(response.getheaders())
                if response.status != 200:
                    # Never follow redirects or read potentially sensitive
                    # error bodies. An unread response cannot be reused.
                    self.close()
                    return HttpResponse(response.status, b"", headers)
                body = response.read(MAX_RESPONSE_BYTES + 1)
                if len(body) > MAX_RESPONSE_BYTES:
                    # The parser reports the bounded-response error. Discard
                    # this socket so unread bytes cannot contaminate a reply.
                    self.close()
                elif response.length not in (None, 0):
                    # read(amt) does not itself reject a truncated fixed-size
                    # body, which might otherwise parse as valid JSON.
                    raise IncompleteRead(b"")
                elif response.will_close:
                    self.close()
                return HttpResponse(response.status, body, headers)
        except Exception:
            # Reset state on any transport failure; no request is retried.
            self.close()
            raise


def _retry_after(headers: Mapping[str, str]) -> float | None:
    value = next((v for k, v in headers.items() if k.lower() == "retry-after"), None)
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate response field")
        result[key] = value
    return result


def _number(value: Any) -> bool:
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _validate_choice_policy(choice_policy: str) -> None:
    if not isinstance(choice_policy, str) or choice_policy not in {"strict", "argmax"}:
        raise ValueError("choice_policy must be strict or argmax.")


def _parse_choice(answer: Any, choices: tuple[str, ...], question: str,
                  provider_name: str, choice_policy: str):
    error = f"{provider_name} returned an invalid {question} choice answer."
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise JevResponseError(error, validation_code="choice_type", question=question)
    choice = answer.get("choice")
    confidence = answer.get("confidence")
    probabilities = answer.get("probabilities")
    if not isinstance(choice, str) or choice not in choices:
        raise JevResponseError(error, validation_code="choice_unknown", question=question)
    if not _number(confidence) or not 0 <= confidence <= 1:
        raise JevResponseError(error, validation_code="confidence_invalid", question=question)
    if not isinstance(probabilities, dict) or set(probabilities) != set(choices):
        raise JevResponseError(error, validation_code="probability_keys", question=question)
    if any(not _number(p) or not 0 <= p <= 1 for p in probabilities.values()):
        raise JevResponseError(error, validation_code="probability_values", question=question)
    # Allow small provider rounding, but reject malformed distributions.
    probability_sum = sum(probabilities.values())
    if not math.isclose(probability_sum, 1.0, abs_tol=0.02):
        raise JevResponseError(error, validation_code="probability_sum", question=question,
                               diagnostics={"sum": probability_sum})
    max_probability = max(probabilities.values())
    if choice_policy == "strict" and probabilities[choice] + 1e-6 < max_probability:
        raise JevResponseError(error, validation_code="choice_not_max", question=question,
                               diagnostics={"selected_probability": probabilities[choice],
                                            "max_probability": max_probability})
    # Gameplay can explicitly consume the validated distribution directly.
    # This is application policy, not a claim that the provider reply obeyed
    # its choice/maximum contract or that the discrepancy was rounding.
    correction = None
    if choice_policy == "argmax" and probabilities[choice] < max_probability:
        effective_choice = next(option for option in choices
                                if probabilities[option] == max_probability)
        correction = {
            "question": question,
            "reported_choice": choice,
            "effective_choice": effective_choice,
            "selected_probability": float(probabilities[choice]),
            "max_probability": float(max_probability),
            "gap": float(max_probability - probabilities[choice]),
            "probabilities": {option: float(probabilities[option]) for option in choices},
        }
        choice = effective_choice
    return choice, float(confidence), {k: float(v) for k, v in probabilities.items()}, correction


def parse_action_response(body: bytes, *, latency_ms: float = 0.0,
                          provider: str = "typesafe", choice_policy: str = "strict") -> Decision:
    """Validate Choice answers and token usage shared by both Jev providers.

    OpenRouter also returns id, provider, and usage.cost metadata. Those fields
    are not needed for action control and are deliberately not retained.
    The explicit argmax policy selects from the validated distribution when
    the provider's choice disagrees, recording the discrepancy separately.
    """
    _validate_choice_policy(choice_policy)
    provider_name = _provider_config(provider)[0]
    if not isinstance(body, bytes):
        raise JevResponseError(f"{provider_name} response is not a bounded JSON response.",
                               validation_code="response_type")
    if len(body) > MAX_RESPONSE_BYTES:
        raise JevResponseError(f"{provider_name} response is not a bounded JSON response.",
                               validation_code="response_size")
    try:
        payload = json.loads(body.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, ValueError, RecursionError):
        raise JevResponseError(f"{provider_name} response is not valid JSON.",
                               validation_code="invalid_json") from None
    if not isinstance(payload, dict):
        raise JevResponseError(f"{provider_name} response must be an object.",
                               validation_code="response_shape")
    answers = payload.get("answers")
    model = payload.get("model")
    usage = payload.get("usage")
    if not isinstance(answers, dict) or set(answers) != {"move", "shoot"}:
        raise JevResponseError(f"{provider_name} must return exactly move and shoot answers.",
                               validation_code="answer_keys")
    if not isinstance(model, str) or not model.strip():
        raise JevResponseError(f"{provider_name} response is missing its model.",
                               validation_code="model_missing")
    if not isinstance(usage, dict) or any(
        not isinstance(usage.get(k), int) or isinstance(usage.get(k), bool) or usage[k] < 0
        for k in ("input_tokens", "output_tokens")
    ):
        raise JevResponseError(f"{provider_name} response has invalid token usage.",
                               validation_code="usage_invalid")
    move, move_confidence, move_probs, move_correction = _parse_choice(
        answers["move"], MOVES, "move", provider_name, choice_policy)
    shoot, shoot_confidence, shoot_probs, shoot_correction = _parse_choice(
        answers["shoot"], SHOOTS, "shoot", provider_name, choice_policy)
    return Decision(move, shoot, latency_ms, model, move_confidence, shoot_confidence,
                    move_probs, shoot_probs, {k: usage[k] for k in ("input_tokens", "output_tokens")},
                    tuple(correction for correction in (move_correction, shoot_correction)
                          if correction is not None))


class JevClient:
    """A single request per decide call, with a finite timeout and no retries.

    timeout is the socket timeout; completed responses that exceed that
    elapsed duration are rejected too. The controller must separately discard
    decisions older than its tighter game-state freshness threshold.
    A default connection is reused across sequential calls. Call close() after
    the worker has stopped to release it; injected transports remain caller-owned.
    """

    def __init__(self, api_key: str | None = None, *, provider: str = "typesafe", model: str | None = None,
                 timeout: float = 1.0, transport: Transport | None = None,
                 clock: Callable[[], float] = time.perf_counter, choice_policy: str = "strict"):
        _validate_choice_policy(choice_policy)
        provider_name, key_variable, endpoint, default_model = _provider_config(provider)
        key = api_key if api_key is not None else os.environ.get(key_variable, "")
        if not isinstance(key, str) or not key.strip():
            raise JevCredentialsError(f"Set {key_variable} before using Jev through {provider_name}.")
        if any(ord(c) < 33 or ord(c) > 126 for c in key):
            raise JevCredentialsError(f"{key_variable} has invalid characters.")
        if not _number(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number of seconds.")
        model = default_model if model is None else model
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string.")
        self._api_key = key
        self.provider = provider
        self._provider_name = provider_name
        self._endpoint = endpoint
        self.model = model
        self.choice_policy = choice_policy
        self.timeout = float(timeout)
        self._owned_transport = _PersistentHTTPTransport(endpoint) if transport is None else None
        self._transport = self._owned_transport if transport is None else transport
        self._clock = clock

    def close(self) -> None:
        if self._owned_transport is not None:
            self._owned_transport.close()

    def warm_connect(self) -> float | None:
        """Prepare the owned TLS connection without sending the API key.

        Returns connection time in milliseconds, or None for an injected
        transport, which remains caller-owned. This makes no paid API call
        and does not validate credentials. Call before starting the worker.
        """
        if self._owned_transport is None:
            return None
        started = self._clock()
        try:
            self._owned_transport.warm_connect(self.timeout)
        except (TimeoutError, socket.timeout):
            raise JevTimeoutError(f"{self._provider_name} connection preparation timed out.",
                                  latency_ms=(self._clock()-started)*1000) from None
        except (OSError, HTTPException):
            raise JevTransportError(f"{self._provider_name} connection preparation failed.",
                                    latency_ms=(self._clock()-started)*1000) from None
        elapsed = (self._clock()-started)*1000
        if elapsed > self.timeout*1000:
            self._owned_transport.close()
            raise JevTimeoutError(f"{self._provider_name} connection preparation exceeded the timeout.",
                                  latency_ms=elapsed)
        return elapsed

    def decide(self, state: Mapping[str, Any]) -> Decision:
        payload = build_action_request(state, self.model)
        body, elapsed = self.post(payload)
        try:
            return parse_action_response(body, latency_ms=elapsed, provider=self.provider,
                                         choice_policy=self.choice_policy)
        except JevResponseError as exc:
            exc.latency_ms = elapsed
            raise

    def post(self, payload: Mapping[str, Any]) -> tuple[bytes, float]:
        """Send one JSON request to the configured provider; never replay it.

        Shared by action and tactical-goal clients. Parsing and application
        freshness remain the caller's responsibility; connection ownership,
        bounded reads, redacted errors and timing are identical for both.
        """
        request = Request(self._endpoint,
                          data=json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8"),
                          headers={"Authorization": f"Bearer {self._api_key}",
                                   "Content-Type": "application/json", "Accept": "application/json"},
                          method="POST")
        started = self._clock()
        try:
            response = self._transport(request, self.timeout)
        except (TimeoutError, socket.timeout):
            raise JevTimeoutError(f"{self._provider_name} request timed out.",
                                  latency_ms=(self._clock() - started) * 1000) from None
        except HTTPError as exc:
            raise JevHTTPError(exc.code, latency_ms=(self._clock() - started) * 1000,
                               retry_after=_retry_after(exc.headers or {}), provider=self.provider) from None
        except URLError as exc:
            error_cls = JevTimeoutError if isinstance(exc.reason, (TimeoutError, socket.timeout)) else JevTransportError
            raise error_cls(f"{self._provider_name} connection failed or timed out.",
                            latency_ms=(self._clock() - started) * 1000) from None
        except (OSError, HTTPException):
            raise JevTransportError(f"{self._provider_name} connection failed.",
                                    latency_ms=(self._clock() - started) * 1000) from None
        elapsed = (self._clock() - started) * 1000
        if elapsed > self.timeout * 1000:
            raise JevTimeoutError(f"{self._provider_name} response arrived after the timeout.", latency_ms=elapsed)
        if not isinstance(response, HttpResponse):
            raise JevResponseError(f"{self._provider_name} transport returned an invalid response.",
                                   latency_ms=elapsed, validation_code="response_type")
        if response.status != 200:
            raise JevHTTPError(response.status, latency_ms=elapsed, retry_after=_retry_after(response.headers),
                               provider=self.provider)
        return response.body, elapsed
