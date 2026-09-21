"""Offline contract tests: never call providers or require real credentials."""

from http.client import HTTPException, HTTPResponse, RemoteDisconnected
import io
import json
import os
import ssl
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

from jev_isaac.jev import (
    ENDPOINT, MAX_RESPONSE_BYTES, OPENROUTER_ENDPOINT, OPENROUTER_DEFAULT_MODEL,
    MOVES, SHOOTS, Decision, HttpResponse, JevClient, JevCredentialsError,
    JevHTTPError, JevResponseError, JevTimeoutError, JevTransportError,
    _PersistentHTTPTransport, build_action_request, parse_action_response,
)


def response_payload(move="up_left", shoot="right"):
    def answer(choice, options):
        return {"type": "choice", "choice": choice, "confidence": 1.0,
                "probabilities": {option: float(option == choice) for option in options}}
    return {"model": "jev-1.13.0", "answers": {
        "move": answer(move, MOVES), "shoot": answer(shoot, SHOOTS)},
        "usage": {"input_tokens": 700, "output_tokens": 80}}


def encoded(payload=None):
    return json.dumps(response_payload() if payload is None else payload).encode()


def wire_response(body=None, *, status=200, headers=(), declared_length=None, chunked=False):
    """Use the real HTTP body reader against an in-memory socket."""
    body = encoded() if body is None else body
    if chunked:
        framing = b"Transfer-Encoding: chunked\r\n"
        body = f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n"
    else:
        length = len(body) if declared_length is None else declared_length
        framing = f"Content-Length: {length}\r\n".encode()
    wire = (f"HTTP/1.1 {status} Test\r\n".encode() + framing +
            b"".join(f"{name}: {value}\r\n".encode() for name, value in headers) +
            b"\r\n" + body)
    response = HTTPResponse(Mock(makefile=lambda *args: io.BytesIO(wire)))
    response.begin()
    return response


class PersistentTransportTests(unittest.TestCase):
    def test_warm_connect_verifies_tls_without_http_then_reuses_connection(self):
        with patch("jev_isaac.jev.HTTPSConnection") as constructor:
            connection = constructor.return_value
            connection.getresponse.return_value = wire_response()
            ticks = iter([1.0, 1.4, 2.0, 2.01, 3.0, 3.3])
            client = JevClient("offline-secret-key", timeout=2.0, clock=lambda: next(ticks))
            self.assertAlmostEqual(client.warm_connect(), 400.0)
            self.assertAlmostEqual(client.warm_connect(), 10.0)
            constructor.assert_called_once()
            self.assertEqual(constructor.call_args.args, ("api.typesafe.ai",))
            self.assertEqual(constructor.call_args.kwargs["timeout"], 2.0)
            context = constructor.call_args.kwargs["context"]
            self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
            self.assertTrue(context.check_hostname)
            connection.connect.assert_called_once_with()
            connection.request.assert_not_called()
            connection.getresponse.assert_not_called()
            self.assertNotIn("offline-secret-key", str(constructor.mock_calls))
            decision = client.decide({"frame": 1})
            self.assertAlmostEqual(decision.latency_ms, 300.0)
            constructor.assert_called_once()
            connection.request.assert_called_once()
            client.close()

    def test_warm_failure_is_redacted_closed_and_does_not_retry(self):
        for failure, expected in ((TimeoutError("offline-secret-key"), JevTimeoutError),
                                  (ssl.SSLCertVerificationError("offline-secret-key"), JevTransportError),
                                  (HTTPException("offline-secret-key"), JevTransportError)):
            with self.subTest(failure=type(failure)), patch("jev_isaac.jev.HTTPSConnection") as constructor:
                first, second = Mock(), Mock()
                constructor.side_effect = [first, second]
                first.connect.side_effect = failure
                second.getresponse.return_value = wire_response()
                client = JevClient("offline-secret-key")
                with self.assertRaises(expected) as result:
                    client.warm_connect()
                self.assertNotIn("offline-secret-key", str(result.exception))
                self.assertTrue(result.exception.__suppress_context__)
                constructor.assert_called_once()
                first.connect.assert_called_once_with()
                first.request.assert_not_called()
                first.close.assert_called_once_with()
                # A later ordinary decision may open a new connection.
                client.decide({})
                self.assertEqual(constructor.call_count, 2)
                second.request.assert_called_once()
                client.close()

    def test_idle_unused_warm_socket_is_discarded_before_first_post(self):
        with patch("jev_isaac.jev.HTTPSConnection") as constructor, \
             patch("jev_isaac.jev.time.monotonic", side_effect=[10.0, 15.1, 15.2, 15.3, 15.4]):
            warmed, used = Mock(), Mock()
            constructor.side_effect = [warmed, used]
            used.getresponse.side_effect = [wire_response(), wire_response()]
            client = JevClient("offline-key")
            client.warm_connect()
            client.decide({"frame": 1})
            # A nearby next request still reuses the established connection.
            client.decide({"frame": 2})
            self.assertEqual(constructor.call_count, 2)
            warmed.connect.assert_called_once_with()
            warmed.request.assert_not_called()
            warmed.close.assert_called_once_with()
            self.assertEqual(used.request.call_count, 2)
            client.close()

    def test_used_connection_is_replaced_after_pause_before_sending_new_post(self):
        with patch("jev_isaac.jev.HTTPSConnection") as constructor, \
             patch("jev_isaac.jev.time.monotonic", side_effect=[10., 10.5, 10.6, 30., 30.2]):
            first, second = Mock(), Mock()
            constructor.side_effect = [first, second]
            first.getresponse.side_effect = [wire_response(), wire_response()]
            second.getresponse.return_value = wire_response()
            client = JevClient("offline-key")
            client.decide({"frame": 1})
            client.decide({"frame": 2})
            self.assertEqual(constructor.call_count, 1)
            client.decide({"frame": 3})
            self.assertEqual(constructor.call_count, 2)
            self.assertEqual(first.request.call_count, 2)
            second.request.assert_called_once()
            first.close.assert_called_once()
            self.assertEqual(json.loads(second.request.call_args.kwargs["body"])["state"]["frame"], 3)
            client.close()

    def test_warm_elapsed_timeout_closes_connection_without_sending_request(self):
        with patch("jev_isaac.jev.HTTPSConnection") as constructor:
            ticks = iter([1.0, 3.1])
            client = JevClient("offline-key", timeout=2.0, clock=lambda: next(ticks))
            with self.assertRaises(JevTimeoutError) as result:
                client.warm_connect()
            self.assertAlmostEqual(result.exception.latency_ms, 2100.0)
            constructor.return_value.connect.assert_called_once_with()
            constructor.return_value.close.assert_called_once_with()
            constructor.return_value.request.assert_not_called()

    def test_warm_injected_transport_is_untouched(self):
        transport = Mock()
        client = JevClient("offline-key", transport=transport)
        self.assertIsNone(client.warm_connect())
        self.assertEqual(transport.mock_calls, [])
        client.close()
        self.assertEqual(transport.mock_calls, [])

    def test_transport_warm_rejects_unbounded_timeout_before_connecting(self):
        with patch("jev_isaac.jev.HTTPSConnection") as constructor:
            transport = _PersistentHTTPTransport(ENDPOINT)
            for timeout in (0, -1, float("inf"), float("nan"), True):
                with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                    transport.warm_connect(timeout)
            constructor.assert_not_called()

    def test_lazy_verified_connection_is_reused_and_close_is_idempotent(self):
        with patch("jev_isaac.jev.HTTPSConnection") as constructor:
            connection = constructor.return_value
            connection.getresponse.side_effect = [wire_response(), wire_response()]
            ticks = iter([1.0, 1.717, 2.0, 2.302])
            client = JevClient("offline-key", clock=lambda: next(ticks))
            constructor.assert_not_called()
            first = client.decide({"frame": 1})
            second = client.decide({"frame": 2})
            constructor.assert_called_once()
            self.assertEqual(constructor.call_args.args, ("api.typesafe.ai",))
            context = constructor.call_args.kwargs["context"]
            self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
            self.assertTrue(context.check_hostname)
            self.assertEqual(connection.request.call_count, 2)
            for call in connection.request.call_args_list:
                self.assertEqual(call.args, ("POST", "/v1/systemone"))
                self.assertEqual(call.kwargs["headers"]["Authorization"], "Bearer offline-key")
            self.assertEqual(connection.sock.settimeout.call_count, 2)
            self.assertAlmostEqual(first.latency_ms, 717.0)
            self.assertAlmostEqual(second.latency_ms, 302.0)
            connection.close.assert_not_called()
            client.close()
            client.close()
            connection.close.assert_called_once()

    def test_server_close_reconnects_only_for_next_decision(self):
        with patch("jev_isaac.jev.HTTPSConnection") as constructor:
            first, second = Mock(), Mock()
            constructor.side_effect = [first, second]
            first.getresponse.return_value = wire_response(headers=[("Connection", "close")])
            second.getresponse.return_value = wire_response()
            client = JevClient("offline-key")
            client.decide({})
            self.assertEqual(constructor.call_count, 1)
            first.close.assert_called_once()
            client.decide({})
            self.assertEqual(constructor.call_count, 2)
            first.request.assert_called_once()
            second.request.assert_called_once()
            client.close()

    def test_idle_disconnect_is_not_retried_but_next_call_can_reconnect(self):
        with patch("jev_isaac.jev.HTTPSConnection") as constructor:
            first, second = Mock(), Mock()
            constructor.side_effect = [first, second]
            first.getresponse.side_effect = [wire_response(), RemoteDisconnected("secret-key")]
            second.getresponse.return_value = wire_response()
            client = JevClient("secret-key")
            client.decide({})
            with self.assertRaises(JevTransportError) as failure:
                client.decide({})
            self.assertNotIn("secret-key", str(failure.exception))
            self.assertTrue(failure.exception.__suppress_context__)
            self.assertEqual(constructor.call_count, 1)
            self.assertEqual(first.request.call_count, 2)
            first.close.assert_called_once()
            client.decide({})
            self.assertEqual(constructor.call_count, 2)
            second.request.assert_called_once()
            client.close()

    def test_redirects_and_errors_never_read_body_or_forward_key(self):
        for status in (302, 401, 429):
            with self.subTest(status=status), patch("jev_isaac.jev.HTTPSConnection") as constructor:
                response = wire_response(b"secret-key", status=status,
                                         headers=[("Location", "https://example.com/"),
                                                  ("Retry-After", "2")])
                response.read = Mock(side_effect=AssertionError("must not read error body"))
                constructor.return_value.getresponse.return_value = response
                client = JevClient("secret-key", provider="openrouter")
                with self.assertRaises(JevHTTPError) as failure:
                    client.decide({})
                self.assertEqual(failure.exception.status, status)
                self.assertEqual(failure.exception.retry_after, 2.0)
                self.assertNotIn("secret-key", str(failure.exception))
                constructor.assert_called_once()
                self.assertEqual(constructor.call_args.args, ("openrouter.ai",))
                constructor.return_value.request.assert_called_once()
                constructor.return_value.close.assert_called_once()
                response.read.assert_not_called()

    def test_fixed_and_chunked_oversized_responses_are_bounded_and_closed(self):
        for chunked in (False, True):
            with self.subTest(chunked=chunked), patch("jev_isaac.jev.HTTPSConnection") as constructor:
                response = wire_response(b" " * (MAX_RESPONSE_BYTES + 500), chunked=chunked)
                real_read = response.read
                response.read = Mock(side_effect=real_read)
                constructor.return_value.getresponse.return_value = response
                client = JevClient("offline-key")
                with self.assertRaises(JevResponseError):
                    client.decide({})
                response.read.assert_called_once_with(MAX_RESPONSE_BYTES + 1)
                constructor.return_value.close.assert_called_once()
                constructor.return_value.request.assert_called_once()

    def test_truncated_body_cannot_be_accepted_even_if_partial_json_is_valid(self):
        with patch("jev_isaac.jev.HTTPSConnection") as constructor:
            constructor.return_value.getresponse.return_value = wire_response(
                declared_length=len(encoded()) + 20)
            client = JevClient("offline-key")
            with self.assertRaises(JevTransportError):
                client.decide({})
            constructor.return_value.close.assert_called_once()
            constructor.return_value.request.assert_called_once()

    def test_transport_failure_closes_without_retry_and_redacts_details(self):
        for failure, expected in ((TimeoutError("secret-key"), JevTimeoutError),
                                  (ssl.SSLCertVerificationError("secret-key"), JevTransportError),
                                  (HTTPException("secret-key"), JevTransportError)):
            with self.subTest(failure=type(failure)), patch("jev_isaac.jev.HTTPSConnection") as constructor:
                constructor.return_value.getresponse.side_effect = failure
                client = JevClient("secret-key")
                with self.assertRaises(expected) as result:
                    client.decide({})
                self.assertNotIn("secret-key", str(result.exception))
                self.assertTrue(result.exception.__suppress_context__)
                constructor.return_value.request.assert_called_once()
                constructor.return_value.close.assert_called_once()

    def test_transport_rejects_endpoint_changes_before_connecting(self):
        with patch("jev_isaac.jev.HTTPSConnection") as constructor:
            with self.assertRaises(ValueError):
                _PersistentHTTPTransport("http://api.typesafe.ai/v1/systemone")
            transport = _PersistentHTTPTransport(ENDPOINT)
            for url in (OPENROUTER_ENDPOINT, "https://example.com/v1/systemone"):
                with self.subTest(url=url), self.assertRaises(ValueError):
                    transport(Request(url, data=b"{}", method="POST"), 1.0)
            constructor.assert_not_called()

    def test_injected_transport_is_preserved_and_remains_caller_owned(self):
        transport = Mock(return_value=HttpResponse(200, encoded()))
        # A falsey callable is still an explicit transport override.
        class FalseyTransport:
            def __bool__(self):
                return False

            def __call__(self, request, timeout):
                return transport(request, timeout)

        client = JevClient("offline-key", transport=FalseyTransport())
        client.decide({})
        client.close()
        transport.assert_called_once()
        transport.close.assert_not_called()


class JevTests(unittest.TestCase):
    def test_compact_grid_state_roundtrips_mixed_hazards_without_mutation(self):
        grids = [{"kind": "grid", "index": i + 100, "type": 1 + i % 3,
                  "collision": i % 4, "state": i % 2,
                  "x": 40.0 * i, "y": 80.5 + i, "radius": 20}
                 for i in range(30)]
        unfamiliar = [
            {"kind": "creep", "x": 123.25, "y": 45, "radius": 8, "extra": [1, 2]},
            {**grids[0], "future_field": {"danger": True}},
            {**grids[1], "radius": 30},
            {**grids[2], "collision": "unknown"},
            {"kind": "grid", "index": 901},
            None,
        ]
        hazards = grids[:5] + unfamiliar[:3] + grids[5:20] + unfamiliar[3:] + grids[20:]
        state = {"player": {"x": 10, "y": 20}, "hazards": hazards,
                 "paused": True, "enabled": False, "truncated": True,
                 "truncated_arrays": {"hazards": True, "enemies": False},
                 "future_state": {"unknown": [1, 2, 3]}}
        original = json.loads(json.dumps(state))
        request = build_action_request(state)
        compact = request["state"]
        self.assertEqual(state, original)
        self.assertEqual(compact["hazards"], unfamiliar)
        self.assertEqual(compact["truncated_arrays"], original["truncated_arrays"])
        self.assertEqual(compact["future_state"], original["future_state"])
        self.assertTrue(compact["paused"])
        self.assertFalse(compact["enabled"])
        self.assertTrue(compact["truncated"])
        table = compact["grid_hazards"]
        self.assertEqual(len(table["rows"]), len(grids))
        expanded = [None] * len(hazards)
        for row in table["rows"]:
            item = dict(zip(table["columns"], row))
            offset = item.pop("hazard_offset")
            expanded[offset] = {**table["shared"], **item}
        remaining = iter(compact["hazards"])
        # The original offset distinguishes preserved nulls from unfilled slots.
        grid_offsets = {row[0] for row in table["rows"]}
        for offset in range(len(expanded)):
            if offset not in grid_offsets:
                expanded[offset] = next(remaining)
        self.assertEqual(expanded, original["hazards"])
        restored = {key: value for key, value in compact.items() if key != "grid_hazards"}
        restored["hazards"] = expanded
        self.assertEqual(restored, original)
        for question in request["questions"].values():
            self.assertIn("`grid_hazards`", question["instructions"])
            self.assertIn("`columns`", question["instructions"])
            self.assertIn("not a coordinate", question["instructions"])
        # Mutating the source after the build cannot change the snapshot.
        state["hazards"][0]["x"] = -999
        state["future_state"]["unknown"].append(4)
        self.assertEqual(expanded, original["hazards"])
        self.assertEqual(compact["future_state"], original["future_state"])

    def test_grid_compaction_falls_back_for_small_lists_and_unfamiliar_shapes(self):
        grid = {"kind": "grid", "index": 1, "type": 1, "collision": 4,
                "state": 0, "x": 40, "y": 80, "radius": 20}
        for hazards in ([], [grid], None, {"unexpected": "object"},
                        [{**grid, "radius": True}] * 30,
                        [{**grid, "x": "unknown"}] * 30,
                        [{**grid, "new_field": 123}] * 30):
            with self.subTest(hazards=hazards):
                state = {"hazards": hazards, "truncated": True}
                request = build_action_request(state)
                self.assertEqual(request["state"], state)
                self.assertNotIn("`grid_hazards`", request["questions"]["move"]["instructions"])
        state = {"hazards": [grid] * 30, "grid_hazards": {"future": "meaning"}}
        self.assertEqual(build_action_request(state)["state"], state)

    def test_compact_grid_complete_request_is_smaller_including_both_questions(self):
        hazards = [{"kind": "grid", "index": i, "type": 1, "collision": 4,
                    "state": 0, "x": i * 40, "y": 80, "radius": 20}
                   for i in range(128)]
        state = {"hazards": hazards}
        compact = build_action_request(state)
        with patch("jev_isaac.jev._compact_grid_hazards", return_value=None):
            original = build_action_request(state)
        serialize = lambda value: json.dumps(value, separators=(",", ":")).encode("utf-8")
        self.assertLess(len(serialize(compact)), len(serialize(original)))
        for question_name in ("move", "shoot"):
            self.assertEqual(compact["questions"][question_name]["criteria"],
                             original["questions"][question_name]["criteria"])

    def test_builder_preserves_world_units_and_groups_decisions(self):
        state = {"session": "private-id", "room_id": "room-1", "frame": 90,
                 "player": {"x": 320, "y": 240, "vx": 2},
                 "room": {"top_left": {"x": 40, "y": 80}, "bottom_right": {"x": 600, "y": 480}},
                 "enemies": [], "projectiles": [], "hazards": []}
        request = build_action_request(state)
        self.assertEqual(set(request), {"model", "state", "questions"})
        self.assertEqual(request["state"]["player"]["x"], 320)
        self.assertNotIn("session", request["state"])
        self.assertNotIn("room_id", request["state"])
        self.assertEqual(set(request["questions"]["move"]["criteria"]), set(MOVES))
        self.assertEqual(set(request["questions"]["shoot"]["criteria"]), set(SHOOTS))
        state["player"]["x"] = 1
        self.assertEqual(request["state"]["player"]["x"], 320)
        with self.assertRaises(ValueError):
            build_action_request({"player": {"x": float("nan")}})

    def test_environment_key_and_request_contract_with_latency(self):
        calls = []
        def transport(request, timeout):
            calls.append((request, timeout))
            return HttpResponse(200, encoded())
        ticks = iter([2.0, 2.125])
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "offline-test-key"}):
            client = JevClient(transport=transport, clock=lambda: next(ticks))
        result = client.decide({"frame": 100})
        request, timeout = calls[0]
        self.assertEqual(request.full_url, ENDPOINT)
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer offline-test-key")
        self.assertEqual(json.loads(request.data)["model"], "jev-latest")
        self.assertEqual(timeout, 1.0)
        self.assertEqual((result.move, result.shoot, result.latency_ms), ("up_left", "right", 125.0))
        self.assertEqual(result.usage["input_tokens"], 700)
        self.assertNotIn("offline-test-key", repr(client))

    def test_missing_or_invalid_credentials_fail_before_transport(self):
        with patch.dict(os.environ, {}, clear=True):
            for key in (None, "", "has\nnewline"):
                with self.subTest(key_is_missing=key is None):
                    with self.assertRaises(JevCredentialsError):
                        JevClient(api_key=key)

    def test_openrouter_uses_own_key_endpoint_and_documented_contract(self):
        calls = []
        # OpenRouter documents these extra fields and snake_case token counts.
        payload = response_payload()
        payload.update(model="typesafe/jev-1.13-20260917", provider="TypeSafe", id="gen-dec-offline")
        payload["usage"]["cost"] = 0.0000294

        def transport(request, timeout):
            calls.append(request)
            return HttpResponse(200, encoded(payload))

        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "wrong-provider-key",
                                    "OPENROUTER_API_KEY": "openrouter-test-key"}, clear=True):
            client = JevClient(provider="openrouter", transport=transport)
        result = client.decide({"frame": 100})
        self.assertEqual(client.provider, "openrouter")
        self.assertEqual(client.model, OPENROUTER_DEFAULT_MODEL)
        self.assertEqual(calls[0].full_url, OPENROUTER_ENDPOINT)
        self.assertEqual(calls[0].get_header("Authorization"), "Bearer openrouter-test-key")
        request = json.loads(calls[0].data)
        self.assertEqual(request["model"], "typesafe/jev-1.13")
        self.assertEqual(set(request), {"model", "state", "questions"})
        self.assertEqual(result.model, "typesafe/jev-1.13-20260917")
        self.assertEqual(result.usage, {"input_tokens": 700, "output_tokens": 80})
        self.assertNotIn("openrouter-test-key", repr(client))

    def test_provider_selection_never_falls_back_to_another_providers_key(self):
        for provider, other_variable, own_variable in (
            ("typesafe", "OPENROUTER_API_KEY", "TYPESAFE_API_KEY"),
            ("openrouter", "TYPESAFE_API_KEY", "OPENROUTER_API_KEY"),
        ):
            with self.subTest(provider=provider), patch.dict(os.environ, {other_variable: "unrelated-key"}, clear=True):
                with self.assertRaisesRegex(JevCredentialsError, own_variable) as error:
                    JevClient(provider=provider)
                self.assertNotIn("unrelated-key", str(error.exception))

    def test_explicit_provider_key_and_model_override_environment(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "environment-key"}):
            client = JevClient("explicit-key", provider="openrouter", model="typesafe/jev-future")
        self.assertEqual(client.model, "typesafe/jev-future")
        self.assertEqual(client._api_key, "explicit-key")
        for provider in ("unknown", "https://example.com", None, []):
            with self.subTest(provider=provider), self.assertRaises(ValueError):
                JevClient("offline-key", provider=provider)

    def test_openrouter_invalid_credentials_and_responses_are_labeled(self):
        with self.assertRaisesRegex(JevCredentialsError, "OPENROUTER_API_KEY"):
            JevClient("has\nnewline", provider="openrouter")
        for payload in (b"invalid secret-key", encoded({}), encoded({**response_payload(), "usage": {"inputTokens": 700, "outputTokens": 80}})):
            client = JevClient("secret-key", provider="openrouter",
                               transport=lambda req, timeout: HttpResponse(200, payload))
            with self.subTest(payload=payload), self.assertRaises(JevResponseError) as error:
                client.decide({})
            self.assertIn("OpenRouter", str(error.exception))
            self.assertNotIn("secret-key", str(error.exception))

    def test_openrouter_billing_and_redirect_errors_do_not_retry(self):
        for status, phrase in ((402, "insufficient credits"), (302, "request failed")):
            calls = []

            def transport(request, timeout):
                calls.append(request)
                return HttpResponse(status, b"secret-key reflected by provider",
                                    {"Location": "https://example.com/"})

            client = JevClient("secret-key", provider="openrouter", transport=transport)
            with self.subTest(status=status), self.assertRaises(JevHTTPError) as error:
                client.decide({})
            self.assertIn(f"OpenRouter HTTP {status}: {phrase}", str(error.exception))
            self.assertNotIn("secret-key", str(error.exception))
            self.assertFalse(error.exception.retryable)
            self.assertEqual(len(calls), 1)

    def test_openrouter_transport_failures_are_redacted_and_labeled(self):
        for failure, expected in ((TimeoutError("secret-key"), JevTimeoutError),
                                  (URLError("secret-key"), JevTransportError),
                                  (OSError("secret-key"), JevTransportError),
                                  (HTTPError(OPENROUTER_ENDPOINT, 429, "secret-key", {}, None), JevHTTPError)):
            def transport(request, timeout):
                raise failure
            client = JevClient("secret-key", provider="openrouter", transport=transport)
            with self.subTest(failure=type(failure)), self.assertRaises(expected) as error:
                client.decide({})
            self.assertIn("OpenRouter", str(error.exception))
            self.assertNotIn("secret-key", str(error.exception))
            self.assertTrue(error.exception.__suppress_context__)

    def test_rejects_unknown_choices_and_malformed_distributions_with_safe_codes(self):
        mutations = [
            (lambda a: a.update(choice="secret-key"), "choice_unknown"),
            (lambda a: a.update(type="secret-key"), "choice_type"),
            (lambda a: a.update(confidence=True), "confidence_invalid"),
            (lambda a: a.update(confidence=float("nan")), "confidence_invalid"),
            (lambda a: a.update(confidence=10 ** 400), "confidence_invalid"),
            (lambda a: a["probabilities"].pop("none"), "probability_keys"),
            (lambda a: a["probabilities"].update({"secret-key": 0}), "probability_keys"),
            (lambda a: a["probabilities"].update(none=True), "probability_values"),
            (lambda a: a["probabilities"].update(none=float("inf")), "probability_values"),
            (lambda a: a["probabilities"].update(none=-0.1), "probability_values"),
            (lambda a: a["probabilities"].update(none=0.5), "probability_sum"),
            (lambda a: a["probabilities"].update({a["choice"]: 0, "left": 1}), "choice_not_max"),
        ]
        for question in ("move", "shoot"):
            for mutate, code in mutations:
                data = response_payload()
                data["model"] = "secret-key"
                mutate(data["answers"][question])
                with self.subTest(question=question, code=code):
                    with self.assertRaises(JevResponseError) as failure:
                        parse_action_response(encoded(data))
                    error = failure.exception
                    self.assertEqual(error.validation_code, code)
                    self.assertEqual(error.question, question)
                    self.assertEqual(str(error), f"TypeSafe returned an invalid {question} choice answer.")
                    if code == "probability_sum":
                        self.assertEqual(error.diagnostics, {"sum": 1.5})
                    elif code == "choice_not_max":
                        self.assertEqual(error.diagnostics, {"selected_probability": 0.0,
                                                             "max_probability": 1.0})
                    else:
                        self.assertEqual(error.diagnostics, {})
                    self.assertNotIn("secret-key", repr(vars(error)))
                    self.assertNotIn("secret-key", str(error))

    def test_choice_rounding_and_tie_tolerances_are_unchanged(self):
        data = response_payload()
        data["answers"]["move"]["probabilities"].update(up_left=0.5, right=0.5000005)
        data["answers"]["shoot"]["probabilities"].update(right=0.99)
        decision = parse_action_response(encoded(data))
        self.assertEqual((decision.move, decision.shoot), ("up_left", "right"))

    def test_argmax_is_explicit_and_preserves_original_judgment_and_usage(self):
        data = response_payload()
        move = data["answers"]["move"]
        move["confidence"] = 0.03
        move["probabilities"] = dict(zip(MOVES, (0.01, 0.01, 0.17, 0.16, 0.16,
                                               0.16, 0.16, 0.16, 0.01)))
        move["untrusted_extra"] = "secret-key"
        original = json.loads(json.dumps(data))
        for policy in (None, "strict"):
            kwargs = {} if policy is None else {"choice_policy": policy}
            with self.subTest(policy=policy), self.assertRaises(JevResponseError) as failure:
                parse_action_response(encoded(data), **kwargs)
            self.assertEqual(failure.exception.validation_code, "choice_not_max")
        decision = parse_action_response(encoded(data), choice_policy="argmax", latency_ms=313)
        self.assertEqual((decision.move, decision.shoot), ("right", "right"))
        self.assertEqual(decision.move_probabilities, original["answers"]["move"]["probabilities"])
        self.assertEqual(decision.shoot_probabilities, original["answers"]["shoot"]["probabilities"])
        self.assertEqual(decision.usage, original["usage"])
        self.assertEqual(decision.model, original["model"])
        self.assertEqual(decision.latency_ms, 313)
        self.assertEqual(decision.move_confidence, 0.03)
        self.assertEqual(len(decision.choice_corrections), 1)
        correction = decision.choice_corrections[0]
        self.assertEqual(set(correction), {"question", "reported_choice", "effective_choice",
                                          "selected_probability", "max_probability", "gap",
                                          "probabilities"})
        self.assertEqual(correction["question"], "move")
        self.assertEqual(correction["reported_choice"], "up_left")
        self.assertEqual(correction["effective_choice"], "right")
        self.assertEqual(correction["selected_probability"], 0.16)
        self.assertEqual(correction["max_probability"], 0.17)
        self.assertAlmostEqual(correction["gap"], 0.01)
        self.assertEqual(correction["probabilities"], decision.move_probabilities)
        self.assertNotIn("secret-key", repr(decision.choice_corrections))
        self.assertEqual(data, original)

    def test_argmax_corrects_both_questions_with_bounded_metadata(self):
        data = response_payload()
        data["answers"]["move"]["probabilities"].update(up_left=0.25, left=0.75)
        data["answers"]["shoot"]["probabilities"].update(right=0.10, down=0.90)
        decision = parse_action_response(encoded(data), choice_policy="argmax")
        self.assertEqual((decision.move, decision.shoot), ("left", "down"))
        self.assertEqual(tuple(c["question"] for c in decision.choice_corrections), ("move", "shoot"))
        self.assertEqual(tuple(c["reported_choice"] for c in decision.choice_corrections),
                         ("up_left", "right"))
        self.assertEqual(tuple(c["effective_choice"] for c in decision.choice_corrections),
                         ("left", "down"))
        self.assertEqual(tuple(len(c["probabilities"]) for c in decision.choice_corrections), (9, 5))

    def test_argmax_ties_preserve_provider_choice_or_use_declared_option_order(self):
        data = response_payload()
        for question in ("move", "shoot"):
            probabilities = data["answers"][question]["probabilities"]
            probabilities.update({option: 0.0 for option in probabilities})
        # Reverse the response ordering: it must not become the tie breaker.
        data["answers"]["move"]["probabilities"].update(right=0.5, down=0.5)
        data["answers"]["shoot"]["probabilities"].update(left=0.5, right=0.5)
        for question in ("move", "shoot"):
            probabilities = data["answers"][question]["probabilities"]
            data["answers"][question]["probabilities"] = dict(reversed(list(probabilities.items())))
        decision = parse_action_response(encoded(data), choice_policy="argmax")
        self.assertEqual((decision.move, decision.shoot), ("right", "right"))
        self.assertEqual(len(decision.choice_corrections), 1)
        self.assertEqual(decision.choice_corrections[0]["question"], "move")

    def test_argmax_selects_actual_maximum_without_widening_strict_tolerance(self):
        data = response_payload()
        data["answers"]["move"]["probabilities"].update(up_left=0.5, right=0.5000005)
        strict = parse_action_response(encoded(data))
        argmax = parse_action_response(encoded(data), choice_policy="argmax")
        self.assertEqual(strict.move, "up_left")
        self.assertEqual(strict.choice_corrections, ())
        self.assertEqual(argmax.move, "right")
        self.assertEqual(len(argmax.choice_corrections), 1)

    def test_argmax_does_not_repair_invalid_answers_or_response_contract(self):
        mutations = [
            (lambda a: a.update(choice="secret-key"), "choice_unknown"),
            (lambda a: a.update(type="secret-key"), "choice_type"),
            (lambda a: a.update(confidence=True), "confidence_invalid"),
            (lambda a: a.update(confidence=float("nan")), "confidence_invalid"),
            (lambda a: a["probabilities"].pop("none"), "probability_keys"),
            (lambda a: a["probabilities"].update({"secret-key": 0}), "probability_keys"),
            (lambda a: a["probabilities"].update(none=True), "probability_values"),
            (lambda a: a["probabilities"].update(none=float("inf")), "probability_values"),
            (lambda a: a["probabilities"].update(none=-0.1), "probability_values"),
            (lambda a: a["probabilities"].update(none=0.5), "probability_sum"),
        ]
        for question in ("move", "shoot"):
            for mutate, code in mutations:
                data = response_payload()
                mutate(data["answers"][question])
                with self.subTest(question=question, code=code), self.assertRaises(JevResponseError) as failure:
                    parse_action_response(encoded(data), choice_policy="argmax")
                self.assertEqual(failure.exception.validation_code, code)
        malformed_bodies = [
            (b"not JSON", "invalid_json"),
            (b'{"answers":{},"answers":{}}', "invalid_json"),
            (b" " * (MAX_RESPONSE_BYTES + 1), "response_size"),
            (encoded({**response_payload(), "model": None}), "model_missing"),
            (encoded({**response_payload(), "usage": {"input_tokens": True, "output_tokens": 1}}),
             "usage_invalid"),
        ]
        # A corrected move must never permit a missing/malformed shoot answer.
        data = response_payload()
        data["answers"]["move"]["probabilities"].update(up_left=0.25, left=0.75)
        del data["answers"]["shoot"]
        malformed_bodies.append((encoded(data), "answer_keys"))
        data["answers"]["shoot"] = {"type": "choice", "choice": "secret-key"}
        malformed_bodies.append((encoded(data), "choice_unknown"))
        for body, code in malformed_bodies:
            with self.subTest(code=code), self.assertRaises(JevResponseError) as failure:
                parse_action_response(body, choice_policy="argmax")
            self.assertEqual(failure.exception.validation_code, code)

    def test_client_passes_explicit_policy_without_retry_or_request_changes(self):
        data = response_payload()
        data["answers"]["shoot"]["probabilities"].update(right=0.25, left=0.75)
        transport = Mock(return_value=HttpResponse(200, encoded(data)))
        ticks = iter([1.0, 1.313])
        client = JevClient("offline-key", choice_policy="argmax", transport=transport,
                           clock=lambda: next(ticks))
        decision = client.decide({"frame": 100})
        self.assertEqual(client.choice_policy, "argmax")
        self.assertEqual(decision.shoot, "left")
        self.assertAlmostEqual(decision.latency_ms, 313)
        self.assertEqual(len(decision.choice_corrections), 1)
        transport.assert_called_once()
        self.assertEqual(json.loads(transport.call_args.args[0].data), build_action_request({"frame": 100}))
        # Default clients retain contract-error behavior and never retry it.
        transport.reset_mock()
        strict_client = JevClient("offline-key", transport=transport)
        self.assertEqual(strict_client.choice_policy, "strict")
        with self.assertRaises(JevResponseError) as failure:
            strict_client.decide({"frame": 101})
        self.assertEqual(failure.exception.validation_code, "choice_not_max")
        transport.assert_called_once()

    def test_invalid_choice_policy_fails_locally_without_echo_or_transport(self):
        transport = Mock()
        for policy in (None, [], {}, True, "secret-key"):
            with self.subTest(policy=policy):
                with self.assertRaises(ValueError) as failure:
                    JevClient("offline-key", choice_policy=policy, transport=transport)
                self.assertNotIn("secret-key", str(failure.exception))
                with self.assertRaises(ValueError):
                    parse_action_response(encoded(), choice_policy=policy)
        transport.assert_not_called()

    def test_decision_original_constructor_and_valid_replies_have_no_corrections(self):
        original_constructor = Decision("none", "none", 0.0, "test", 0.0, 0.0, {}, {}, {})
        self.assertEqual(original_constructor.choice_corrections, ())
        for policy in ("strict", "argmax"):
            with self.subTest(policy=policy):
                decision = parse_action_response(encoded(), choice_policy=policy)
                self.assertEqual(decision.choice_corrections, ())

    def test_response_diagnostic_metadata_is_allowlisted_and_finite(self):
        for untrusted in ("secret-key", ["secret-key"]):
            error = JevResponseError("safe failure", validation_code=untrusted,
                                     question=untrusted, diagnostics={
                                         "secret-key": 1, "sum": float("nan"),
                                         "selected_probability": "secret-key",
                                         "max_probability": float("inf"),
                                     })
            self.assertEqual(error.validation_code, "response_shape")
            self.assertIsNone(error.question)
            self.assertEqual(error.diagnostics, {})
            self.assertNotIn("secret-key", repr(vars(error)))
        error = JevResponseError("safe failure", validation_code="probability_sum",
                                 question="move", diagnostics={"sum": 1.25,
                                                              "selected_probability": True,
                                                              "max_probability": 10 ** 400})
        self.assertEqual(error.diagnostics, {"sum": 1.25})

    def test_both_answers_required_before_any_action(self):
        data = response_payload()
        del data["answers"]["shoot"]
        with self.assertRaises(JevResponseError) as failure:
            parse_action_response(encoded(data))
        self.assertEqual(failure.exception.validation_code, "answer_keys")
        data = response_payload()
        data["usage"]["input_tokens"] = True
        with self.assertRaises(JevResponseError) as failure:
            parse_action_response(encoded(data))
        self.assertEqual(failure.exception.validation_code, "usage_invalid")

    def test_invalid_json_duplicate_fields_and_size_fail_closed(self):
        bodies = (
            ("secret-key", "response_type"),
            (b"not json secret-key", "invalid_json"),
            (b"\xff", "invalid_json"),
            (b"[]", "response_shape"),
            (b'{"answers":{},"answers":{}}', "invalid_json"),
            (b" " * 65_537, "response_size"),
            (encoded({"answers": "secret-key"}), "answer_keys"),
            (encoded({**response_payload(), "model": None}), "model_missing"),
            (encoded({**response_payload(), "usage": "secret-key"}), "usage_invalid"),
        )
        for body, code in bodies:
            with self.subTest(code=code):
                with self.assertRaises(JevResponseError) as failure:
                    parse_action_response(body)
                self.assertEqual(failure.exception.validation_code, code)
                self.assertIsNone(failure.exception.question)
                self.assertEqual(failure.exception.diagnostics, {})
                self.assertNotIn("secret-key", repr(vars(failure.exception)))
                self.assertNotIn("secret-key", str(failure.exception))

    def test_http_failures_are_redacted_and_not_retried(self):
        for status, retryable in ((401, False), (422, False), (429, True), (529, True), (302, False)):
            calls = []
            def transport(request, timeout):
                calls.append(1)
                return HttpResponse(status, b"secret-key reflected in provider error", {"Retry-After": "2"})
            client = JevClient(api_key="secret-key", transport=transport)
            with self.assertRaises(JevHTTPError) as error:
                client.decide({})
            self.assertEqual(error.exception.status, status)
            self.assertEqual(error.exception.retryable, retryable)
            self.assertEqual(error.exception.retry_after, 2.0)
            self.assertNotIn("secret-key", str(error.exception))
            self.assertEqual(len(calls), 1)

    def test_timeout_and_connection_failures_are_redacted(self):
        for failure, expected in ((TimeoutError("secret-key"), JevTimeoutError),
                                  (URLError(TimeoutError("secret-key")), JevTimeoutError),
                                  (URLError("secret-key"), JevTransportError),
                                  (OSError("secret-key"), JevTransportError)):
            def transport(request, timeout):
                raise failure
            client = JevClient(api_key="secret-key", transport=transport)
            with self.assertRaises(expected) as error:
                client.decide({})
            self.assertNotIn("secret-key", str(error.exception))
            self.assertTrue(error.exception.__suppress_context__)

    def test_late_transport_success_is_discarded(self):
        ticks = iter([1.0, 2.1])
        client = JevClient(api_key="offline", clock=lambda: next(ticks),
                           timeout=1.0, transport=lambda req, timeout: HttpResponse(200, encoded()))
        with self.assertRaises(JevTimeoutError):
            client.decide({})

    def test_invalid_transport_result_has_safe_code_and_elapsed_latency(self):
        ticks = iter([1.0, 1.313])
        client = JevClient(api_key="secret-key", clock=lambda: next(ticks),
                           transport=lambda req, timeout: {"body": "secret-key"})
        with self.assertRaises(JevResponseError) as failure:
            client.decide({})
        self.assertEqual(failure.exception.validation_code, "response_type")
        self.assertIsNone(failure.exception.question)
        self.assertEqual(failure.exception.diagnostics, {})
        self.assertAlmostEqual(failure.exception.latency_ms, 313)
        self.assertNotIn("secret-key", repr(vars(failure.exception)))
        self.assertNotIn("secret-key", str(failure.exception))

    def test_parser_error_keeps_diagnostics_when_client_attaches_elapsed_latency(self):
        data = response_payload()
        data["answers"]["shoot"]["probabilities"].update(right=0.25, left=0.75)
        ticks = iter([1.0, 1.313])
        client = JevClient(api_key="offline", clock=lambda: next(ticks),
                           transport=lambda req, timeout: HttpResponse(200, encoded(data)))
        with self.assertRaises(JevResponseError) as failure:
            client.decide({})
        self.assertEqual(failure.exception.validation_code, "choice_not_max")
        self.assertEqual(failure.exception.question, "shoot")
        self.assertEqual(failure.exception.diagnostics,
                         {"selected_probability": 0.25, "max_probability": 0.75})
        self.assertAlmostEqual(failure.exception.latency_ms, 313)

    def test_invalid_timeouts(self):
        for timeout in (0, -1, True, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                JevClient(api_key="offline", timeout=timeout)


if __name__ == "__main__":
    unittest.main()
