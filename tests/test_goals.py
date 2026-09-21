"""Offline goal contracts; no provider calls or real credentials."""
from copy import deepcopy
import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import Mock, patch

from jev_isaac.goals import GoalClient, GoalDecision, build_goal_request, parse_goal_response
from jev_isaac.jev import (
    ENDPOINT, OPENROUTER_ENDPOINT, MAX_RESPONSE_BYTES, HttpResponse,
    JevHTTPError, JevResponseError, JevTransportError, JevTimeoutError,
)


def enemy(ident, x=180, y=160, **changes):
    return {"id": ident, "x": x, "y": y, "vx": 0, "vy": 0, "hp": 10,
            "vulnerable": True, "radius": 12, **changes}


def state(enemies=None):
    return {"session": "local-only", "room_id": "room", "run_id": "run",
            "enabled": True, "paused": False, "frame": 30,
            "player": {"x": 100, "y": 160, "radius": 10, "dead": False},
            "room": {"top_left": {"x": 40, "y": 40},
                     "bottom_right": {"x": 600, "y": 400}, "clear": False},
            "enemies": [enemy("target-A")] if enemies is None else enemies,
            "hazards": [], "projectiles": []}


def response(choice="enemy_0", options=("hold", "evade", "enemy_0")):
    return {"model": "jev-test", "answers": {"goal": {
        "type": "choice", "choice": choice, "confidence": 1.0,
        "probabilities": {key: float(key == choice) for key in options}}},
        "usage": {"input_tokens": 411, "output_tokens": 51}}


def encoded(payload=None):
    return json.dumps(response() if payload is None else payload).encode()


class GoalRequestTests(unittest.TestCase):
    def test_tactical_choice_keeps_movement_and_shooting_independent(self):
        request = build_goal_request(state())
        question = request["questions"]["goal"]
        self.assertIn("Movement and shooting are independent", question["instructions"])
        self.assertIn("Evade and hold do not disable shooting", question["instructions"])
        self.assertIn("shoot simultaneously", question["criteria"]["evade"])
        self.assertIn("Do not pursue or reposition", question["criteria"]["hold"])
        self.assertEqual(set(request["questions"]), {"goal"})

    def test_nearest_eight_binding_is_bounded_and_entity_order_independent(self):
        observed = state([enemy(f"target-{i:02}", x=120+i*15) for i in range(12)])
        request = build_goal_request(observed)
        candidates = request["state"]["goal_candidates"]
        self.assertEqual(candidates, [{"option": f"enemy_{i}", "id": f"target-{i:02}"}
                                      for i in range(8)])
        self.assertEqual(set(request["questions"]["goal"]["criteria"]),
                         {"hold", "evade", *(f"enemy_{i}" for i in range(8))})
        observed["enemies"].reverse()
        self.assertEqual(build_goal_request(observed)["state"]["goal_candidates"], candidates)

    def test_invalid_dead_invulnerable_and_ambiguous_entities_are_not_options(self):
        observed = state([enemy("good"), enemy("zero", hp=0), enemy("dead", dead=True),
                          enemy("immune", vulnerable=False), enemy("unknown", vulnerable=None),
                          enemy("duplicate"), enemy("duplicate", x=280), enemy(""),
                          enemy("x" * 129), enemy("bad\nid"), enemy(19),
                          enemy("boolhp", hp=True), enemy("bad-position", x=None)])
        candidates = build_goal_request(observed)["state"]["goal_candidates"]
        self.assertEqual(candidates, [{"option": "enemy_0", "id": "good"}])

    def test_copied_state_keeps_unfamiliar_fields_without_input_mutation(self):
        observed = state()
        observed["goal_candidates"] = {"unexpected": [1, 2]}
        original = deepcopy(observed)
        request = build_goal_request(observed)
        self.assertEqual(observed, original)
        copied = request["state"]["observation"]
        self.assertEqual(copied["goal_candidates"], {"unexpected": [1, 2]})
        self.assertFalse({"session", "room_id", "run_id"} & set(copied))
        observed["enemies"][0]["id"] = "changed"
        self.assertEqual(copied["enemies"][0]["id"], "target-A")
        self.assertEqual(request["state"]["goal_candidates"][0]["id"], "target-A")

    def test_large_grid_is_losslessly_compacted_without_discarding_other_hazards(self):
        observed = state()
        observed["hazards"] = [{"kind": "grid", "radius": 20, "index": i,
                                "x": 20*i, "y": 200, "type": 2, "collision": 3, "state": 1}
                               for i in range(64)]
        observed["hazards"].insert(3, {"kind": "fire", "x": 42, "y": 44})
        request = build_goal_request(observed)
        compact = request["state"]["observation"]
        table = compact["grid_hazards"]
        restored = list(compact["hazards"])
        for values in table["rows"]:
            row = dict(zip(table["columns"], values))
            offset = row.pop("hazard_offset")
            restored.insert(offset, {**table["shared"], **row})
        self.assertEqual(restored, observed["hazards"])
        self.assertIn("Within `observation`", request["questions"]["goal"]["instructions"])

    def test_no_candidate_keeps_hold_and_evade_and_nonfinite_state_rejected(self):
        request = build_goal_request(state([]))
        self.assertEqual(set(request["questions"]["goal"]["criteria"]), {"hold", "evade"})
        for value in (float("nan"), float("inf"), {1, 2}):
            with self.subTest(value=type(value)), self.assertRaises(ValueError):
                build_goal_request({"invalid": value})


class GoalResponseTests(unittest.TestCase):
    def test_goal_resolves_only_the_source_map_and_usage_is_retained(self):
        result = parse_goal_response(encoded(), {"enemy_0": "original-id"}, latency_ms=320)
        self.assertEqual((result.kind, result.target_id), ("engage", "original-id"))
        self.assertEqual(result.latency_ms, 320)
        self.assertEqual(result.usage, {"input_tokens": 411, "output_tokens": 51})
        self.assertEqual(result.choice_corrections, ())
        for choice in ("hold", "evade"):
            result = parse_goal_response(encoded(response(choice)), {"enemy_0": "original-id"})
            self.assertEqual((result.kind, result.target_id), (choice, None))

    def test_argmax_records_discrepancy_without_dropping_usage(self):
        payload = response("hold")
        payload["answers"]["goal"]["probabilities"] = {"hold": .1, "evade": .2, "enemy_0": .7}
        result = parse_goal_response(encoded(payload), {"enemy_0": "target-A"})
        self.assertEqual((result.kind, result.target_id), ("engage", "target-A"))
        correction, = result.choice_corrections
        self.assertEqual(correction["question"], "goal")
        self.assertEqual(correction["reported_choice"], "hold")
        self.assertEqual(correction["effective_choice"], "enemy_0")
        self.assertEqual(result.usage["input_tokens"], 411)
        with self.assertRaises(JevResponseError) as caught:
            parse_goal_response(encoded(payload), {"enemy_0": "target-A"}, choice_policy="strict")
        self.assertEqual(caught.exception.validation_code, "choice_not_max")
        self.assertEqual(caught.exception.question, "goal")

    def test_unknown_goal_target_id_or_extra_command_is_rejected(self):
        for choice in ("engage", "left", "target-A", "enemy_1", "delete"):
            payload = response()
            payload["answers"]["goal"]["choice"] = choice
            with self.subTest(choice=choice), self.assertRaises(JevResponseError):
                parse_goal_response(encoded(payload), {"enemy_0": "target-A"})
        for key, value in (("target_id", "invented-id"), ("duration", 100), ("move", "left")):
            payload = response()
            payload["answers"]["goal"][key] = value
            with self.subTest(key=key), self.assertRaises(JevResponseError):
                parse_goal_response(encoded(payload), {"enemy_0": "target-A"})

    def test_incomplete_nonfinite_and_invalid_distributions_are_rejected(self):
        bads = [({"hold": .2, "enemy_0": .8}, 1),
                ({"hold": .2, "evade": .2, "enemy_0": .2}, 1),
                ({"hold": float("nan"), "evade": 0, "enemy_0": 1}, 1),
                ({"hold": -.1, "evade": 0, "enemy_0": 1.1}, 1),
                ({"hold": 0, "evade": 0, "enemy_0": True}, 1),
                ({"hold": 0, "evade": 0, "enemy_0": 1}, True),
                ({"hold": 0, "evade": 0, "enemy_0": 1}, 2)]
        for probabilities, confidence in bads:
            payload = response()
            payload["answers"]["goal"].update(probabilities=probabilities, confidence=confidence)
            with self.subTest(probabilities=probabilities), self.assertRaises(JevResponseError):
                parse_goal_response(encoded(payload), {"enemy_0": "target-A"})

    def test_bounded_duplicate_json_and_invalid_envelope_are_rejected(self):
        bodies = ["not-bytes", b" " * (MAX_RESPONSE_BYTES+1), b"\xff", b"[]",
                  b'{"answers":{},"answers":{}}']
        for body in bodies:
            with self.subTest(kind=type(body)), self.assertRaises(JevResponseError):
                parse_goal_response(body, {"enemy_0": "target-A"})
        for key, value in (("answers", {"move": {}}), ("model", ""),
                           ("usage", {"input_tokens": True, "output_tokens": 1}),
                           ("usage", {"input_tokens": 1, "output_tokens": -1})):
            payload = response()
            payload[key] = value
            with self.subTest(key=key), self.assertRaises(JevResponseError):
                parse_goal_response(encoded(payload), {"enemy_0": "target-A"})

    def test_invalid_local_bindings_and_goal_shapes_are_rejected(self):
        bindings = [{"target-A": "id"}, {"enemy_8": "id"}, {"enemy_0": ""},
                    {"enemy_0": "same", "enemy_1": "same"}, {"enemy_0": 1}]
        for mapping in bindings:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                parse_goal_response(encoded(), mapping)
        for kind, ident in (("left", None), ("hold", "target"), ("engage", None)):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                GoalDecision(kind, ident, 0, "test", {})


class GoalClientTests(unittest.TestCase):
    def test_client_uses_fixed_endpoint_and_binds_before_source_changes(self):
        for provider, endpoint in (("typesafe", ENDPOINT), ("openrouter", OPENROUTER_ENDPOINT)):
            observed = state()
            def transport(request, timeout):
                self.assertEqual(request.full_url, endpoint)
                self.assertEqual(request.get_method(), "POST")
                self.assertEqual(timeout, 2.0)
                self.assertEqual(request.get_header("Authorization"), "Bearer offline-key")
                payload = json.loads(request.data)
                self.assertNotIn("offline-key", request.data.decode())
                self.assertEqual(set(payload["questions"]), {"goal"})
                observed["enemies"][0]["id"] = "new-entity"
                return HttpResponse(200, encoded())
            ticks = iter([1.0, 1.35])
            client = GoalClient("offline-key", provider=provider, transport=transport,
                                clock=lambda: next(ticks))
            result = client.decide(observed)
            self.assertEqual(result.target_id, "target-A")
            self.assertAlmostEqual(result.latency_ms, 350)
            client.close()

    def test_failure_has_no_key_body_logging_and_no_retry(self):
        for failure, expected in ((OSError("secret-value"), JevTransportError),
                                  (TimeoutError("secret-value"), JevTimeoutError)):
            transport = Mock(side_effect=failure)
            output = io.StringIO()
            with redirect_stdout(output), redirect_stderr(output):
                with self.assertRaises(expected) as caught:
                    GoalClient("secret-value", transport=transport).decide(state())
            self.assertNotIn("secret-value", str(caught.exception))
            self.assertEqual(output.getvalue(), "")
            transport.assert_called_once()
        for status, body, expected in ((403, b"secret-value", JevHTTPError),
                                      (200, b"secret-value", JevResponseError)):
            transport = Mock(return_value=HttpResponse(status, body))
            with self.assertRaises(expected) as caught:
                GoalClient("secret-value", transport=transport).decide(state())
            self.assertNotIn("secret-value", str(caught.exception))
            transport.assert_called_once()

    def test_elapsed_timeout_and_invalid_response_retain_safe_timing(self):
        ticks = iter([1.0, 3.1])
        with self.assertRaises(JevTimeoutError):
            GoalClient("offline", transport=lambda *_: HttpResponse(200, encoded()),
                       clock=lambda: next(ticks)).decide(state())
        ticks = iter([1.0, 1.3])
        with self.assertRaises(JevResponseError) as caught:
            GoalClient("offline", transport=lambda *_: HttpResponse(200, b"{}"),
                       clock=lambda: next(ticks)).decide(state())
        self.assertAlmostEqual(caught.exception.latency_ms, 300)

    def test_owned_transport_is_reused_and_closed_through_shared_lifecycle(self):
        with patch("jev_isaac.jev._PersistentHTTPTransport") as constructor:
            transport = constructor.return_value
            transport.return_value = HttpResponse(200, encoded())
            client = GoalClient("offline")
            client.decide(state())
            client.decide(state())
            self.assertEqual(transport.call_count, 2)
            constructor.assert_called_once_with(ENDPOINT)
            client.close()
            transport.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
