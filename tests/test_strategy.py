"""Offline bound adventure/combat request contracts; no service calls."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_goals import state
from jev_isaac.jev import HttpResponse, JevResponseError, MAX_RESPONSE_BYTES
from jev_isaac.strategy import AdventureClient, EquippedGoalDecision, StrategyDecision


def answer(choice, options):
    return {"type": "choice", "choice": choice, "confidence": 1.,
            "probabilities": {key: float(key == choice) for key in options}}


def result(goal="enemy_0", ability=None):
    payload = {"model": "jev-offline", "usage": {"input_tokens": 400, "output_tokens": 50},
               "answers": {"goal": answer(goal, ("hold", "evade", "enemy_0"))}}
    if ability is not None:
        payload["answers"]["ability"] = answer(ability, ("none", "ability_0"))
    return payload


def choice(key="collect:pill:70:1"):
    return {"key": key, "kind": "collect", "target_id": "pill", "point": [200, 280],
            "cost": {}, "description": "Carry an unidentified pill", "details": {"pill_known": False}}


class StrategyTests(unittest.TestCase):
    def call(self, observed, payload, inspect=None, **options):
        calls = []
        def transport(request, timeout):
            copied = json.loads(request.data)
            calls.append(copied)
            if inspect:
                inspect(copied)
            return HttpResponse(200, payload if isinstance(payload, bytes) else json.dumps(payload).encode())
        client = AdventureClient("offline-key", transport=transport, **options)
        try:
            returned = client.decide(observed)
        finally:
            client.close()
        self.assertEqual(len(calls), 1)
        return returned, calls[0]

    def test_adventure_binds_only_offered_identity_and_keeps_map_inventory_context(self):
        observed = state()
        observed["_adventure_options"] = [choice()]
        observed["known_map"] = {"visited": [84, 83], "open_frontiers": [82]}
        observed["player"]["inventory"] = [{"id": 1, "count": 1}]
        before = copy.deepcopy(observed)
        returned, request = self.call(observed, result())
        self.assertIsInstance(returned, StrategyDecision)
        self.assertEqual((returned.kind, returned.target_id), ("adventure", "collect:pill:70:1"))
        self.assertEqual(set(request["questions"]), {"goal"})
        self.assertEqual(request["state"]["goal_candidates"], [])
        self.assertEqual(request["state"]["adventure_candidates"][0]["option"], "enemy_0")
        copied = request["state"]["observation"]
        self.assertEqual(copied["known_map"], before["known_map"])
        self.assertEqual(copied["player"]["inventory"], before["player"]["inventory"])
        self.assertNotIn("_adventure_options", copied)
        self.assertNotIn("_ability_options", copied)
        self.assertEqual(observed, before)
        self.assertEqual(returned.usage, {"input_tokens": 400, "output_tokens": 50})

    def test_inflight_mutation_cannot_rebind_adventure_selection(self):
        observed = state()
        observed["_adventure_options"] = [choice()]
        def mutate(request):
            observed["_adventure_options"][0]["key"] = "different"
        returned, _ = self.call(observed, result(), inspect=mutate)
        self.assertEqual(returned.target_id, "collect:pill:70:1")

    def test_hold_and_evade_skip_adventure_offers_without_command(self):
        for selected in ("hold", "evade"):
            observed = state()
            observed["_adventure_options"] = [choice()]
            returned, _ = self.call(observed, result(selected))
            self.assertEqual(returned.kind, "adventure")
            self.assertIsNone(returned.target_id)

    def test_combined_combat_and_ability_use_independent_bound_options(self):
        observed = state()
        observed["_ability_options"] = [choice("active:34")]
        returned, request = self.call(observed, result(ability="ability_0"))
        self.assertIsInstance(returned, EquippedGoalDecision)
        self.assertEqual((returned.kind, returned.target_id, returned.ability_key),
                         ("engage", "target-A", "active:34"))
        self.assertEqual(set(request["questions"]), {"goal", "ability"})
        self.assertEqual(request["state"]["ability_candidates"][0]["option"], "ability_0")
        self.assertEqual(returned.usage["input_tokens"], 400)
        returned, _ = self.call(observed, result("evade", "none"))
        self.assertEqual(returned.kind, "evade")
        self.assertIsNone(returned.ability_key)

    def test_no_ability_offers_retains_existing_tactical_contract(self):
        returned, request = self.call(state(), result())
        self.assertEqual((returned.kind, returned.target_id), ("engage", "target-A"))
        self.assertEqual(set(request["questions"]), {"goal"})

    def test_argmax_corrections_are_retained_for_both_questions(self):
        observed = state()
        observed["_ability_options"] = [choice("active:34")]
        payload = result("hold", "none")
        payload["answers"]["goal"]["probabilities"] = {"hold": .1, "evade": .1, "enemy_0": .8}
        payload["answers"]["ability"]["probabilities"] = {"none": .2, "ability_0": .8}
        returned, _ = self.call(observed, payload)
        self.assertEqual(returned.target_id, "target-A")
        self.assertEqual(returned.ability_key, "active:34")
        self.assertEqual({v["question"] for v in returned.choice_corrections}, {"goal", "ability"})
        with self.assertRaises(JevResponseError):
            self.call(observed, payload, choice_policy="strict")

    def test_unknown_ability_or_generated_command_rejected(self):
        observed = state()
        observed["_ability_options"] = [choice("active:34")]
        for mutate in (lambda p: p["answers"]["ability"].update(choice="bomb"),
                       lambda p: p["answers"]["ability"].update(choice="ability_1"),
                       lambda p: p["answers"]["ability"].update(duration=100),
                       lambda p: p["answers"].update(move={"choice": "left"}),
                       lambda p: p["answers"].pop("goal")):
            payload = result(ability="ability_0")
            mutate(payload)
            with self.assertRaises(JevResponseError):
                self.call(observed, payload)

    def test_duplicate_oversized_and_nonfinite_combined_responses_rejected(self):
        observed = state()
        observed["_ability_options"] = [choice("active:34")]
        bodies = [b" "*(MAX_RESPONSE_BYTES+1), b"[]", b'{}',
                  b'{"answers":{},"answers":{}}', b'\xff']
        bad = result(ability="ability_0")
        bad["answers"]["goal"]["confidence"] = float("nan")
        bodies.append(json.dumps(bad).encode())
        for body in bodies:
            with self.subTest(body=body[:40]), self.assertRaises(JevResponseError):
                self.call(observed, body)

    def test_duplicate_excessive_and_nonfinite_local_options_rejected_before_post(self):
        for offers in ([choice(), choice()], [choice(str(i)) for i in range(9)],
                       [choice("")], [dict(choice(), bad=float("nan"))]):
            observed = state()
            observed["_adventure_options"] = offers
            with self.assertRaises(ValueError):
                self.call(observed, result())


if __name__ == "__main__":
    unittest.main()
