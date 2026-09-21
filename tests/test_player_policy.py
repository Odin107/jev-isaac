"""Jev owns firing direction and source-bound activities; no live API calls."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).resolve().parents[1]/"src")]
from test_goals import state, enemy
from test_strategy import choice
from jev_isaac.player_policy import PlayerClient
from jev_isaac.jev import HttpResponse, JevResponseError
from jev_isaac.navigation import compute_action
from test_navigation import state as moving_state, enemy as moving_enemy


def reply(request, choices):
    return {"model": "jev-offline", "usage": {"input_tokens": 400, "output_tokens": 60},
            "answers": {key: {"type": "choice", "choice": choices.get(key, next(iter(q["criteria"]))),
                "confidence": 1., "probabilities": {option: float(option == choices.get(key, next(iter(q["criteria"]))))
                                                     for option in q["criteria"]}}
                        for key, q in request["questions"].items()}}


def call(data, choices, mutate=None):
    sent = []
    def transport(request, timeout):
        payload = json.loads(request.data)
        sent.append(payload)
        answer = reply(payload, choices)
        if mutate:
            mutate(answer)
        return HttpResponse(200, json.dumps(answer).encode())
    client = PlayerClient("offline-test", transport=transport)
    try:
        decision = client.decide(data)
    finally:
        client.close()
    return decision, sent[0]


class PlayerPolicyTests(unittest.TestCase):
    def test_hold_and_fire_are_independent_explicit_model_choices(self):
        decision, request = call(state(), {"goal": "hold", "fire": "right"})
        self.assertEqual((decision.kind, decision.fire_direction), ("hold", "right"))
        self.assertEqual(set(request["questions"]), {"goal", "fire"})
        self.assertEqual(set(request["questions"]["fire"]["criteria"]), {"none", "left", "right", "up", "down"})
        self.assertIn("no implied shooting", request["questions"]["goal"]["criteria"]["hold"])

    def test_all_observed_vulnerable_targets_are_offered_beyond_the_old_eight(self):
        data = state([enemy(f"enemy-{i:02}", x=140+i*10) for i in range(20)])
        decision, request = call(data, {"goal": "enemy_19", "fire": "none"})
        self.assertEqual(len(request["state"]["goal_candidates"]), 20)
        self.assertEqual(decision.target_id, "enemy-19")
        self.assertEqual(decision.fire_direction, "none")

    def test_momentum_is_explicit_without_inventing_a_multiplier_or_trajectory(self):
        data = state()
        data["player"].update(vx=2.5, vy=-.4, shot_speed=1.2)
        before = copy.deepcopy(data)
        _, request = call(data, {"goal": "hold", "fire": "down"})
        self.assertIn("diagonally", request["questions"]["fire"]["instructions"])
        self.assertIn("uncalibrated", request["questions"]["fire"]["instructions"])
        self.assertEqual(request["state"]["game_context"]["shooting_motion"]["player_velocity"], {"vx": 2.5, "vy": -.4})
        self.assertEqual(data, before)

    def test_activity_can_select_ninth_offer_or_wait_without_automatic_exploration(self):
        data = state()
        data["_adventure_options"] = [choice(f"collect:{i}") for i in range(12)]
        decision, request = call(data, {"activity": "action_10"})
        self.assertEqual(decision.target_id, "collect:10")
        self.assertEqual(len(request["state"]["activity_candidates"]), 12)
        self.assertEqual(set(request["questions"]), {"activity"})
        waiting, _ = call(data, {"activity": "wait"})
        self.assertIsNone(waiting.target_id)
        self.assertNotIn("_adventure_options", request["state"]["observation"])

    def test_ability_shares_one_request_with_movement_and_fire(self):
        data = state()
        data["_ability_options"] = [choice("active:34")]
        decision, request = call(data, {"goal": "evade", "fire": "up", "ability": "ability_0"})
        self.assertEqual(set(request["questions"]), {"goal", "fire", "ability"})
        self.assertEqual((decision.kind, decision.fire_direction, decision.ability_key), ("evade", "up", "active:34"))

    def test_missing_extra_unbound_and_nonfinite_answers_are_rejected(self):
        mutations = [lambda d: d["answers"].pop("fire"),
                     lambda d: d["answers"].update(move=d["answers"]["goal"]),
                     lambda d: d["answers"]["fire"].update(choice="diagonal"),
                     lambda d: d["answers"]["fire"]["probabilities"].update(right=float("nan")),
                     lambda d: d["answers"]["fire"].update(target_id="invented"),
                     lambda d: d["usage"].update(input_tokens=True)]
        for mutate in mutations:
            with self.subTest(mutate=mutate), self.assertRaises(JevResponseError):
                call(state(), {"goal": "hold", "fire": "none"}, mutate)

    def test_source_mutation_cannot_rebind_activity(self):
        data = state()
        data["_adventure_options"] = [choice("original")]
        decision, _ = call(data, {"activity": "action_0"},
                           lambda _: data["_adventure_options"][0].update(key="different"))
        self.assertEqual(decision.target_id, "original")

    def test_local_executor_never_adds_or_substitutes_fire(self):
        data = moving_state()
        data["enemies"] = [moving_enemy(450, 250)]
        for goal in ("hold", "evade", "engage"):
            for fire in ("none", "left", "right", "up", "down"):
                with self.subTest(goal=goal, fire=fire):
                    action = compute_action(data, goal, "target" if goal == "engage" else None, fire_direction=fire)
                    self.assertEqual(action.shoot, fire)
        data["player"]["vx"] = 2
        self.assertEqual(compute_action(data, "hold", fire_direction="up").shoot, "up")

    def test_emergency_dodge_reports_override_but_keeps_jev_firing_button(self):
        data = moving_state()
        data["projectiles"] = [{"x": 340, "y": 250, "vx": -8, "vy": 0, "radius": 5}]
        action = compute_action(data, "hold", fire_direction="none")
        self.assertNotEqual(action.move, "none")
        self.assertEqual(action.shoot, "none")
        self.assertEqual(action.override, "emergency collision avoidance")

    def test_disabled_or_incomplete_state_still_revokes_fire(self):
        for field, value in (("paused", True), ("enabled", False), ("truncated", True)):
            data = moving_state()
            data[field] = value
            action = compute_action(data, "hold", fire_direction="right")
            self.assertEqual((action.move, action.shoot), ("none", "none"))


if __name__ == "__main__":
    unittest.main()
