"""Target-bound back-off choices, parallel input and freshness; no paid calls."""
import copy
import unittest
from unittest.mock import patch

from test_goals import enemy, state, build_goal_request, encoded, response
from test_player_policy import call
from test_player_controller import combat, frame, run, OLD, NEW
from test_strategy import choice
from jev_isaac.controller import Action
from jev_isaac.goals import GoalDecision, parse_goal_response
from jev_isaac.jev import JevResponseError


class BackOffPolicyTests(unittest.TestCase):
    def test_each_observed_target_has_two_explicitly_bound_movement_choices(self):
        observed = state([enemy("east", x=180), enemy("north", x=100, y=100)])
        for option, kind in (("enemy_1", "engage"), ("back_off_enemy_1", "back_off")):
            with self.subTest(option=option):
                decision, request = call(observed, {"goal": option, "fire": "left"})
                self.assertEqual((decision.kind, decision.target_id, decision.fire_direction),
                                 (kind, "north", "left"))
                candidates = request["state"]["goal_candidates"]
                self.assertEqual({c["option"]: (c["id"], c["kind"]) for c in candidates},
                    {"enemy_0": ("east", "engage"), "enemy_1": ("north", "engage"),
                     "back_off_enemy_0": ("east", "back_off"),
                     "back_off_enemy_1": ("north", "back_off")})

    def test_all_64_targets_have_engage_and_back_off_with_bounded_choice_count(self):
        observed = state([enemy(f"target-{index:02}") for index in range(64)])
        decision, request = call(observed, {"goal": "back_off_enemy_63", "fire": "none"})
        self.assertEqual((decision.kind, decision.target_id), ("back_off", "target-63"))
        criteria = request["questions"]["goal"]["criteria"]
        self.assertEqual(len(criteria), 130)
        self.assertEqual(len(request["state"]["goal_candidates"]), 128)
        self.assertEqual({"hold", "evade"}, set(criteria) - {
            c["option"] for c in request["state"]["goal_candidates"]})

    def test_invalid_targets_cannot_become_back_off_options(self):
        observed = state([enemy("dead", dead=True), enemy("empty", hp=0),
                          enemy("invulnerable", vulnerable=False), enemy("duplicate"),
                          enemy("duplicate", x=220)])
        _, request = call(observed, {"goal": "hold", "fire": "none"})
        self.assertEqual(request["state"]["goal_candidates"], [])
        self.assertEqual(set(request["questions"]["goal"]["criteria"]), {"hold", "evade"})

    def test_source_mutation_cannot_rebind_back_off_target(self):
        observed = state([enemy("first"), enemy("second")])
        before = copy.deepcopy(observed)
        def mutate(_):
            observed["enemies"].reverse()
            observed["enemies"][1].update(id="replacement")
        decision, request = call(observed, {"goal": "back_off_enemy_0", "fire": "up"}, mutate)
        self.assertEqual(decision.target_id, "first")
        self.assertEqual(request["state"]["observation"]["enemies"], before["enemies"])

    def test_argmax_correction_resolves_back_off_and_records_its_original_choice(self):
        def mutate(answer):
            answer["answers"]["goal"].update(choice="enemy_0", confidence=.7,
                probabilities={"hold": .1, "evade": .1, "enemy_0": .1, "back_off_enemy_0": .7})
        decision, _ = call(state(), {"goal": "enemy_0", "fire": "down"}, mutate)
        self.assertEqual((decision.kind, decision.target_id, decision.fire_direction),
                         ("back_off", "target-A", "down"))
        correction, = decision.choice_corrections
        self.assertEqual(correction["reported_choice"], "enemy_0")
        self.assertEqual(correction["effective_choice"], "back_off_enemy_0")

    def test_back_off_does_not_choose_or_remove_parallel_fire_and_ability(self):
        observed = state()
        observed["_ability_options"] = [choice("active:34")]
        for fire in ("none", "left", "right", "up", "down"):
            with self.subTest(fire=fire):
                decision, request = call(observed, {"goal": "back_off_enemy_0", "fire": fire,
                                                     "ability": "ability_0"})
                self.assertEqual(set(request["questions"]), {"goal", "fire", "ability"})
                self.assertEqual((decision.fire_direction, decision.ability_key), (fire, "active:34"))
                self.assertEqual(decision.fire_judgment["probabilities"][fire], 1.)

    def test_unbound_back_off_and_invented_target_answers_are_rejected(self):
        for mutate in (lambda answer: answer["answers"]["goal"].update(choice="back_off_enemy_1"),
                       lambda answer: answer["answers"]["goal"].update(target_id="invented")):
            with self.subTest(mutate=mutate), self.assertRaises(JevResponseError):
                call(state(), {"goal": "back_off_enemy_0", "fire": "none"}, mutate)
        for target in (None, "", "bad\nid", 17):
            with self.subTest(target=target), self.assertRaises(ValueError):
                GoalDecision("back_off", target, 0, "offline", {})

    def test_legacy_goal_request_and_parser_do_not_offer_back_off(self):
        request = build_goal_request(state())
        self.assertEqual(set(request["questions"]["goal"]["criteria"]),
                         {"hold", "evade", "enemy_0"})
        with self.assertRaises(JevResponseError):
            parse_goal_response(encoded(response("back_off_enemy_0")), {"enemy_0": "target-A"})


class BackOffControllerTests(unittest.TestCase):
    def test_controller_accepts_bound_goal_and_forwards_independent_fire(self):
        observed = combat()
        seen = []
        def execute(data, kind, target=None, *, fire_direction):
            if kind == "back_off":
                seen.append((kind, target, fire_direction))
                return Action("left", fire_direction)
            return Action("none", fire_direction)
        with patch("jev_isaac.navigation.compute_action", side_effect=execute):
            transport, requests, _, result = run(
                [(index / 10, frame(observed, 30 + index * 3), OLD) for index in range(6)],
                {"goal": "back_off_enemy_0", "fire": "up"}, duration=.6)
        self.assertTrue(requests)
        self.assertGreater(result["goal_updates"], 0)
        self.assertEqual(result["errors"], 0)
        self.assertEqual(set(seen), {("back_off", "target", "up")})
        self.assertTrue(any((packet["move"], packet["shoot"]) == ("left", "up")
                            for _, packet, _ in transport.sent))
        self.assertEqual(result["player_decisions"][0]["kind"], "back_off")

    def test_inflight_back_off_is_discarded_after_pause_and_new_arm(self):
        observed = combat()
        events = [(0, frame(observed, 30), OLD), (.1, frame(observed, 33, paused=True), OLD),
                  (.2, frame(observed, 36, enabled=False), OLD),
                  (.25, frame(observed, 37, session="fresh-arm"), NEW)]
        with patch("jev_isaac.navigation.compute_action", return_value=Action("none", "none")) as execute:
            transport, requests, _, result = run(events,
                {"goal": "back_off_enemy_0", "fire": "up"}, delay=.3, duration=.48)
        self.assertEqual(len(requests), 1)
        self.assertGreaterEqual(result["stale_discarded"], 1)
        self.assertEqual(result["goal_updates"], 0)
        self.assertEqual(result["errors"], 0)
        self.assertTrue(all(call.args[1] == "hold" for call in execute.call_args_list))
        self.assertTrue(all((packet["move"], packet["shoot"]) == ("none", "none")
                            for _, packet, _ in transport.sent))


if __name__ == "__main__":
    unittest.main()
