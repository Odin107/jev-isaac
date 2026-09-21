"""Real joint wire choices and observation feedback through the controller."""
import copy
import json
import unittest

from test_goals import state, enemy
from test_player_policy import call
from test_player_controller import combat, frame, run, OLD, NEW
from jev_isaac.jev import JevResponseError, MAX_RESPONSE_BYTES


class JointCombatTests(unittest.TestCase):
    def test_every_target_maneuver_and_fire_combination_is_available(self):
        data = state([enemy(f"target-{i:02}") for i in range(64)])
        decision, request = call(data, {"combat_group": "group_2",
            "combat_0": "enemy_0__down", "combat_1": "enemy_24__up",
            "combat_2": "back_off_enemy_63__left"})
        self.assertEqual((decision.kind, decision.target_id, decision.fire_direction),
                         ("back_off", "target-63", "left"))
        options = set().union(*(q["criteria"] for k, q in request["questions"].items()
                                if k != "combat_group"))
        expected = {f"{goal}__{fire}" for goal in (
            ["hold", "evade"] + [f"enemy_{i}" for i in range(64)]
            + [f"back_off_enemy_{i}" for i in range(64)])
            for fire in ("none", "left", "right", "up", "down")}
        self.assertEqual(options, expected)
        self.assertTrue(all(len(q["criteria"]) <= 255 for q in request["questions"].values()))
        self.assertIsNone(decision.fire_judgment)  # Joint probabilities are not fire-only probabilities.
        self.assertEqual(decision.combat_judgment["group_selection"]["effective_choice"], "group_2")
        from test_player_policy import reply
        self.assertLess(len(json.dumps(reply(request, {})).encode()), MAX_RESPONSE_BYTES)

    def test_single_question_boundary_and_no_target_case(self):
        for count in (0, 1, 24, 25):
            with self.subTest(count=count):
                _, request = call(state([enemy(f"t{i:02}") for i in range(count)]), {})
                if count <= 24:
                    self.assertEqual(set(request["questions"]), {"combat"})
                    self.assertEqual(len(request["questions"]["combat"]["criteria"]), (2*count+2)*5)
                else:
                    self.assertEqual(set(request["questions"]), {"combat_group", "combat_0", "combat_1"})

    def test_joint_argmax_correction_changes_all_components_together(self):
        def mutate(answer):
            a = answer["answers"]["combat"]
            probabilities = {k: 0. for k in a["probabilities"]}
            probabilities.update(enemy_0__left=.2, back_off_enemy_1__down=.8)
            a.update(choice="enemy_0__left", confidence=.6, probabilities=probabilities)
        decision, _ = call(state([enemy("a"), enemy("b")]),
                           {"combat": "enemy_0__left"}, mutate)
        self.assertEqual((decision.kind, decision.target_id, decision.fire_direction), ("back_off", "b", "down"))
        self.assertEqual(decision.combat_judgment["reported_choice"], "enemy_0__left")
        self.assertEqual(decision.choice_corrections[0]["effective_choice"], "back_off_enemy_1__down")

    def test_group_correction_routes_the_matching_complete_action(self):
        def mutate(answer):
            answer["answers"]["combat_group"].update(choice="group_0", confidence=.5,
                probabilities={"group_0": .2, "group_1": .8})
        decision, _ = call(state([enemy(f"t{i:02}") for i in range(25)]),
            {"combat_group": "group_0", "combat_0": "enemy_0__right",
             "combat_1": "back_off_enemy_24__up"}, mutate)
        self.assertEqual((decision.kind, decision.target_id, decision.fire_direction), ("back_off", "t24", "up"))

    def test_joint_choice_cannot_rebind_to_mutated_observation(self):
        data = state([enemy("a"), enemy("b")])
        decision, request = call(data, {"combat": "enemy_1__down"},
            lambda _: data["enemies"][1].update(id="c"))
        self.assertEqual(decision.target_id, "b")
        self.assertEqual(request["state"]["observation"]["enemies"][1]["id"], "b")

    def test_unbound_or_missing_group_answers_cannot_execute(self):
        data = state([enemy(f"t{i:02}") for i in range(25)])
        for mutate in (lambda a: a["answers"].pop("combat_1"),
                       lambda a: a["answers"]["combat_0"].update(choice="enemy_24__right"),
                       lambda a: a["answers"]["combat_group"].update(choice="group_2")):
            with self.subTest(mutate=mutate), self.assertRaises(JevResponseError):
                call(data, {}, mutate)


class CombatFeedbackIntegrationTests(unittest.TestCase):
    def test_history_and_execution_feedback_reaches_next_decision_and_report(self):
        data = combat()
        data["control"] = {"shoot": "right", "applied_move": "left", "requested_move": "left"}
        events = []
        for index in range(67):
            observed = frame(data, 100+index)
            observed["player"].update(x=300+index, hearts=6 if index < 30 else 5)
            observed["enemies"][0]["hp"] = 10 if index < 30 else 7
            observed["control"]["source_frame"] = 100+index-1
            events.append((index/30, observed, OLD))
        _, requests, _, result = run(events, {"combat": "hold__up"}, duration=2.21)
        self.assertEqual(result["errors"], 0)
        context = requests[-1]["state"]["controller_context"]
        feedback = context["combat_feedback"]
        self.assertEqual(feedback["status"], "observed")
        self.assertEqual(feedback["enemies"]["reported"][0]["decrease_observed"], 3)
        self.assertEqual(feedback["player"]["health_half_hearts"]["hearts"]["decrease_observed"], 1)
        self.assertGreater(feedback["player"]["movement"]["net_displacement"]["distance"], 0)
        self.assertEqual(feedback["inputs"]["latest"]["shoot"], "right")  # Echo, not invented from the chosen 'up'.
        self.assertEqual(context["last_local_command"]["shoot"], "up")
        self.assertEqual(context["previous_goal"]["fire_direction"], "up")
        self.assertEqual(context["decision_timing"]["request_rate_limit_hz"], 2)
        self.assertEqual(result["player_decisions"][-1]["combat_feedback_at_request"], feedback)

    def test_new_arm_does_not_inherit_previous_health_or_input_history(self):
        data = combat()
        after = copy.deepcopy(data)
        after["player"]["hearts"] = 1
        after["enemies"][0]["hp"] = 1
        events = [(0, frame(data, 30), OLD), (.3, frame(data, 39), OLD),
                  (.6, frame(data, 48, enabled=False), OLD),
                  (.8, frame(after, 50, session="new-arm"), NEW),
                  (1.1, frame(after, 59, session="new-arm"), NEW),
                  (1.4, frame(after, 68, session="new-arm"), NEW)]
        _, requests, _, result = run(events, {"combat": "hold__none"}, duration=1.45)
        self.assertEqual(result["errors"], 0)
        refreshed = [r["state"]["controller_context"]["combat_feedback"] for r in requests
                     if r["state"]["observation"]["frame"] >= 50]
        self.assertTrue(refreshed)
        for feedback in refreshed:
            self.assertIn(feedback["status"], ("insufficient_history", "observed"))
            self.assertGreaterEqual(feedback["window"]["start_frame"], 50)
            self.assertIn(feedback["enemies"]["reported"][0]["decrease_observed"], (None, 0))


if __name__ == "__main__":
    unittest.main()
