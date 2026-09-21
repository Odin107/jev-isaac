"""Observed combat outcomes remain bounded, detached, and epistemically narrow."""
import json
import unittest

from jev_isaac.combat_feedback import CombatFeedback


def observation(frame=0, hp=10, **updates):
    state = {"session": "session", "run_id": "run", "room_id": "room", "frame": frame,
             "floor": {"id": "floor", "dimension": 0, "room_index": 1}, "enabled": True, "paused": False,
             "player": {"x": 100, "y": 100, "hearts": 6, "soul_hearts": 2, "dead": False},
             "enemies": [{"id": "enemy", "hp": hp}],
             "control": {"requested_move": "right", "applied_move": "right", "shoot": "left",
                         "source_frame": frame, "move_stop_reason": "none"}}
    state.update(updates)
    return state


class CombatFeedbackTests(unittest.TestCase):
    def feedback(self, *states, **kwargs):
        tracker = CombatFeedback(**kwargs)
        for state in states:
            tracker.observe(state)
        return tracker.context(states[-1])

    def test_health_decrease_and_recovery_are_separate_observed_changes(self):
        first, middle, final = observation(), observation(15, 7), observation(30, 9)
        middle["player"].update(hearts=4, soul_hearts=3)
        final["player"].update(hearts=5, soul_hearts=1)
        context = self.feedback(first, middle, final)
        enemy = context["enemies"]["reported"][0]
        self.assertEqual((enemy["decrease_observed"], enemy["increase_observed"]), (3, 2))
        self.assertEqual(enemy["status"], "decreased_and_increased")
        red = context["player"]["health_half_hearts"]["hearts"]
        soul = context["player"]["health_half_hearts"]["soul_hearts"]
        self.assertEqual((red["decrease_observed"], red["increase_observed"]), (2, 1))
        self.assertEqual((soul["decrease_observed"], soul["increase_observed"]), (2, 1))
        self.assertIn("not necessarily Isaac", context["limits"])
        self.assertNotIn("hits", enemy)
        self.assertNotIn("kills", context)

    def test_input_held_at_samples_and_zero_displacement_are_both_visible(self):
        context = self.feedback(observation(), observation(10), observation(20))
        movement = context["player"]["movement"]
        self.assertEqual(movement["net_displacement"], {"dx": 0, "dy": 0, "distance": 0})
        self.assertEqual(movement["moving_input_at_both_endpoints"],
                         {"intervals": 2, "span_frames_sum": 20, "sampled_distance": 0})
        self.assertEqual(context["inputs"]["latest_sampled_streak"]["shoot"],
                         {"value": "left", "span_frames": 20, "samples": 3})
        self.assertEqual(context["enemies"]["reported"][0]["status"], "unchanged_at_samples")

    def test_return_to_start_is_not_stationary_and_release_can_have_momentum(self):
        first, middle, final = observation(), observation(10), observation(20)
        middle["player"]["x"] = 120
        for state in (first, middle, final):
            state["control"]["applied_move"] = "none"
        context = self.feedback(first, middle, final)
        movement = context["player"]["movement"]
        self.assertEqual(movement["net_displacement"]["distance"], 0)
        self.assertEqual(movement["sampled_path_distance_lower_bound"], 40)
        self.assertEqual(movement["moving_input_at_both_endpoints"]["intervals"], 0)

    def test_requested_movement_and_actual_pulse_stop_remain_distinct(self):
        first, second = observation(), observation(5)
        second["control"].update(applied_move="none", move_stop_reason="distance")
        context = self.feedback(first, second)
        self.assertEqual(context["inputs"]["latest"]["requested_move"], "right")
        self.assertEqual(context["inputs"]["latest"]["applied_move"], "none")
        self.assertEqual(context["inputs"]["latest"]["move_stop_reason"], "distance")
        self.assertEqual(context["player"]["movement"]["moving_input_at_both_endpoints"]["intervals"], 0)

    def test_unknown_controls_break_sampled_streaks(self):
        context = self.feedback(observation(), observation(4, control=None), observation(8))
        self.assertEqual(context["inputs"]["sample_counts"]["shoot"], {"left": 2, "unknown": 1})
        self.assertEqual(context["inputs"]["latest_sampled_streak"]["shoot"]["span_frames"], 0)
        self.assertEqual(context["player"]["movement"]["moving_input_at_both_endpoints"]["intervals"], 0)

    def test_missing_hp_absence_and_duplicate_ids_break_comparison(self):
        for rows in ([{"id": "enemy"}], [], [{"id": "enemy", "hp": 8}, {"id": "enemy", "hp": 2}]):
            with self.subTest(rows=rows):
                context = self.feedback(observation(), observation(5, enemies=rows), observation(10, 1))
                enemy = context["enemies"]["reported"][0]
                self.assertEqual(enemy["status"], "not_comparable")
                self.assertIsNone(enemy["decrease_observed"])

    def test_truncated_or_missing_exports_do_not_report_disappearance_as_damage(self):
        for extra in ({"enemies": [], "truncated_arrays": {"enemies": True}}, {"enemies": None}):
            with self.subTest(extra=extra):
                context = self.feedback(observation(), observation(10, **extra))
                enemy = context["enemies"]["reported"][0]
                self.assertFalse(enemy["present_in_latest_export"])
                self.assertIsNone(enemy["latest_hp"])
                self.assertIsNone(enemy["decrease_observed"])
                self.assertEqual(context["enemies"]["latest_coverage"], "missing" if extra["enemies"] is None else "partial")
        context = self.feedback(observation(), observation(10, 7, truncated_arrays={"enemies": True}))
        self.assertEqual(context["enemies"]["reported"][0]["decrease_observed"], 3)

    def test_room_run_floor_and_dimension_changes_reset_history(self):
        edits = [("session", "next"), ("run_id", "next"), ("room_id", "next"),
                 ("floor.id", "next"), ("floor.dimension", 1), ("floor.room_index", 2)]
        for field, value in edits:
            with self.subTest(field=field):
                first, second = observation(), observation(10, 2)
                if field.startswith("floor."):
                    second["floor"][field.split(".")[1]] = value
                else:
                    second[field] = value
                context = self.feedback(first, second)
                self.assertEqual(context["status"], "insufficient_history")
                self.assertEqual(context["window"]["sample_count"], 1)
                self.assertIsNone(context["enemies"]["reported"][0]["decrease_observed"])

    def test_pause_disable_death_and_explicit_reset_do_not_bridge_outcomes(self):
        for field in ("paused", "enabled", "dead", "reset"):
            with self.subTest(field=field):
                tracker = CombatFeedback()
                tracker.observe(observation())
                inactive = observation(5, 7)
                if field == "dead":
                    inactive["player"]["dead"] = True
                elif field != "reset":
                    inactive[field] = field == "paused"
                if field == "reset":
                    tracker.reset()
                else:
                    tracker.observe(inactive)
                self.assertEqual(tracker.context(inactive)["status"], "no_matching_observation_history")
                resumed = observation(10, 2)
                tracker.observe(resumed)
                self.assertEqual(tracker.context(resumed)["window"]["sample_count"], 1)

    def test_duplicate_frames_do_not_invent_time_and_regressed_clock_resets(self):
        tracker = CombatFeedback()
        tracker.observe(observation(10, 10))
        tracker.observe(observation(10, 1))
        last = observation(11, 8)
        tracker.observe(last)
        self.assertEqual(tracker.context(last)["enemies"]["reported"][0]["decrease_observed"], 2)
        last = observation(5, 1)
        tracker.observe(last)
        self.assertEqual(tracker.context(last)["window"]["sample_count"], 1)

    def test_rolling_window_evicts_old_outcomes_and_is_memory_bounded(self):
        tracker = CombatFeedback(window_frames=60)
        for frame in range(100):
            last = observation(frame, 10 if frame == 0 else 1)
            tracker.observe(last)
        context = tracker.context(last)
        self.assertEqual(context["window"]["sample_count"], 61)
        self.assertEqual(context["window"]["start_frame"], 39)
        self.assertEqual(context["enemies"]["reported"][0]["decrease_observed"], 0)
        last = observation(500)
        tracker.observe(last)
        self.assertEqual(tracker.context(last)["window"]["sample_count"], 1)

    def test_context_requires_latest_identity_and_frame_without_mutating_history(self):
        tracker = CombatFeedback()
        last = observation(10)
        tracker.observe(last)
        for other in (observation(9), observation(11), observation(10, room_id="other")):
            self.assertEqual(tracker.context(other), {"status": "no_matching_observation_history"})
        self.assertEqual(tracker.context(last)["window"]["sample_count"], 1)

    def test_input_and_output_mutation_cannot_change_recorded_facts(self):
        tracker = CombatFeedback()
        first, last = observation(), observation(10, 7)
        tracker.observe(first)
        first["enemies"][0]["hp"] = 1000
        first["control"]["shoot"] = "down"
        tracker.observe(last)
        expected = tracker.context(last)
        mutated = tracker.context(last)
        mutated["inputs"]["latest"]["shoot"] = "down"
        mutated["enemies"]["reported"][0]["decrease_observed"] = 999
        last["enemies"][0]["hp"] = 0
        self.assertEqual(tracker.context(last), expected)

    def test_invalid_values_stay_unknown_and_result_remains_finite_json(self):
        first, last = observation(), observation(10, float("nan"))
        last["player"].update(x=float("inf"), hearts=True, soul_hearts=10**1000)
        last["control"].update(shoot=[], source_frame=100, applied_move=False)
        context = self.feedback(first, last)
        self.assertIsNone(context["player"]["movement"]["net_displacement"])
        self.assertIsNone(context["player"]["health_half_hearts"]["hearts"]["latest"])
        self.assertIsNone(context["enemies"]["reported"][0]["decrease_observed"])
        self.assertNotIn("shoot", context["inputs"]["latest"])
        self.assertNotIn("source_frame", context["inputs"]["latest"])
        json.dumps(context, allow_nan=False)

    def test_enemy_rows_are_bounded_and_prioritize_current_ids(self):
        first = observation(enemies=[{"id": "gone", "hp": 10}])
        last = observation(10, enemies=[{"id": f"e{i:03}", "hp": 8} for i in range(300)])
        context = self.feedback(first, last, enemy_limit=2)
        self.assertEqual([row["id"] for row in context["enemies"]["reported"]], ["e000", "e001"])
        self.assertEqual(context["enemies"]["omitted_count"], 255)
        self.assertEqual(context["enemies"]["latest_coverage"], "partial")


if __name__ == "__main__":
    unittest.main()
