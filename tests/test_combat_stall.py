"""No-target live combat cannot silently spend an entire attempt holding."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_isaac.combat_stall import CombatStallWatchdog, NO_COMBAT_OBJECTIVE, no_living_enemies
from jev_isaac.demo import sample


def state(frame=30, **changes):
    observed = sample(frame)
    observed.update(enemies=[], run_id="run", room_id="uncleared", session="arm1",
                    floor={"id": "floor1", "dimension": 0})
    observed.update(changes)
    return observed


class CombatStallTests(unittest.TestCase):
    def test_no_enemy_predicate_requires_complete_metadata_but_ignores_eligibility(self):
        self.assertTrue(no_living_enemies(state()))
        self.assertTrue(no_living_enemies(state(enabled=False, paused=True)))
        for mutate in (
                lambda s: s.pop("enemies"), lambda s: s.pop("hazards"), lambda s: s.pop("projectiles"),
                lambda s: s.pop("truncated"), lambda s: s.update(truncated=True),
                lambda s: s.update(truncated_arrays={"enemies": 0}),
                lambda s: s.update(truncated_arrays={"enemies": True}),
                lambda s: s.update(enemies=[{"x": 1, "y": 1, "hp": None}]),
                lambda s: s.update(enemies=[{"x": 1, "y": 1, "hp": 0, "dead": 1}]),
                lambda s: s.update(enemies=[{"hp": 0}]),
                lambda s: s.update(hazards=[{"x": float("nan"), "y": 1}])):
            value = state()
            mutate(value)
            self.assertFalse(no_living_enemies(value))
        self.assertFalse(no_living_enemies(state(enemies=[{"x": 1, "y": 1, "hp": 1, "vulnerable": False}])))
        self.assertTrue(no_living_enemies(state(enemies=[{"x": 1, "y": 1, "hp": 0}])))

    def test_frame_only_changes_do_not_extend_five_second_grace(self):
        watch = CombatStallWatchdog()
        for second in range(5):
            self.assertIsNone(watch.observe(state(30+second*30), second))
        result = watch.observe(state(180), 5)
        self.assertEqual(result["reason"], NO_COMBAT_OBJECTIVE)
        self.assertEqual(result["timeout_kind"], "no_progress")
        self.assertEqual(result["fresh_observations"], 6)
        self.assertFalse(result["room_clear"])
        self.assertIsNone(watch.observe(state(210), 6))

    def test_actual_switch_room_stall_reproduces_without_declaring_clear(self):
        path = Path(__file__).parent / "fixtures" / "combat-switch-stall.json"
        observed = json.loads(path.read_text())
        observed.update(enabled=True, paused=False)
        original = copy.deepcopy(observed)
        watch = CombatStallWatchdog()
        self.assertIsNone(watch.observe(observed, 0))
        observed["frame"] += 150
        result = watch.observe(observed, 5)
        self.assertEqual(result["reason"], NO_COMBAT_OBJECTIVE)
        self.assertEqual(result["living_enemy_count"], 0)
        self.assertEqual(result["player"]["x"], 80)
        self.assertFalse(observed["room"]["clear"])
        self.assertEqual(observed["hazards"], original["hazards"])

    def test_supported_local_switch_objective_bypasses_and_resets_guard(self):
        watch = CombatStallWatchdog()
        watch.observe(state(30), 0)
        self.assertIsNone(watch.observe(state(180), 5, local_objective=True))
        self.assertIsNone(watch.observe(state(330), 10, local_objective=True))
        self.assertIsNone(watch.observe(state(480), 15))
        self.assertIsNotNone(watch.observe(state(630), 20))

    def test_living_invulnerable_enemy_never_counts_as_empty_room(self):
        for vulnerable in (True, False, None):
            watch = CombatStallWatchdog()
            watch.observe(state(), 0)
            value = state(180, enemies=[{"id": "enemy", "hp": 10, "x": 400, "y": 280,
                                         "vulnerable": vulnerable}])
            self.assertIsNone(watch.observe(value, 5))
            value["frame"] += 1000
            self.assertIsNone(watch.observe(value, 40))
            self.assertIsNone(watch.observe(state(1500), 41))

    def test_short_spawn_gap_does_not_report_and_real_enemy_resets_grace(self):
        watch = CombatStallWatchdog()
        self.assertIsNone(watch.observe(state(), 0))
        self.assertIsNone(watch.observe(state(150), 4))
        self.assertIsNone(watch.observe(state(180, enemies=[{"hp": 1, "x": 400, "y": 280}]), 5))
        self.assertIsNone(watch.observe(state(210), 6))
        self.assertIsNone(watch.observe(state(330), 10))
        self.assertIsNotNone(watch.observe(state(360), 11))

    def test_small_motion_and_control_echoes_do_not_count_as_progress(self):
        watch = CombatStallWatchdog()
        value = state()
        start_x = value["player"]["x"]
        watch.observe(value, 0)
        for second in range(1, 6):
            value["frame"] += 30
            value["player"]["x"] = start_x + (2 if second % 2 else -2)
            value["control"] = {"source_frame": value["frame"]-1}
            result = watch.observe(value, second)
        self.assertIsNotNone(result)

    def test_meaningful_pose_change_gets_grace_but_targetless_motion_remains_bounded(self):
        watch = CombatStallWatchdog()
        value = state()
        watch.observe(value, 0)
        for second in (4, 8):
            value["frame"] += 120
            value["player"]["x"] += 8
            self.assertIsNone(watch.observe(value, second))
        value["frame"] += 120
        value["player"]["x"] += 8
        result = watch.observe(value, 12)
        self.assertEqual(result["timeout_kind"], "episode")
        self.assertEqual(result["no_progress_seconds"], 0)

    def test_duplicate_or_reordered_frames_do_not_advance_guard(self):
        watch = CombatStallWatchdog()
        watch.observe(state(30), 0)
        self.assertIsNone(watch.observe(state(30), 5))
        self.assertIsNone(watch.observe(state(29), 6))
        self.assertIsNotNone(watch.observe(state(31), 7))

    def test_inactive_incomplete_and_malformed_states_reset_grace(self):
        for mutate in (
                lambda s: s.update(enabled=False), lambda s: s.update(paused=True),
                lambda s: s["player"].update(dead=True), lambda s: s["room"].update(clear=True),
                lambda s: s.update(truncated=True), lambda s: s.update(truncated_arrays={"hazards": True}),
                lambda s: s.update(enemies=None), lambda s: s.update(enemies=[{"hp": None}]),
                lambda s: s["player"].update(x=float("nan"))):
            watch = CombatStallWatchdog()
            watch.observe(state(), 0)
            value = state(180)
            mutate(value)
            self.assertIsNone(watch.observe(value, 5))
            self.assertIsNone(watch.observe(state(210), 6))
            self.assertIsNotNone(watch.observe(state(360), 11))

    def test_room_run_floor_dimension_and_session_changes_start_new_grace(self):
        for mutate in (
                lambda s: s.update(room_id="different"), lambda s: s.update(run_id="different"),
                lambda s: s.update(session="arm2"), lambda s: s["floor"].update(id="different"),
                lambda s: s["floor"].update(dimension=1)):
            watch = CombatStallWatchdog()
            watch.observe(state(), 0)
            value = state(180)
            mutate(value)
            self.assertIsNone(watch.observe(value, 5))
            value["frame"] += 150
            self.assertIsNotNone(watch.observe(value, 10))

    def test_explicit_reset_rearms_one_shot_diagnostic(self):
        watch = CombatStallWatchdog()
        watch.observe(state(), 0)
        self.assertIsNotNone(watch.observe(state(180), 5))
        watch.reset()
        self.assertIsNone(watch.observe(state(181), 6))
        self.assertIsNotNone(watch.observe(state(331), 11))


if __name__ == "__main__":
    unittest.main()
