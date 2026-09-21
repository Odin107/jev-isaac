"""Chosen retreats preserve range, clear movement and independent firing."""
import copy
import math
import unittest

from test_navigation import enemy, grid, state
from jev_isaac.navigation import _VECTORS, _clear, compute_action


class BackOffNavigationTests(unittest.TestCase):
    def test_aligned_target_can_be_given_more_space_without_changing_fire(self):
        observed = state(300, 280)
        observed["enemies"] = [enemy(400, 280)]
        engage = compute_action(observed, "engage", "target", fire_direction="right")
        retreat = compute_action(observed, "back_off", "target", fire_direction="right")
        self.assertEqual((engage.move, engage.shoot), ("none", "right"))
        self.assertEqual((retreat.move, retreat.shoot), ("left", "right"))
        self.assertIsNone(retreat.override)

    def test_retreat_works_on_all_four_sides_of_the_selected_target(self):
        for point, expected in (((400, 280), "left"), ((200, 280), "right"),
                                ((300, 180), "down"), ((300, 380), "up")):
            with self.subTest(target=point):
                observed = state(300, 280)
                observed["enemies"] = [enemy(*point)]
                result = compute_action(observed, "back_off", "target", fire_direction="none")
                self.assertEqual(result.move, expected)
                self.assertEqual(result.shoot, "none")
                self.assertIsNone(result.override)

    def test_retreat_uses_selected_target_instead_of_another_enemy(self):
        observed = state(300, 280)
        observed["enemies"] = [enemy(180, 280, "other"), enemy(400, 280)]
        self.assertEqual(compute_action(observed, "back_off", "target",
                                        fire_direction="right").move, "left")
        self.assertEqual(compute_action(observed, "back_off", "other",
                                        fire_direction="left").move, "right")

    def test_diagonal_target_allows_a_farther_firing_alignment(self):
        observed = state(300, 280)
        observed["enemies"] = [enemy(380, 250)]
        result = compute_action(observed, "back_off", "target", fire_direction="right")
        self.assertEqual(result.move, "up_left")
        vector = _VECTORS[result.move]
        after = (300 + vector[0] * 12, 280 + vector[1] * 12)
        self.assertGreater(math.dist(after, (380, 250)), math.dist((300, 280), (380, 250)))
        self.assertLess(abs(after[1] - 250), 30)

    def test_blocked_retreat_holds_instead_of_routing_toward_target(self):
        observed = state(300, 280)
        observed["enemies"] = [enemy(400, 280)]
        observed["hazards"] = [grid(260, 280)]
        result = compute_action(observed, "back_off", "target", fire_direction="right")
        self.assertEqual((result.move, result.shoot), ("none", "right"))
        self.assertIsNone(result.override)

    def test_blocked_firing_lane_does_not_trigger_an_inward_detour(self):
        observed = state(300, 280)
        observed["enemies"] = [enemy(400, 280)]
        observed["hazards"] = [grid(350, 280)]
        result = compute_action(observed, "back_off", "target", fire_direction="none")
        self.assertEqual((result.move, result.shoot), ("none", "none"))

    def test_pit_in_firing_lane_still_allows_a_clear_retreat(self):
        observed = state(300, 280)
        observed["enemies"] = [enemy(400, 280)]
        observed["hazards"] = [grid(350, 280, collision=1)]
        result = compute_action(observed, "back_off", "target", fire_direction="right")
        self.assertEqual((result.move, result.shoot), ("left", "right"))

    def test_shorter_retreat_is_possible_near_room_boundary(self):
        observed = state(90, 280)
        observed["enemies"] = [enemy(200, 280)]
        result = compute_action(observed, "back_off", "target", fire_direction="right")
        self.assertEqual(result.move, "left")
        observed["player"]["x"] = 78
        result = compute_action(observed, "back_off", "target", fire_direction="right")
        self.assertEqual(result.move, "none")

    def test_beyond_range_holds_but_imminent_danger_can_still_dodge(self):
        observed = state(300, 280)
        observed["enemies"] = [enemy(550, 280)]
        result = compute_action(observed, "back_off", "target", fire_direction="up")
        self.assertEqual((result.move, result.shoot), ("none", "up"))
        observed["projectiles"] = [{"x": 340, "y": 280, "vx": -6, "vy": 0, "radius": 5}]
        result = compute_action(observed, "back_off", "target", fire_direction="up")
        self.assertNotEqual(result.move, "none")
        self.assertEqual(result.shoot, "up")
        self.assertEqual(result.override, "emergency collision avoidance")

    def test_unavailable_or_ambiguous_target_does_not_pursue_another_enemy(self):
        for condition in ("missing", "zero_hp", "dead", "invulnerable", "duplicate"):
            with self.subTest(condition=condition):
                observed = state(300, 280)
                observed["enemies"] = [enemy(480, 350, "other")]
                if condition != "missing":
                    target = enemy(400, 280)
                    if condition == "zero_hp":
                        target["hp"] = 0
                    elif condition == "dead":
                        target["dead"] = True
                    elif condition == "invulnerable":
                        target["vulnerable"] = False
                    observed["enemies"].append(target)
                    if condition == "duplicate":
                        observed["enemies"].append(enemy(200, 280))
                result = compute_action(observed, "back_off", "target", fire_direction="left")
                self.assertEqual((result.move, result.shoot), ("none", "left"))
                self.assertIsNone(result.override)

    def test_jev_fire_is_preserved_even_when_it_is_not_toward_the_target(self):
        observed = state(300, 280)
        observed["enemies"] = [enemy(400, 280)]
        for fire in ("none", "left", "right", "up", "down"):
            with self.subTest(fire=fire):
                result = compute_action(observed, "back_off", "target", fire_direction=fire)
                self.assertEqual((result.move, result.shoot), ("left", fire))

    def test_inactive_or_incomplete_observation_is_neutral(self):
        changes = [lambda s: s.update(enabled=False), lambda s: s.update(paused=True),
                   lambda s: s["player"].update(dead=True),
                   lambda s: s["room"].update(clear=True), lambda s: s.update(truncated=True)]
        changes.extend(lambda s, field=field: s.update(truncated_arrays={field: True})
                       for field in ("enemies", "projectiles", "hazards"))
        for change in changes:
            observed = state(300, 280)
            observed["enemies"] = [enemy(400, 280)]
            change(observed)
            result = compute_action(observed, "back_off", "target", fire_direction="right")
            self.assertEqual((result.move, result.shoot), ("none", "none"))

    def test_repeated_retreat_is_bounded_and_never_approaches_target(self):
        for tear_range, maximum in ((None, 220), (150, 130)):
            with self.subTest(tear_range=tear_range):
                observed = state(300, 280)
                observed["enemies"] = [enemy(400, 280)]
                observed["player"]["weapon_type"] = 1
                if tear_range is not None:
                    observed["player"]["tear_range"] = tear_range
                distances = [100.0]
                for _ in range(80):
                    before = copy.deepcopy(observed)
                    result = compute_action(observed, "back_off", "target", fire_direction="right")
                    self.assertEqual(observed, before)
                    self.assertEqual(result.shoot, "right")
                    self.assertIsNone(result.override)
                    if result.move == "none":
                        break
                    start = observed["player"]["x"], observed["player"]["y"]
                    vector = _VECTORS[result.move]
                    end = start[0] + vector[0] * 4, start[1] + vector[1] * 4
                    self.assertTrue(all(math.isfinite(value) for value in end))
                    self.assertTrue(_clear(start, end, []))
                    self.assertTrue(70 <= end[0] <= 570 and 150 <= end[1] <= 410)
                    distance = math.dist(end, (400, 280))
                    self.assertGreater(distance, distances[-1])
                    self.assertLessEqual(distance, maximum)
                    distances.append(distance)
                    observed["player"].update(x=end[0], y=end[1])
                else:
                    self.fail("Retreat did not stop within its range envelope")
                self.assertGreater(len(distances), 1)
                self.assertGreaterEqual(distances[-1], maximum - 12)

    def test_short_observed_ordinary_tear_range_prevents_more_separation(self):
        observed = state(300, 280)
        observed["enemies"] = [enemy(400, 280)]
        observed["player"].update(weapon_type=1, tear_range=110)
        result = compute_action(observed, "back_off", "target", fire_direction="right")
        self.assertEqual((result.move, result.shoot), ("none", "right"))

    def test_special_weapon_does_not_claim_ordinary_tear_range(self):
        observed = state(300, 280)
        observed["enemies"] = [enemy(400, 280)]
        observed["player"].update(weapon_type=2, tear_range=110)
        result = compute_action(observed, "back_off", "target", fire_direction="right")
        self.assertEqual((result.move, result.shoot), ("left", "right"))


if __name__ == "__main__":
    unittest.main()
