"""Local steering scenarios. No API calls or game controls are used."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_isaac.navigation import compute_action, _VECTORS, _clear, MAX_NODES


def state(x=300, y=250):
    return {"enabled": True, "paused": False, "truncated": False,
            "player": {"x": x, "y": y, "vx": 0, "vy": 0, "radius": 10, "dead": False},
            "room": {"top_left": {"x": 60, "y": 140},
                     "bottom_right": {"x": 580, "y": 420}, "clear": False},
            "enemies": [], "projectiles": [], "hazards": []}


def enemy(x, y, ident="target"):
    return {"id": ident, "x": x, "y": y, "vx": 0, "vy": 0,
            "radius": 13, "hp": 10, "vulnerable": True}


def grid(x, y, collision=3):
    return {"kind": "grid", "x": x, "y": y, "collision": collision, "radius": 20}


def saved(name):
    fixture = {"third": "combat-corner.json", "fifth": "combat-alignment.json"}[name]
    return json.loads((Path(__file__).parent / "fixtures" / fixture).read_text())


class NavigationTests(unittest.TestCase):
    def test_saved_fifth_alignment_stops_and_shoots(self):
        observed = saved("fifth")
        target = observed["enemies"][1]["id"]
        result = compute_action(observed, "engage", target)
        self.assertEqual((result.move, result.shoot), ("left", "none"))
        observed["player"].update(x=520, vx=0)
        result = compute_action(observed, "engage", target)
        self.assertEqual((result.move, result.shoot), ("none", "down"))

    def test_saved_third_near_rock_routes_to_clear_firing_lane(self):
        observed = saved("third")
        observed["player"].update(x=190, y=190, vx=0, vy=0)
        target = observed["enemies"][0]["id"]
        boxes = [(g["x"]-30, g["y"]-30, g["x"]+30, g["y"]+30)
                 for g in observed["hazards"] if g["kind"] == "grid" and g["collision"]]
        initial = compute_action(observed, "engage", target)
        self.assertIn(initial.move, ("down", "down_left"))
        # Replan every small step; assert every simulated segment stays clear.
        for _ in range(180):
            action = compute_action(observed, "engage", target)
            if action.shoot != "none" and action.move == "none":
                break
            self.assertNotEqual(action.move, "none", "Navigator got stuck before reaching a firing lane")
            vector = _VECTORS[action.move]
            start = observed["player"]["x"], observed["player"]["y"]
            end = start[0]+vector[0]*4, start[1]+vector[1]*4
            self.assertTrue(_clear(start, end, boxes))
            observed["player"].update(x=end[0], y=end[1])
        else:
            self.fail("No firing position found")
        self.assertEqual(action.shoot, "up")
        self.assertGreaterEqual(observed["player"]["y"], 220)

    def test_rock_blocks_shooting_and_requires_route(self):
        observed = state(160, 280)
        observed["enemies"] = [enemy(360, 280)]
        observed["hazards"] = [grid(240, 280)]
        result = compute_action(observed, "engage", "target")
        self.assertEqual(result.shoot, "none")
        self.assertIn(result.move, ("up", "down", "up_right", "down_right"))

    def test_pit_blocks_walking_but_not_shots(self):
        observed = state(160, 280)
        observed["enemies"] = [enemy(360, 280)]
        observed["hazards"] = [grid(240, 280, 1)]
        result = compute_action(observed, "engage", "target")
        self.assertEqual((result.move, result.shoot), ("none", "right"))

    def test_short_drift_stops_before_alignment_overshoot(self):
        observed = state(531, 280)
        observed["player"]["vx"] = -4
        observed["enemies"] = [enemy(520, 400)]
        self.assertEqual(compute_action(observed, "engage", "target").move, "none")
        observed["player"]["vx"] = 0
        self.assertEqual(compute_action(observed, "engage", "target").move, "left")

    def test_alignment_releases_immediately_when_target_moves(self):
        observed = state(520, 280)
        observed["enemies"] = [enemy(520, 400)]
        self.assertEqual(compute_action(observed, "engage", "target").shoot, "down")
        observed["enemies"][0]["x"] = 440
        result = compute_action(observed, "engage", "target")
        self.assertEqual(result.shoot, "none")
        self.assertEqual(result.move, "left")

    def test_imminent_projectile_triggers_local_dodge(self):
        observed = state()
        observed["enemies"] = [enemy(480, 250)]
        self.assertEqual(compute_action(observed, "engage", "target").move, "none")
        observed["projectiles"] = [{"x": 340, "y": 250, "vx": -6, "vy": 0, "radius": 5}]
        result = compute_action(observed, "engage", "target")
        self.assertIn(result.move, ("up", "down", "up_left", "down_left"))
        self.assertEqual(result.shoot, "right")

    def test_contact_dodge_does_not_move_through_wall(self):
        observed = state(70, 250)
        observed["enemies"] = [enemy(95, 250)]
        observed["hazards"] = [grid(40, 240, 4), grid(40, 280, 4)]
        result = compute_action(observed, "evade")
        self.assertIn(result.move, ("up", "down", "up_right", "down_right"))
        vector = _VECTORS[result.move]
        self.assertGreaterEqual(observed["player"]["x"]+vector[0]*24, 70)

    def test_fast_crossing_projectile_is_not_missed_between_samples(self):
        observed = state()
        observed["enemies"] = [enemy(480, 250)]
        observed["projectiles"] = [{"x": 350, "y": 250, "vx": -25, "vy": 0, "radius": 5}]
        result = compute_action(observed, "engage", "target")
        self.assertIn(result.move, ("up", "down", "up_left", "down_left", "up_right", "down_right"))

    def test_small_target_deadband_still_finishes_alignment(self):
        observed = state(525, 280)
        observed["enemies"] = [enemy(520, 400)]
        observed["enemies"][0]["radius"] = 2
        self.assertEqual(compute_action(observed, "engage", "target").move, "left")

    def test_fire_and_zero_collision_spikes_are_not_walked_through(self):
        for obstacle in ({"kind": "fire", "x": 240, "y": 280, "radius": 20},
                         dict(grid(240, 280, 0), type=8)):
            observed = state(200, 280)
            observed["enemies"] = [enemy(380, 400)]
            observed["hazards"] = [obstacle]
            result = compute_action(observed, "engage", "target")
            self.assertNotEqual(result.move, "right")
            start = 200, 280
            vector = _VECTORS[result.move]
            end = start[0]+vector[0]*12, start[1]+vector[1]*12
            self.assertTrue(_clear(start, end, [(210, 250, 270, 310)]))

    def test_evade_moves_away_from_nearest_observed_threat(self):
        observed = state()
        observed["enemies"] = [enemy(340, 250)]
        result = compute_action(observed, "evade")
        self.assertEqual(result.move, "left")
        self.assertEqual(result.shoot, "none")

    def test_missing_dead_invulnerable_target_holds(self):
        observed = state()
        observed["enemies"] = [enemy(450, 250)]
        self.assertEqual(compute_action(observed, "engage", "missing").move, "none")
        self.assertEqual(compute_action(observed, "engage").move, "none")
        observed["enemies"][0]["hp"] = 0
        self.assertEqual(compute_action(observed, "engage", "target").shoot, "none")
        observed["enemies"][0].update(hp=10, vulnerable=False)
        self.assertEqual(compute_action(observed, "engage", "target").move, "none")

    def test_invalidated_target_dodges_and_fires_at_a_fresh_aligned_enemy(self):
        for target_state in ("missing", "dead", "invulnerable"):
            with self.subTest(target_state=target_state):
                observed = state()
                # Another enemy has a clear firing lane. Shoot locally without
                # replacing Jev's pursuit target or changing the dodge.
                observed["enemies"] = [enemy(480, 250, "other")]
                if target_state != "missing":
                    target = enemy(300, 400)
                    target.update(hp=0 if target_state == "dead" else 10,
                                  vulnerable=target_state != "invulnerable")
                    observed["enemies"].append(target)
                observed["projectiles"] = [
                    {"x": 340, "y": 250, "vx": -6, "vy": 0, "radius": 5}]
                result = compute_action(observed, "engage", "target")
                self.assertIn(result.move, ("up", "down", "up_left", "down_left"))
                self.assertEqual(result.shoot, "right")

    def test_invalidated_target_does_not_pursue_other_enemy_without_threat(self):
        for target_state in ("missing", "dead", "invulnerable"):
            with self.subTest(target_state=target_state):
                observed = state()
                # The other enemy requires pursuit to acquire a firing lane.
                observed["enemies"] = [enemy(500, 350, "other")]
                if target_state != "missing":
                    target = enemy(450, 250)
                    target.update(hp=0 if target_state == "dead" else 10,
                                  vulnerable=target_state != "invulnerable")
                    observed["enemies"].append(target)
                result = compute_action(observed, "engage", "target")
                self.assertEqual((result.move, result.shoot), ("none", "none"))

    def test_invalidated_target_dodge_still_obeys_walls(self):
        observed = state(70, 250)
        observed["hazards"] = [grid(40, 240, 4), grid(40, 280, 4)]
        observed["projectiles"] = [
            {"x": 110, "y": 250, "vx": -6, "vy": 0, "radius": 5}]
        result = compute_action(observed, "engage", "missing")
        self.assertIn(result.move, ("up", "down", "up_right", "down_right"))
        self.assertEqual(result.shoot, "none")

    def test_invalidated_target_cannot_dodge_when_inactive_or_truncated(self):
        for change in (lambda s: s.update(enabled=False), lambda s: s.update(paused=True),
                       lambda s: s["player"].update(dead=True), lambda s: s["room"].update(clear=True),
                       lambda s: s.update(truncated=True),
                       lambda s: s.update(truncated_arrays={"projectiles": True}),
                       lambda s: s.update(truncated_arrays={"enemies": True}),
                       lambda s: s.update(truncated_arrays={"hazards": True})):
            observed = state()
            observed["projectiles"] = [
                {"x": 340, "y": 250, "vx": -6, "vy": 0, "radius": 5}]
            change(observed)
            result = compute_action(observed, "engage", "missing")
            self.assertEqual((result.move, result.shoot), ("none", "none"))
        observed = state()
        observed["projectiles"] = [
            {"x": 340, "y": 250, "vx": -6, "vy": 0, "radius": 5}]
        for kind, target in (("engage", None), ("engage", "")):
            result = compute_action(observed, kind, target)
            self.assertEqual((result.move, result.shoot), ("none", "none"))

    def test_inactive_goals_are_neutral(self):
        for change in (lambda s: s.update(enabled=False), lambda s: s.update(paused=True),
                       lambda s: s["player"].update(dead=True), lambda s: s["room"].update(clear=True),
                       lambda s: s.update(truncated=True),
                       lambda s: s.update(truncated_arrays={"hazards": True})):
            observed = state()
            observed["enemies"] = [enemy(450, 300)]
            change(observed)
            self.assertEqual(compute_action(observed, "engage", "target").move, "none")

    def test_hold_goal_dodges_and_shoots_without_pursuit(self):
        observed = state()
        observed["enemies"] = [enemy(480, 250)]
        observed["projectiles"] = [
            {"x": 340, "y": 250, "vx": -6, "vy": 0, "radius": 5}]
        result = compute_action(observed, "hold", "target")
        self.assertIn(result.move, ("up", "down", "up_left", "down_left"))
        self.assertEqual(result.shoot, "right")

    def test_hold_goal_only_shoots_without_repositioning_for_unaligned_enemy(self):
        observed = state()
        for target, expected in ((enemy(480, 250), "right"), (enemy(500, 350), "none")):
            observed["enemies"] = [target]
            result = compute_action(observed, "hold", "target")
            self.assertEqual((result.move, result.shoot), ("none", expected))

    def test_evade_and_hold_shoot_without_changing_defensive_movement(self):
        for goal in ("evade", "hold", "engage"):
            observed = state()
            observed["enemies"] = [enemy(480, 250, "aligned")]
            observed["projectiles"] = [
                {"x": 340, "y": 250, "vx": -6, "vy": 0, "radius": 5}]
            firing = compute_action(observed, goal, "missing" if goal == "engage" else None)
            observed["enemies"][0]["vulnerable"] = False
            without_shot = compute_action(observed, goal, "missing" if goal == "engage" else None)
            self.assertNotEqual(firing.move, "none")
            self.assertEqual(firing.move, without_shot.move)
            self.assertEqual(firing.shoot, "right")
            self.assertEqual(without_shot.shoot, "none")

    def test_engage_prefers_selected_target_over_closer_available_shot(self):
        observed = state()
        observed["enemies"] = [enemy(480, 250, "selected"), enemy(300, 330, "closer")]
        self.assertEqual(compute_action(observed, "engage", "selected").shoot, "right")
        self.assertEqual(compute_action(observed, "hold").shoot, "down")

    def test_opportunistic_shot_cannot_redirect_or_cancel_selected_target_pursuit(self):
        observed = state()
        observed["enemies"] = [enemy(500, 350, "selected"), enemy(300, 400, "aligned")]
        firing = compute_action(observed, "engage", "selected")
        observed["enemies"][1]["vulnerable"] = False
        without_shot = compute_action(observed, "engage", "selected")
        self.assertNotEqual(firing.move, "none")
        self.assertEqual(firing.move, without_shot.move)
        self.assertEqual(firing.shoot, "down")
        self.assertEqual(without_shot.shoot, "none")

    def test_opportunistic_aim_is_nearest_then_identity_and_independent_of_entity_order(self):
        observed = state()
        observed["enemies"] = [enemy(480, 250, "b"), enemy(120, 250, "a")]
        for goal in ("hold", "evade"):
            self.assertEqual(compute_action(observed, goal).shoot, "left")
            observed["enemies"].reverse()
            self.assertEqual(compute_action(observed, goal).shoot, "left")
        observed["enemies"].append(enemy(390, 250, "z"))
        self.assertEqual(compute_action(observed, "hold").shoot, "right")

    def test_opportunistic_shot_still_requires_clear_lane_range_and_alignment(self):
        for goal in ("hold", "evade", "engage"):
            observed = state()
            observed["enemies"] = [enemy(480, 250)]
            observed["hazards"] = [grid(390, 250)]
            self.assertEqual(compute_action(observed, goal, "missing").shoot, "none")
            observed["hazards"][0]["collision"] = 1  # A pit does not block ordinary tears.
            self.assertEqual(compute_action(observed, goal, "missing").shoot, "right")
            observed["hazards"] = []
            for point in ((350, 250), (521, 250), (480, 260)):
                observed["enemies"][0].update(x=point[0], y=point[1])
                self.assertEqual(compute_action(observed, goal, "missing").shoot, "none", (goal, point))

    def test_dead_invulnerable_invalid_and_duplicate_id_enemies_never_receive_shots(self):
        for change in (dict(hp=0), dict(hp=-1), dict(hp=float("nan")), dict(dead=True),
                       dict(vulnerable=False), dict(vulnerable=None), dict(id=""),
                       dict(id="bad\nid"), dict(id=[])):
            observed = state()
            target = enemy(480, 250)
            target.update(change)
            observed["enemies"] = [target]
            for goal in ("engage", "hold", "evade"):
                self.assertEqual(compute_action(observed, goal, "target").shoot, "none", (goal, change))
        observed = state()
        observed["enemies"] = [enemy(480, 250), enemy(120, 250)]
        for goal in ("engage", "hold", "evade"):
            self.assertEqual(compute_action(observed, goal, "target").shoot, "none")
        observed["enemies"].append(enemy(300, 400, "unique"))
        self.assertEqual(compute_action(observed, "engage", "target").shoot, "down")

    def test_opportunistic_target_is_rechecked_each_frame_without_stale_fire(self):
        observed = state()
        observed["enemies"] = [enemy(480, 250)]
        self.assertEqual(compute_action(observed, "evade").shoot, "right")
        observed["enemies"][0]["y"] = 280
        self.assertEqual(compute_action(observed, "evade").shoot, "none")
        observed["enemies"] = []
        self.assertEqual(compute_action(observed, "evade").shoot, "none")

    def test_hold_goal_cannot_dodge_when_inactive_or_truncated(self):
        for change in (lambda s: s.update(enabled=False), lambda s: s.update(paused=True),
                       lambda s: s["player"].update(dead=True), lambda s: s["room"].update(clear=True),
                       lambda s: s.update(truncated=True),
                       lambda s: s.update(truncated_arrays={"projectiles": True}),
                       lambda s: s.update(truncated_arrays={"enemies": True}),
                       lambda s: s.update(truncated_arrays={"hazards": True})):
            observed = state()
            observed["projectiles"] = [
                {"x": 340, "y": 250, "vx": -6, "vy": 0, "radius": 5}]
            change(observed)
            result = compute_action(observed, "hold")
            self.assertEqual((result.move, result.shoot), ("none", "none"))

    def test_irregular_room_internal_wall_never_crossed(self):
        observed = state(180, 200)
        observed["enemies"] = [enemy(460, 350)]
        # Continuous internal wall reaching both boundaries: no reachable goal.
        observed["hazards"] = [grid(320, y, 4) for y in range(120, 441, 40)]
        result = compute_action(observed, "engage", "target")
        self.assertEqual((result.move, result.shoot), ("none", "none"))

    def test_input_is_preserved_and_invalid_values_are_bounded(self):
        observed = saved("fifth")
        before = copy.deepcopy(observed)
        compute_action(observed, "engage", observed["enemies"][1]["id"])
        self.assertEqual(observed, before)
        for value in (float("nan"), float("inf"), 10**1000, True, None):
            observed = state()
            observed["player"]["x"] = value
            self.assertEqual(compute_action(observed, "engage", "target").move, "none")
        observed = state()
        observed["enemies"] = [enemy(400, 350)]*65
        self.assertEqual(compute_action(observed, "engage", "target").move, "none")
        observed = state()
        observed["enemies"] = [enemy(400, 350)]
        observed["room"]["bottom_right"].update(x=4000, y=4000)
        self.assertEqual(compute_action(observed, "engage", "target").move, "none")
        self.assertLessEqual(MAX_NODES, 1200)

    def test_malformed_hazard_and_missing_arrays_hold(self):
        observed = state()
        observed["enemies"] = [enemy(450, 300)]
        observed["hazards"] = [grid(400, 300, "wall")]
        self.assertEqual(compute_action(observed, "engage", "target").move, "none")
        del observed["hazards"]
        self.assertEqual(compute_action(observed, "engage", "target").move, "none")


if __name__ == "__main__":
    unittest.main()
