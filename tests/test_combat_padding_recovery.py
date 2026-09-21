"""Bounded geometry recovery, including a counterfactual replay after boss death.

The recorded fixture is unchanged live data. Enabling its dead player is only a
diagnostic replay; a successful movement here does not establish survivability.
"""
import copy
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_isaac.navigation import compute_action, _VECTORS, _recovery_options, _clear


def grid(x=200, y=360, kind=14, collision=3):
    return {"kind": "grid", "type": kind, "collision": collision,
            "radius": 20, "x": x, "y": y}


def state(x=219.75, y=379.75):
    return {"enabled": True, "paused": False,
            "player": {"x": x, "y": y, "vx": 0, "vy": 0, "radius": 10, "dead": False},
            "room": {"clear": False, "top_left": {"x": 60, "y": 140},
                     "bottom_right": {"x": 580, "y": 420}},
            "hazards": [grid()], "enemies": [], "projectiles": []}


def signed_distance(point, box):
    x, y = point
    outside = math.hypot(max(box[0]-x, 0, x-box[2]), max(box[1]-y, 0, y-box[3]))
    return outside if outside else -min(x-box[0], box[2]-x, y-box[1], box[3]-y)


class CombatPaddingRecoveryTests(unittest.TestCase):
    def test_recorded_dead_state_remains_neutral(self):
        observed = json.loads((Path(__file__).parent / "fixtures" / "boss-poop-padding-death.json").read_text())
        self.assertTrue(observed["player"]["dead"])
        action = compute_action(observed, "evade")
        self.assertEqual((action.move, action.shoot), ("none", "none"))

    def test_counterfactual_recorded_boss_state_escapes_conservative_poop_corner(self):
        observed = json.loads((Path(__file__).parent / "fixtures" / "boss-poop-padding-death.json").read_text())
        observed["enabled"] = True
        observed["player"]["dead"] = False
        before = copy.deepcopy(observed)
        action = compute_action(observed, "evade")
        self.assertEqual(action.move, "right")
        self.assertEqual(action.shoot, "none")
        self.assertEqual(observed, before)
        start = observed["player"]["x"], observed["player"]["y"]
        core = (180, 340, 220, 380)
        self.assertLess(signed_distance(start, core), 0)
        vector = _VECTORS[action.move]
        last = signed_distance(start, core)
        for sample in range(1, 241):
            point = start[0]+vector[0]*sample/10, start[1]+vector[1]*sample/10
            current = signed_distance(point, core)
            self.assertGreaterEqual(current, last-1e-9)
            last = current
        self.assertGreater(last, 10)

    def test_every_allowed_ray_increases_signed_core_clearance_at_all_corners(self):
        core, padded = (180, 340, 220, 380), (170, 330, 230, 390)
        for sx in (-1, 1):
            for sy in (-1, 1):
                for offset in (-.5, -.25, 0, 2, 6, 9.5):
                    start = 200+sx*(20+offset), 360+sy*(20+offset)
                    options = _recovery_options(start, (-4*sx, -4*sy), 10,
                                                (70, 150, 570, 410), [padded], {0: core})
                    self.assertTrue(options, (sx, sy, offset))
                    for name in options:
                        vector = _VECTORS[name]
                        last = signed_distance(start, core)
                        for sample in range(1, 241):
                            point = start[0]+vector[0]*sample/10, start[1]+vector[1]*sample/10
                            current = signed_distance(point, core)
                            self.assertGreaterEqual(current, last-1e-9, (start, name))
                            last = current
                        self.assertGreaterEqual(signed_distance(point, padded), 0)

    def test_shallow_padding_face_recovers_but_deep_or_noncorner_core_does_not(self):
        for kind in (2, 14):
            observed = state(229, 360)
            observed["hazards"][0]["type"] = kind
            self.assertEqual(compute_action(observed, "evade").move, "right")
            for point in ((200, 360), (219.49, 379.49), (219.75, 360), (223, 360)):
                observed["player"].update(x=point[0], y=point[1])
                self.assertEqual(compute_action(observed, "evade").move, "none", (kind, point))

    def test_threat_prediction_selects_legal_escape_away_from_incoming_enemy(self):
        for enemy, expected in (((247, 379.75, -6, 0), "down"),
                                ((219.75, 407, 0, -6), "right")):
            observed = state()
            observed["enemies"] = [{"id": "incoming", "hp": 10, "vulnerable": True,
                                    "x": enemy[0], "y": enemy[1], "vx": enemy[2],
                                    "vy": enemy[3], "radius": 10}]
            action = compute_action(observed, "evade")
            self.assertEqual(action.move, expected)
            self.assertEqual(action.shoot, "none")

    def test_recovery_never_exempts_other_hazard_kinds(self):
        for kind, collision in ((8, 0), (9, 0), (17, 0), (18, 0), (23, 0),
                                (7, 1), (15, 4), (16, 5), (2, 1), (3, 3)):
            observed = state()
            observed["hazards"] = [grid(kind=kind, collision=collision)]
            self.assertEqual(compute_action(observed, "evade").move, "none", (kind, collision))
        observed = state()
        observed["hazards"] = [{"kind": "fire", "x": 200, "y": 360, "radius": 20}]
        self.assertEqual(compute_action(observed, "evade").move, "none")

    def test_escaping_one_obstacle_cannot_cross_another_pit_wall_fire_or_floor_exit(self):
        blockers = [grid(260, 380, 7, 1), grid(260, 380, 15, 4),
                    grid(260, 380, 17, 0), grid(260, 380, 18, 0), grid(260, 380, 23, 0),
                    {"kind": "fire", "x": 260, "y": 380, "radius": 20}]
        for blocker in blockers:
            observed = state()
            observed["hazards"].extend((blocker, grid(220, 420)))
            self.assertEqual(compute_action(observed, "evade").move, "none", blocker)

    def test_escaping_overlapping_destructibles_moves_outward_from_both(self):
        observed = state(229.75, 379.75)
        observed["hazards"].append(grid(250, 360, 2))
        self.assertEqual(compute_action(observed, "evade").move, "down")

    def test_escape_stays_inside_player_inset_room_bounds(self):
        observed = state()
        observed["room"]["bottom_right"]["y"] = 400
        observed["hazards"].append(grid(260, 380))
        self.assertEqual(compute_action(observed, "evade").move, "none")

    def test_recovery_replans_normally_after_leaving_padding(self):
        observed = state(235, 379.75)
        action = compute_action(observed, "hold")
        self.assertEqual((action.move, action.shoot), ("none", "none"))
        self.assertTrue(_clear((235, 379.75), (259, 379.75), [(170, 330, 230, 390)]))


if __name__ == "__main__":
    unittest.main()
