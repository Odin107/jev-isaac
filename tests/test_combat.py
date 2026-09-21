"""Offline geometry checks; no game control or model requests."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_isaac.combat import build_combat_context, current_firing_view


def state():
    return {"player": {"x": 300, "y": 250, "radius": 10},
            "room": {"top_left": {"x": 60, "y": 140}, "bottom_right": {"x": 580, "y": 420}},
            "enemies": [], "hazards": []}


def enemy(x, y, ident="target"):
    return {"x": x, "y": y, "id": ident, "radius": 13, "hp": 10, "vulnerable": True}


def grid(x, y, collision=3, index=1):
    return {"kind": "grid", "x": x, "y": y, "radius": 20, "collision": collision, "index": index}


class CombatContextTests(unittest.TestCase):
    def test_current_firing_view_rotates_with_enemy_and_ignores_previous_input(self):
        for direction, point in {"left": (200, 250), "right": (400, 250),
                                 "up": (300, 150), "down": (300, 350)}.items():
            with self.subTest(direction=direction):
                observed = state()
                observed["control"] = {"shoot": "left"}
                observed["enemies"] = [enemy(*point)]
                context = build_combat_context(observed)
                context["firing_positions"] = [{"shoot": "left"}]
                view = current_firing_view(context)
                self.assertTrue(view["grid_complete"])
                for button, row in view["directions"].items():
                    expected = ["target"] if button == direction else []
                    self.assertEqual(row["enemies_on_side"], expected)
                    self.assertEqual(row["aligned_enemies"], expected)
                    self.assertEqual(row["grid_clear_aligned_enemies"], expected)

    def test_current_firing_view_distinguishes_diagonal_blocked_and_unknown(self):
        observed = state()
        observed["enemies"] = [enemy(500, 250, "blocked"), enemy(450, 380, "diagonal"),
                               enemy(200, 250, "dead")]
        observed["enemies"][-1]["dead"] = True
        observed["hazards"] = [grid(400, 250)]
        view = current_firing_view(build_combat_context(observed))["directions"]
        self.assertEqual(set(view["right"]["enemies_on_side"]), {"blocked", "diagonal"})
        self.assertEqual(view["right"]["aligned_enemies"], ["blocked"])
        self.assertEqual(view["right"]["grid_clear_aligned_enemies"], [])
        self.assertEqual(view["down"]["enemies_on_side"], ["diagonal"])
        self.assertEqual(view["down"]["aligned_enemies"], [])
        self.assertEqual(view["left"]["enemies_on_side"], [])
        del observed["hazards"]
        view = current_firing_view(build_combat_context(observed))
        self.assertFalse(view["grid_complete"])
        self.assertIsNone(view["directions"]["right"]["grid_clear_aligned_enemies"])
        self.assertEqual(current_firing_view({}), {"geometry_available": False, "directions": {}})

    def test_saved_corner_has_blocked_left_lane_and_clear_down_alignment(self):
        saved = json.loads((Path(__file__).parent / "fixtures/combat-corner.json").read_text())
        result = build_combat_context(saved)
        target = next(item for item in result["targets"] if item["dx"] < -400)
        self.assertTrue(target["lanes"]["horizontal"]["aligned"])
        self.assertIn(18, target["lanes"]["horizontal"]["blockers"])
        candidate = next(item for item in result["firing_positions"] if item["x"] == 520)
        self.assertEqual(candidate["shoot"], "down")
        self.assertAlmostEqual(candidate["move_distance"], 50, delta=.01)
        self.assertTrue(candidate["range_unknown"])
        self.assertLess(result["moves"]["right"]["clearance"], .01)
        self.assertLess(result["moves"]["up"]["clearance"], .1)
        self.assertTrue(result["moves"]["left"]["probe_clear"])

    def test_pit_blocks_ground_movement_but_not_tears(self):
        observed = state()
        observed["enemies"] = [enemy(500, 250)]
        observed["hazards"] = [grid(350, 250, 1)]
        result = build_combat_context(observed)
        self.assertEqual(result["moves"]["right"]["clearance"], 20)
        self.assertEqual(result["targets"][0]["lanes"]["horizontal"]["blockers"], [])
        observed["hazards"][0]["collision"] = 3
        self.assertEqual(build_combat_context(observed)["targets"][0]["lanes"]["horizontal"]["blockers"], [1])

    def test_boundary_contact_allows_inward_and_tangent_moves(self):
        observed = state()
        observed["player"].update(x=570, y=150)
        observed["hazards"] = [grid(600, 160, 4), grid(560, 120, 4)]
        moves = build_combat_context(observed)["moves"]
        self.assertEqual(moves["right"]["clearance"], 0)
        self.assertEqual(moves["up"]["clearance"], 0)
        self.assertTrue(moves["left"]["probe_clear"])
        self.assertTrue(moves["down_left"]["probe_clear"])
        self.assertEqual(moves["none"]["clearance"], 0)

    def test_diagonal_probe_is_normalized(self):
        observed = state()
        observed["player"].update(x=550, y=390)
        moves = build_combat_context(observed)["moves"]
        self.assertAlmostEqual(moves["down_right"]["clearance"], 28.28, places=2)

    def test_blocked_direct_path_excludes_firing_position(self):
        observed = state()
        observed["enemies"] = [enemy(500, 400)]
        open_candidates = build_combat_context(observed)["firing_positions"]
        self.assertTrue(any(item["x"] == 500 and item["y"] == 250 for item in open_candidates))
        observed["hazards"] = [grid(400, 250)]
        candidates = build_combat_context(observed)["firing_positions"]
        self.assertFalse(any(item["x"] == 500 and item["y"] == 250 for item in candidates))

    def test_shot_blocked_from_candidate_excludes_it(self):
        observed = state()
        observed["enemies"] = [enemy(500, 400)]
        observed["hazards"] = [grid(500, 320)]
        candidates = build_combat_context(observed)["firing_positions"]
        self.assertFalse(any(item["x"] == 500 and item["y"] == 250 for item in candidates))

    def test_candidates_respect_separation_and_player_bounds(self):
        observed = state()
        observed["enemies"] = [enemy(575, 300), enemy(340, 280, "near")]
        candidates = build_combat_context(observed)["firing_positions"]
        self.assertTrue(all(70 <= item["x"] <= 570 and 150 <= item["y"] <= 410
                            for item in candidates))
        self.assertTrue(all(item["target_distance"] > 60 for item in candidates))
        self.assertFalse(any(item["target_id"] == "near" for item in candidates))

    def test_bounded_and_no_input_mutation(self):
        observed = state()
        observed["enemies"] = [enemy(450+i, 350+i, str(i)) for i in range(20)]
        original = copy.deepcopy(observed)
        result = build_combat_context(observed)
        self.assertEqual(observed, original)
        self.assertEqual(len(result["targets"]), 8)
        self.assertLessEqual(len(result["firing_positions"]), 4)
        json.dumps(result, allow_nan=False)

    def test_missing_or_nonfinite_data_and_incomplete_grid(self):
        self.assertEqual(build_combat_context({}), {})
        observed = state()
        observed["player"]["x"] = float("nan")
        self.assertEqual(build_combat_context(observed), {})
        observed = state()
        observed["enemies"] = [enemy(500, 400), enemy(float("inf"), 300)]
        del observed["hazards"]
        result = build_combat_context(observed)
        self.assertFalse(result["grid_complete"])
        self.assertEqual(result["firing_positions"], [])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
