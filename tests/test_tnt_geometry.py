"""Pure geometry checks; explosion distances remain explicitly estimated."""
import copy
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_isaac.exploration import _point_waypoint, _room_geometry, _validated
from jev_isaac.navigation import _clear, _free
from jev_isaac.tnt_geometry import plan_demolition, shot_direction


class TntGeometryTests(unittest.TestCase):
    def setUp(self):
        self.state = json.loads((Path(__file__).parent / "fixtures" / "floor-switch-tnt-live.json").read_text())
        self.state.update(enabled=True, paused=False)

    def plan(self):
        boxes, _, _, phase = _room_geometry(self.state, 10)
        return plan_demolition(self.state, (400, 160), boxes, phase)

    def test_actual_crowded_room_gets_centered_firing_and_widest_reachable_retreat(self):
        before = copy.deepcopy(self.state)
        plan = self.plan()
        self.assertIsNotNone(plan)
        self.assertEqual((plan.target_index, plan.target_point), (54, (400, 240)))
        self.assertEqual((plan.firing_point, plan.shoot), ((188, 240), "right"))
        self.assertEqual(plan.retreat_point, (78, 280))
        self.assertEqual(len(plan.chain_points), 14)
        self.assertIn((360, 360), plan.chain_points)  # Observed movable barrel.
        clearance = min(math.dist(plan.retreat_point, point) for point in plan.chain_points)
        self.assertAlmostEqual(clearance, math.sqrt(42**2+80**2))
        self.assertLess(clearance, 105+10)  # Preferred margin is not a false impossibility gate.
        self.assertEqual(self.state, before)

    def test_every_movement_route_uses_original_barrels_and_closed_doors(self):
        plan = self.plan()
        parsed = _validated(self.state)
        boxes, _, _, phase = _room_geometry(self.state, 10)
        self.assertIsNotNone(_point_waypoint(parsed[3], plan.firing_point, parsed[6], boxes, phase))
        self.assertIsNotNone(_point_waypoint(plan.firing_point, plan.retreat_point, parsed[6], boxes, phase))
        self.assertFalse(_free((40, 280), boxes))
        self.assertFalse(_free((600, 280), boxes))
        self.assertIsNone(_point_waypoint(parsed[3], (400, 160), parsed[6], boxes, phase))
        hypothetical = copy.deepcopy(self.state)
        hypothetical["hazards"] = [h for h in hypothetical["hazards"] if h.get("index") != plan.target_index]
        new_boxes, _, _, new_phase = _room_geometry(hypothetical, 10)
        self.assertIsNotNone(_point_waypoint(parsed[3], (400, 160), parsed[6], new_boxes, new_phase))

    def test_collision_two_and_movable_tnt_block_shots_to_selected_barrel(self):
        plan = self.plan()
        self.assertEqual(shot_direction(self.state, plan, plan.firing_point), "right")
        for item in (
            {"kind": "grid", "type": 12, "index": 999, "variant": 0, "state": 0,
             "collision": 2, "radius": 20, "x": 280, "y": 240},
            {"kind": "tnt", "id": "new-movable", "type": 292, "variant": 0,
             "radius": 20, "x": 280, "y": 240, "vx": 0, "vy": 0},
        ):
            self.state["hazards"].append(item)
            self.assertEqual(shot_direction(self.state, plan, plan.firing_point), "none")
            self.state["hazards"].pop()

    def test_fresh_range_alignment_damage_and_identity_are_checked(self):
        plan = self.plan()
        self.assertEqual(shot_direction(self.state, 54, (261, 240)), "none")  # Stand-off under140.
        self.assertEqual(shot_direction(self.state, 54, (179, 240)), "none")  # Beyond capped range220.
        self.assertEqual(shot_direction(self.state, 54, (188, 249)), "none")
        target = next(h for h in self.state["hazards"] if h.get("index") == 54)
        for damage in (1, 2, 3):
            target["state"] = damage
            self.assertEqual(shot_direction(self.state, plan, (180, 240)), "right")
        self.state["player"]["tear_range"] = 200
        self.assertEqual(shot_direction(self.state, plan, (180, 240)), "none")
        self.state["player"]["tear_range"] = 260
        target.update(state=4, collision=0)
        self.assertEqual(shot_direction(self.state, plan, (180, 240)), "none")

    def test_moving_tnt_and_ambiguous_target_ids_do_not_create_a_static_plan(self):
        movable = next(h for h in self.state["hazards"] if h.get("kind") == "tnt")
        movable["vx"] = .6
        self.assertIsNone(self.plan())
        movable["vx"] = 0
        target = next(h for h in self.state["hazards"] if h.get("index") == 54)
        self.state["hazards"].append(copy.deepcopy(target))
        self.assertIsNone(self.plan())

    def test_firing_point_keeps_the_full_settling_tolerance_inside_aim_and_range(self):
        plan = self.plan()
        for dx in (-6, 0, 6):
            for dy in (-6, 0, 6):
                point = plan.firing_point[0]+dx, plan.firing_point[1]+dy
                self.assertEqual(shot_direction(self.state, plan, point), "right")
        self.state["player"]["tear_range"] = 155
        self.assertIsNone(self.plan())

    def test_malformed_target_is_never_authorized_to_shoot(self):
        original = next(h for h in self.state["hazards"] if h.get("index") == 54)
        for field, value in (("variant", 1), ("variant", False), ("state", True),
                             ("state", 4), ("collision", 3), ("type", 5)):
            before = copy.deepcopy(original)
            original[field] = value
            self.assertEqual(shot_direction(self.state, 54, (180, 240)), "none")
            original.clear()
            original.update(before)

    def test_demolition_is_not_offered_when_a_walking_route_exists_or_plate_unknown(self):
        self.state["switches"][0].update(x=320, y=280)
        boxes, _, _, phase = _room_geometry(self.state, 10)
        self.assertIsNone(plan_demolition(self.state, (320, 280), boxes, phase))
        self.state["switches"][0].update(x=400, y=160, variant=1)
        self.assertIsNone(self.plan())

    def test_hypothetical_removal_preserves_an_overlapping_unrelated_wall(self):
        self.state["hazards"].append({"kind": "grid", "type": 15, "variant": 0,
                                       "state": 0, "collision": 4, "radius": 20,
                                       "x": 400, "y": 240, "index": 1000})
        self.assertIsNone(self.plan())


if __name__ == "__main__":
    unittest.main()
