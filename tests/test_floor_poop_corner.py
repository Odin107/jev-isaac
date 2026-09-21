"""Replay the recorded room-83 navigation stop, without game or API calls."""
import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_isaac.exploration import _door_move, _room_geometry, _validated
from jev_isaac.navigation import _VECTORS, _clear, _free
from jev_isaac.protocol import Observation


def distance_from_core(point, solid):
    return math.hypot(max(solid[0]-point[0], 0, point[0]-solid[2]),
                      max(solid[1]-point[1], 0, point[1]-solid[3]))


class FloorPoopCornerTests(unittest.TestCase):
    def setUp(self):
        fixture = Path(__file__).parent / "fixtures" / "floor-poop-corner-stop.json"
        self.state = Observation.decode(fixture.read_bytes()).data
        self.poop = next(item for item in self.state["hazards"] if item.get("index") == 122)

    def parsed_door(self):
        parsed = _validated(self.state, allow_shop=True)
        self.assertIsNotNone(parsed)
        return parsed, next(door for door in parsed[-1] if door.target_index == 68)

    def action(self):
        parsed, door = self.parsed_door()
        return _door_move(self.state, parsed, door)

    def test_authentic_failure_recovers_outward_from_confirmed_ordinary_poop(self):
        before = copy.deepcopy(self.state)
        self.assertEqual(self.state["frame"], 27816)
        self.assertEqual(self.state["control"]["source_frame"], 27811)
        self.assertFalse(self.state["player"]["dead"])
        self.assertEqual((self.poop["type"], self.poop["variant"], self.poop["state"]), (14, 0, 500))
        parsed, door = self.parsed_door()
        boxes, _, recoverable, _ = _room_geometry(self.state, parsed[4], door=door)
        self.assertFalse(_free(parsed[3], boxes))
        move = self.action()
        self.assertIn(move, ("right", "down", "down_right"))
        vector = _VECTORS[move]
        solid = (420, 260, 460, 300)
        last = distance_from_core(parsed[3], solid)
        self.assertGreater(last, 0)
        for sample in range(1, 241):
            point = parsed[3][0]+vector[0]*sample/10, parsed[3][1]+vector[1]*sample/10
            distance = distance_from_core(point, solid)
            self.assertGreaterEqual(distance, last-1e-9)
            last = distance
        self.assertTrue(_free(point, boxes))
        self.assertEqual(self.state, before)

    def test_same_corner_of_an_ordinary_rock_uses_the_shared_recovery(self):
        self.poop["type"] = 2
        self.assertIn(self.action(), ("right", "down", "down_right"))

    def test_unknown_or_special_poop_does_not_gain_an_escape_permission(self):
        original = copy.deepcopy(self.poop)
        for field, value in (("variant", 1), ("variant", 5), ("variant", -1),
                             ("variant", False), ("tear_destructible", False),
                             ("state", 1000), ("state", -1), ("state", False)):
            with self.subTest(field=field, value=value):
                self.poop.update(original)
                self.poop[field] = value
                self.assertIsNone(self.action())
        for field in ("variant", "state", "tear_destructible"):
            with self.subTest(missing=field):
                self.poop.clear()
                self.poop.update(original)
                del self.poop[field]
                self.assertIsNone(self.action())

    def test_deep_body_or_face_overlap_still_does_not_authorize_a_route(self):
        for x, y in ((440, 280), (455, 280), (460.1, 280), (459.49, 299.49)):
            with self.subTest(point=(x, y)):
                self.state["player"].update(x=x, y=y, vx=0, vy=0)
                self.assertIsNone(self.action())

    def test_identical_fire_box_keeps_its_own_blocking_identity(self):
        # Radius 12 plus the fire margin 8 equals the poop radius 20.
        self.state["hazards"].insert(0, {
            "kind": "fire", "x": 440, "y": 280, "radius": 12,
            "vx": 0, "vy": 0, "type": 33, "variant": 0,
        })
        self.assertIsNone(self.action())

    def test_paid_pickup_exclusion_still_blocks_corner_recovery(self):
        self.state["pickups"].append({
            "id": "paid-at-failed-corner", "type": 5, "variant": 20,
            "subtype": 1, "x": self.state["player"]["x"],
            "y": self.state["player"]["y"], "radius": 10,
            "price": 5, "shop_item": True, "options_index": 0,
            "wait": 0, "collectible_kind": 0,
        })
        self.assertIsNone(self.action())

    def test_outward_recovery_cannot_cross_another_obstacle(self):
        # Seal every outward ray with unrelated solid walls.
        self.state["hazards"].extend([
            {"kind": "grid", "type": 15, "collision": 4, "radius": 20,
             "x": 500, "y": 300},
            {"kind": "grid", "type": 15, "collision": 4, "radius": 20,
             "x": 460, "y": 340},
        ])
        self.assertIsNone(self.action())

    def test_recovered_corner_resumes_route_to_the_original_door(self):
        # This verifies geometry and route continuity. The four-unit stepping
        # deliberately does not pretend to reproduce Isaac's inertial physics.
        solid = (420, 260, 460, 300)
        for tick in range(900):
            parsed, door = self.parsed_door()
            move = _door_move(self.state, parsed, door)
            self.assertIsNotNone(move, f"Route failed at geometric step {tick}")
            self.assertNotEqual(move, "none", f"Route stalled at geometric step {tick}")
            vector = _VECTORS[move]
            start = parsed[3]
            end = start[0]+vector[0]*4, start[1]+vector[1]*4
            boxes, _, _, _ = _room_geometry(self.state, parsed[4], door=door)
            if not _free(start, boxes):
                self.assertGreaterEqual(distance_from_core(end, solid), distance_from_core(start, solid))
            else:
                self.assertTrue(_clear(start, end, boxes))
            self.state["player"].update(x=end[0], y=end[1], vx=0, vy=0)
            self.state["frame"] += 1
            if end[0] < 40:
                self.assertLessEqual(abs(end[1]-280), 10)
                return
        self.fail("Did not reach the originally selected doorway")


if __name__ == "__main__":
    unittest.main()
