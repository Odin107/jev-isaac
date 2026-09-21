"""Doorway regressions from the first failed live floor trial; no API/game IO."""
import copy
from collections import deque
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_isaac.exploration import FloorNavigator
from jev_isaac.navigation import _VECTORS, _clear


def recorded_state():
    # Exact final observed geometry, position and velocity, with fake identities.
    return json.loads((Path(__file__).parent / "fixtures" / "floor-door-first.json").read_text())


def rotate_clockwise(observed):
    """Rotate the entire geometry around room center, including all wall cells."""
    result = copy.deepcopy(observed)

    def point(item, velocity=False):
        item["x"], item["y"] = 320-(item["y"]-280), 280+(item["x"]-320)
        if velocity:
            item["vx"], item["vy"] = -item.get("vy", 0), item.get("vx", 0)

    point(result["player"], True)
    for item in result["hazards"] + result["enemies"] + result["projectiles"]:
        point(item, "vx" in item or "vy" in item)
    for item in result["doors"]:
        point(item)
        item["slot"] = (item["slot"]+1) % 4
    a, b = result["room"]["top_left"], result["room"]["bottom_right"]
    point(a)
    point(b)
    a["x"], b["x"] = sorted((a["x"], b["x"]))
    a["y"], b["y"] = sorted((a["y"], b["y"]))
    return result


class DoorRegressionTests(unittest.TestCase):
    def test_recorded_boundary_overshoot_keeps_moving_toward_open_door(self):
        observed = recorded_state()
        before = copy.deepcopy(observed)
        # This is 0.107 units beyond the ordinary room inset and 7.754 off-center.
        self.assertGreater(observed["player"]["x"], 570)
        self.assertGreater(observed["player"]["y"]-280, 7)
        result = FloorNavigator().step(observed, 0)
        self.assertIsNone(result.stop_reason)
        self.assertGreater(_VECTORS[result.move][0], 0)
        self.assertLessEqual(_VECTORS[result.move][1], 0)
        self.assertEqual(result.shoot, "none")
        self.assertEqual(observed, before)

    def test_all_door_orientations_tolerate_the_same_safe_boundary_drift(self):
        observed = recorded_state()
        observed["doors"] = observed["doors"][:1]
        outward = (1, 0)
        for orientation in range(4):
            with self.subTest(orientation=orientation):
                result = FloorNavigator().step(observed, 0)
                self.assertIsNone(result.stop_reason)
                vector = _VECTORS[result.move]
                self.assertGreater(vector[0]*outward[0]+vector[1]*outward[1], 0)
                self.assertEqual(result.shoot, "none")
            observed = rotate_clockwise(observed)
            outward = -outward[1], outward[0]

    def test_portal_mouth_allows_diagonal_centering_including_upward_door(self):
        observed = recorded_state()
        observed["doors"] = observed["doors"][:1]
        # At the rear of the portal corridor, the short diagonal step must
        # remain available while correcting a five-unit sideways offset.
        observed["player"].update(x=545, y=275, vx=0, vy=0)
        expected = (2**-.5, 2**-.5)
        for orientation in range(4):
            with self.subTest(orientation=orientation):
                result = FloorNavigator().step(observed, 0)
                self.assertIsNone(result.stop_reason)
                self.assertEqual(_VECTORS[result.move], expected)
            observed = rotate_clockwise(observed)
            expected = -expected[1], expected[0]

    def test_portal_exception_does_not_allow_an_ordinary_wall(self):
        observed = recorded_state()
        for hazard in observed["hazards"]:
            if hazard["kind"] == "grid" and (hazard["x"], hazard["y"]) == (600, 280):
                hazard["type"] = 15
        result = FloorNavigator().step(observed, 0)
        self.assertEqual(result.stop_reason, "no safe route to open door")
        self.assertEqual(result.move, "none")

    def test_portal_exception_does_not_allow_adjacent_wall_or_unselected_door(self):
        for point in ((570.10729980469, 310), (320, 430)):
            with self.subTest(point=point):
                observed = recorded_state()
                observed["player"].update(x=point[0], y=point[1], vx=0, vy=0)
                result = FloorNavigator().step(observed, 0)
                self.assertEqual(result.stop_reason, "no safe route to open door")
                self.assertEqual(result.move, "none")

    def test_observed_doorway_is_crossed_with_inertia_and_two_tick_input_delay(self):
        observed = recorded_state()
        navigator = FloorNavigator()
        # A bounded synthetic dynamics model, not a claim to reproduce the game:
        # retain velocity, accelerate toward 4 units/tick, and apply input late.
        pending = deque(["up_left", "up_left"])
        player = observed["player"]
        boxes = []
        for hazard in observed["hazards"]:
            if hazard["kind"] == "grid" and hazard["type"] == 16 and (hazard["x"], hazard["y"]) == (600, 280):
                continue
            padding = hazard["radius"]+player["radius"]+(0 if hazard["kind"] == "grid" else 8)
            boxes.append((hazard["x"]-padding, hazard["y"]-padding,
                          hazard["x"]+padding, hazard["y"]+padding))
        for tick in range(120):
            observed["frame"] += 1
            action = navigator.step(observed, tick/30)
            self.assertIsNone(action.stop_reason, f"Stopped on tick {tick}")
            pending.append(action.move)
            applied = _VECTORS[pending.popleft()]
            start = player["x"], player["y"]
            player["vx"] = player["vx"]*.65+applied[0]*1.4
            player["vy"] = player["vy"]*.65+applied[1]*1.4
            end = start[0]+player["vx"], start[1]+player["vy"]
            self.assertTrue(_clear(start, end, boxes), f"Wall contact on tick {tick}: {end}")
            player["x"], player["y"] = end
            if end[0] > 608:
                self.assertLess(abs(end[1]-280), 10)
                break
        else:
            self.fail("Navigator did not cross the door with delayed, inertial motion")


if __name__ == "__main__":
    unittest.main()
