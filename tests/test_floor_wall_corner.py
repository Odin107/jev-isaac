"""Exact room-98 stop: outside a plain wall, inside its padded corner."""
import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_isaac.exploration import _door_move, _room_geometry, _validated
from jev_isaac.navigation import _VECTORS, _clear, _free
from jev_isaac.protocol import Observation


class FloorWallCornerTests(unittest.TestCase):
    def setUp(self):
        self.state = Observation.decode((Path(__file__).parent / "fixtures" / "floor-wall-corner-stop.json").read_bytes()).data
        self.wall = next(h for h in self.state["hazards"] if h.get("index") == 237)

    def action(self):
        parsed = _validated(self.state)
        return _door_move(self.state, parsed, next(d for d in parsed[-1] if d.target_index == 113))

    def test_exact_wall_padding_stop_has_outward_recovery_for_each_open_door(self):
        state = self.state
        before = copy.deepcopy(state)
        self.assertEqual((state["frame"], state["floor"]["room_index"]), (28808, 98))
        self.assertEqual((self.wall["type"], self.wall["collision"]), (15, 4))
        parsed = _validated(state)
        core = (540, 420, 580, 460)
        self.assertTrue(_free(parsed[3], [core]))
        for door in parsed[-1]:
            boxes = _room_geometry(state, parsed[4], door=door)[0]
            self.assertFalse(_free(parsed[3], boxes))
            move = _door_move(state, parsed, door)
            self.assertEqual(move, "up_right")
            vector = _VECTORS[move]
            end = parsed[3][0]+vector[0]*24, parsed[3][1]+vector[1]*24
            self.assertTrue(_clear(parsed[3], end, [core]))
            self.assertTrue(_free(end, boxes))
        self.assertEqual(state, before)

    def test_unknown_wall_or_door_metadata_never_receives_the_exemption(self):
        original = copy.deepcopy(self.wall)
        for changes in ({"type": 16}, {"variant": 1}, {"variant": False},
                        {"collision": 5}, {"state": 1}, {"state": False}):
            with self.subTest(changes=changes):
                self.wall.update(original)
                self.wall.update(changes)
                self.assertIsNone(self.action())
        for field in ("variant", "state"):
            self.wall.clear()
            self.wall.update(original)
            del self.wall[field]
            self.assertIsNone(self.action())

    def test_wall_body_overlap_still_fails_closed(self):
        for point in ((560, 440), (579.9, 420.1), (575, 430)):
            self.state["player"].update(x=point[0], y=point[1], vx=0, vy=0)
            self.assertIsNone(self.action())

    def test_recovery_can_resume_the_selected_room113_route(self):
        for tick in range(450):
            parsed = _validated(self.state)
            door = next(d for d in parsed[-1] if d.target_index == 113)
            move = _door_move(self.state, parsed, door)
            self.assertIsNotNone(move, f"route failed at step {tick}")
            self.assertNotEqual(move, "none", f"route stalled at step {tick}")
            vector = _VECTORS[move]
            end = parsed[3][0]+vector[0]*4, parsed[3][1]+vector[1]*4
            boxes = _room_geometry(self.state, parsed[4], door=door)[0]
            if _free(parsed[3], boxes):
                self.assertTrue(_clear(parsed[3], end, boxes))
            self.state["player"].update(x=end[0], y=end[1], vx=0, vy=0)
            self.state["frame"] += 1
            if end[0] > 1120:
                self.assertLessEqual(abs(end[1]-560), 10)
                return
        self.fail("did not reach the originally selected door")


if __name__ == "__main__":
    unittest.main()
