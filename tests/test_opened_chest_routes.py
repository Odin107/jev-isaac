"""Regression for the live room-83 stop after opening an ordinary chest."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_isaac.exploration import FloorNavigator, _door_move, _pickup_move, _room_geometry, _validated
from jev_isaac.navigation import _VECTORS, _clear, _free
from jev_isaac.pickups import exclusion_boxes
from jev_isaac.protocol import Observation


class OpenedChestRouteTests(unittest.TestCase):
    def setUp(self):
        fixture = Path(__file__).parent / "fixtures" / "floor-opened-chest-stop.json"
        self.data = Observation.decode(fixture.read_bytes()).data
        self.chest = next(item for item in self.data["pickups"] if item["variant"] == 50)
        self.parsed = _validated(self.data, allow_shop=True)

    def assert_safe_step(self, move, boxes):
        self.assertIn(move, _VECTORS)
        self.assertNotEqual(move, "none")
        start = self.parsed[3]
        vector = _VECTORS[move]
        end = start[0] + vector[0] * 12, start[1] + vector[1] * 12
        self.assertTrue(_clear(start, end, boxes))

    def test_saved_open_chest_position_has_routes_to_both_ordinary_doors(self):
        before = copy.deepcopy(self.data)
        self.assertEqual(self.data["floor"]["room_index"], 83)
        self.assertEqual(self.data["player"]["hearts"], 6)
        self.assertEqual(self.chest["subtype"], 0)
        self.assertEqual({door.slot for door in self.parsed[-1]}, {0, 6})
        for door in self.parsed[-1]:
            with self.subTest(slot=door.slot):
                boxes, _, _, _ = _room_geometry(self.data, self.parsed[4], door=door)
                self.assertTrue(_free(self.parsed[3], boxes))
                self.assert_safe_step(_door_move(self.data, self.parsed, door), boxes)
        self.assertEqual(self.data, before)

    def test_all_three_spawned_supplies_have_safe_first_steps(self):
        supplies = [item for item in self.data["pickups"] if item is not self.chest]
        self.assertEqual(sorted(item["variant"] for item in supplies), [20, 20, 40])
        boxes, _, _, _ = _room_geometry(self.data, self.parsed[4])
        for item in supplies:
            with self.subTest(id=item["id"]):
                self.assert_safe_step(_pickup_move(self.data, self.parsed, item), boxes)

    def test_floor_controller_collects_chest_reward_instead_of_stopping(self):
        controller = FloorNavigator(adventure_mode=True)
        result = controller.step(self.data, 0)
        self.assertEqual(result.status, "waiting for room rewards")
        self.data["frame"] += 19
        result = controller.step(self.data, 19 / 30)
        self.assertIsNone(result.stop_reason)
        self.assertNotEqual(result.move, "none")
        self.assertEqual(result.shoot, "none")
        self.assertTrue(result.status.startswith("collecting pickup"))

    def test_same_position_inside_a_closed_chest_exclusion_remains_blocked(self):
        self.chest["subtype"] = 1
        self.assertFalse(_free(self.parsed[3], exclusion_boxes(self.data, self.parsed[4])))
        for door in self.parsed[-1]:
            self.assertIsNone(_door_move(self.data, self.parsed, door))

    def test_special_paid_shop_and_option_chests_keep_their_exclusion(self):
        cases = [{"variant": variant} for variant in (51, 52, 53, 54, 55, 56, 57, 58, 60, 360)]
        cases += [{"subtype": 1}, {"subtype": 2}, {"price": 5}, {"price": -1},
                  {"shop_item": True}, {"options_index": 1}]
        for changes in cases:
            with self.subTest(changes=changes):
                item = dict(self.chest, **changes)
                boxes = exclusion_boxes({"pickups": [item]}, self.parsed[4])
                self.assertEqual(len(boxes), 1)
                self.assertFalse(_free(self.parsed[3], boxes))


if __name__ == "__main__":
    unittest.main()
