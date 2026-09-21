"""Reject ambiguous visited-room facts at the real JSON boundary."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_isaac.demo import sample
from jev_isaac.protocol import Observation


def packet():
    data = sample()
    data["floor"] = {"id": "stage:seed", "room_index": 84, "dimension": 0}
    data["visited_rooms"] = [{"room_index": 84, "list_index": 1, "type": 1,
                              "clear": True, "visited_count": 2,
                              "room_indices": [84, 85]}]
    return data


def decode(data):
    return Observation.decode(json.dumps(data).encode())


class VisitedProtocolTests(unittest.TestCase):
    def test_optional_facts_and_large_room_aliases_survive_json(self):
        self.assertNotIn("visited_rooms", decode(sample()).data)
        data = packet()
        self.assertEqual(decode(data).data["visited_rooms"], data["visited_rooms"])
        del data["visited_rooms"][0]["room_indices"]
        decode(data)
        data["visited_rooms"] = []
        decode(data)

    def test_invalid_or_unvisited_metadata_is_rejected(self):
        for key, values in {
            "room_index": [True, -1, 169, 1.0, None],
            "list_index": [True, -1, 2**31, 1.0, None],
            "type": [True, 0, 2**31, 1.0, None],
            "clear": [0, 1, "false", None],
            "visited_count": [True, 0, -1, 2**31, 1.0, None],
            "room_indices": [None, {}, [], [84, 84], [85], [84, True],
                             [84, -1], [84, 169], [84, 85, 86, 87, 88]],
        }.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    data = packet()
                    data["visited_rooms"][0][key] = value
                    decode(data)

    def test_conflicting_descriptors_and_overlapping_aliases_are_rejected(self):
        for duplicate in (dict(room_index=86, list_index=1, room_indices=[86]),
                          dict(room_index=85, list_index=2, room_indices=[85]),
                          dict(room_index=86, list_index=2, room_indices=[85, 86])):
            data = packet()
            other = copy.deepcopy(data["visited_rooms"][0])
            other.update(duplicate, clear=False)
            data["visited_rooms"].append(other)
            with self.subTest(duplicate=duplicate), self.assertRaises(ValueError):
                decode(data)

    def test_array_and_floor_context_are_bounded(self):
        for value in (None, {}, [None], packet()["visited_rooms"] * 170):
            data = packet()
            data["visited_rooms"] = value
            with self.subTest(value=repr(value)[:50]), self.assertRaises(ValueError):
                decode(data)
        for value in (True, -1, 3, 0.0, None):
            data = packet()
            data["floor"]["dimension"] = value
            with self.subTest(dimension=value), self.assertRaises(ValueError):
                decode(data)
        data = packet()
        del data["floor"]
        with self.assertRaises(ValueError):
            decode(data)


if __name__ == "__main__":
    unittest.main()
