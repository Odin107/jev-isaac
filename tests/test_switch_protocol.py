import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_isaac.demo import sample
from jev_isaac.protocol import Observation


class SwitchProtocolTests(unittest.TestCase):
    def state(self):
        state = sample()
        state["room"]["has_trigger_pressure_plates"] = True
        state["switches"] = [{"index": 24, "type": 20, "variant": 0, "state": 0,
                              "collision": 0, "x": 400, "y": 160}]
        return state

    def decode(self, state):
        return Observation.decode(json.dumps(state).encode()).data

    def test_known_and_missing_switch_metadata_round_trip_without_defaults(self):
        state = self.state()
        self.assertEqual(self.decode(state)["switches"], state["switches"])
        del state["switches"][0]["variant"]
        del state["switches"][0]["state"]
        self.assertNotIn("variant", self.decode(state)["switches"][0])
        self.assertNotIn("state", self.decode(state)["switches"][0])
        state["switches"] = []
        self.assertEqual(self.decode(state)["switches"], [])
        self.assertNotIn("switches", self.decode(sample()))

    def test_malformed_switches_and_room_trigger_flag_are_rejected(self):
        values = {"index": [True, -1, 4096, None], "type": [True, 19, 20.0],
                  "collision": [False, -1, 6], "variant": [False, -1, 100001, None],
                  "state": [False, -1, 100001, None], "x": [True, "400", float("inf")]}
        for field, choices in values.items():
            for value in choices:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    state = self.state()
                    state["switches"][0][field] = value
                    self.decode(state)
        for value in (0, 1, None, "true"):
            state = self.state()
            state["room"]["has_trigger_pressure_plates"] = value
            with self.subTest(trigger=value), self.assertRaises(ValueError):
                self.decode(state)

    def test_duplicate_or_unbounded_switch_arrays_are_rejected(self):
        for value in (None, {}, [None]):
            state = self.state()
            state["switches"] = value
            with self.assertRaises(ValueError):
                self.decode(state)
        state = self.state()
        state["switches"].append(copy.deepcopy(state["switches"][0]))
        with self.assertRaises(ValueError):
            self.decode(state)
        state["switches"] = [dict(state["switches"][0], index=i) for i in range(33)]
        with self.assertRaises(ValueError):
            self.decode(state)


if __name__ == "__main__":
    unittest.main()
