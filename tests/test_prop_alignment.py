"""Recorded shuffle geometry plus explicit, approximate inertia simulations."""
import copy
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_prop_integration import executor
from test_props import fire, poop, props
from test_pickups import pickup
from jev_isaac.navigation import _VECTORS, _clear
from jev_isaac.props import _geometry, _parsed, _selected


class PropAlignmentTests(unittest.TestCase):
    def fixture(self):
        return json.loads((Path(__file__).parent / "fixtures" / "prop-alignment-shuffle.json").read_text())

    def test_recorded_in_lane_drift_now_gets_a_correction(self):
        fixture = self.fixture()
        state = fixture["observation"]
        self.assertEqual(state["frame"], 27570)
        self.assertEqual(len(fixture["recorded_controls"]), 63)
        self.assertTrue(all(c["shoot"] == "none" for c in fixture["recorded_controls"]))
        self.assertLess(abs(state["player"]["y"]-280), 8)
        control, _ = executor(state)
        action = control.step(state, .1)
        self.assertEqual((action.move, action.shoot), ("up", "none"))
        self.assertEqual(control.prop_approach.destination, (548.97790527344, 280))

    def test_recorded_start_settles_and_keeps_firing_with_delayed_inputs(self):
        # Derived from the recorded release decay, not the game's physics
        # engine. Cover additional delay explicitly rather than claiming live
        # verification from this bounded numerical replay.
        for delay in (0, 1, 2, 4):
            with self.subTest(observation_delay=delay):
                state = self.fixture()["observation"]
                control, choice = executor(state)
                queue, shots, first_shot = ["none"]*delay, 0, None
                for frame in range(80):
                    state["frame"] += 1
                    action = control.step(state, (frame+1)/30)
                    self.assertIsNotNone(control.plan)
                    self.assertEqual(control.prop_approach.destination, (548.97790527344, 280))
                    if action.shoot != "none":
                        first_shot = frame if first_shot is None else first_shot
                        self.assertEqual(action.shoot, "left")
                        self.assertLessEqual(math.hypot(state["player"]["vx"], state["player"]["vy"]), .5)
                        shots += 1
                    queue.append(action.move)
                    vector = _VECTORS[queue.pop(0)]
                    player = state["player"]
                    start = (player["x"], player["y"])
                    end = (start[0]+player["vx"]*1.88, start[1]+player["vy"]*1.88)
                    boxes = _geometry(state, _parsed(state), _selected(state, choice))[1]
                    self.assertTrue(_clear(start, end, boxes))
                    player.update(x=end[0], y=end[1], vx=.775*player["vx"]+.99*vector[0],
                                  vy=.775*player["vy"]+.99*vector[1])
                self.assertLess(first_shot, 25)
                self.assertGreater(shots, 50)

    def test_away_and_back_never_renews_a_settled_prop_watchdog(self):
        state = props(poop(), x=200)
        control, _ = executor(state)
        control.step(state, .1)
        for now, x in ((1, 220), (2, 200), (3, 220), (3.9, 200)):
            state["frame"] += 1
            state["player"]["x"] = x
            control.step(state, now)
            self.assertEqual(control.prop_approach.destination, (200, 280))
            self.assertEqual(control.progress_at, .1)
        self.assertIn("no observed progress", control.step(state, 4.11).status)
        self.assertIsNone(control.plan)

    def test_only_a_new_best_distance_to_the_stable_waypoint_renews_progress(self):
        state = props(poop(), x=200, y=330)
        control, _ = executor(state)
        control.step(state, .1)
        for now, y, expected in ((.5, 310, .5), (2, 330, .5), (3.9, 310, .5)):
            state["frame"] += 1
            state["player"]["y"] = y
            control.step(state, now)
            self.assertEqual(control.progress_at, expected)
        self.assertIn("no observed progress", control.step(state, 4.51).status)

    def test_only_damage_not_healing_or_state_regression_renews_progress(self):
        for target, field, values in ((fire(), "hp", (14, 15, 14)),
                                      (poop(), "state", (500, 250, 500))):
            with self.subTest(kind=target["kind"]):
                state = props(target, x=200)
                control, _ = executor(state)
                control.step(state, .1)
                for now, value in zip((1, 2, 4.9), values):
                    state["hazards"][0][field] = value
                    control.step(state, now)
                    self.assertEqual(control.progress_at, 1)
                self.assertIn("no observed progress", control.step(state, 5.01).status)

    def test_new_pickup_at_fixed_destination_cancels_instead_of_switching_sides(self):
        state = props(poop(), x=200, y=330)
        control, _ = executor(state)
        control.step(state, .1)
        state["pickups"] = [pickup("new-paid", x=200, y=280, price=5, shop_item=True)]
        action = control.step(state, .2)
        self.assertEqual((action.move, action.shoot), ("none", "none"))
        self.assertIsNone(control.plan)
        self.assertEqual(control.completed, 0)

    def test_only_complete_unchanged_context_can_confirm_destruction(self):
        changes = (
            (lambda s: s["hazards"].clear(), True),
            (lambda s: s["hazards"][0].update(state=1000, collision=0, tear_destructible=False), True),
            (lambda s: s["hazards"][0].update(tear_destructible=False), False),
            (lambda s: s["hazards"][0].update(state=True), False),
            (lambda s: s["hazards"][0].pop("index"), False),
            (lambda s: (s["hazards"].clear(), s.update(truncated=True)), False),
            (lambda s: (s["hazards"].clear(), s["floor"].update(id="new-floor")), False),
        )
        for change, success in changes:
            state = props(poop(), x=200)
            control, _ = executor(state)
            control.step(state, .1)
            change(state)
            control.step(state, .2)
            self.assertEqual(control.completed, int(success))
            self.assertIsNone(control.plan)


if __name__ == "__main__":
    unittest.main()
