"""Exact post-demolition door stop and bounded offline movement replays."""
from collections import deque
import copy
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).resolve().parents[1] / "src")]
from test_door_regression import rotate_clockwise
from test_floor_drift_regression import physics_substep
from test_pickups import pickup
from jev_isaac.exploration import FloorNavigator, _door_move, _point_waypoint, _room_geometry, _validated
from jev_isaac.navigation import _VECTORS, _clear, _free
from jev_isaac.protocol import Observation


def recorded_state(*, drift=False):
    name = "post-tnt-door-drift-stop.json" if drift else "post-tnt-door-stop.json"
    path = Path(__file__).parent / "fixtures" / name
    return Observation.decode(path.read_bytes()).data


def boss_geometry(state):
    parsed = _validated(state)
    door = next(d for d in parsed[-1] if d.target_index == 126)
    return parsed, door, _room_geometry(state, parsed[4], door=door)


class PostTntNavigationTests(unittest.TestCase):
    def test_exact_stop_has_clear_offset_portal_despite_blocked_center(self):
        state = recorded_state()
        before = copy.deepcopy(state)
        parsed, door, (boxes, _, _, phase) = boss_geometry(state)
        self.assertEqual((state["frame"], state["floor"]["room_index"]), (36648, 113))
        self.assertTrue(state["room"]["clear"])
        self.assertEqual(state["pickups"], [])
        self.assertEqual([h["kind"] for h in state["hazards"] if h["kind"] != "grid"], ["tnt"])
        self.assertFalse(_free((320, 400), boxes))
        self.assertIsNone(_point_waypoint(parsed[3], (320, 400), parsed[6], boxes, phase))
        self.assertTrue(_free((312.5, 400), boxes))
        self.assertTrue(_clear((312.5, 400), (320, 468), boxes))
        self.assertEqual(_door_move(state, parsed, door), "down_left")
        self.assertEqual(state, before)

    def test_offset_door_approach_rotates_with_the_observed_geometry(self):
        state = recorded_state()
        for orientation in range(4):
            with self.subTest(orientation=orientation):
                parsed, door, (boxes, _, _, _) = boss_geometry(state)
                move = _door_move(state, parsed, door)
                self.assertNotIn(move, (None, "none"))
                vector = _VECTORS[move]
                end = tuple(parsed[3][i]+vector[i]*12 for i in range(2))
                self.assertTrue(_clear(parsed[3], end, boxes))
            state = rotate_clockwise(state)

    def test_exact_stop_reaches_boss_with_zero_one_and_two_delayed_inputs(self):
        # This is a synthetic inertial stress model, not the game's physics.
        for delay in (0, 1, 2):
            with self.subTest(delay=delay):
                self.replay(delay=delay, two_substeps=False)

    def test_recorded_stop_reaches_boss_with_recording_calibrated_two_substeps(self):
        # This model comes from the earlier recorded floor-drift regression.
        self.replay(delay=1, two_substeps=True)

    def test_first_drifting_door_stop_also_reaches_the_boss(self):
        state = recorded_state(drift=True)
        self.assertEqual(state["frame"], 36380)
        self.assertNotEqual(state["player"]["vx"], 0)
        for delay in (0, 1, 2):
            with self.subTest(delay=delay):
                self.replay(delay=delay, two_substeps=False, drift=True)
        self.replay(delay=1, two_substeps=True, drift=True)

    def replay(self, *, delay, two_substeps, drift=False):
        state = recorded_state(drift=drift)
        navigator = FloorNavigator()
        pending = deque(["none"]*delay)
        for tick in range(240):
            state["frame"] += 1
            action = navigator.step(state, tick/30)
            self.assertIsNone(action.stop_reason, (tick, state["player"], action))
            self.assertEqual(action.shoot, "none")
            self.assertIn(action.status, ("waiting for room rewards", "entering room 126"))
            pending.append(action.move)
            applied = _VECTORS[pending.popleft()]
            boxes = boss_geometry(state)[2][0]
            player = state["player"]
            for substep in range(2 if two_substeps else 1):
                start = player["x"], player["y"]
                if two_substeps:
                    physics_substep(player, applied)
                else:
                    player["vx"] = .65*player["vx"]+1.4*applied[0]
                    player["vy"] = .65*player["vy"]+1.4*applied[1]
                    player["x"] += player["vx"]
                    player["y"] += player["vy"]
                end = player["x"], player["y"]
                self.assertTrue(_clear(start, end, boxes), (tick, substep, start, end))
                if end[1] > 448:
                    self.assertLessEqual(abs(end[0]-320), 10)
                    incoming = copy.deepcopy(state)
                    incoming.update(frame=state["frame"]+1, room_id="post-tnt:boss-arrival")
                    incoming["floor"].update(room_index=126, room_list_index=13)
                    incoming["room"].update(type=5, clear=False)
                    incoming["player"].update(x=320, y=280, vx=0, vy=0)
                    incoming["doors"] = []
                    self.assertEqual(navigator.step(incoming, (tick+1)/30).status, "combat")
                    self.assertEqual(navigator.stats["doors_traversed"], 1)
                    return
        self.fail("Did not reach the observed boss doorway within the replay budget")

    def test_blocked_mouth_never_ignores_paid_pickups_pits_or_walls(self):
        for blocker in (
                {"kind": "grid", "type": 7, "collision": 1, "x": 320, "y": 400, "radius": 20},
                {"kind": "grid", "type": 17, "collision": 0, "x": 320, "y": 400, "radius": 20},
                {"kind": "grid", "type": 15, "variant": 0, "state": 0,
                 "collision": 4, "x": 320, "y": 440, "radius": 20},
                pickup("paid-at-mouth", x=320, y=400, price=5, shop_item=True),
                pickup("unselected-trinket", x=320, y=400, variant=350, subtype=41)):
            with self.subTest(blocker=blocker):
                state = recorded_state()
                state["hazards" if "kind" in blocker else "pickups"].append(blocker)
                parsed, door, _ = boss_geometry(state)
                self.assertIsNone(_door_move(state, parsed, door))

    def test_fresh_geometry_replans_offset_approach_without_removing_tnt(self):
        state = recorded_state()
        parsed, door, _ = boss_geometry(state)
        self.assertIsNotNone(_door_move(state, parsed, door))
        tnt = next(h for h in state["hazards"] if h["kind"] == "tnt")
        tnt["x"] = 284.54568481445  # The same obstruction now covers the left side.
        parsed, door, (boxes, _, _, _) = boss_geometry(state)
        self.assertFalse(_free((320, 400), boxes))
        self.assertTrue(_clear((327.5, 400), (320, 468), boxes))
        self.assertIsNotNone(_door_move(state, parsed, door))
        self.assertEqual(len([h for h in state["hazards"] if h["kind"] == "tnt"]), 1)


if __name__ == "__main__":
    unittest.main()
