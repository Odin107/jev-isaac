"""Moving trap routes, including a passive capture of the user's room 86.

The captured snapshot is from the same run, after the reported stop and a
manual move to room center. It is not presented as the exact failing frame.
All tests run offline; no game inputs or model requests are made.
"""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_exploration import door, grid, state
from test_clear_room_hazards import npc
from test_pickups import observed, pickup
from jev_isaac.exploration import (
    FloorNavigator, _door_move, _moving_npc_wait, _room_geometry, _validated,
)
from jev_isaac.navigation import _VECTORS, _clear


def blocked_route():
    data = state(doors=[door(2, 85)])
    data["enemies"] = [npc(x=560, y=280, vy=2)]
    return data


class MovingTrapRouteTests(unittest.TestCase):
    def test_captured_room_uses_safe_other_frontier_instead_of_stopping(self):
        data = json.loads((Path(__file__).parent / "fixtures" / "floor-moving-trap-room.json").read_text())
        data.update(enabled=True, paused=False)
        before = copy.deepcopy(data)
        parsed = _validated(data)
        left, down = parsed[-1]
        self.assertEqual(left.target_index, 85)
        self.assertIsNone(_door_move(data, parsed, left))
        self.assertEqual(down.target_index, 99)
        self.assertIsNotNone(_door_move(data, parsed, down))
        nav = FloorNavigator()
        self.assertEqual(nav.step(data, 0).status, "waiting for room rewards")
        data["frame"] += 19
        result = nav.step(data, .61)
        self.assertIsNone(result.stop_reason)
        self.assertEqual(result.status, "entering room 99")
        self.assertEqual(nav._pending, down)
        self.assertNotEqual(result.move, "none")
        self.assertEqual(result.shoot, "none")
        boxes, _, _, _ = _room_geometry(data, parsed[4], door=down)
        direction = _VECTORS[result.move]
        end = tuple(v + d * 12 for v, d in zip(parsed[3], direction))
        self.assertTrue(_clear(parsed[3], end, boxes))
        before["frame"] = data["frame"]
        self.assertEqual(data, before)

    def test_single_transient_route_waits_then_uses_fresh_clearance(self):
        data, nav = blocked_route(), FloorNavigator()
        first = nav.step(data, 0)
        self.assertIsNone(first.stop_reason)
        self.assertEqual(first.move, "none")
        self.assertIn("moving trap", first.status)
        data["frame"] += 1
        data["enemies"][0].update(y=160, vy=-2)
        result = nav.step(data, .5)
        self.assertIsNone(result.stop_reason)
        self.assertEqual(result.move, "right")
        self.assertEqual(result.status, "entering room 85")

    def test_persistent_blockage_is_still_bounded(self):
        data, nav = blocked_route(), FloorNavigator()
        self.assertIsNone(nav.step(data, 0).stop_reason)
        for frame, now in ((2, .8), (3, 1.9)):
            data["frame"] = frame
            self.assertIsNone(nav.step(data, now).stop_reason)
        data["frame"] = 4
        result = nav.step(data, 2.01)
        self.assertEqual(result.stop_reason, "no safe route to open door")
        self.assertEqual((result.move, result.shoot), ("none", "none"))

    def test_stationary_trap_cannot_be_removed_for_the_route_proof(self):
        for only_stationary in (False, True):
            with self.subTest(only_stationary=only_stationary):
                data = blocked_route()
                if only_stationary:
                    data["enemies"][0].update(vx=0, vy=0)
                else:
                    data["enemies"].append(npc(id="fixed", x=560, y=280))
                result = FloorNavigator().step(data, 0)
                self.assertEqual(result.stop_reason, "no safe route to open door")

    def test_fixed_door_blockers_are_not_hidden_by_a_moving_trap(self):
        for kind in ("wall", "pit", "spikes", "fire", "paid"):
            with self.subTest(kind=kind):
                data = blocked_route()
                if kind == "paid":
                    data = observed(pickup("paid", x=560, price=15, shop_item=True))
                    data["enemies"] = [npc(x=560, y=280, vy=2)]
                elif kind == "fire":
                    data["hazards"] = [{"kind": "fire", "x": 560, "y": 280, "radius": 20}]
                else:
                    typ, collision = {"wall": (15, 4), "pit": (7, 1), "spikes": (8, 0)}[kind]
                    data["hazards"] = [grid(560, 280, kind=typ, collision=collision)]
                parsed = _validated(data)
                self.assertIsNone(_moving_npc_wait(data, parsed, parsed[-1][0]))
                # No fallback input is authorized even before the pickup
                # collector's reward-settling period has elapsed.
                self.assertIsNone(_door_move(data, parsed, parsed[-1][0]))

    def test_waiting_dodge_keeps_the_fixed_boundaries(self):
        data = blocked_route()
        data["player"].update(x=500, y=280)
        data["enemies"][0].update(x=550, y=280, vx=-3, vy=0)
        # This obstacle forbids the otherwise attractive downward escape.
        data["hazards"] = [grid(500, 325, kind=7, collision=1)]
        parsed = _validated(data)
        move = _moving_npc_wait(data, parsed, parsed[-1][0])
        self.assertIsNotNone(move)
        boxes, _, _, _ = _room_geometry(data, parsed[4], door=parsed[-1][0], include_npcs=False)
        vector = _VECTORS[move]
        end = tuple(v + d * 24 for v, d in zip(parsed[3], vector))
        self.assertTrue(_clear(parsed[3], end, boxes))
        self.assertNotIn(move, ("down", "down_left", "down_right", "right"))

    def test_alternate_still_prefers_ordinary_frontier_before_boss(self):
        data = state(doors=[door(0, 83), door(2, 85), door(3, 97, 5)])
        data["enemies"] = [npc(x=80, y=280, vy=2)]
        result = FloorNavigator().step(data, 0)
        self.assertEqual(result.status, "entering room 85")
        self.assertEqual(result.move, "right")

    def test_closed_or_damage_door_cannot_be_an_alternate(self):
        for other in (door(0, 83, opened=False), door(0, 83, locked=True), door(0, 83, 10)):
            with self.subTest(other=other):
                data = blocked_route()
                data["doors"].append(other)
                result = FloorNavigator().step(data, 0)
                self.assertEqual(result.move, "none")
                self.assertIn("moving trap", result.status)

    def test_rerouting_does_not_reset_total_traversal_budget(self):
        data = state(doors=[door(0, 83), door(2, 85)])
        data["enemies"] = [npc(x=80, y=160, vy=-2)]
        nav = FloorNavigator(stuck_timeout=1.5, transition_timeout=2)
        self.assertEqual(nav.step(data, 0).status, "entering room 83")
        data["frame"] = 2
        data["enemies"][0].update(y=280, vy=2)
        self.assertEqual(nav.step(data, 1).status, "entering room 85")
        data["frame"] = 3
        self.assertEqual(nav.step(data, 2.01).stop_reason, "door traversal timed out")

    def test_pause_does_not_spend_the_short_trap_wait(self):
        data, nav = blocked_route(), FloorNavigator()
        self.assertIsNone(nav.step(data, 0).stop_reason)
        data.update(frame=2, paused=True)
        self.assertEqual(nav.step(data, .3).status, "paused")
        data.update(frame=3, paused=False)
        self.assertIsNone(nav.step(data, 10).stop_reason)
        data["frame"] = 4
        self.assertIsNone(nav.step(data, 11).stop_reason)
        data["frame"] = 5
        self.assertEqual(nav.step(data, 12.01).stop_reason, "no safe route to open door")


if __name__ == "__main__":
    unittest.main()
