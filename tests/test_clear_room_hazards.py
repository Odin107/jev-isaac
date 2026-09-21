"""Clear-room NPC and stationary TNT regressions; no game or API actions."""
import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_exploration import door, grid, state
from test_pickups import observed, pickup, ready
from jev_isaac.exploration import FloorNavigator
from jev_isaac.navigation import _VECTORS, _clear


def npc(**extra):
    # WallHugger retains positive health after the room is clear. Vulnerability
    # alone must never grant permission to navigate around an ordinary enemy.
    value = {"id": "wall-hugger", "type": 218, "variant": 0, "subtype": 0,
             "x": 360, "y": 280, "vx": 0, "vy": 0, "radius": 16,
             "hp": 100, "max_hp": 100, "vulnerable": False,
             "keeps_doors_closed": False}
    value.update(extra)
    return value


def circle_clearance(player, entity):
    return math.hypot(player["x"]-entity["x"], player["y"]-entity["y"])-player["radius"]-entity["radius"]


class ClearRoomHazardTests(unittest.TestCase):
    def test_live_npc_explicitly_not_holding_doors_allows_clear_navigation(self):
        data = state(x=220, doors=[door(2, 85)])
        data["enemies"] = [npc()]
        before = copy.deepcopy(data)
        action = FloorNavigator().step(data, 0)
        self.assertIsNone(action.stop_reason)
        self.assertNotEqual(action.move, "none")
        self.assertEqual(action.shoot, "none")
        self.assertEqual(data, before)

    def test_missing_or_true_door_flag_still_conflicts_even_if_invulnerable(self):
        for missing in (False, True):
            data = state(doors=[door(2, 85)])
            entity = npc(keeps_doors_closed=True)
            if missing:
                entity.pop("keeps_doors_closed")
            data["enemies"] = [entity]
            nav = FloorNavigator()
            action = nav.step(data, 0)
            self.assertEqual(action.stop_reason, "conflicting room clear state")
            self.assertEqual(action.move, "none")
            self.assertEqual(nav.stats["rooms_cleared"], 0)

    def test_malformed_door_flags_reject_the_observation(self):
        for flag in (0, 1, None, "false", [], {}):
            with self.subTest(flag=flag):
                data = state(doors=[door(2, 85)])
                data["enemies"] = [npc(keeps_doors_closed=flag)]
                action = FloorNavigator().step(data, 0)
                self.assertEqual(action.move, "none")
                self.assertEqual(action.stop_reason, "incomplete floor observation")

    def test_clear_room_exception_does_not_navigate_during_combat(self):
        data = state(clear=False, doors=[door(2, 85)])
        data["enemies"] = [npc()]
        action = FloorNavigator().step(data, 0)
        self.assertEqual(action.status, "combat")
        self.assertEqual(action.move, "none")

    def test_door_route_avoids_static_alive_npc_body(self):
        self._traverse(moving=False, target_pickup=False)

    def test_pickup_route_avoids_alive_npc_then_resumes_door(self):
        self._traverse(moving=False, target_pickup=True)

    def test_moving_npc_replans_without_crossing_its_body(self):
        self._traverse(moving=True, target_pickup=False)

    def test_stationary_tnt_is_avoided_without_waiting_for_it_to_disappear(self):
        for target_pickup in (False, True):
            with self.subTest(target_pickup=target_pickup):
                self._traverse(moving=False, target_pickup=target_pickup, tnt=True)

    def test_moving_npc_escape_preserves_pit_rock_and_paid_pickup_boundaries(self):
        for kind in ("pit", "rock", "paid"):
            for target_pickup in (False, True):
                with self.subTest(kind=kind, target_pickup=target_pickup):
                    data = observed(x=327.6, y=319.6)
                    data["enemies"] = [npc(x=360, y=276, vy=.5)]
                    if target_pickup:
                        data["pickups"].append(pickup("reward", x=480))
                    if kind == "paid":
                        data["pickups"].append(pickup("paid", x=300, y=350,
                                                        price=15, shop_item=True))
                        box = (272, 322, 328, 378)
                    else:
                        data["hazards"] = [grid(300, 350, kind=7 if kind == "pit" else 2,
                                                collision=1 if kind == "pit" else 3)]
                        box = (270, 320, 330, 380)
                    # Without the fixed obstacle the escape heads down-left,
                    # directly through this obstacle. Both collection and door
                    # navigation must choose another short safe direction.
                    action = ready(FloorNavigator(), data)
                    self.assertIsNone(action.stop_reason)
                    self.assertNotEqual(action.move, "none")
                    vector = _VECTORS[action.move]
                    start = data["player"]["x"], data["player"]["y"]
                    end = start[0]+vector[0]*24, start[1]+vector[1]*24
                    self.assertTrue(_clear(start, end, [box]), (kind, action, end))

    def test_npc_escape_cannot_rescue_an_overlap_with_a_fixed_blocker(self):
        for kind in ("pit", "rock", "paid"):
            with self.subTest(kind=kind):
                data = observed(x=327.6, y=319.6)
                data["enemies"] = [npc(x=360, y=276, vy=.5)]
                if kind == "paid":
                    data["pickups"].append(pickup("paid", x=327.6, y=319.6,
                                                    price=15, shop_item=True))
                else:
                    data["hazards"] = [grid(327.6, 319.6, kind=7 if kind == "pit" else 2,
                                            collision=1 if kind == "pit" else 3)]
                action = ready(FloorNavigator(), data)
                self.assertEqual(action.move, "none")
                self.assertEqual(action.stop_reason, "no safe route to open door")

    def _traverse(self, *, moving, target_pickup, tnt=False):
        data = observed(pickup("reward", x=480), x=200) if target_pickup else state(x=200, doors=[door(2, 85)])
        obstacle = npc(y=240 if moving else 280, vy=.5 if moving else 0)
        if tnt:
            obstacle = {"id": "tnt", "kind": "tnt", "type": 292,
                        "x": 360, "y": 280, "vx": 0, "vy": 0, "radius": 20}
            data["hazards"] = [obstacle]
        else:
            data["enemies"] = [obstacle]
        player, navigator, collected = data["player"], FloorNavigator(), False
        deviation = 0
        for tick in range(220):
            data["frame"] = tick+1
            action = navigator.step(data, tick/30)
            self.assertIsNone(action.stop_reason, (tick, action, player, obstacle))
            self.assertNotEqual(action.status, "waiting for lingering hazards")
            self.assertEqual(action.shoot, "none")
            direction = _VECTORS[action.move]
            # Check both bodies at substeps, rather than only testing a returned
            # direction or checking the planner's own predicted obstacle box.
            for _ in range(2):
                player["x"] += 2*direction[0]
                player["y"] += 2*direction[1]
                if moving:
                    obstacle["y"] += obstacle["vy"]
                    if obstacle["y"] >= 340:
                        obstacle["vy"] = -.5
                    elif obstacle["y"] <= 220:
                        obstacle["vy"] = .5
                self.assertGreaterEqual(circle_clearance(player, obstacle), -1e-6,
                                        (tick, action.move, player, obstacle))
                deviation = max(deviation, abs(player["y"]-280))
                if target_pickup and not collected and math.dist((player["x"], player["y"]), (480, 280)) <= 20:
                    collected = True
                    data["pickups"].clear()
            if player["x"] > 608:
                self.assertGreater(deviation, 15, "Route did not visibly avoid the obstacle")
                if target_pickup:
                    self.assertTrue(collected, "Left room before reaching reward")
                    self.assertEqual(navigator.pickup_stats["targets_disappeared_or_changed"], 1)
                return
        self.fail("Did not safely traverse around persistent clear-room hazard")


if __name__ == "__main__":
    unittest.main()
