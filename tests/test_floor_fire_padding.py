"""Observed Technology run: a live fire's extra buffer stranded every exit."""
import copy
import json
import math
from pathlib import Path
import unittest

from jev_isaac.exploration import _door_move, _room_geometry, _validated
from jev_isaac.navigation import _VECTORS, _clear, _free
from jev_isaac.player_navigation import PlayerNavigator


class FirePaddingTests(unittest.TestCase):
    def setUp(self):
        self.state = json.loads((Path(__file__).parent / "fixtures" /
                                 "floor-fire-padding.json").read_text(encoding="utf-8-sig"))

    def move(self, slot=7):
        parsed = _validated(self.state, allow_shop=True, allow_secret=True)
        selected = next(door for door in parsed[-1] if door.slot == slot)
        return _door_move(self.state, parsed, selected)

    def test_observed_living_fire_allows_outward_step_for_each_selected_exit(self):
        before = copy.deepcopy(self.state)
        self.assertEqual(self.state["player"]["weapon_type"], 3)
        fire = next(h for h in self.state["hazards"] if h["kind"] == "fire" and h["x"] == 920)
        self.assertEqual(fire["hp"], 2)
        for slot in (0, 2, 7):
            with self.subTest(slot=slot):
                move = self.move(slot)
                self.assertIn(move, ("up", "up_left", "up_right"))
                start = self.state["player"]["x"], self.state["player"]["y"]
                vector = _VECTORS[move]
                distance = math.dist(start, (fire["x"], fire["y"]))
                for tick in range(1, 25):
                    end = start[0]+vector[0]*tick, start[1]+vector[1]*tick
                    self.assertGreater(math.dist(end, (fire["x"], fire["y"])), distance)
                    distance = math.dist(end, (fire["x"], fire["y"]))
                boxes, _, _, _ = _room_geometry(self.state, 10)
                self.assertTrue(_free(end, boxes))
        self.assertEqual(self.state, before)

    def test_player_executes_selected_door_without_shooting_or_abandoning_it(self):
        nav = PlayerNavigator()
        nav.step(self.state, 0)
        self.state["frame"] += 20
        nav.step(self.state, .7)
        choice = next(c for c in nav.adventure_options if c.key == "enter:7:71")
        self.assertTrue(nav.accept_adventure(choice.key, self.state, .71))
        action = nav.step(self.state, .72)
        self.assertIn(action.move, ("up", "up_left", "up_right"))
        self.assertEqual(action.shoot, "none")
        self.assertIsNone(action.stop_reason)
        self.assertEqual(nav.activity_failures, [])

    def test_contact_overlap_or_momentum_toward_fire_never_gains_permission(self):
        for position in ({"x": 920, "y": 376}, {"x": 920, "y": 400}, {"vy": 1}):
            with self.subTest(position=position):
                original = copy.deepcopy(self.state["player"])
                self.state["player"].update(position)
                self.assertIsNone(self.move())
                self.state["player"] = original

    def test_unknown_special_moving_or_unobserved_fire_keeps_full_exclusion(self):
        fire = next(h for h in self.state["hazards"] if h["kind"] == "fire" and h["x"] == 920)
        original = dict(fire)
        for change in ({"variant": 2}, {"variant": True}, {"type": 32}, {"vx": 1},
                       {"hp": 0}, {"hp": 6}, {"max_hp": None}, {"tear_destructible": False},
                       {"radius": None}, {"vy": None}):
            with self.subTest(change=change):
                fire.update(change)
                # Malformed observations must be rejected before planning.
                parsed = _validated(self.state, allow_shop=True)
                if parsed is not None:
                    self.assertIsNone(self.move())
                fire.clear()
                fire.update(original)

    def test_another_hazard_at_the_same_point_is_not_removed_with_fire_buffer(self):
        for kind in ("bomb", "laser", "creep", "fire"):
            with self.subTest(kind=kind):
                self.state["hazards"].append({"kind": kind, "x": 920, "y": 400,
                                              "radius": 13, "vx": 0, "vy": 0})
                self.assertIsNone(self.move())
                self.state["hazards"].pop()

    def test_paid_pickup_and_wall_obstacles_remain_excluded(self):
        player = self.state["player"]
        self.state["pickups"].append({"id": "paid", "type": 5, "variant": 20, "subtype": 1,
            "x": player["x"], "y": player["y"], "radius": 10, "wait": 0,
            "price": 5, "shop_item": True, "options_index": 0, "collectible_kind": 0})
        self.assertIsNone(self.move())
        self.state["pickups"].pop()
        self.state["hazards"].append({"kind": "grid", "type": 15, "variant": 0, "state": 0,
            "collision": 4, "x": player["x"], "y": 345, "radius": 20})
        self.assertIsNone(self.move())

    def test_escape_resumes_the_selected_door_route_around_the_still_living_fire(self):
        for tick in range(200):
            parsed = _validated(self.state, allow_shop=True)
            door = next(d for d in parsed[-1] if d.slot == 7)
            move = _door_move(self.state, parsed, door)
            self.assertIsNotNone(move, f"route failed at step {tick}")
            self.assertNotEqual(move, "none")
            start, vector = parsed[3], _VECTORS[move]
            end = start[0]+vector[0]*4, start[1]+vector[1]*4
            boxes, _, _, _ = _room_geometry(self.state, parsed[4], door=door)
            if _free(start, boxes):
                self.assertTrue(_clear(start, end, boxes))
            self.state["player"].update(x=end[0], y=end[1], vx=0, vy=0)
            self.state["frame"] += 1
            if end[1] > 440:
                self.assertLessEqual(abs(end[0]-840), 10)
                return
        self.fail("Never reached Jev's selected doorway")


if __name__ == "__main__":
    unittest.main()
