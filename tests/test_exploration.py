"""Offline floor and doorway scenarios; no API or live game controls."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_isaac.exploration import FloorNavigator
from jev_isaac.navigation import _VECTORS, _clear


def door(slot, target, kind=1, *, opened=True, locked=False):
    x, y = ((40, 280), (320, 120), (600, 280), (320, 440))[slot % 4]
    return {"slot": slot, "x": x, "y": y, "open": opened, "locked": locked,
            "target_index": target, "target_type": kind}


def state(index=84, *, frame=1, kind=1, clear=True, doors=None, x=320, y=280):
    return {"session": "run:arm1", "run_id": "run", "room_id": f"visit:{index}",
            "frame": frame, "enabled": True, "paused": False, "truncated": False,
            "floor": {"id": "stage1:seed", "room_index": index},
            "room": {"type": kind, "clear": clear,
                     "top_left": {"x": 60, "y": 140}, "bottom_right": {"x": 580, "y": 420}},
            "player": {"x": x, "y": y, "vx": 0, "vy": 0, "radius": 10, "dead": False},
            "doors": list(doors or []), "enemies": [], "projectiles": [], "hazards": []}


def grid(x, y, *, kind=2, collision=3):
    return {"kind": "grid", "type": kind, "collision": collision,
            "radius": 20, "x": x, "y": y}


class ExplorationTests(unittest.TestCase):
    def test_clear_room_approaches_and_crosses_open_door(self):
        nav = FloorNavigator()
        observed = state(doors=[door(2, 85)])
        for frame in range(1, 90):
            observed["frame"] = frame
            action = nav.step(observed, frame/30)
            self.assertIsNone(action.stop_reason)
            self.assertEqual(action.shoot, "none")
            self.assertEqual(action.move, "right")
            observed["player"]["x"] += 4
            if observed["player"]["x"] > 608:
                break
        else:
            self.fail("Navigator never crossed the door boundary")
        incoming = state(85, frame=100, clear=False, doors=[door(0, 84, opened=False)])
        self.assertEqual(nav.step(incoming, 4).status, "combat")
        self.assertEqual(nav.stats, {"rooms_visited": 2, "rooms_cleared": 1,
                                     "doors_traversed": 1, "boss_cleared": False})

    def test_all_directions_and_second_door_slots(self):
        for slot in range(8):
            nav = FloorNavigator()
            observed = state(doors=[door(slot, 85)])
            result = nav.step(observed, 0)
            self.assertEqual(result.move, ("left", "up", "right", "down")[slot % 4])

    def test_obstacle_route_is_replanned_without_crossing_rocks(self):
        nav = FloorNavigator()
        observed = state(doors=[door(2, 85)], x=240, y=280)
        observed["hazards"] = [grid(400, 280)]
        boxes = [(370, 250, 430, 310)]
        for frame in range(1, 180):
            observed["frame"] = frame
            action = nav.step(observed, frame/30)
            self.assertIsNone(action.stop_reason)
            self.assertNotEqual(action.move, "none")
            vector = _VECTORS[action.move]
            start = observed["player"]["x"], observed["player"]["y"]
            end = start[0]+vector[0]*4, start[1]+vector[1]*4
            self.assertTrue(_clear(start, end, boxes))
            observed["player"].update(x=end[0], y=end[1])
            if end[0] > 608:
                break
        else:
            self.fail("Navigator failed to reach the doorway around the rock")

    def test_only_selected_open_door_grid_is_crossable(self):
        nav = FloorNavigator()
        observed = state(doors=[door(2, 85)], x=560)
        observed["hazards"] = [grid(600, 280, kind=16, collision=4),
                                grid(600, 240, kind=15, collision=4),
                                grid(600, 320, kind=15, collision=4)]
        self.assertEqual(nav.step(observed, 0).move, "right")
        observed = state(doors=[door(2, 85)], x=560)
        observed["hazards"] = [grid(600, 280, kind=15, collision=4)]
        self.assertEqual(FloorNavigator().step(observed, 0).stop_reason, "no safe route to open door")

    def test_door_corridor_requires_alignment_and_does_not_cut_wall(self):
        nav = FloorNavigator()
        observed = state(doors=[door(2, 85)], x=560, y=310)
        observed["hazards"] = [grid(600, 240, kind=15, collision=4),
                                grid(600, 320, kind=15, collision=4)]
        result = nav.step(observed, 0)
        self.assertIn(result.move, ("up", "up_left"))
        observed["frame"] += 1
        observed["player"].update(x=584, y=310)
        self.assertEqual(nav.step(observed, .1).stop_reason, "no safe route to open door")

    def test_backtracks_to_other_known_branch_before_boss(self):
        nav = FloorNavigator()
        a = state(84, doors=[door(0, 83), door(2, 85), door(3, 97, 5)])
        self.assertEqual(nav.step(a, 0).status, "entering room 83")
        b = state(83, frame=2, doors=[door(2, 84)])
        self.assertEqual(nav.step(b, .1).status, "entering room 84")
        a["frame"], a["room_id"] = 3, "visit:84:again"
        self.assertEqual(nav.step(a, .2).status, "entering room 85")
        c = state(85, frame=4, doors=[door(0, 84)])
        self.assertEqual(nav.step(c, .3).status, "entering room 84")
        a["frame"], a["room_id"] = 5, "visit:84:third"
        self.assertEqual(nav.step(a, .4).status, "entering room 97")
        boss = state(97, frame=6, kind=5, clear=False, doors=[door(1, 84, opened=False)])
        self.assertEqual(nav.step(boss, .5).status, "combat")
        boss["frame"] = 7
        boss["room"]["clear"] = True
        boss["doors"][0]["open"] = True
        action = nav.step(boss, .6)
        self.assertEqual(action.stop_reason, "floor cleared")
        self.assertEqual(action.move, "none")
        self.assertEqual(nav.stats["rooms_cleared"], 4)
        self.assertTrue(nav.stats["boss_cleared"])

    def test_remote_normal_frontier_beats_local_boss(self):
        nav = FloorNavigator()
        a = state(84, doors=[door(0, 83), door(2, 85)])
        nav.step(a, 0)
        b = state(83, frame=2, doors=[door(2, 84), door(0, 82, 5)])
        self.assertEqual(nav.step(b, .1).status, "entering room 84")

    def test_large_room_entry_quadrants_share_one_visited_room(self):
        nav = FloorNavigator()
        large = state(84, doors=[door(2, 86)])
        large["floor"]["room_list_index"] = 1
        self.assertEqual(nav.step(large, 0).status, "entering room 86")
        neighbor = state(86, frame=2, doors=[door(0, 85)])
        neighbor["floor"]["room_list_index"] = 2
        self.assertEqual(nav.step(neighbor, .1).status, "entering room 85")
        # Returning through another side reports entry quadrant85 for the
        # same large room previously entered at84. Only two rooms were visited.
        large["floor"]["room_index"] = 85
        large.update(frame=3, room_id="large:secondvisit")
        result = nav.step(large, .2)
        self.assertIsNone(result.stop_reason)
        self.assertEqual(result.move, "none")
        self.assertEqual(nav.stats["rooms_visited"], 2)
        self.assertEqual(nav.stats["doors_traversed"], 2)

    def test_observe_exposes_stop_before_a_combat_request(self):
        nav = FloorNavigator()
        observed = state(clear=False)
        nav.observe(observed)
        self.assertIsNone(nav.stop_reason)
        observed["truncated"] = True
        nav.observe(observed)
        self.assertEqual(nav.stop_reason, "incomplete floor observation")

    def test_open_treasure_room_is_allowed_but_not_locked_treasure(self):
        observed = state(doors=[door(0, 83, 4, locked=True), door(2, 85, 4)])
        self.assertEqual(FloorNavigator().step(observed, 0).status, "entering room 85")

    def test_special_locked_closed_and_invalid_destinations_are_excluded(self):
        for excluded in (door(0, 83, 2), door(0, 83, 7), door(0, 83, 10),
                         door(0, 83, 11), door(0, 83, 13), door(0, 83, 14),
                         door(0, 83, locked=True), door(0, 83, opened=False),
                         door(0, -1), door(0, 169)):
            observed = state(doors=[excluded, door(2, 85)])
            self.assertEqual(FloorNavigator().step(observed, 0).status, "entering room 85")

    def test_missing_open_doors_waits_then_stops_without_claiming_floor_clear(self):
        nav = FloorNavigator()
        observed = state()
        self.assertIsNone(nav.step(observed, 0).stop_reason)
        observed["frame"] += 1
        self.assertEqual(nav.step(observed, 2.1).stop_reason, "no accessible unexplored rooms")

    def test_door_opening_animation_is_given_time(self):
        nav = FloorNavigator()
        observed = state(doors=[door(2, 85, opened=False)])
        self.assertEqual(nav.step(observed, 0).move, "none")
        observed["frame"] = 2
        observed["doors"][0]["open"] = True
        self.assertEqual(nav.step(observed, .5).move, "right")

    def test_stuck_and_total_transition_watchdogs(self):
        nav = FloorNavigator(stuck_timeout=1, transition_timeout=2)
        observed = state(doors=[door(2, 85)])
        nav.step(observed, 0)
        observed["frame"] = 2
        self.assertEqual(nav.step(observed, 1.1).stop_reason, "stuck while approaching door")
        nav = FloorNavigator(stuck_timeout=1, transition_timeout=2)
        observed = state(doors=[door(2, 85)])
        for frame in range(1, 5):
            observed["frame"] = frame
            observed["player"]["x"] += 12
            result = nav.step(observed, (frame-1)*.75)
        self.assertEqual(result.stop_reason, "door traversal timed out")

    def test_monotonic_clock_handles_long_computer_uptime(self):
        observed = state(doors=[door(2, 85)])
        self.assertEqual(FloorNavigator().step(observed, 100_000_000).move, "right")
        for value in (float("inf"), float("nan"), True):
            self.assertEqual(FloorNavigator().step(observed, value).stop_reason, "invalid exploration clock")

    def test_closing_door_invalidates_pending_traversal(self):
        nav = FloorNavigator()
        observed = state(doors=[door(2, 85)])
        nav.step(observed, 0)
        observed["frame"] = 2
        observed["doors"][0]["locked"] = True
        self.assertEqual(nav.step(observed, .1).move, "none")

    def test_pause_preserves_expected_arrival_even_on_same_frame(self):
        nav = FloorNavigator()
        observed = state(doors=[door(2, 85)])
        nav.step(observed, 0)
        paused = copy.deepcopy(observed)
        paused.update(paused=True, floor_transition=True)
        self.assertEqual(nav.step(paused, 5).move, "none")
        arriving = state(85, frame=1, clear=False)
        nav.observe(arriving)
        arriving["frame"] = 2
        self.assertEqual(nav.step(arriving, 6).status, "combat")
        self.assertEqual(nav.stats["doors_traversed"], 1)

    def test_pause_time_does_not_trigger_stuck_watchdog(self):
        nav = FloorNavigator()
        observed = state(doors=[door(2, 85)])
        nav.step(observed, 0)
        observed["paused"] = True
        nav.observe(observed)
        observed.update(paused=False, frame=2)
        self.assertEqual(nav.step(observed, 30).move, "right")

    def test_freshness_identity_and_unexpected_arrival_guards(self):
        for change, reason in (
                (lambda s: s.update(frame=0), "stale floor observation"),
                (lambda s: s["floor"].update(id="floor2"), "floor or run changed"),
                (lambda s: s.update(run_id="other"), "floor or run changed"),
                (lambda s: s.update(room_id="another visit"), "unexpected room visit"),
                (lambda s: s["floor"].update(room_index=90), "unexpected room transition")):
            nav = FloorNavigator()
            observed = state(doors=[door(2, 85)])
            nav.step(observed, 0)
            observed["frame"] = 2
            change(observed)
            self.assertEqual(nav.step(observed, .1).stop_reason, reason)

    def test_observe_then_step_is_valid_but_duplicate_step_is_neutral(self):
        nav = FloorNavigator()
        observed = state(doors=[door(2, 85)])
        nav.observe(observed)
        self.assertEqual(nav.step(observed, 0).move, "right")
        self.assertEqual(nav.step(observed, .01).move, "none")

    def test_combat_or_disarmed_state_never_navigates(self):
        for change in (lambda s: s.update(enabled=False), lambda s: s.update(paused=True),
                       lambda s: s["room"].update(clear=False)):
            observed = state(doors=[door(2, 85)])
            change(observed)
            self.assertEqual(FloorNavigator().step(observed, 0).move, "none")

    def test_incomplete_or_conflicting_observations_never_mark_clear(self):
        for change in (lambda s: s.update(truncated=True),
                       lambda s: s.update(truncated_arrays={"doors": True}),
                       lambda s: s.pop("hazards"),
                       lambda s: s.pop("doors"),
                       lambda s: s["room"].pop("clear"),
                       lambda s: s["room"].update(clear=1),
                       lambda s: s["player"].update(x=float("nan")),
                       lambda s: s["doors"][0].update(target_index=True),
                       lambda s: s["enemies"].append({"x": 200, "y": 200, "hp": 10})):
            observed = state(doors=[door(2, 85)])
            change(observed)
            nav = FloorNavigator()
            result = nav.step(observed, 0)
            self.assertEqual(result.move, "none")
            self.assertIsNotNone(result.stop_reason)
            self.assertEqual(nav.stats["rooms_cleared"], 0)

    def test_floor_exits_are_obstacles_even_without_collision(self):
        for kind in (8, 9, 17, 18, 23):
            observed = state(doors=[door(2, 85)], x=360)
            observed["hazards"] = [grid(400, 280, kind=kind, collision=0)]
            result = FloorNavigator().step(observed, 0)
            self.assertIsNone(result.stop_reason)
            self.assertNotEqual(result.move, "right")

    def test_lingering_projectiles_dodge_without_pursuit_and_input_is_unchanged(self):
        observed = state(doors=[door(2, 85)])
        observed["projectiles"] = [{"x": 360, "y": 280, "vx": -6, "vy": 0, "radius": 5}]
        before = copy.deepcopy(observed)
        result = FloorNavigator().step(observed, 0)
        self.assertEqual(result.status, "dodging lingering projectiles")
        self.assertIn(result.move, ("up", "down", "up_left", "down_left"))
        self.assertEqual(result.shoot, "none")
        self.assertEqual(observed, before)


if __name__ == "__main__":
    unittest.main()
