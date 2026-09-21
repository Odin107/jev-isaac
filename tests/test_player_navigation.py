"""No activity, puzzle shot or alternate door starts without Jev's selection."""
import copy
from collections import deque
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).resolve().parents[1]/"src")]
from test_adventure import ready
from test_exploration import door, grid
from test_pickups import pickup
from test_switches import required, plate
from test_tnt_regressions import recorded
from test_post_tnt_navigation import recorded_state, boss_geometry
from test_floor_drift_regression import physics_substep
from jev_isaac.player_navigation import PlayerNavigator
from jev_isaac.navigation import _VECTORS, _clear


def offers(nav, data):
    first = nav.step(data, 0)
    assert (first.move, first.shoot) == ("none", "none")
    data["frame"] += 18
    nav.step(data, .6)
    return nav.adventure_options


class PlayerNavigationTests(unittest.TestCase):
    def test_no_free_pickup_or_open_door_is_selected_while_waiting(self):
        data = ready(pickup())
        nav = PlayerNavigator()
        choices = offers(nav, data)
        self.assertTrue(any(c.kind == "collect" for c in choices))
        self.assertTrue(any(c.kind == "enter_door" for c in choices))
        self.assertIsNone(nav._intent)
        self.assertIsNone(nav._pending)
        self.assertEqual(nav.pickup_stats["attempts"], 0)
        nav.accept_adventure(None, data, .7)
        for step in range(1, 20):
            data["frame"] += 1
            action = nav.step(data, .7+step/30)
            self.assertEqual((action.move, action.shoot), ("none", "none"))
        self.assertIsNone(nav._pending)

    def test_jev_can_choose_boss_before_other_rooms_or_nearby_supplies(self):
        data = ready(pickup(x=240, y=180))
        data["doors"] = [door(0, 83), door(2, 85, kind=5)]
        nav = PlayerNavigator()
        chosen = next(c for c in offers(nav, data) if c.kind == "enter_door" and c.details["target_type"] == 5)
        self.assertTrue(nav.accept_adventure(chosen.key, data, .7))
        data["frame"] += 1
        action = nav.step(data, .8)
        self.assertEqual(action.move, "right")
        self.assertEqual(action.shoot, "none")
        self.assertEqual(nav._pending.target_index, 85)
        self.assertEqual(nav.pickup_stats["attempts"], 0)

    def test_collect_is_selected_then_completion_returns_control_to_jev(self):
        data = ready(pickup())
        nav = PlayerNavigator()
        selected = next(c for c in offers(nav, data) if c.kind == "collect")
        self.assertTrue(nav.accept_adventure(selected.key, data, .7))
        data["frame"] += 1
        self.assertEqual(nav.step(data, .8).move, "left")
        data["pickups"] = []
        data["frame"] += 1
        action = nav.step(data, .9)
        self.assertEqual((action.move, action.shoot), ("none", "none"))
        self.assertIsNone(nav._intent)
        self.assertIsNone(nav._pending)
        data["frame"] += 15
        self.assertEqual(nav.step(data, 1.4).move, "none")
        self.assertTrue(nav.adventure_options)

    def test_closed_selected_door_does_not_choose_another_open_door(self):
        data = ready()
        data["doors"] = [door(0, 83), door(2, 85)]
        nav = PlayerNavigator()
        selected = next(c for c in offers(nav, data) if c.key == "enter:2:85")
        nav.accept_adventure(selected.key, data, .7)
        data["doors"][1]["open"] = False
        data["frame"] += 1
        action = nav.step(data, .8)
        self.assertEqual((action.move, action.shoot), ("none", "none"))
        self.assertIsNone(action.stop_reason)
        self.assertIsNone(nav._pending)
        self.assertEqual(nav.player_events[-1]["event"], "failed")

    def test_unoffered_or_changed_destination_cannot_execute(self):
        data = ready()
        nav = PlayerNavigator()
        offers(nav, data)
        self.assertFalse(nav.accept_adventure("enter:7:0", data, .7))
        nav.step(data, .8)
        data["doors"][0]["target_index"] = 86
        self.assertFalse(nav.accept_adventure("enter:2:85", data, .9))
        self.assertIsNone(nav._pending)

    def test_selected_switch_does_not_automatically_select_another(self):
        data = required(plate(12, 240, 200), plate(24, 440, 360))
        data["capabilities"]["interaction_control"] = 1
        nav = PlayerNavigator()
        offers(nav, data)
        self.assertTrue(nav.accept_adventure("switch:24", data, .7))
        data["frame"] += 1
        self.assertIsNone(nav.step(data, .8).stop_reason)
        self.assertEqual(nav._selected_switch.target[0], 24)
        data["switches"][1]["state"] = 3
        data["frame"] += 1
        action = nav.step(data, .9)
        self.assertEqual(action.move, "none")
        self.assertIsNone(nav._intent)
        self.assertEqual(data["switches"][0]["state"], 0)

    def test_blocked_switch_does_not_authorize_tnt_shooting(self):
        data, nav = recorded(), PlayerNavigator()
        choices = offers(nav, data)
        self.assertTrue(any(c.kind == "demolish_tnt" for c in choices))
        self.assertIsNone(nav._tnt)
        self.assertTrue(nav.accept_adventure("switch:24", data, .7))
        data["frame"] += 1
        action = nav.step(data, .8)
        self.assertEqual((action.move, action.shoot), ("none", "none"))
        self.assertIsNone(nav._tnt)
        self.assertIsNone(nav._selected_switch.demolition)
        self.assertEqual(nav.player_events[-1]["event"], "failed")

    def test_jev_selected_tnt_does_not_automatically_press_plate_afterward(self):
        data, nav = recorded(), PlayerNavigator()
        choices = offers(nav, data)
        selected = next(c for c in choices if c.kind == "demolish_tnt" and c.details["tnt_index"] == 54)
        self.assertTrue(nav.accept_adventure(selected.key, data, .7))
        now = .8
        for _ in range(16):
            data["frame"] += 8
            action = nav.step(data, now)
            self.assertIsNone(action.stop_reason)
            if action.shoot != "none":
                break
            waypoint = nav._tnt.waypoint
            data["player"].update(x=waypoint[0], y=waypoint[1], vx=0, vy=0)
            now += 8/30
        else:
            self.fail("selected TNT never fired")
        self.assertEqual(nav._tnt.plan.target_index, 54)
        self.assertEqual(action.hold_frames, 3)
        retreat = nav._tnt.plan.retreat_point
        data["hazards"] = [h for h in data["hazards"] if h.get("index") != 54]
        data["player"].update(x=retreat[0], y=retreat[1], vx=0, vy=0)
        data["frame"] += 70
        now += 2.4
        nav.step(data, now)
        data["frame"] += 24
        action = nav.step(data, now+.8)
        self.assertEqual((action.move, action.shoot), ("none", "none"))
        self.assertIsNone(nav._intent)
        self.assertEqual(data["switches"][0]["state"], 0)
        data["frame"] += 15
        self.assertEqual(nav.step(data, now+1.3).move, "none")
        self.assertTrue(any(c.kind == "press_switch" for c in nav.adventure_options))

    def test_changed_tnt_position_revokes_source_bound_demolition(self):
        data, nav = recorded(), PlayerNavigator()
        selected = next(c for c in offers(nav, data) if c.kind == "demolish_tnt")
        target = next(h for h in data["hazards"] if h.get("index") == selected.details["tnt_index"])
        target["x"] += 40
        self.assertFalse(nav.accept_adventure(selected.key, data, .7))
        self.assertIsNone(nav._tnt)

    def test_rearm_preserves_map_but_revokes_selected_door(self):
        data, nav = ready(), PlayerNavigator()
        selected = next(c for c in offers(nav, data) if c.kind == "enter_door")
        nav.accept_adventure(selected.key, data, .7)
        data.update(session="fresh-arm", frame=data["frame"]+1)
        fresh = nav.rearmed(data)
        self.assertIsInstance(fresh, PlayerNavigator)
        self.assertIsNone(fresh._intent)
        self.assertIsNone(fresh._pending)
        self.assertEqual(fresh.stats, nav.stats)
        self.assertEqual(fresh.step(data, .8).move, "none")

    def test_jev_can_choose_floor_exit_without_local_completion_order(self):
        data = ready(kind=5)
        data["hazards"] = [grid(320, 360, kind=17, collision=0)]
        nav = PlayerNavigator(continue_floors=True)
        choices = offers(nav, data)
        self.assertTrue(any(c.kind == "enter_door" for c in choices))
        self.assertTrue(any(c.kind == "descend" for c in choices))
        selected = next(c for c in choices if c.kind == "descend")
        self.assertTrue(nav.accept_adventure(selected.key, data, .7))
        self.assertFalse(nav.descent_requested)
        data["frame"] += 1
        nav.step(data, .8)
        self.assertIsNotNone(nav._intent)

    def test_jev_selected_door_replays_the_actual_post_tnt_stop(self):
        data, nav = recorded_state(), PlayerNavigator()
        choice = next(c for c in offers(nav, data) if c.key == "enter:3:126")
        self.assertTrue(nav.accept_adventure(choice.key, data, .7))
        delayed = deque(["none"])
        for tick in range(240):
            data["frame"] += 1
            action = nav.step(data, .8+tick/30)
            self.assertIsNone(action.stop_reason)
            self.assertEqual(action.shoot, "none")
            delayed.append(action.move)
            applied = _VECTORS[delayed.popleft()]
            boxes = boss_geometry(data)[2][0]
            player = data["player"]
            for _ in range(2):
                start = player["x"], player["y"]
                physics_substep(player, applied)
                end = player["x"], player["y"]
                self.assertTrue(_clear(start, end, boxes))
                if end[1] > 448:
                    incoming = copy.deepcopy(data)
                    incoming.update(frame=data["frame"]+1, room_id="boss-arrival")
                    incoming["floor"].update(room_index=126, room_list_index=13)
                    incoming["room"].update(type=5, clear=False)
                    incoming["player"].update(x=320, y=280, vx=0, vy=0)
                    incoming["doors"] = []
                    arrival = nav.step(incoming, .8+(tick+1)/30)
                    self.assertIsNone(arrival.stop_reason)
                    self.assertIsNone(nav._intent)
                    self.assertEqual(nav.stats["doors_traversed"], 1)
                    return
        self.fail("Selected boss doorway did not complete the recorded-layout replay")

    def test_rearming_after_tnt_shot_retains_blast_guard_without_reselecting(self):
        data, nav = recorded(), PlayerNavigator()
        selected = next(c for c in offers(nav, data) if c.kind == "demolish_tnt")
        self.assertTrue(nav.accept_adventure(selected.key, data, .7))
        now = .8
        for _ in range(16):
            data["frame"] += 8
            action = nav.step(data, now)
            if action.shoot != "none":
                break
            waypoint = nav._tnt.waypoint
            data["player"].update(x=waypoint[0], y=waypoint[1], vx=0, vy=0)
            now += 8/30
        else:
            self.fail("No selected TNT shot")
        nav.cancel_intent()
        data.update(session="fresh-arm", frame=data["frame"]+1)
        fresh = nav.rearmed(data)
        action = fresh.step(data, now+.1)
        self.assertEqual((action.move, action.shoot), ("none", "none"))
        self.assertIn("committed TNT shot", action.status)
        self.assertIsNone(fresh._intent)
        self.assertEqual(fresh.adventure_options, ())


if __name__ == "__main__":
    unittest.main()
