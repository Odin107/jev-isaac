"""Observed ordinary required plates only; no game or network actions."""
import copy
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_exploration import grid
from test_pickups import observed, pickup
from jev_isaac.navigation import _VECTORS, _clear
from jev_isaac.switches import SwitchNavigator, _geometry


def plate(index=12, x=400, y=280, **extra):
    result = {"index": index, "x": x, "y": y, "type": 20, "variant": 0, "state": 0, "collision": 0}
    result.update(extra)
    return result


def required(*plates, **extra):
    data = observed(clear=False, x=200, **extra)
    data["capabilities"]["room_switches"] = 1
    data["room"]["has_trigger_pressure_plates"] = True
    data["switches"] = list(plates or (plate(),))
    return data


class SwitchTests(unittest.TestCase):
    def recorded_corner(self):
        data = json.loads((Path(__file__).parent / "fixtures" / "floor-poop-corner-stop.json").read_text())
        # Keep exact recorded hazards/player. The plate is a synthetic objective
        # overlay to exercise starting/rearming from this confirmed overlap.
        data.update(enabled=True, paused=False)
        data["room"].update(clear=False, has_trigger_pressure_plates=True)
        data["capabilities"]["room_switches"] = 1
        data["switches"] = [plate(index=124, x=520, y=360)]
        return data

    def test_recorded_padding_corner_allows_initial_and_fresh_rearm_recovery(self):
        data, nav = self.recorded_corner(), SwitchNavigator()
        before = copy.deepcopy(data)
        first = nav.step(data, 0)
        self.assertIsNone(first.stop_reason)
        self.assertEqual((first.move, first.shoot), ("down_right", "none"))
        self.assertEqual(data, before)
        nav.reset()
        data["session"] += ":new-arm"
        rearmed = nav.step(data, 1)
        self.assertEqual((rearmed.move, rearmed.shoot), ("down_right", "none"))
        self.assertIsNone(rearmed.stop_reason)
        vector = _VECTORS[rearmed.move]
        data["player"].update(x=data["player"]["x"]+24*vector[0],
                              y=data["player"]["y"]+24*vector[1], vx=0, vy=0)
        for tick in range(1, 80):
            data["frame"] += 1
            action = nav.step(data, 1+tick/30)
            self.assertIsNone(action.stop_reason)
            self.assertEqual(action.shoot, "none")
            start = data["player"]["x"], data["player"]["y"]
            v = _VECTORS[action.move]
            end = start[0]+v[0]*4, start[1]+v[1]*4
            boxes = _geometry(data, 10, nav.target)[0]
            self.assertTrue(_clear(start, end, boxes))
            data["player"].update(x=end[0], y=end[1])
            if math.dist(end, (520, 360)) <= 4:
                return
        self.fail("corner recovery did not resume route to selected plate")

    def test_initial_recovery_retains_unsupported_and_paid_overlap_exclusions(self):
        changes = (
            lambda d: next(h for h in d["hazards"] if h.get("index") == 122).update(variant=1),
            lambda d: next(h for h in d["hazards"] if h.get("index") == 122).update(type=12),
            lambda d: d["player"].update(x=440, y=280, vx=0, vy=0),
            lambda d: d["pickups"].append(pickup("paid-at-start", x=d["player"]["x"],
                                                 y=d["player"]["y"], price=5, shop_item=True)),
        )
        for change in changes:
            data = self.recorded_corner()
            change(data)
            action = SwitchNavigator().step(data, 0)
            self.assertEqual((action.move, action.shoot), ("none", "none"))
            self.assertEqual(action.stop_reason, "no safe route to required room switch")

    def test_initial_padding_escape_needs_a_route_afterward_and_cannot_loop_forever(self):
        data, nav = self.recorded_corner(), SwitchNavigator()
        data["hazards"].extend(grid(500, y, kind=15, collision=4) for y in range(120, 441, 40))
        self.assertEqual(nav.step(data, 0).stop_reason, "no safe route to required room switch")
        data, nav = self.recorded_corner(), SwitchNavigator()
        self.assertIsNone(nav.step(data, 0).stop_reason)
        data["frame"] += 121
        self.assertEqual(nav.step(data, 4.01).stop_reason, "room switch approach stalled")

    def test_reachable_plate_approaches_without_shooting_then_waits_for_clear(self):
        data, nav = required(), SwitchNavigator()
        for frame in range(1, 100):
            data["frame"] = frame
            action = nav.step(data, frame/30)
            self.assertTrue(nav.has_objective)
            self.assertIsNone(action.stop_reason)
            self.assertEqual(action.shoot, "none")
            vector = _VECTORS[action.move]
            data["player"]["x"] += vector[0]*4
            data["player"]["y"] += vector[1]*4
            if math.dist((data["player"]["x"], data["player"]["y"]), (400, 280)) <= 4:
                break
        else:
            self.fail("never approached plate")
        data["switches"][0]["state"] = 3
        self.assertIn("waiting", nav.step(data, 3).status)
        self.assertEqual(nav.step(data, 5.01).stop_reason, "room switch activation timed out")
        data["room"]["clear"] = True
        self.assertIsNone(nav.step(data, 5.1))
        self.assertFalse(nav.has_objective)

    def test_rock_route_preserves_each_verified_segment(self):
        data, nav = required(), SwitchNavigator()
        data["hazards"] = [grid(300, 280)]
        for frame in range(1, 150):
            data["frame"] = frame
            action = nav.step(data, frame/30)
            self.assertIsNone(action.stop_reason)
            self.assertEqual(action.shoot, "none")
            start = data["player"]["x"], data["player"]["y"]
            v = _VECTORS[action.move]
            end = start[0]+v[0]*4, start[1]+v[1]*4
            self.assertTrue(_clear(start, end, [(270, 250, 330, 310)]))
            data["player"].update(x=end[0], y=end[1])
            if math.dist(end, (400, 280)) <= 4:
                return
        self.fail("rock detour never reached switch")

    def test_saved_tnt_room_uses_live_plate_facts_to_begin_demolition(self):
        data = json.loads((Path(__file__).parent / "fixtures" / "floor-switch-tnt-room-stop.json").read_text())
        self.assertFalse(data["room"]["clear"])
        self.assertEqual(data["enemies"], [])
        self.assertIsNone(SwitchNavigator().step(data, 0))  # Old export has no plate facts.
        data = json.loads((Path(__file__).parent / "fixtures" / "floor-switch-tnt-live.json").read_text())
        data.update(enabled=True, paused=False)
        before = copy.deepcopy(data)
        nav = SwitchNavigator()
        result = nav.step(data, 0)
        self.assertTrue(nav.has_objective)
        self.assertNotEqual(result.move, "none")
        self.assertEqual(result.shoot, "none")
        self.assertIsNone(result.stop_reason)
        self.assertTrue(nav.demolition.active)
        self.assertEqual(data, before)
        # A newly observed opening allows ordinary switch movement. It must
        # not begin another demolition when walking already reaches the plate.
        data["hazards"] = [h for h in data["hazards"] if h.get("index") != 54]
        direct = SwitchNavigator()
        self.assertIsNone(direct.step(data, 0).stop_reason)
        self.assertIsNone(direct.demolition)

    def test_special_missing_unknown_or_malformed_plate_metadata_is_not_authorized(self):
        for changes in ({"variant": 1}, {"variant": 2}, {"variant": 3}, {"variant": 9},
                        {"variant": 10}, {"variant": False}, {"state": True},
                        {"state": 1}, {"state": 2}, {"collision": 1}, {"index": -1},
                        {"type": 21}, {"x": float("nan")}):
            data, nav = required(plate(**changes)), SwitchNavigator()
            self.assertIsNone(nav.step(data, 0), changes)
            self.assertFalse(nav.has_objective)
        for field in ("variant", "state", "collision", "index", "x"):
            data = required()
            del data["switches"][0][field]
            self.assertIsNone(SwitchNavigator().step(data, 0))
        data = required(plate(), plate())
        self.assertIsNone(SwitchNavigator().step(data, 0))

    def test_context_flag_complete_data_and_no_active_combat_are_required(self):
        changes = (lambda d: d["room"].pop("has_trigger_pressure_plates"),
                   lambda d: d["room"].update(has_trigger_pressure_plates=False),
                   lambda d: d["room"].update(has_trigger_pressure_plates=1),
                   lambda d: d["capabilities"].update(room_switches=True),
                   lambda d: d["room"].update(type=11),
                   lambda d: d.update(truncated=True),
                   lambda d: d.update(truncated_arrays={"switches": True}),
                   lambda d: d["enemies"].append({"x": 400, "y": 280, "hp": 5}),
                   lambda d: d["projectiles"].append({"x": 400, "y": 280}),
                   lambda d: d["player"].update(dead=True))
        for change in changes:
            data, nav = required(), SwitchNavigator()
            change(data)
            self.assertIsNone(nav.step(data, 0))
            self.assertFalse(nav.has_objective)

    def test_paid_pickup_new_hazard_and_unselected_dangerous_plate_remain_excluded(self):
        for change in (lambda d: d["pickups"].append(pickup("paid", x=400, price=5, shop_item=True)),
                       lambda d: d["switches"].append(plate(index=13, x=400, variant=9)),
                       lambda d: d["hazards"].append({"kind": "bomb", "x": 300, "y": 280})):
            data, nav = required(), SwitchNavigator()
            nav.step(data, 0)
            change(data)
            action = nav.step(data, .1)
            self.assertEqual((action.move, action.shoot), ("none", "none"))
            self.assertEqual(action.stop_reason, "no safe route to required room switch")

    def test_away_and_back_never_resets_progress(self):
        data, nav = required(), SwitchNavigator()
        nav.step(data, 0)
        for now, x in ((1, 220), (2, 200), (4.9, 220)):
            data["frame"] += 1
            data["player"]["x"] = x
            nav.step(data, now)
            self.assertEqual(nav.progress_at, 1)
        self.assertEqual(nav.step(data, 5.01).stop_reason, "room switch approach stalled")

    def test_pause_and_rearm_clear_pending_movement_and_keep_timeout_bounded(self):
        data, nav = required(), SwitchNavigator()
        nav.step(data, 0)
        data["paused"] = True
        self.assertEqual(nav.step(data, 1).move, "none")
        data["paused"] = False
        self.assertIsNone(nav.step(data, 101).stop_reason)
        self.assertEqual(nav.step(data, 104.01).stop_reason, "room switch approach stalled")
        data["session"] = "run:arm2"
        self.assertIsNone(nav.step(data, 105).stop_reason)
        nav.reset()
        self.assertFalse(nav.has_objective)
        self.assertIsNone(nav.target)

    def test_selected_target_cannot_move_or_disappear_silently(self):
        for change in (lambda d: d["switches"][0].update(x=440),
                       lambda d: d["switches"][0].update(index=50)):
            data, nav = required(), SwitchNavigator()
            nav.step(data, 0)
            change(data)
            action = nav.step(data, .1)
            self.assertEqual(action.move, "none")
            self.assertEqual(action.stop_reason, "no safe route to required room switch")


if __name__ == "__main__":
    unittest.main()
