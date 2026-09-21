"""Observed TNT puzzle approach, short firing, retreat and completion contracts.

Waypoints are advanced by explicit synthetic observations. These tests check
the controller's decisions and geometry, not actual game explosion physics.
"""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_isaac.navigation import _clear
from jev_isaac.switches import SwitchNavigator, _geometry
from jev_isaac.tnt import TntDemolition


def recorded():
    data = json.loads((Path(__file__).parent / "fixtures" / "floor-switch-tnt-live.json").read_text())
    data.update(enabled=True, paused=False)
    return data


def geometry(data):
    row = data["switches"][0]
    boxes, _, _, phase = _geometry(data, data["player"]["radius"], (row["index"], row["x"], row["y"]))
    return (row["x"], row["y"]), boxes, phase


def step(demolition, data, now):
    return demolition.step(data, now, *geometry(data))


def approach_until_fire(demolition, data):
    """Advance only along the planner's currently clear, observed route segments."""
    now = 0.0
    trace = []
    for _ in range(12):
        before = copy.deepcopy(data)
        action = step(demolition, data, now)
        trace.append((now, copy.deepcopy(data), action))
        if action is None or action.stop_reason or action.shoot != "none":
            return action, now, trace
        if data != before:
            raise AssertionError("Demolition mutated the observation")
        waypoint = demolition.waypoint
        if waypoint is None:
            raise AssertionError("Approach has no verified waypoint")
        start = data["player"]["x"], data["player"]["y"]
        if not _clear(start, waypoint, geometry(data)[1]):
            raise AssertionError("Approach traverses an observed obstacle")
        data["player"].update(x=waypoint[0], y=waypoint[1], vx=0, vy=0)
        data["frame"] += 8
        now += 8/30
    raise AssertionError("TNT approach never reached a firing decision")


class TntRegressionTests(unittest.TestCase):
    def test_exact_live_metadata_enters_demolition_without_a_false_route_stop(self):
        data = recorded()
        self.assertEqual(data["frame"], 35291)
        self.assertEqual(data["switches"][0]["index"], 24)
        self.assertEqual(data["switches"][0]["state"], 0)
        self.assertIn(366, {item["id"] for item in data["player"]["inventory"]})
        before = copy.deepcopy(data)
        nav = SwitchNavigator()
        action = nav.step(data, 0)
        self.assertIsNotNone(action)
        self.assertIsNone(action.stop_reason)
        self.assertNotEqual(action.move, "none")
        self.assertEqual(action.shoot, "none")
        self.assertTrue(nav.demolition.active)
        self.assertEqual(data, before)

    def test_approach_has_no_early_shots_then_fires_only_three_frames_when_settled(self):
        data, demolition = recorded(), TntDemolition()
        action, _, trace = approach_until_fire(demolition, data)
        self.assertIsNone(action.stop_reason)
        self.assertNotEqual(action.shoot, "none")
        self.assertEqual(action.hold_frames, 3)
        self.assertEqual(action.move, "none")
        self.assertTrue(all(item.shoot == "none" for _, _, item in trace[:-1]))
        self.assertEqual(demolition.pulses, 1)
        self.assertEqual(demolition.phase, "retreat")
        self.assertEqual(data["player"]["vx"], 0)
        self.assertEqual(data["player"]["vy"], 0)

    def test_disappearance_after_shot_still_requires_retreat_and_settling(self):
        data, demolition = recorded(), TntDemolition()
        fire, fired_at, _ = approach_until_fire(demolition, data)
        self.assertNotEqual(fire.shoot, "none")
        target = demolition.plan.target_index
        data["hazards"] = [h for h in data["hazards"] if not (h.get("kind") == "grid" and h.get("index") == target)]
        data["frame"] += 1
        action = step(demolition, data, fired_at+1/30)
        self.assertIsNone(action.stop_reason)
        self.assertEqual(action.shoot, "none")
        self.assertIn("retreat", action.status.lower())
        self.assertNotEqual(action.move, "none")
        self.assertTrue(demolition.active)
        self.assertFalse(demolition.complete)
        # Observe each chosen retreat segment, then allow both the post-shot
        # and stable-geometry windows to pass. No clearance is invented.
        for offset in (.3, .6, .9):
            point = demolition.waypoint
            data["player"].update(x=point[0], y=point[1], vx=0, vy=0)
            data["frame"] += 9
            action = step(demolition, data, fired_at+offset)
            self.assertIsNone(action.stop_reason)
            self.assertEqual(action.shoot, "none")
        data["frame"] += 60
        action = step(demolition, data, fired_at+2.1)
        self.assertIsNone(action.stop_reason)
        self.assertEqual(action.shoot, "none")
        self.assertTrue(demolition.complete)
        self.assertFalse(demolition.active)
        self.assertFalse(data["room"]["clear"])

    def test_surviving_damaged_barrel_is_not_deleted_or_declared_complete(self):
        data, demolition = recorded(), TntDemolition()
        _, fired_at, _ = approach_until_fire(demolition, data)
        target = next(h for h in data["hazards"] if h.get("index") == demolition.plan.target_index)
        target["state"] = 1
        preserved = copy.deepcopy(data["hazards"])
        # Visit the current route's waypoint before the retreat destination.
        for offset in (1/30, .4, .8, 2.1):
            data["frame"] += 12
            action = step(demolition, data, fired_at+offset)
            self.assertIsNone(action.stop_reason)
            self.assertEqual(action.shoot, "none")
            waypoint = demolition.waypoint
            if waypoint is not None:
                data["player"].update(x=waypoint[0], y=waypoint[1], vx=0, vy=0)
        self.assertEqual(data["hazards"], preserved)
        self.assertFalse(demolition.complete)
        self.assertTrue(demolition.active)

    def test_wall_time_does_not_replace_game_frames_after_a_shot(self):
        data, demolition = recorded(), TntDemolition()
        _, fired_at, _ = approach_until_fire(demolition, data)
        fire_frame = data["frame"]
        data["hazards"] = [h for h in data["hazards"]
                           if not (h.get("kind") == "grid" and h.get("index") == demolition.plan.target_index)]
        # A fresh observation places Isaac at the planned retreat. Wall time
        # may advance while the game is paused, so it cannot prove shot expiry.
        x, y = demolition.plan.retreat_point
        data["player"].update(x=x, y=y, vx=0, vy=0)
        data["frame"] = fire_frame+1
        self.assertIsNone(step(demolition, data, fired_at+.1).stop_reason)
        data["frame"] = fire_frame+2
        action = step(demolition, data, fired_at+3)
        self.assertEqual((action.move, action.shoot), ("none", "none"))
        self.assertIsNone(action.stop_reason)
        self.assertFalse(demolition.complete)
        self.assertEqual(demolition.phase, "wait")
        data["frame"] = fire_frame+60
        action = step(demolition, data, fired_at+3.1)
        self.assertIsNone(action.stop_reason)
        self.assertTrue(demolition.complete)

    def test_late_hazard_change_needs_23_stable_game_frames(self):
        data, demolition = recorded(), TntDemolition()
        _, fired_at, _ = approach_until_fire(demolition, data)
        fire_frame = data["frame"]
        x, y = demolition.plan.retreat_point
        data["player"].update(x=x, y=y, vx=0, vy=0)
        data["hazards"] = [h for h in data["hazards"]
                           if not (h.get("kind") == "grid" and h.get("index") == demolition.plan.target_index)]
        data["frame"] = fire_frame+60
        action = step(demolition, data, fired_at+3)
        self.assertIsNone(action.stop_reason)
        for frame, offset in ((fire_frame+61, 4), (fire_frame+82, 4.1)):
            data["frame"] = frame
            action = step(demolition, data, fired_at+offset)
            self.assertEqual((action.move, action.shoot), ("none", "none"))
            self.assertIsNone(action.stop_reason)
            self.assertFalse(demolition.complete)
        data["frame"] = fire_frame+83
        action = step(demolition, data, fired_at+4.2)
        self.assertIsNone(action.stop_reason)
        self.assertTrue(demolition.complete)

    def test_reset_and_fresh_session_preserve_the_prior_shot_window(self):
        for reset in (False, True):
            with self.subTest(explicit_reset=reset):
                data, demolition = recorded(), TntDemolition()
                _, fired_at, trace = approach_until_fire(demolition, data)
                nav = SwitchNavigator()
                for now, observed, _ in trace:
                    action = nav.step(observed, now)
                    self.assertIsNone(action.stop_reason)
                self.assertNotEqual(action.shoot, "none")
                fire_frame = data["frame"]
                if reset:
                    nav.reset()
                    self.assertIsNotNone(nav.shot_guard)
                data["session"] = "fresh-arm-after-shot"
                for frame, offset in ((fire_frame+2, 3), (fire_frame+59, 3.1)):
                    data["frame"] = frame
                    action = nav.step(data, fired_at+offset)
                    self.assertEqual((action.move, action.shoot), ("none", "none"))
                    self.assertIsNone(action.stop_reason)
                    self.assertIn("earlier TNT shots", action.status)
                    self.assertIsNone(nav.demolition)
                data["frame"] = fire_frame+60
                action = nav.step(data, fired_at+3.2)
                self.assertIsNone(action.stop_reason)
                self.assertIsNone(nav.shot_guard)
                self.assertIsNotNone(nav.demolition)
                self.assertTrue(nav.demolition.active)

    def test_public_switch_navigation_continues_after_observed_tnt_destruction(self):
        data, reference = recorded(), TntDemolition()
        _, fired_at, trace = approach_until_fire(reference, data)
        nav = SwitchNavigator()
        for now, observed, _ in trace:
            before = copy.deepcopy(observed)
            action = nav.step(observed, now)
            self.assertIsNone(action.stop_reason)
            self.assertEqual(observed, before)
        self.assertNotEqual(action.shoot, "none")
        self.assertEqual(nav.demolition.plan.target_index, 54)
        # These explicit fixture events stand in for fresh observations; this
        # test does not infer engine explosions, switch presses or room clear.
        data["hazards"] = [h for h in data["hazards"]
                           if not (h.get("kind") == "grid" and h.get("index") == 54)]
        data["frame"] += 1
        now = fired_at+1/30
        for _ in range(12):
            before = copy.deepcopy(data)
            action = nav.step(data, now)
            self.assertIsNone(action.stop_reason)
            self.assertEqual(action.shoot, "none")
            self.assertEqual(data, before)
            self.assertFalse(data["room"]["clear"])
            if nav.demolition.complete:
                break
            self.assertTrue(nav.demolition.active)
            waypoint = nav.demolition.waypoint
            self.assertIsNotNone(waypoint)
            start = data["player"]["x"], data["player"]["y"]
            self.assertTrue(_clear(start, waypoint, geometry(data)[1]))
            data["player"].update(x=waypoint[0], y=waypoint[1], vx=0, vy=0)
            data["frame"] += 9
            now += .3
        else:
            self.fail("Public switch navigation did not finish the bounded TNT retreat")
        self.assertGreaterEqual(data["frame"]-nav.demolition.last_fire_frame, 60)
        self.assertGreaterEqual(data["frame"]-nav.demolition.changed_frame, 23)
        self.assertGreaterEqual(now-fired_at, 2)
        data["frame"] += 1
        now += 1/30
        action = nav.step(data, now)
        self.assertIsNone(action.stop_reason)
        self.assertIsNone(nav.demolition)
        self.assertEqual(nav.target, (24, 400, 160))
        self.assertEqual(action.status, "pressing required room switch")
        self.assertNotEqual(action.move, "none")
        self.assertEqual(action.shoot, "none")
        for _ in range(12):
            start = data["player"]["x"], data["player"]["y"]
            if start == (400, 160):
                break
            self.assertIsNotNone(nav.waypoint)
            self.assertTrue(_clear(start, nav.waypoint, geometry(data)[1]))
            data["player"].update(x=nav.waypoint[0], y=nav.waypoint[1], vx=0, vy=0)
            data["frame"] += 6
            now += .2
            before = copy.deepcopy(data)
            action = nav.step(data, now)
            self.assertIsNone(action.stop_reason)
            self.assertEqual(action.shoot, "none")
            self.assertEqual(data, before)
        else:
            self.fail("Public switch navigation did not reach the observed required plate")
        self.assertEqual(data["switches"][0]["state"], 0)
        self.assertFalse(data["room"]["clear"])
        data["switches"][0]["state"] = 3
        data["frame"] += 1
        now += 1/30
        action = nav.step(data, now)
        self.assertIsNone(action.stop_reason)
        self.assertEqual((action.move, action.shoot), ("none", "none"))
        self.assertEqual(action.status, "required switches pressed; waiting for room to clear")
        self.assertTrue(nav.has_objective)
        self.assertFalse(data["room"]["clear"])
        data["room"]["clear"] = True
        data["frame"] += 1
        action = nav.step(data, now+1/30)
        self.assertIsNone(action)
        self.assertFalse(nav.has_objective)

    def test_unknown_weapon_effect_or_incomplete_inventory_never_fires(self):
        for mutate in (
                lambda d: d["player"].update(weapon_type=2),
                lambda d: d["player"].update(weapon_type=True),
                lambda d: d["player"].update(weapon_types=[1, 2]),
                lambda d: d["player"].update(inventory_truncated=True),
                lambda d: d["player"].update(inventory=[{"id": 149, "count": 1}]),
                lambda d: d["player"].pop("tear_range")):
            data = recorded()
            mutate(data)
            demolition = TntDemolition()
            action = step(demolition, data, 0)
            self.assertTrue(action is None or action.shoot == "none")
            self.assertEqual(demolition.pulses, 0)

    def test_pause_truncation_enemy_and_session_changes_cancel_before_fire(self):
        for mutate in (
                lambda d: d.update(paused=True), lambda d: d.update(enabled=False),
                lambda d: d.update(truncated=True), lambda d: d.update(session="new-arm"),
                lambda d: d["enemies"].append({"id": "new", "x": 400, "y": 280,
                                                "vx": 0, "vy": 0, "radius": 10,
                                                "hp": 5, "vulnerable": True})):
            data, demolition = recorded(), TntDemolition()
            self.assertIsNotNone(step(demolition, data, 0))
            mutate(data)
            data["frame"] += 1
            action = step(demolition, data, .1)
            self.assertEqual(action.shoot, "none")
            self.assertIsNotNone(action.stop_reason)
            self.assertEqual(demolition.pulses, 0)

    def test_moved_or_replaced_target_cannot_receive_stale_shooting(self):
        for mutate in (lambda target: target.update(x=target["x"]+40),
                       lambda target: target.update(type=2, collision=3),
                       lambda target: target.update(variant=1)):
            data, demolition = recorded(), TntDemolition()
            step(demolition, data, 0)
            target = next(h for h in data["hazards"] if h.get("index") == demolition.plan.target_index)
            mutate(target)
            data["frame"] += 1
            action = step(demolition, data, .1)
            self.assertEqual(action.shoot, "none")
            self.assertIsNotNone(action.stop_reason)
            self.assertEqual(demolition.pulses, 0)

    def test_duplicate_and_older_frames_never_repeat_fire_pulse(self):
        data, demolition = recorded(), TntDemolition()
        _, fired_at, _ = approach_until_fire(demolition, data)
        for offset in (0, -1):
            replay = copy.deepcopy(data)
            replay["frame"] += offset
            action = step(demolition, replay, fired_at+.1)
            self.assertEqual(action.shoot, "none")
            self.assertEqual(demolition.pulses, 1)

    def test_changed_chain_cannot_use_a_previous_firing_plan(self):
        for changed in ("grid_position", "movable_velocity", "new_explosive"):
            with self.subTest(changed=changed):
                data, demolition = recorded(), TntDemolition()
                self.assertIsNotNone(step(demolition, data, 0))
                if changed == "grid_position":
                    other = next(h for h in data["hazards"] if h.get("kind") == "grid"
                                 and h.get("type") == 12
                                 and h.get("index") != demolition.plan.target_index)
                    other["x"] += 5
                elif changed == "movable_velocity":
                    other = next(h for h in data["hazards"] if h.get("kind") == "tnt")
                    other["vx"] = 1
                else:
                    other = copy.deepcopy(next(h for h in data["hazards"] if h.get("kind") == "tnt"))
                    other.update(id="new-tnt", x=500, y=400)
                    data["hazards"].append(other)
                data["frame"] += 1
                action = step(demolition, data, .1)
                self.assertEqual(action.shoot, "none")
                self.assertIsNotNone(action.stop_reason)
                self.assertEqual(demolition.pulses, 0)


if __name__ == "__main__":
    unittest.main()
