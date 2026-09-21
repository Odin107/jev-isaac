"""Replay the two saved live TNT stops without simulating explosion physics."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_isaac.combat import _inside
from jev_isaac.exploration import _validated
from jev_isaac.navigation import _clear, _free
from jev_isaac.switches import _geometry
from jev_isaac.tnt import TntDemolition, _explosives
from jev_isaac.tnt_geometry import TntPlan, plan_demolition, shot_direction


def recorded(index):
    path = Path(__file__).parent / "fixtures" / "tnt-live-stops.json"
    return json.loads(path.read_text())["stops"][index]["observation"]


def geometry(state):
    row = state["switches"][0]
    boxes, _, _, phase = _geometry(state, state["player"]["radius"],
                                   (row["index"], row["x"], row["y"]))
    return (row["x"], row["y"]), boxes, phase


def pending_retreat(state, *, retreat=(78, 280)):
    """Restore only the shot/retreat state evidenced by the saved command log."""
    demolition = TntDemolition()
    demolition.plan = TntPlan(54, (400, 240), (188, 240), retreat, "right", ())
    demolition.context = (state["run_id"], state["session"], state["room_id"],
                          state["floor"]["id"], state["floor"].get("dimension"), (400, 160))
    demolition.active = True
    demolition.phase = "retreat"
    demolition.started = demolition.progress_at = demolition.changed_at = demolition.fired_at = 0
    demolition.last_frame = state["frame"]-1
    demolition.last_fire_frame = state["frame"]-2
    demolition.explosives = _explosives(state)
    demolition.pulses, demolition.damage = 1, 3
    return demolition


class TntLiveStopTests(unittest.TestCase):
    def test_recorded_boundary_drift_gets_one_clear_inward_pulse(self):
        state = recorded(0)
        self.assertEqual(state["frame"], 35756)
        demolition = pending_retreat(state, retreat=(70, 280))
        parsed = _validated(state)
        start, velocity, bounds = parsed[3], parsed[5], parsed[6]
        neutral = start[0]+velocity[0]*2, start[1]+velocity[1]*2
        self.assertTrue(_inside(start, bounds))
        self.assertFalse(_inside(neutral, bounds))
        before = copy.deepcopy(state)
        action = demolition.step(state, .1, *geometry(state))
        self.assertIsNone(action.stop_reason)
        self.assertEqual((action.move, action.shoot, action.hold_frames), ("right", "none", 1))
        predicted = neutral[0]+2, neutral[1]
        self.assertTrue(_inside(predicted, bounds))
        self.assertTrue(_clear(start, predicted, geometry(state)[1]))
        self.assertEqual(state, before)
        # Fresh evidence of corrected drift resumes settling without rearming.
        state["frame"] += 1
        state["player"].update(x=72, y=280, vx=.2, vy=0)
        action = demolition.step(state, .2, *geometry(state))
        self.assertIsNone(action.stop_reason)
        self.assertEqual((action.move, action.shoot), ("none", "none"))

    def test_boundary_correction_cannot_cross_an_obstacle_or_fast_momentum(self):
        state = recorded(0)
        demolition = pending_retreat(state, retreat=(70, 280))
        point, boxes, phase = geometry(state)
        boxes.append((71, 278, 73, 281))
        action = demolition.step(state, .1, point, boxes, phase)
        self.assertIsNotNone(action.stop_reason)
        self.assertEqual((action.move, action.shoot), ("none", "none"))
        state["player"]["vx"] = -5
        demolition = pending_retreat(state, retreat=(70, 280))
        action = demolition.step(state, .1, *geometry(state))
        self.assertIsNotNone(action.stop_reason)
        self.assertEqual(action.shoot, "none")

    def test_new_retreat_plan_reserves_settling_space_inside_room_bounds(self):
        state = recorded(0)
        point, boxes, phase = geometry(state)
        plan = plan_demolition(state, point, boxes, phase)
        self.assertEqual(plan.retreat_point, (78, 280))
        bounds = _validated(state)[6]
        for dx, dy in ((-6, 0), (6, 0), (0, -6), (0, 6)):
            self.assertTrue(_inside((plan.retreat_point[0]+dx, plan.retreat_point[1]+dy), bounds))

    def test_recorded_explosion_state_with_collision_keeps_retreating(self):
        state = recorded(1)
        self.assertEqual(state["frame"], 35953)
        target = next(h for h in state["hazards"] if h.get("index") == 54)
        self.assertEqual((target["state"], target["collision"]), (4, 2))
        demolition = pending_retreat(state)
        before = copy.deepcopy(state)
        action = demolition.step(state, .1, *geometry(state))
        self.assertIsNone(action.stop_reason)
        self.assertEqual(action.shoot, "none")
        self.assertNotEqual(action.move, "none")
        self.assertFalse(demolition.complete)
        self.assertFalse(_free((400, 240), geometry(state)[1]))
        self.assertEqual(shot_direction(state, demolition.plan, (188, 240)), "none")
        self.assertEqual(state, before)
        # Reaching retreat does not authorize another shot at EXPLODED or
        # declare passage open while its observed collision remains present.
        state["player"].update(x=78, y=280, vx=0, vy=0)
        state["frame"] += 70
        action = demolition.step(state, 3, *geometry(state))
        self.assertIsNone(action.stop_reason)
        self.assertEqual((action.move, action.shoot), ("none", "none"))
        self.assertFalse(demolition.complete)
        self.assertEqual(demolition.phase, "wait")
        self.assertEqual(demolition.pulses, 1)
        # Explicit later collision removal, followed by fresh stable frames,
        # completes the existing plan with no F8 and no invented destruction.
        target["collision"] = 0
        state["frame"] += 1
        self.assertIsNone(demolition.step(state, 3.1, *geometry(state)).stop_reason)
        self.assertFalse(demolition.complete)
        state["frame"] += 23
        action = demolition.step(state, 3.9, *geometry(state))
        self.assertIsNone(action.stop_reason)
        self.assertTrue(demolition.complete)

    def test_explosion_seen_during_approach_starts_retreat_without_shooting(self):
        state = recorded(1)
        demolition = pending_retreat(state)
        demolition.phase = "approach"
        demolition.last_fire_frame = demolition.fired_at = None
        action = demolition.step(state, .1, *geometry(state))
        self.assertIsNone(action.stop_reason)
        self.assertEqual(action.shoot, "none")
        self.assertEqual(demolition.phase, "retreat")
        self.assertEqual(demolition.last_fire_frame, state["frame"])

    def test_unknown_changes_still_reject_identity_and_stuck_explosion_is_bounded(self):
        for changes in ({"state": 5}, {"state": True}, {"collision": 1},
                        {"variant": 1}, {"x": 440}, {"type": 2},
                        {"state": 3, "collision": 0}):
            with self.subTest(changes=changes):
                state = recorded(1)
                demolition = pending_retreat(state)
                target = next(h for h in state["hazards"] if h.get("index") == 54)
                target.update(changes)
                action = demolition.step(state, .1, *geometry(state))
                self.assertIsNotNone(action.stop_reason)
                self.assertEqual(action.shoot, "none")
        state = recorded(1)
        demolition = pending_retreat(state)
        self.assertIsNone(demolition.step(state, .1, *geometry(state)).stop_reason)
        state["frame"] += 1800
        state["player"].update(x=78, y=280, vx=0, vy=0)
        action = demolition.step(state, 60, *geometry(state))
        self.assertIsNotNone(action.stop_reason)
        self.assertEqual(action.shoot, "none")
        self.assertFalse(demolition.complete)


if __name__ == "__main__":
    unittest.main()
