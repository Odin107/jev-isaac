"""Recorded cleared-room drift and delayed steering regressions; no game/API IO.

The fixture is the exact room82 geometry/position/velocity at the failure, with
replacement identities. The motion replay is a stress model calibrated against
the recorded straight acceleration, not an Isaac physics emulator.
"""
import copy
from collections import deque
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_isaac.exploration import FloorNavigator, _door_move, _validated
from jev_isaac.navigation import _VECTORS


def recorded_state():
    return json.loads((Path(__file__).parent / "fixtures" / "floor-rock-drift.json").read_text())


def toward_bottom(observed):
    parsed = _validated(observed)
    chosen = next(d for d in parsed[-1] if d.slot == 3)
    return _door_move(observed, parsed, chosen)


def recorded_rock(observed):
    return next(h for h in observed["hazards"] if h.get("index") == 98)


def rock(x, y):
    return {"kind": "grid", "type": 2, "collision": 3,
            "radius": 20, "x": x, "y": y}


def prime_previous_room(navigator, observed):
    """Remember the actual entry room so exploration chooses the bottom exit."""
    previous = copy.deepcopy(observed)
    previous["floor"].update(room_index=83, room_list_index=8)
    previous.update(frame=observed["frame"]-1, room_id="fixture-room83:visit1")
    previous["player"].update(x=80, y=280, vx=0, vy=0)
    previous["hazards"] = []
    previous["doors"] = [{"slot": 0, "x": 40, "y": 280, "open": True,
                           "locked": False, "target_index": 82, "target_type": 1}]
    navigator.step(previous, 0)


def physics_substep(player, applied):
    # Isaac's observed position advances about twice the reported velocity per
    # observation. These two substeps match sustained diagonal acceleration:
    # first displacement .3748 and final velocity .7046, versus .3735/.7023 live.
    player["x"] += player["vx"]
    player["y"] += player["vy"]
    for axis, component in (("vx", applied[0]), ("vy", applied[1])):
        player[axis] = player[axis]*(.88 if component else .79)+.53*component


def solid_core_overlap(point, hazard):
    # Reported grid cells are conservative squares. Test against their core,
    # independently of the production padding/recovery calculation. The replay
    # cannot prove the game's exact player-versus-rock collision shape.
    half = hazard["radius"]
    return abs(point[0]-hazard["x"]) < half and abs(point[1]-hazard["y"]) < half


def circle_grid_clearance(player, hazard):
    """Player-circle distance from the reported solid square, less radius."""
    dx = max(abs(player["x"]-hazard["x"])-hazard["radius"], 0)
    dy = max(abs(player["y"]-hazard["y"])-hazard["radius"], 0)
    return math.hypot(dx, dy)-player["radius"]


class FloorDriftRegressionTests(unittest.TestCase):
    def test_exact_recorded_shallow_rock_padding_drift_recovers_outward(self):
        observed = recorded_state()
        before = copy.deepcopy(observed)
        player = observed["player"]
        self.assertAlmostEqual(390-player["x"], .48937988281)
        self.assertFalse(solid_core_overlap((player["x"], player["y"]), recorded_rock(observed)))
        move = toward_bottom(observed)
        self.assertIn(move, ("right", "down", "down_right"))
        self.assertEqual(observed, before)

    def test_recovery_resumes_the_selected_door_route_after_leaving_padding(self):
        observed = recorded_state()
        navigator = FloorNavigator()
        prime_previous_room(navigator, observed)
        first = navigator.step(observed, 1)
        self.assertIsNone(first.stop_reason)
        self.assertEqual(first.status, "entering room 95")
        observed["frame"] += 1
        observed["player"].update(x=402, y=397, vx=0, vy=0)
        action = navigator.step(observed, 1.1)
        self.assertIsNone(action.stop_reason)
        self.assertEqual(action.status, "entering room 95")
        self.assertNotEqual(action.move, "none")

    def test_deep_or_actual_rock_overlap_is_not_an_escape_exception(self):
        for point in ((360, 360), (380.1, 360)):
            with self.subTest(point=point):
                observed = recorded_state()
                observed["player"].update(x=point[0], y=point[1], vx=0, vy=0)
                self.assertIn(toward_bottom(observed), (None, "none"))

    def test_only_ordinary_solid_rocks_allow_shallow_padding_recovery(self):
        for kind, collision in ((8, 0), (9, 0), (17, 0), (18, 0), (23, 0),
                                (7, 1), (15, 4), (16, 5), (2, 1), (3, 3)):
            with self.subTest(kind=kind, collision=collision):
                observed = recorded_state()
                recorded_rock(observed).update(type=kind, collision=collision)
                self.assertIn(toward_bottom(observed), (None, "none"))
        observed = recorded_state()
        recorded_rock(observed).update(kind="fire", radius=20, vx=0, vy=0)
        self.assertIn(toward_bottom(observed), (None, "none"))

    def test_shallow_rock_recovery_cannot_enter_neighboring_obstacles(self):
        observed = recorded_state()
        observed["hazards"].extend((rock(420, 380), rock(400, 420)))
        self.assertIn(toward_bottom(observed), (None, "none"))

    def test_recorded_layout_is_traversed_with_inertia_and_delayed_input(self):
        observed = recorded_state()
        observed["player"].update(x=560, y=280, vx=0, vy=0)
        self._assert_delayed_traversal(observed, "none")

    def test_exact_failure_recovers_and_traverses_with_momentum_and_delayed_input(self):
        observed = recorded_state()
        # Retain the exact recorded velocity and the command already accepted by
        # the game when navigation failed. No teleport to a convenient waypoint.
        self._assert_delayed_traversal(observed, observed["control"]["applied_move"])

    def _assert_delayed_traversal(self, observed, initial_command):
        navigator = FloorNavigator()
        prime_previous_room(navigator, observed)
        # Input takes effect one observation later, with two physics steps per
        # observation. This deliberately differs from immediate constant-speed
        # movement and reproduces the live trace's delayed change of direction.
        pending = deque([initial_command])
        player = observed["player"]
        # The failed live position overlaps our circle-versus-square estimate by
        # just .136 units at the rock corner. Permit only that existing overlap,
        # and require each substep to reduce it until it has fully cleared. The
        # normal entrance replay starts with no allowed overlap at all.
        existing_overlap = {i: circle_grid_clearance(player, h)
                            for i, h in enumerate(observed["hazards"])
                            if h["kind"] == "grid" and circle_grid_clearance(player, h) < 0}
        for tick in range(600):
            observed["frame"] += 1
            action = navigator.step(observed, (tick+1)/30)
            self.assertIsNone(action.stop_reason, f"Stopped on tick {tick}: {action.stop_reason}")
            self.assertEqual(action.shoot, "none")
            self.assertEqual(action.status, "entering room 95")
            pending.append(action.move)
            applied = _VECTORS[pending.popleft()]
            for substep in range(2):
                physics_substep(player, applied)
                point = player["x"], player["y"]
                for index, hazard in enumerate(observed["hazards"]):
                    if hazard["kind"] == "grid":
                        if hazard["type"] == 16 and (hazard["x"], hazard["y"]) == (320, 440):
                            continue
                        clearance = circle_grid_clearance(player, hazard)
                        self.assertGreaterEqual(clearance, existing_overlap.get(index, 0)-1e-6,
                                                f"Entered solid cell at tick {tick}/{substep}: {point}, {hazard}")
                        if clearance < 0:
                            existing_overlap[index] = clearance
                        else:
                            existing_overlap.pop(index, None)
                    elif hazard["kind"] == "fire":
                        self.assertGreaterEqual(math.dist(point, (hazard["x"], hazard["y"])),
                                                player["radius"]+hazard["radius"],
                                                f"Entered fire at tick {tick}/{substep}")
                if player["y"] > 448:
                    self.assertLessEqual(abs(player["x"]-320), 10)
                    return
        self.fail("Navigator did not cross the bottom doorway with delayed, inertial motion")


if __name__ == "__main__":
    unittest.main()
