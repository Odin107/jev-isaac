"""Exercise real UDP floor orchestration separately from path planning."""
import json
from pathlib import Path
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_controller import LocalGame
from test_goal_controller import goal, wait_for
from jev_isaac.controller import Action
from jev_isaac.demo import sample
from jev_isaac.protocol import Observation, encode_action

CAPS = {"movement_pulses": 1, "local_goal_control": 1, "floor_control": 1}


def changes(clear=False, room_id="room-1", **extra):
    room = dict(sample()["room"], clear=clear, type=1)
    return dict(capabilities=CAPS, floor={"id": "floor-1", "room_index": 1},
                run_id="run-1", room_id=room_id, room=room, doors=[], **extra)


class StubExplorer:
    stats = {"rooms_visited": 1}
    stop_reason = None
    def observe(self, state):
        pass
    def step(self, state, now):
        return SimpleNamespace(move="right", shoot="none", stop_reason=None)


class FloorControllerTests(unittest.TestCase):
    def test_floor_packet_and_clear_room_eligibility(self):
        data = sample()
        data.update(changes(clear=True))
        obs = Observation.decode(json.dumps(data).encode())
        self.assertTrue(obs.controllable)
        self.assertFalse(obs.active)
        self.assertIs(json.loads(encode_action(obs, "right", "none", floor_mode=True))["floor_mode"], True)
        with self.assertRaises(ValueError):
            encode_action(obs, "none", "none", floor_mode=1)

    def test_clear_room_navigation_does_not_call_api(self):
        calls = []
        with patch("jev_isaac.exploration.FloorNavigator", return_value=StubExplorer()):
            game = LocalGame(lambda state: calls.append(state) or goal(), goal_mode=True,
                             floor_mode=True, duration=.14)
            game.send(30, **changes(clear=True))
            packets = game.collect()
        self.assertEqual(calls, [])
        self.assertTrue(any(p["move"] == "right" and p["floor_mode"] is True for p in packets))
        self.assertIs(packets[-1]["floor_mode"], False)

    def test_room_clear_replaces_goal_with_one_floor_packet(self):
        with patch("jev_isaac.exploration.FloorNavigator", return_value=StubExplorer()), \
             patch("jev_isaac.navigation.compute_action", return_value=Action("left", "up")):
            game = LocalGame(lambda _: goal(), goal_mode=True, floor_mode=True,
                             max_hz=1, duration=.24)
            game.send(30, **changes())
            wait_for(lambda: game.controller.stats.goal_updates == 1)
            game.send(33, **changes(clear=True))
            wait_for(lambda: game.controller.stats.expired_goals == 1)
            game.send(34, **changes(clear=True))
            packets = game.collect()
        clear_packets = [p for p in packets if p["frame"] == 33]
        self.assertEqual(len(clear_packets), 1)
        self.assertEqual(clear_packets[0]["move"], "right")
        self.assertIs(clear_packets[0]["floor_mode"], True)

    def test_old_room_cloud_reply_is_not_applied_in_new_room(self):
        entered, release = threading.Event(), threading.Event()
        def policy(_):
            entered.set()
            release.wait(.5)
            return goal()
        with patch("jev_isaac.exploration.FloorNavigator", return_value=StubExplorer()):
            game = LocalGame(policy, goal_mode=True, floor_mode=True, max_hz=1, duration=.18)
            game.send(30, **changes())
            self.assertTrue(entered.wait(.2))
            game.send(32, **changes(room_id="room-2"))
            time.sleep(.025)
            release.set()
            packets = game.collect()
        self.assertEqual(game.result["goal_updates"], 0)
        self.assertEqual(game.result["stale_discarded"], 1)
        self.assertFalse(any(p["hold_frames"] > 1 for p in packets))

    def test_f8_off_stops_floor_without_more_calls(self):
        with patch("jev_isaac.exploration.FloorNavigator", return_value=StubExplorer()):
            game = LocalGame(lambda _: goal(), goal_mode=True, floor_mode=True, duration=.3)
            game.send(30, **changes(clear=True))
            wait_for(lambda: game.controller.stats.actions_sent == 1)
            game.send(31, **changes(clear=True, enabled=False))
            game.collect()
        self.assertEqual(game.result["stop_reason"], "control disabled")
        self.assertEqual(game.result["decisions"], 0)

    def test_transition_pause_suppresses_controls_without_ending_floor(self):
        with patch("jev_isaac.exploration.FloorNavigator", return_value=StubExplorer()):
            game = LocalGame(lambda _: goal(), goal_mode=True, floor_mode=True, duration=.2)
            game.send(30, **changes(clear=True))
            wait_for(lambda: game.controller.stats.actions_sent == 1)
            game.send(31, **changes(clear=True, paused=True))
            time.sleep(.03)
            game.send(32, **changes(clear=True, room_id="room-2"))
            packets = game.collect()
        self.assertEqual(game.result["stop_reason"], "duration reached")
        self.assertFalse(any(p["frame"] == 31 for p in packets))
        self.assertTrue(any(p["frame"] == 32 and p["move"] == "right" for p in packets))

    def test_local_guard_moves_before_first_cloud_reply(self):
        release = threading.Event()
        def policy(_):
            release.wait(.5)
            return goal()
        with patch("jev_isaac.navigation.compute_action", return_value=Action("down", "none")):
            game = LocalGame(policy, goal_mode=True, startup_guard=True, duration=.2)
            try:
                game.send(30, capabilities=CAPS)
                wait_for(lambda: game.controller.stats.startup_local_actions == 1)
                self.assertEqual(game.controller.stats.goal_updates, 0)
                self.assertLess(game.controller.stats.first_nonneutral_action_ms, 100)
            finally:
                release.set()
                game.collect()


if __name__ == "__main__":
    unittest.main()
