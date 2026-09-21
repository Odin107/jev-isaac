"""Real local UDP tests for control that continues while goal inference waits."""
import json
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_controller import LocalGame
from jev_isaac.controller import Action
from jev_isaac.goals import GoalDecision

CAPABILITIES = {"movement_pulses": 1, "local_goal_control": 1}


def goal():
    return GoalDecision("engage", "target", 10, "offline", {"input_tokens": 10, "output_tokens": 2})


def wait_for(predicate, seconds=.5):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        time.sleep(.002)
    if not predicate():
        raise AssertionError("Controller condition not reached")


class GoalControlTests(unittest.TestCase):
    def test_local_steering_changes_while_next_cloud_goal_is_pending(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        def policy(state):
            calls.append(state["frame"])
            if len(calls) > 1:
                entered.set()
                release.wait(1)
            return goal()
        with patch("jev_isaac.navigation.compute_action",
                   side_effect=lambda state, *_: Action("right" if state["frame"] < 35 else "none", "down")):
            game = LocalGame(policy, goal_mode=True, max_hz=10, duration=.45,
                             hold_frames=15, move_frames=6, move_distance=20)
            try:
                game.send(30, capabilities=CAPABILITIES)
                wait_for(lambda: game.controller.stats.goal_updates == 1)
                for frame in range(31, 39):
                    game.send(frame, capabilities=CAPABILITIES)
                    time.sleep(.03)
                self.assertTrue(entered.is_set())
                self.assertEqual(game.controller.stats.goal_updates, 1)
                self.assertGreaterEqual(game.controller.stats.actions_sent, 7)
            finally:
                release.set()
                actions = game.collect()
        moving = [packet for packet in actions if packet["hold_frames"] > 1]
        self.assertTrue(any(packet["move"] == "right" for packet in moving))
        self.assertTrue(any(packet["move"] == "none" and packet["shoot"] == "down" for packet in moving))
        self.assertEqual(len({packet["frame"] for packet in moving}), len(moving))

    def test_goal_expires_without_queued_movements(self):
        with patch("jev_isaac.navigation.compute_action", return_value=Action("right", "down")):
            game = LocalGame(lambda _: goal(), goal_mode=True, max_hz=1, goal_max_age=.12,
                             duration=.32, hold_frames=15, move_frames=6, move_distance=20)
            game.send(30, capabilities=CAPABILITIES)
            wait_for(lambda: game.controller.stats.goal_updates == 1)
            for frame in range(31, 40):
                game.send(frame, capabilities=CAPABILITIES)
                time.sleep(.025)
            actions = game.collect()
        self.assertEqual(game.result["goal_updates"], 1)
        self.assertEqual(game.result["expired_goals"], 1)
        self.assertTrue(any(packet["hold_frames"] == 1 and packet["frame"] > 30 for packet in actions))
        self.assertFalse(any(packet["hold_frames"] > 1 and packet["frame"] >= 37 for packet in actions))

    def test_pause_and_rearm_cannot_reuse_previous_goal(self):
        with patch("jev_isaac.navigation.compute_action", return_value=Action("right", "down")):
            game = LocalGame(lambda _: goal(), goal_mode=True, max_hz=1,
                             duration=.3, hold_frames=15, move_frames=6, move_distance=20)
            game.send(30, capabilities=CAPABILITIES)
            wait_for(lambda: game.controller.stats.actions_sent >= 1)
            game.send(33, paused=True, capabilities=CAPABILITIES)
            wait_for(lambda: game.controller.stats.expired_goals == 1)
            game.send(36, capabilities=CAPABILITIES)
            actions = game.collect()
        self.assertEqual(game.result["goal_updates"], 1)
        self.assertFalse(any(packet["hold_frames"] > 1 and packet["frame"] >= 33 for packet in actions))

    def test_goal_mode_checks_new_mod_before_any_api_call(self):
        calls = []
        game = LocalGame(lambda state: calls.append(state) or goal(), goal_mode=True,
                         hold_frames=15, move_frames=6, move_distance=20)
        game.send(30, capabilities={"movement_pulses": 1})
        game.collect()
        self.assertEqual(calls, [])
        self.assertEqual(game.result["stop_reason"], "mod update required")


if __name__ == "__main__":
    unittest.main()
