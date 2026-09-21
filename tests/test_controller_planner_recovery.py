"""Planner failures release once and retain the same authenticated attempt."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_controller_arming import ImmediatePool, NEW, OLD, ScheduledSocket
from test_controller_recovery import observed
from test_goal_controller import goal
from jev_isaac.controller import Action, Controller


PRIVATE_ERROR = "planner detail that must never be logged"


def frame(stage, number=30, **changes):
    data = observed(number, clear=stage in ("floor_step", "floor_observe"), **changes)
    if stage == "room_objective":
        data["enemies"] = []
    return data


def run(stage, events, *, duration=.4, delay=0, **options):
    transport = ScheduledSocket(events)
    calls, messages, checkpoints, navigators = [], [], [], []

    def fails(data, place):
        if stage == place and data["session"] == "arm0":
            raise RuntimeError(PRIVATE_ERROR)

    def policy(data):
        calls.append((transport.now, copy.deepcopy(data)))
        return goal()

    class Pool(ImmediatePool):
        def submit(self, function, *args):
            value, ready = function(*args), transport.now+delay
            return SimpleNamespace(done=lambda: transport.now >= ready, result=lambda: value)

    class Navigator:
        stop_reason = None
        stats = {"rooms_visited": 1}
        descent_requested = False
        adventure_options = ()

        def __init__(self, **kwargs):
            navigators.append(self)

        def observe_interaction_pause(self, data, now):
            pass

        def observe(self, data):
            fails(data, "floor_observe")

        def rearmed(self, data):
            return Navigator()

        def step(self, data, now):
            fails(data, "floor_step")
            return SimpleNamespace(move="right", shoot="none", stop_reason=None)

    class Switches:
        has_objective = False

        def reset(self):
            self.has_objective = False

        def step(self, data, now):
            self.has_objective = True
            fails(data, "room_objective")
            return SimpleNamespace(move="right", shoot="none", stop_reason=None)

    def local(data, *args):
        fails(data, "local_goal" if stage == "local_goal" else "local_startup")
        return Action("right", "up")

    with patch("jev_isaac.controller.socket.socket", return_value=transport) as sockets, \
         patch("jev_isaac.controller.time.monotonic", side_effect=lambda: transport.now), \
         patch("jev_isaac.controller.concurrent.futures.ThreadPoolExecutor", Pool), \
         patch("jev_isaac.exploration.FloorNavigator", Navigator), \
         patch("jev_isaac.switches.SwitchNavigator", Switches), \
         patch("jev_isaac.navigation.compute_action", side_effect=local):
        result = Controller(policy, duration=duration, max_hz=10, max_latency=.5,
                            goal_mode=True, floor_mode=True, stay_ready=True,
                            startup_guard=stage == "local_startup",
                            on_navigation_stop=checkpoints.append,
                            logger=messages.append, **options).run()
        sockets.assert_called_once()
    return transport, calls, messages, checkpoints, navigators, result


class PlannerRecoveryTests(unittest.TestCase):
    def test_each_local_planner_failure_waits_for_fresh_f8_on_the_same_listener(self):
        for stage, reason in (("room_objective", "room objective navigation failed"),
                              ("local_goal", "local navigation failed"),
                              ("local_startup", "local navigation failed"),
                              ("floor_step", "floor navigation failed"),
                              ("floor_observe", "floor navigation failed")):
            with self.subTest(stage=stage):
                transport, calls, messages, checkpoints, navigators, result = run(stage, [
                    (0, frame(stage), OLD),
                    (.04, frame(stage, 31), OLD),
                    (.08, frame(stage, 32, enabled=False), OLD),
                    (.12, frame(stage, 33), OLD),
                    (.2, frame(stage, 34, session="fresh-arm"), NEW),
                    (.3, frame(stage, 35, session="fresh-arm"), NEW),
                ])
                self.assertEqual(result["errors"], 1)
                self.assertEqual(result["stop_reason"], "duration reached")
                self.assertEqual((result["navigation_stops"], result["navigation_rearms"]), (1, 1))
                self.assertEqual(len(navigators), 2)
                self.assertEqual(len(checkpoints), 1)
                self.assertEqual(checkpoints[0]["reason"], reason)
                self.assertTrue(checkpoints[0]["recoverable"])
                self.assertFalse(any(PRIVATE_ERROR in message for message in messages))
                releases = [(when, packet) for when, packet, _ in transport.sent
                            if packet.get("stop_reason") == "navigation"]
                self.assertEqual(len(releases), 1)
                released_at, release = releases[0]
                self.assertEqual((release["move"], release["shoot"], release["floor_mode"]),
                                 ("none", "none", False))
                self.assertFalse(any(released_at < when < .2 for when, _, _ in transport.sent))
                self.assertFalse(any(released_at < when < .2 for when, _ in calls))
                self.assertTrue(any(when >= .2 and peer == NEW and packet["move"] != "none"
                                    for when, packet, peer in transport.sent))
                self.assertAlmostEqual(transport.now, .4, delta=.011)

    def test_planner_recovery_does_not_renew_the_request_cap(self):
        stage = "local_goal"
        transport, calls, _, checkpoints, navigators, result = run(stage, [
            (0, frame(stage), OLD),
            (.1, frame(stage, 31), OLD),
            (.2, frame(stage, 32, session="fresh-arm"), NEW),
            (.3, frame(stage, 33, session="fresh-arm"), NEW),
            (.4, frame(stage, 34, session="fresh-arm"), NEW),
        ], duration=.5, max_calls=2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["decisions"], 2)
        self.assertEqual(result["stop_reason"], "request cap reached")
        self.assertEqual(result["navigation_rearms"], 1)
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(len(navigators), 2)
        self.assertFalse(any(when >= .3 for when, _, _ in transport.sent))

    def test_inflight_reply_is_counted_but_not_applied_after_a_planner_failure(self):
        transport, calls, _, checkpoints, _, result = run("floor_step", [
            (0, frame("local_goal"), OLD),
            (.02, frame("floor_step", 31), OLD),
            (.08, frame("local_goal", 32), OLD),
            (.2, frame("local_goal", 33, session="fresh-arm"), NEW),
            (.25, frame("local_goal", 34, session="fresh-arm"), NEW),
        ], delay=.1, max_calls=2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["responses_with_usage"], 2)
        self.assertEqual(result["stale_discarded"], 1)
        self.assertEqual(result["goal_updates"], 1)
        self.assertEqual(len(checkpoints), 1)
        self.assertFalse(any(.02 < when < .2 for when, _, _ in transport.sent))

    def test_planner_recovery_keeps_the_original_armed_deadline(self):
        stage = "floor_step"
        transport, calls, _, checkpoints, _, result = run(stage, [
            (0, frame(stage, enabled=False), OLD),
            (5, frame(stage, 31), OLD),
            (5.1, frame(stage, 32, session="fresh-arm"), NEW),
            (5.2, frame(stage, 33, session="fresh-arm"), NEW),
        ], duration=.3, wait_for_arm=True)
        self.assertEqual(calls, [])
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(result["navigation_rearms"], 1)
        self.assertEqual(result["stop_reason"], "duration reached")
        self.assertAlmostEqual(transport.now, 5.3, delta=.011)


if __name__ == "__main__":
    unittest.main()
