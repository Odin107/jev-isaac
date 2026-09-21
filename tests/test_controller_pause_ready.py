"""Bounded floor pause/rearm scenarios with fake time, transport and policy."""
import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_controller_arming import OLD, NEW, ScheduledSocket, ImmediatePool, state
from test_floor_controller import changes
from test_goal_controller import goal
from jev_isaac.cli import main
from jev_isaac.controller import Action, Controller
from jev_isaac.exploration import FloorNavigator


def observed(frame, **extra):
    data = state(frame)
    data.update(changes())
    data.update(extra)
    return data


def run_floor(events, *, completion_delay=0, **options):
    transport, calls, messages = ScheduledSocket(events), [], []
    def policy(data):
        calls.append((transport.now, data))
        return goal()
    class Pool(ImmediatePool):
        def submit(self, function, *args):
            value, ready_at = function(*args), transport.now+completion_delay
            return SimpleNamespace(done=lambda: transport.now >= ready_at, result=lambda: value)
    with patch("jev_isaac.controller.socket.socket", return_value=transport), \
         patch("jev_isaac.controller.time.monotonic", side_effect=lambda: transport.now), \
         patch("jev_isaac.controller.concurrent.futures.ThreadPoolExecutor", Pool), \
         patch("jev_isaac.navigation.compute_action", return_value=Action("right", "up")), \
         patch("jev_isaac.exploration.FloorNavigator", wraps=FloorNavigator) as explorer:
        result = Controller(policy, duration=options.pop("duration", .4), max_hz=10,
                            goal_mode=True, floor_mode=True, logger=messages.append,
                            **options).run()
        explorer.assert_called()
    return transport, calls, messages, result


class ControllerPauseReadyTests(unittest.TestCase):
    def test_pause_waits_quietly_then_rearms_on_a_new_endpoint_with_the_same_map(self):
        transport, calls, messages, result = run_floor([
            (0, observed(30), OLD),
            (.05, observed(31, enabled=False, paused=True), OLD),
            (.1, observed(32, enabled=False, paused=True), OLD),
            (.15, observed(33, enabled=False), OLD),
            (.2, observed(34, session="new-arm"), NEW),
        ], stay_ready=True, wait_for_arm=True)
        self.assertFalse(any(.05 <= when < .2 for when, _ in calls))
        self.assertFalse(any(.05 <= when < .2 for when, _, _ in transport.sent))
        self.assertTrue(any(when >= .2 for when, _ in calls))
        self.assertTrue(any(packet["move"] == "right" and address == NEW
                            for _, packet, address in transport.sent))
        self.assertEqual(result["floor_progress"]["rooms_visited"], 1)
        self.assertEqual(result["stop_reason"], "duration reached")
        self.assertEqual(sum("still ready" in message for message in messages), 1)
        self.assertAlmostEqual(transport.now, .4, delta=.011)

    def test_inflight_reply_completed_while_disabled_is_counted_but_never_applied(self):
        transport, calls, _, result = run_floor([
            (0, observed(30), OLD),
            (.05, observed(31, enabled=False, paused=True), OLD),
            (.15, observed(32, enabled=False), OLD),
            (.2, observed(33, session="new-arm"), NEW),
        ], completion_delay=.1, stay_ready=True, wait_for_arm=True)
        self.assertEqual(result["stale_discarded"], 1)
        self.assertEqual(result["responses_with_usage"], 2)
        self.assertEqual(result["goal_updates"], 1)
        self.assertFalse(any(.05 <= when < .2 for when, _ in calls))
        self.assertFalse(any(when < .2 for when, _, _ in transport.sent))

    def test_waiting_and_rearming_do_not_extend_the_original_deadline(self):
        transport, _, _, result = run_floor([
            (0, observed(30, enabled=False), OLD),
            (5, observed(31, session="first-arm"), OLD),
            (5.05, observed(32, enabled=False), OLD),
            (5.25, observed(33, session="second-arm"), NEW),
        ], duration=.3, stay_ready=True, wait_for_arm=True)
        self.assertAlmostEqual(transport.now, 5.3, delta=.011)
        self.assertEqual(result["stop_reason"], "duration reached")

    def test_rearming_keeps_the_original_request_cap(self):
        _, calls, _, result = run_floor([
            (0, observed(30), OLD),
            (.05, observed(31, enabled=False), OLD),
            (.2, observed(32, session="second-arm"), NEW),
            (.35, observed(33, session="second-arm"), NEW),
        ], duration=.5, max_calls=2, stay_ready=True, wait_for_arm=True)
        self.assertEqual(result["stop_reason"], "request cap reached")
        self.assertEqual(result["decisions"], 2)
        self.assertEqual(len(calls), 2)

    def test_default_floor_controller_still_stops_when_disarmed(self):
        transport, calls, _, result = run_floor([
            (0, observed(30), OLD), (.05, observed(31, enabled=False), OLD),
            (.2, observed(32, session="second-arm"), NEW),
        ])
        self.assertEqual(result["stop_reason"], "control disabled")
        self.assertAlmostEqual(transport.now, .05)
        self.assertEqual(len(calls), 1)

    def test_death_run_and_floor_changes_still_end_a_ready_session(self):
        for kind, reason in (("death", "player died"), ("run", "run changed"), ("floor", "floor changed")):
            with self.subTest(kind=kind):
                changed = observed(31, enabled=False)
                if kind == "death": changed["player"]["dead"] = True
                elif kind == "run": changed["run_id"] = "other-run"
                else: changed["floor"]["id"] = "other-floor"
                _, calls, _, result = run_floor([(0, observed(30), OLD), (.05, changed, OLD)],
                                                stay_ready=True, wait_for_arm=True)
                self.assertEqual(result["stop_reason"], reason)
                self.assertEqual(len(calls), 1)

    def test_stay_ready_requires_floor_mode_and_cli_enables_wait_for_arm(self):
        with self.assertRaises(ValueError):
            Controller(lambda _: goal(), stay_ready=True)
        with patch("jev_isaac.key_prompt.prompt_key_window") as prompt, \
             patch("jev_isaac.cli.Controller") as controller, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["run", "--policy", "jev-goal", "--stay-ready", "--key-window"]), 1)
            prompt.assert_not_called()
            controller.assert_not_called()
        with patch("jev_isaac.key_prompt.prompt_key_window", return_value="offline-key"), \
             patch("jev_isaac.goals.GoalClient") as client, \
             patch("jev_isaac.cli.Controller") as controller, \
             contextlib.redirect_stdout(io.StringIO()):
            client.return_value.model = "offline"
            controller.return_value.run.return_value = {"stop_reason": "duration reached"}
            self.assertEqual(main(["run", "--policy", "jev-goal", "--floor", "--stay-ready", "--key-window"]), 0)
            self.assertTrue(controller.call_args.kwargs["stay_ready"])
            self.assertTrue(controller.call_args.kwargs["wait_for_arm"])

    def test_manual_restart_retains_key_and_excludes_the_old_run_instead_of_the_new_one(self):
        with patch("jev_isaac.key_prompt.prompt_key_window", return_value="offline-key") as prompt, \
             patch("jev_isaac.goals.GoalClient") as client, \
             patch("jev_isaac.cli.Controller") as controller, \
             contextlib.redirect_stdout(io.StringIO()):
            client.return_value.model = "offline"
            controller.return_value.run.side_effect = [
                {"stop_reason": "run changed", "first_observation": {"run_id": "old-run"},
                 "last_observation": {"run_id": "new-run"}},
                {"stop_reason": "duration reached", "first_observation": {"run_id": "new-run"},
                 "last_observation": {"run_id": "new-run"}},
            ]
            self.assertEqual(main(["run", "--policy", "jev-goal", "--floor", "--stay-ready", "--key-window"]), 0)
            prompt.assert_called_once_with("TypeSafe")
            client.assert_called_once()
            self.assertEqual(controller.call_count, 2)
            self.assertIsNone(controller.call_args_list[0].kwargs.get("excluded_run_id"))
            self.assertEqual(controller.call_args_list[1].kwargs["excluded_run_id"], "old-run")
            client.return_value.warm_connect.assert_called_once()
            self.assertEqual(client.return_value.close.call_count, 2)


if __name__ == "__main__":
    unittest.main()
