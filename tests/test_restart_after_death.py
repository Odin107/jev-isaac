"""Restart-session lifecycle checks using fake clients and local fake time only."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_controller_arming import OLD, NEW, run_scenario, state
from jev_isaac.cli import main


def attempt(reason, run_id="dead-run"):
    return {"stop_reason": reason, "decisions": 2,
            "last_observation": {"run_id": run_id, "player": {"dead": reason == "player died"}}}


class RestartAfterDeathTests(unittest.TestCase):
    def test_death_reuses_key_and_client_but_starts_a_new_controller_and_preserves_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder)/"live-floor.json"
            preserved = Path(folder)/"live-floor-attempt-earlier.json"
            preserved.write_text('{"previous_session":true}')
            first = attempt("player died")
            final = attempt("control disabled", "new-run")
            with patch("jev_isaac.key_prompt.prompt_key_window", return_value="offline-session-secret") as prompt, \
                 patch("jev_isaac.goals.GoalClient") as client_factory, \
                 patch("jev_isaac.cli.Controller") as controller, \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                client = client_factory.return_value
                client.model = "offline-model"
                def second_run():
                    self.assertGreaterEqual(client.close.call_count, 1)
                    self.assertEqual(json.loads(report.read_text())["stop_reason"], "player died")
                    archived = [p for p in Path(folder).glob("live-floor-attempt-*.json") if p != preserved]
                    self.assertEqual(len(archived), 1)
                    self.assertEqual(json.loads(archived[0].read_text())["stop_reason"], "player died")
                    return copy.deepcopy(final)
                controller.side_effect = [SimpleNamespace(run=lambda: copy.deepcopy(first)),
                                          SimpleNamespace(run=second_run)]
                self.assertEqual(main(["run", "--policy", "jev-goal", "--floor", "--restart-on-death",
                                       "--key-window", "--report", str(report)]), 0)
                prompt.assert_called_once_with("TypeSafe")
                client_factory.assert_called_once()
                client.warm_connect.assert_called_once()
                self.assertEqual(client.close.call_count, 2)
                self.assertEqual(controller.call_count, 2)
                initial, restarted = controller.call_args_list
                self.assertIsNone(initial.kwargs.get("excluded_run_id"))
                self.assertEqual(restarted.kwargs["excluded_run_id"], "dead-run")
                self.assertTrue(initial.kwargs["wait_for_arm"] and restarted.kwargs["wait_for_arm"])
                self.assertIs(initial.args[0], restarted.args[0])
                self.assertEqual(initial.kwargs["duration"], restarted.kwargs["duration"])
                archived = [p for p in Path(folder).glob("live-floor-attempt-*.json") if p != preserved]
                self.assertEqual(len(archived), 2)
                self.assertCountEqual([json.loads(p.read_text())["stop_reason"] for p in archived],
                                      ["player died", "control disabled"])
                self.assertEqual(json.loads(report.read_text())["stop_reason"], "control disabled")
                self.assertEqual(preserved.read_text(), '{"previous_session":true}')
                all_text = output.getvalue()+"".join(p.read_text() for p in Path(folder).glob("*.json"))
                self.assertNotIn("offline-session-secret", all_text)

    def test_non_death_stops_never_repeat(self):
        for reason in ("control disabled", "request cap reached", "duration reached", "decision failed",
                       "three consecutive invalid replies", "floor navigation failed",
                       "no safe route to open door", "floor cleared", "user stopped"):
            with self.subTest(reason=reason), \
                 patch("jev_isaac.key_prompt.prompt_key_window", return_value="offline-key"), \
                 patch("jev_isaac.goals.GoalClient") as client, \
                 patch("jev_isaac.cli.Controller") as controller, \
                 contextlib.redirect_stdout(io.StringIO()):
                client.return_value.model = "offline"
                controller.return_value.run.return_value = attempt(reason)
                self.assertEqual(main(["run", "--policy", "jev-goal", "--floor",
                                       "--restart-on-death", "--key-window"]), 0)
                controller.assert_called_once()
                client.return_value.close.assert_called_once()

    def test_death_without_a_valid_run_id_does_not_guess_or_repeat(self):
        for run_id in (None, "", 42):
            with self.subTest(run_id=run_id), \
                 patch("jev_isaac.key_prompt.prompt_key_window", return_value="offline-key"), \
                 patch("jev_isaac.goals.GoalClient") as client, \
                 patch("jev_isaac.cli.Controller") as controller, \
                 contextlib.redirect_stdout(io.StringIO()):
                client.return_value.model = "offline"
                controller.return_value.run.return_value = attempt("player died", run_id)
                self.assertEqual(main(["run", "--policy", "jev-goal", "--floor",
                                       "--restart-on-death", "--key-window"]), 0)
                controller.assert_called_once()

    def test_restart_option_requires_floor_mode_before_requesting_a_key(self):
        with patch("jev_isaac.key_prompt.prompt_key_window") as prompt, \
             patch("jev_isaac.goals.GoalClient") as client, \
             patch("jev_isaac.cli.Controller") as controller, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["run", "--policy", "jev-goal", "--restart-on-death", "--key-window"]), 1)
            prompt.assert_not_called()
            client.assert_not_called()
            controller.assert_not_called()

    def test_default_floor_mode_still_returns_after_a_death(self):
        with patch("jev_isaac.key_prompt.prompt_key_window", return_value="offline-key"), \
             patch("jev_isaac.goals.GoalClient") as client, \
             patch("jev_isaac.cli.Controller") as controller, \
             contextlib.redirect_stdout(io.StringIO()):
            client.return_value.model = "offline"
            controller.return_value.run.return_value = attempt("player died")
            self.assertEqual(main(["run", "--policy", "jev-goal", "--floor", "--key-window"]), 0)
            controller.assert_called_once()
            client.return_value.close.assert_called_once()

    def test_excluded_dead_run_cannot_arm_or_control_even_with_a_new_session_or_alive_flag(self):
        dead = state(run_id="dead-run", session="old-arm")
        dead["player"]["dead"] = True
        forged_alive = state(31, run_id="dead-run", session="new-arm")
        missing = state(32, session="missing-run")
        missing.pop("run_id")
        empty = state(33, session="empty-run", run_id="")
        transport, calls, result = run_scenario([
            (0, dead, OLD), (.1, forged_alive, OLD), (.2, missing, OLD), (.3, empty, OLD),
            (4, state(1, enabled=False, run_id="fresh-run", session="fresh-arm0"), NEW),
            (5, state(2, run_id="fresh-run", session="fresh-arm1"), NEW),
            (5.1, forged_alive, OLD),
        ], excluded_run_id="dead-run", wait_for_arm=True)
        self.assertAlmostEqual(transport.now, 5.3, delta=.011)
        self.assertTrue(calls)
        self.assertTrue(all(when >= 5 and data["run_id"] == "fresh-run" for when, data in calls))
        self.assertTrue(all(when >= 5 and packet["session"] == "fresh-arm1"
                            for when, packet, _ in transport.sent))
        self.assertEqual(result["first_observation"]["run_id"], "fresh-run")
        self.assertEqual(result["last_observation"]["run_id"], "fresh-run")


if __name__ == "__main__":
    unittest.main()
