import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_isaac.cli import main
from jev_isaac.jev import JevHTTPError


class AccessCheckTests(unittest.TestCase):
    def test_goal_mode_uses_goal_client_and_local_control(self):
        with patch("jev_isaac.key_prompt.prompt_key_window", return_value="offline-goal-key"), \
             patch("jev_isaac.goals.GoalClient") as client, \
             patch("jev_isaac.cli.Controller") as controller, \
             contextlib.redirect_stdout(io.StringIO()) as capture:
            client.return_value.model = "offline"
            controller.return_value.run.return_value = {}
            self.assertEqual(main(["run", "--policy", "jev-goal", "--key-window"]), 0)
            self.assertTrue(controller.call_args.kwargs["goal_mode"])
            self.assertEqual(controller.call_args.kwargs["max_hz"], 2)
            self.assertEqual(controller.call_args.kwargs["max_latency"], .5)
            self.assertNotIn("offline-goal-key", capture.getvalue())
            client.return_value.close.assert_called_once()

    def test_live_jev_selects_measured_timing_and_closes_connection(self):
        for failure in (False, True):
            with self.subTest(failure=failure), \
                 patch("jev_isaac.key_prompt.prompt_key_window", return_value="live-test-key"), \
                 patch("jev_isaac.jev.JevClient") as client, \
                 patch("jev_isaac.cli.Controller") as controller, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                client.return_value.model = "jev-test"
                controller.return_value.run.return_value = {}
                if failure:
                    controller.return_value.run.side_effect = OSError("Synthetic listener failure")
                self.assertEqual(main(["run", "--policy", "jev", "--key-window"]), 1 if failure else 0)
                self.assertEqual(controller.call_args.kwargs["max_latency"], .5)
                self.assertEqual(controller.call_args.kwargs["hold_frames"], 15)
                self.assertEqual(controller.call_args.kwargs["move_frames"], 6)
                self.assertEqual(controller.call_args.kwargs["move_distance"], 20)
                self.assertEqual(client.call_args.kwargs["choice_policy"], "argmax")
                self.assertEqual(client.call_args.kwargs["timeout"], 2)
                client.return_value.close.assert_called_once()

    def test_timing_window_requests_only_four_synthetic_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "timing.json"
            with patch("jev_isaac.key_prompt.prompt_key_window", return_value="timing-test-key"), \
                 patch("jev_isaac.timing.run_timing", return_value={"success":True,"requests":4}) as timing, \
                 patch("jev_isaac.cli.Controller") as controller, \
                 contextlib.redirect_stdout(io.StringIO()) as capture:
                self.assertEqual(main(["measure-latency", "--key-window", "--report", str(report)]), 0)
                timing.assert_called_once_with("timing-test-key", provider="typesafe", model=None, samples=4)
                controller.assert_not_called()
                self.assertNotIn("timing-test-key", capture.getvalue() + report.read_text())

    def test_window_key_makes_one_request_without_terminal_prompt(self):
        with patch.dict("os.environ", {"TYPESAFE_API_KEY":"old-environment-key"}, clear=True), \
             patch("jev_isaac.key_prompt.prompt_key_window", return_value="window-test-key") as prompt, \
             patch("getpass.getpass") as terminal, patch("jev_isaac.jev.JevClient") as client, \
             contextlib.redirect_stdout(io.StringIO()) as capture:
            client.return_value.decide.return_value = SimpleNamespace(
                model="jev-test", latency_ms=45, move="left", shoot="right",
                usage={"input_tokens":123, "output_tokens":12})
            self.assertEqual(main(["check-access", "--key-window"]), 0)
            prompt.assert_called_once_with("TypeSafe")
            terminal.assert_not_called()
            client.return_value.decide.assert_called_once()
            self.assertEqual(client.call_args.kwargs["api_key"], "window-test-key")
            self.assertNotIn("window-test-key", capture.getvalue())

    def test_window_cancel_makes_no_request_for_check_or_gameplay(self):
        for command in (["check-access"], ["run", "--policy", "jev"],
                        ["run", "--policy", "jev-goal"], ["measure-latency"]):
            with self.subTest(command=command), \
                 patch("jev_isaac.key_prompt.prompt_key_window", return_value=""), \
                 patch("jev_isaac.jev.JevClient") as client, \
                 patch("jev_isaac.cli.Controller") as controller, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(command + ["--key-window"]), 1)
                client.assert_not_called()
                controller.assert_not_called()

    def test_window_unavailable_makes_no_request(self):
        with patch("jev_isaac.key_prompt.prompt_key_window", side_effect=RuntimeError("Key window unavailable")), \
             patch("jev_isaac.jev.JevClient") as client, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["check-access", "--key-window"]), 1)
            client.assert_not_called()

    def test_hidden_key_makes_one_request_and_never_enters_report(self):
        with tempfile.TemporaryDirectory() as folder:
            report = Path(folder) / "report.json"
            capture = io.StringIO()
            with patch.dict("os.environ", {}, clear=True), patch("getpass.getpass", return_value="test-secret-123"), \
                 patch("jev_isaac.jev.JevClient") as client, contextlib.redirect_stdout(capture):
                client.return_value.decide.return_value = SimpleNamespace(
                    model="jev-test", latency_ms=45, move="left", shoot="right",
                    usage={"input_tokens":123, "output_tokens":12})
                status = main(["check-access", "--report", str(report)])
            self.assertEqual(status, 0)
            client.return_value.decide.assert_called_once()
            self.assertEqual(client.call_args.kwargs["provider"], "typesafe")
            self.assertNotIn("test-secret-123", capture.getvalue() + report.read_text())
            self.assertEqual(json.loads(report.read_text())["requests"], 1)

    def test_provider_error_reports_failure_without_retry(self):
        with patch.dict("os.environ", {}, clear=True), patch("getpass.getpass", return_value="test-key"), \
             patch("jev_isaac.jev.JevClient") as client, contextlib.redirect_stdout(io.StringIO()):
            client.return_value.decide.side_effect = JevHTTPError(401)
            self.assertEqual(main(["check-access"]), 1)
            client.return_value.decide.assert_called_once()

    def test_blank_key_makes_no_request(self):
        with patch.dict("os.environ", {}, clear=True), patch("getpass.getpass", return_value=""), \
             patch("jev_isaac.jev.JevClient") as client, contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["check-access"]), 1)
            client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
