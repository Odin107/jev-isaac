"""Adventure launcher wiring only; no API, key UI, or game access."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_isaac.cli import main


class AdventureCliTests(unittest.TestCase):
    def test_invalid_combinations_fail_before_key_or_controller(self):
        combinations = [
            ["--adventure"], ["--policy", "jev-goal", "--adventure"],
            ["--policy", "jev", "--floor", "--adventure"],
            ["--policy", "jev-goal", "--floor", "--continue-floors"],
            ["--policy", "jev-goal", "--floor", "--adventure", "--one-room"],
            ["--continue-floors"],
        ]
        for options in combinations:
            with self.subTest(options=options), patch("jev_isaac.cli.read_key") as key, \
                 patch("jev_isaac.cli.Controller") as controller, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["run", *options]), 1)
                key.assert_not_called()
                controller.assert_not_called()

    def test_adventure_selects_shared_client_and_preserves_limits(self):
        for continuing in (False, True):
            with self.subTest(continuing=continuing), \
                 patch("jev_isaac.cli.read_key", return_value="offline-key"), \
                 patch("jev_isaac.strategy.AdventureClient") as adventure, \
                 patch("jev_isaac.goals.GoalClient") as tactical, \
                 patch("jev_isaac.cli.Controller") as controller, \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                adventure.return_value.model = "offline-adventure"
                controller.return_value.run.return_value = {}
                options = ["run", "--policy", "jev-goal", "--floor", "--adventure",
                           "--wait-for-arm", "--restart-on-death", "--stay-ready",
                           "--duration", "900", "--max-calls", "1200"]
                if continuing:
                    options.append("--continue-floors")
                self.assertEqual(main(options), 0)
                tactical.assert_not_called()
                self.assertEqual(controller.call_args.args[0], adventure.return_value.decide)
                kwargs = controller.call_args.kwargs
                self.assertTrue(kwargs["adventure_mode"])
                self.assertEqual(kwargs["continue_floors"], continuing)
                self.assertTrue(kwargs["floor_mode"])
                self.assertTrue(kwargs["goal_mode"])
                self.assertTrue(kwargs["wait_for_arm"])
                self.assertTrue(kwargs["stay_ready"])
                self.assertEqual((kwargs["duration"], kwargs["max_calls"], kwargs["max_hz"]), (900, 1200, 2))
                adventure.return_value.warm_connect.assert_called_once()
                adventure.return_value.close.assert_called_once()
                self.assertNotIn("offline-key", output.getvalue())

    def test_existing_room_and_floor_modes_keep_tactical_client(self):
        for floor in (False, True):
            with self.subTest(floor=floor), patch("jev_isaac.cli.read_key", return_value="offline-key"), \
                 patch("jev_isaac.strategy.AdventureClient") as adventure, \
                 patch("jev_isaac.goals.GoalClient") as tactical, \
                 patch("jev_isaac.cli.Controller") as controller, \
                 contextlib.redirect_stdout(io.StringIO()):
                tactical.return_value.model = "offline-tactical"
                controller.return_value.run.return_value = {}
                options = ["run", "--policy", "jev-goal"] + (["--floor"] if floor else [])
                self.assertEqual(main(options), 0)
                adventure.assert_not_called()
                tactical.assert_called_once()
                self.assertFalse(controller.call_args.kwargs["adventure_mode"])
                self.assertFalse(controller.call_args.kwargs["continue_floors"])

    def test_cancelled_adventure_key_never_constructs_client(self):
        with patch("jev_isaac.cli.read_key", return_value=""), \
             patch("jev_isaac.strategy.AdventureClient") as client, \
             patch("jev_isaac.cli.Controller") as controller, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["run", "--policy", "jev-goal", "--floor", "--adventure"]), 1)
            client.assert_not_called()
            controller.assert_not_called()

    def test_adventure_client_closes_when_listener_fails(self):
        with patch("jev_isaac.cli.read_key", return_value="offline-key"), \
             patch("jev_isaac.strategy.AdventureClient") as client, \
             patch("jev_isaac.cli.Controller") as controller, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            controller.return_value.run.side_effect = OSError("offline listener failure")
            self.assertEqual(main(["run", "--policy", "jev-goal", "--floor", "--adventure"]), 1)
            client.return_value.close.assert_called_once()

    def test_launcher_branches_enable_adventure_without_expanding_budgets(self):
        launcher = (Path(__file__).resolve().parents[1] / "start-floor.cmd").read_text()
        commands = [line for line in launcher.splitlines() if "launch.py run" in line]
        self.assertEqual(len(commands), 2)
        for command in commands:
            for option in ("--policy jev-goal", "--floor", "--adventure", "--continue-floors",
                           "--duration 900", "--max-calls 1200", "--wait-for-arm",
                           "--restart-on-death", "--stay-ready", "--key-window"):
                self.assertIn(option, command)


if __name__ == "__main__":
    unittest.main()
