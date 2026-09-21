"""Stopped navigation segments retain bounded, immutable reproduction evidence."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_controller_recovery import OLD, NEW, ROUTE_FAILURE, observed, run
from jev_isaac.controller import Stats
from jev_isaac.protocol import Observation


class NavigationDiagnosticSnapshotTests(unittest.TestCase):
    def test_snapshot_is_independent_of_later_observation_stats_and_controls(self):
        data = observed(40, fault=ROUTE_FAILURE)
        navigator = SimpleNamespace(
            stats={"rooms_visited": 3, "doors_traversed": 2},
            pickup_stats={"attempts": 4},
            adventure_stats={"selected": 2, "events": [{"reason": "prop changed"}]})
        stats = Stats(decisions=12, actions_sent=18,
                      recent_local_controls=[{"frame": 39, "player": {"x": 100}}])
        before = copy.deepcopy(data)
        stats.record_navigation_stop(ROUTE_FAILURE, Observation(data), navigator, recoverable=True)
        data["player"]["x"] += 500
        navigator.stats["rooms_visited"] = 0
        navigator.pickup_stats["attempts"] = 0
        navigator.adventure_stats["events"][0]["reason"] = "changed later"
        stats.recent_local_controls[0]["player"]["x"] = 900

        saved = stats.navigation_stop_snapshots[0]
        self.assertEqual(saved["observation"], before)
        self.assertEqual(saved["floor_progress"], {"rooms_visited": 3, "doors_traversed": 2})
        self.assertEqual(saved["pickup_progress"], {"attempts": 4})
        self.assertEqual(saved["adventure_progress"]["events"][0]["reason"], "prop changed")
        self.assertEqual(saved["recent_local_controls"][0]["player"]["x"], 100)
        self.assertEqual((saved["decisions"], saved["actions_sent"]), (12, 18))
        self.assertEqual(saved["run_id"], before["run_id"])
        self.assertEqual(saved["floor_id"], before["floor"]["id"])
        self.assertEqual(json.loads(json.dumps(saved)), saved)

    def test_only_latest_eight_stops_and_120_controls_are_retained(self):
        stats = Stats(recent_local_controls=[{"frame": frame} for frame in range(150)])
        for frame in range(30, 42):
            stats.record_navigation_stop(str(frame), Observation(observed(frame)), None,
                                         recoverable=False)
        self.assertEqual([s["frame"] for s in stats.navigation_stop_snapshots], list(range(34, 42)))
        for saved in stats.navigation_stop_snapshots:
            self.assertEqual(len(saved["recent_local_controls"]), 120)
            self.assertEqual(saved["recent_local_controls"][0]["frame"], 30)
            self.assertEqual(saved["floor_progress"], {})
            self.assertEqual(saved["pickup_progress"], {})
            self.assertEqual(saved["adventure_progress"], {})

    def test_summary_copies_snapshots_for_report_consumers(self):
        stats = Stats()
        stats.record_navigation_stop(ROUTE_FAILURE, Observation(observed()), None, recoverable=True)
        summary = stats.summary()
        summary["navigation_stop_snapshots"][0]["observation"]["player"]["x"] += 1
        self.assertNotEqual(summary["navigation_stop_snapshots"], stats.navigation_stop_snapshots)


class ControllerNavigationDiagnosticTests(unittest.TestCase):
    def test_later_waiting_packets_do_not_replace_failed_observation(self):
        failure = observed(31, fault=ROUTE_FAILURE)
        latest = observed(40, enabled=False)
        latest["player"]["x"] += 100
        _, _, _, result, _ = run([(0, failure, OLD), (.2, latest, OLD)])
        saved = result["navigation_stop_snapshots"]
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["observation"], failure)
        self.assertEqual(result["last_observation"], latest)
        self.assertTrue(saved[0]["recoverable"])

    def test_rebuild_keeps_failed_segment_separate_from_current_floor_totals(self):
        _, _, _, result, navigators = run([
            (0, observed(30), OLD),
            (.1, observed(31, fault=ROUTE_FAILURE), OLD),
            (.2, observed(32, session="fresh-arm"), NEW),
            (.3, observed(33, session="fresh-arm"), NEW),
        ])
        self.assertEqual(len(navigators), 2)
        saved = result["navigation_stop_snapshots"][0]
        self.assertEqual(saved["floor_progress"]["rooms_visited"], 1)
        self.assertEqual(result["floor_progress"]["rooms_visited"], 1)
        self.assertEqual(saved["recent_local_controls"][-1]["frame"], 30)
        self.assertEqual(result["recent_local_controls"][-1]["frame"], 33)
        self.assertEqual(result["floor_history"], [])

    def test_repeated_recoveries_capture_each_reason_once_and_bound_history(self):
        events = [(i * .03, observed(30 + i, session=f"arm-{i}", fault=ROUTE_FAILURE),
                   NEW if i % 2 else OLD) for i in range(10)]
        _, _, _, result, navigators = run(events)
        self.assertEqual(result["navigation_stops"], 10)
        self.assertEqual(len(navigators), 10)
        self.assertEqual([s["frame"] for s in result["navigation_stop_snapshots"]], list(range(32, 40)))

    def test_observe_and_terminal_failures_keep_snapshots_without_changing_recovery_counts(self):
        for during_observe, stay_ready, reason in (
                (True, True, ROUTE_FAILURE), (False, False, ROUTE_FAILURE),
                (False, True, "incomplete floor observation")):
            with self.subTest(observe=during_observe, ready=stay_ready, reason=reason):
                _, _, _, result, _ = run([(0, observed(fault=reason), OLD)],
                                        fault_during_observe=during_observe, stay_ready=stay_ready)
                saved = result["navigation_stop_snapshots"]
                self.assertEqual(len(saved), 1)
                self.assertEqual(saved[0]["reason"], reason)
                should_recover = stay_ready and reason in (ROUTE_FAILURE, "incomplete floor observation")
                self.assertEqual(saved[0]["recoverable"], should_recover)
                self.assertEqual(result["navigation_stops"], int(should_recover))

    def test_full_snapshots_are_returned_but_omitted_from_console_summary(self):
        _, _, messages, result, _ = run([(0, observed(fault=ROUTE_FAILURE), OLD)])
        self.assertIn("navigation_stop_snapshots", result)
        console_summary = json.loads(messages[-1])
        self.assertNotIn("navigation_stop_snapshots", console_summary)
        self.assertEqual(console_summary["navigation_stops"], 1)

    def test_checkpoint_receives_independent_snapshot_before_waiting_or_rearm(self):
        checkpoints = []
        def checkpoint(snapshot):
            checkpoints.append(copy.deepcopy(snapshot))
            snapshot["observation"]["player"]["x"] += 500
        first = observed(30, fault=ROUTE_FAILURE)
        second = observed(32, session="fresh-arm", fault="door traversal timed out")
        _, _, _, result, _ = run([(0, first, OLD), (.1, second, NEW)],
                                on_navigation_stop=checkpoint)
        self.assertEqual([item["frame"] for item in checkpoints], [30, 32])
        self.assertEqual([item["reason"] for item in checkpoints],
                         [ROUTE_FAILURE, "door traversal timed out"])
        self.assertEqual(result["navigation_stop_snapshots"], checkpoints)
        self.assertEqual(checkpoints[0]["observation"], first)
        self.assertEqual(checkpoints[1]["observation"], second)

    def test_checkpoint_failure_does_not_break_recovery_or_count_as_api_error(self):
        def broken_checkpoint(snapshot):
            raise OSError("private diagnostic detail must not appear")
        transport, _, messages, result, navigators = run([
            (0, observed(fault=ROUTE_FAILURE), OLD),
            (.1, observed(31, session="fresh-arm"), NEW),
        ], on_navigation_stop=broken_checkpoint)
        self.assertEqual(result["errors"], 0)
        self.assertEqual(result["navigation_rearms"], 1)
        self.assertEqual(len(navigators), 2)
        self.assertEqual(transport.sent[0][1]["move"], "none")
        self.assertIs(transport.sent[0][1]["floor_mode"], False)
        self.assertEqual(transport.sent[0][1]["stop_reason"], "navigation")
        self.assertTrue(any("Navigation checkpoint failed (OSError)." == message for message in messages))
        self.assertFalse(any("private diagnostic detail" in message for message in messages))

    def test_terminal_navigation_stop_also_checkpoints_immediately(self):
        checkpoints = []
        _, _, _, result, _ = run([(0, observed(fault="incomplete floor observation"), OLD)],
                                on_navigation_stop=checkpoints.append, stay_ready=False)
        self.assertEqual(len(checkpoints), 1)
        self.assertIs(checkpoints[0]["recoverable"], False)
        self.assertEqual(checkpoints[0], result["navigation_stop_snapshots"][0])


if __name__ == "__main__":
    unittest.main()
