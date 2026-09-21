"""Known route failures keep one listener and budget, with explicit fresh rearm."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_controller_arming import ImmediatePool, OLD, NEW, ScheduledSocket
from test_controller_pause_ready import observed as combat_observed
from test_goal_controller import goal
from jev_isaac.controller import Action, Controller, _NavigationRecovery
from jev_isaac.protocol import Observation


ROUTE_FAILURE = "no safe route to open door"


def observed(frame=30, *, clear=True, fault=None, **changes):
    data = combat_observed(frame, **changes)
    data["room"]["clear"] = clear
    if clear:
        data["enemies"] = []
    if fault:
        data["route_fault"] = fault
    return data


def run(events, *, delay=0, fault_during_observe=False, policy=None, local=None, **options):
    transport, calls, messages, navigators = ScheduledSocket(events), [], [], []

    def call(data):
        calls.append((transport.now, data))
        return policy(data) if policy else goal()

    class Pool(ImmediatePool):
        def submit(self, fn, *args):
            # Real executors deliver exceptions from result(), not submit().
            try:
                value, error = fn(*args), None
            except Exception as exc:
                value, error = None, exc
            completed_at = transport.now + delay
            def result():
                if error is not None:
                    raise error
                return value
            return SimpleNamespace(done=lambda: transport.now >= completed_at, result=result)

    class Navigator:
        stop_reason = None
        stats = {"rooms_visited": 1}
        descent_requested = False
        adventure_options = ()

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.frames = []
            navigators.append(self)

        def observe_interaction_pause(self, state, now):
            pass

        def rearmed(self, state):
            fresh = Navigator(**self.kwargs)
            fresh.rearmed_from = self
            fresh.rearm_state = state
            return fresh

        def observe(self, state):
            self.frames.append((state["session"], state["frame"]))
            if fault_during_observe and state.get("route_fault"):
                self.stop_reason = state["route_fault"]

        def step(self, state, now):
            self.stop_reason = state.get("route_fault")
            return SimpleNamespace(move="none" if self.stop_reason else "right",
                                   shoot="none", stop_reason=self.stop_reason)

    with patch("jev_isaac.controller.socket.socket", return_value=transport), \
         patch("jev_isaac.controller.time.monotonic", side_effect=lambda: transport.now), \
         patch("jev_isaac.controller.concurrent.futures.ThreadPoolExecutor", Pool), \
         patch("jev_isaac.exploration.FloorNavigator", Navigator), \
         patch("jev_isaac.navigation.compute_action", side_effect=local or (lambda *_: Action("left", "up"))):
        result = Controller(call, duration=options.pop("duration", .4),
                            max_hz=options.pop("max_hz", 10), max_latency=.5,
                            goal_mode=True, floor_mode=True,
                            stay_ready=options.pop("stay_ready", True),
                            logger=messages.append, **options).run()
    return transport, calls, messages, result, navigators


class NavigationRecoveryIdentityTests(unittest.TestCase):
    def test_only_unseen_live_arm_on_same_run_floor_and_fresh_frame_is_accepted(self):
        recovery = _NavigationRecovery("run-1", "floor-1", frozenset({"arm0", "failed-arm"}), 40)
        current = observed(40, session="new-arm")
        self.assertTrue(recovery.accepts(Observation(current)))
        for changes in ({"session": "failed-arm"}, {"session": "arm0"}, {"frame": 39},
                        {"enabled": False}, {"paused": True}, {"run_id": "other-run"}):
            candidate = copy.deepcopy(current)
            candidate.update(changes)
            self.assertFalse(recovery.accepts(Observation(candidate)), changes)
        candidate = copy.deepcopy(current)
        candidate["player"]["dead"] = True
        self.assertFalse(recovery.accepts(Observation(candidate)))
        candidate = copy.deepcopy(current)
        candidate["floor"]["id"] = "other-floor"
        self.assertFalse(recovery.accepts(Observation(candidate)))

    def test_later_waiting_frames_raise_replay_floor_but_room_may_change_manually(self):
        recovery = _NavigationRecovery("run-1", "floor-1", frozenset({"arm0"}), 30)
        recovery.observe(Observation(observed(50)))
        self.assertFalse(recovery.accepts(Observation(observed(49, session="new-arm"))))
        current = observed(50, session="new-arm", room_id="manually-entered-room")
        current["floor"]["room_index"] = 2
        self.assertTrue(recovery.accepts(Observation(current)))


class ControllerRecoveryTests(unittest.TestCase):
    def test_known_failures_release_once_and_ignore_same_enabled_session(self):
        for reason in (ROUTE_FAILURE, "stuck while approaching door", "door traversal timed out",
                       "pickup animation timed out"):
            with self.subTest(reason=reason):
                transport, calls, messages, result, navigators = run([
                    (0, observed(fault=reason), OLD),
                    (.05, observed(31), OLD),
                    (.1, observed(32, clear=False), OLD),
                    (.2, observed(33, paused=True, enabled=False), OLD),
                    (.3, observed(34, clear=False), OLD),
                ], startup_guard=True)
                self.assertEqual(calls, [])
                self.assertEqual(len(transport.sent), 1)
                self.assertEqual(transport.sent[0][1]["move"], "none")
                self.assertEqual(transport.sent[0][1]["shoot"], "none")
                self.assertIs(transport.sent[0][1]["floor_mode"], False)
                self.assertEqual(len(navigators), 1)
                self.assertEqual(result["navigation_stops"], 1)
                self.assertEqual(result["navigation_rearms"], 0)
                self.assertEqual(result["last_navigation_stop"]["reason"], reason)
                self.assertEqual(result["stop_reason"], "duration reached")
                self.assertEqual(sum("Navigation paused:" in m for m in messages), 1)
                self.assertTrue(any("30 requests remain" in m and "timer keeps running" in m for m in messages))
                self.assertAlmostEqual(transport.now, .4, delta=.011)

    def test_new_arm_replaces_endpoint_immediately_and_rebuilds_only_on_rearm(self):
        transport, calls, messages, result, navigators = run([
            (0, observed(fault=ROUTE_FAILURE), OLD),
            (.05, observed(31), OLD),
            (.1, observed(32, session="fresh-arm"), NEW),
            (.15, observed(33, session="fresh-arm"), NEW),
        ])
        self.assertEqual(calls, [])
        self.assertEqual(len(navigators), 2)
        self.assertIs(navigators[1].rearmed_from, navigators[0])
        self.assertEqual(navigators[0].frames, [("arm0", 30)])
        self.assertEqual(navigators[1].frames[0], ("fresh-arm", 32))
        self.assertEqual(result["navigation_rearms"], 1)
        self.assertTrue(any(when == .1 and packet["move"] == "right" and address == NEW
                            for when, packet, address in transport.sent))
        self.assertFalse(any(0 < when < .1 for when, _, _ in transport.sent))
        self.assertTrue(any("same remaining time and request limits" in m for m in messages))

    def test_previously_seen_arm_and_lower_frame_replays_do_not_restart(self):
        transport, calls, _, result, navigators = run([
            (0, observed(30, enabled=False, session="older-arm"), OLD),
            (.01, observed(31, session="failed-arm", fault=ROUTE_FAILURE), OLD),
            (.05, observed(35, session="failed-arm"), OLD),
            (.1, observed(36, session="older-arm", clear=False), OLD),
            (.15, observed(34, session="fresh-arm", clear=False), OLD),
            (.2, observed(37, session="fresh-arm"), OLD),
        ])
        self.assertEqual(calls, [])
        self.assertEqual(len(navigators), 2)
        self.assertEqual(result["navigation_rearms"], 1)
        self.assertFalse(any(.01 < when < .2 for when, _, _ in transport.sent))

    def test_new_disabled_or_paused_session_waits_for_controllable_observation(self):
        transport, calls, _, result, _ = run([
            (0, observed(fault=ROUTE_FAILURE), OLD),
            (.05, observed(31, session="fresh-arm", enabled=False), OLD),
            (.1, observed(32, session="fresh-arm", paused=True), OLD),
            (.2, observed(33, session="fresh-arm"), OLD),
        ])
        self.assertEqual(calls, [])
        self.assertEqual(result["navigation_rearms"], 1)
        self.assertFalse(any(0 < when < .2 for when, _, _ in transport.sent))

    def test_manual_room_change_is_adopted_only_by_the_new_navigator(self):
        incoming = observed(33, session="fresh-arm", room_id="new-room")
        incoming["floor"]["room_index"] = 2
        _, _, _, result, navigators = run([
            (0, observed(fault=ROUTE_FAILURE), OLD),
            (.1, observed(32, room_id="new-room"), OLD),
            (.2, incoming, NEW),
        ])
        self.assertEqual(result["stop_reason"], "duration reached")
        self.assertEqual(len(navigators), 2)
        self.assertEqual(navigators[0].frames, [("arm0", 30)])
        self.assertEqual(navigators[1].frames[0], ("fresh-arm", 33))

    def test_waiting_and_repeated_recovery_share_original_timer(self):
        transport, calls, _, result, navigators = run([
            (0, observed(enabled=False), OLD),
            (5, observed(31, session="first-arm", fault=ROUTE_FAILURE), OLD),
            (5.1, observed(32, session="second-arm", fault="door traversal timed out"), NEW),
            (5.2, observed(33, session="third-arm"), OLD),
        ], duration=.3, wait_for_arm=True)
        self.assertAlmostEqual(transport.now, 5.3, delta=.011)
        self.assertEqual(len(navigators), 3)
        self.assertEqual(calls, [])
        self.assertEqual((result["navigation_stops"], result["navigation_rearms"]), (2, 2))

    def test_inflight_reply_is_counted_discarded_and_request_budget_is_not_renewed(self):
        transport, calls, _, result, _ = run([
            (0, observed(clear=False), OLD),
            (.02, observed(31, fault=ROUTE_FAILURE), OLD),
            (.1, observed(32, clear=False), OLD),
            (.2, observed(33, clear=False, session="fresh-arm"), NEW),
            (.3, observed(34, clear=False, session="fresh-arm"), NEW),
        ], delay=.05, max_calls=2)
        self.assertEqual(len(calls), 2)
        self.assertEqual([data["session"] for _, data in calls], ["arm0", "fresh-arm"])
        self.assertEqual(result["stop_reason"], "request cap reached")
        self.assertEqual(result["decisions"], 2)
        self.assertEqual(result["responses_with_usage"], 2)
        self.assertEqual(result["stale_discarded"], 1)
        self.assertFalse(any(.02 < when < .2 for when, _, _ in transport.sent))

    def test_request_cap_stops_even_while_waiting_and_prevents_manual_retry(self):
        _, calls, _, result, navigators = run([
            (0, observed(clear=False), OLD),
            (.02, observed(31, fault=ROUTE_FAILURE), OLD),
            (.2, observed(32, clear=False, session="fresh-arm"), NEW),
        ], delay=.05, max_calls=1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["stop_reason"], "request cap reached")
        self.assertEqual(result["navigation_rearms"], 0)
        self.assertEqual(len(navigators), 1)

    def test_old_goal_is_not_reused_when_new_arm_request_is_still_pending(self):
        dispatches = []
        def local(data, kind, target):
            dispatches.append((data["session"], data["frame"], kind, target))
            return Action("left", "up")
        _, _, _, result, _ = run([
            (0, observed(clear=False), OLD),
            (.08, observed(31, clear=False), OLD),
            (.1, observed(32, fault=ROUTE_FAILURE), OLD),
            (.15, observed(33, clear=False), OLD),
            (.2, observed(34, clear=False, session="fresh-arm"), NEW),
            (.22, observed(35, clear=False, session="fresh-arm"), NEW),
        ], delay=.05, local=local)
        self.assertTrue(any(session == "arm0" for session, *_ in dispatches))
        self.assertFalse(any(session == "fresh-arm" and frame == 34 for session, frame, *_ in dispatches))
        self.assertEqual(result["navigation_rearms"], 1)

    def test_observe_failure_uses_the_same_recovery_path(self):
        _, calls, _, result, navigators = run([
            (0, observed(fault=ROUTE_FAILURE), OLD),
            (.2, observed(31, session="fresh-arm"), NEW),
        ], fault_during_observe=True)
        self.assertEqual(calls, [])
        self.assertEqual(len(navigators), 2)
        self.assertEqual((result["navigation_stops"], result["navigation_rearms"]), (1, 1))

    def test_unknown_stop_and_no_stay_ready_still_exit(self):
        for reason, stay_ready in ((ROUTE_FAILURE, False), ("incomplete floor observation", True),
                                   ("no accessible unexplored rooms", True), ("floor cleared", True),
                                   ("bomb retreat blocked", True)):
            with self.subTest(reason=reason, stay_ready=stay_ready):
                _, calls, _, result, navigators = run([
                    (0, observed(fault=reason), OLD),
                    (.2, observed(31, session="fresh-arm"), NEW),
                ], stay_ready=stay_ready)
                self.assertEqual(result["stop_reason"], reason)
                self.assertEqual(result["navigation_stops"], 0)
                self.assertEqual(len(navigators), 1)
                self.assertEqual(calls, [])

    def test_death_run_and_floor_changes_remain_terminal_during_recovery(self):
        for kind, reason in (("dead", "player died"), ("run", "run changed"), ("floor", "floor changed")):
            changed = observed(31)
            if kind == "dead":
                changed["player"]["dead"] = True
            elif kind == "run":
                changed["run_id"] = "other-run"
            else:
                changed["floor"]["id"] = "other-floor"
            _, _, _, result, navigators = run([(0, observed(fault=ROUTE_FAILURE), OLD),
                                               (.1, changed, OLD)])
            self.assertEqual(result["stop_reason"], reason)
            self.assertEqual(len(navigators), 1)

    def test_inflight_api_error_is_terminal_and_not_retried_by_recovery(self):
        def failed_policy(_):
            raise RuntimeError("not logged")
        _, calls, messages, result, navigators = run([
            (0, observed(clear=False), OLD),
            (.02, observed(31, fault=ROUTE_FAILURE), OLD),
            (.2, observed(32, session="fresh-arm", clear=False), NEW),
        ], delay=.05, policy=failed_policy)
        self.assertEqual(result["stop_reason"], "decision failed")
        self.assertEqual(result["errors"], 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(navigators), 1)
        self.assertFalse(any("not logged" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
