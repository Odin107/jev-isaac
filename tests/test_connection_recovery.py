"""A failed provider connection retains the listener without replaying calls."""
import copy
import unittest

from test_controller_arming import OLD, NEW
from test_controller_recovery import observed, run
from test_goal_controller import goal
from jev_isaac.jev import JevTransportError, JevTimeoutError


class ConnectionRecoveryTests(unittest.TestCase):
    def test_transport_failure_waits_for_fresh_f8_without_renewing_budget(self):
        for failure in (JevTransportError, JevTimeoutError):
            with self.subTest(failure=failure):
                count = 0
                def policy(data):
                    nonlocal count
                    count += 1
                    if count == 1:
                        raise failure("do-not-log-sensitive-text")
                    return goal()
                events = [(0, observed(30, clear=False), OLD),
                          (.1, observed(33, clear=False), OLD),
                          (.2, observed(36, clear=False, enabled=False, paused=True), OLD),
                          (.5, observed(36, clear=False), OLD),
                          (.7, observed(38, clear=False, session="fresh-arm"), NEW),
                          (.8, observed(41, clear=False, session="fresh-arm"), NEW)]
                transport, calls, logs, report, _ = run(events, policy=policy, duration=1., max_hz=2)
                self.assertEqual(report["stop_reason"], "duration reached")
                self.assertEqual((report["errors"], report["connection_stops"], report["connection_rearms"]), (1, 1, 1))
                self.assertEqual(report["navigation_stops"], 0)
                self.assertEqual(len(calls), 2)
                self.assertEqual(calls[-1][1]["session"], "fresh-arm")
                release = [p for t, p, _ in transport.sent if t < .7 and p.get("floor_mode") is False]
                self.assertEqual(len(release), 1)
                self.assertEqual((release[0]["move"], release[0]["shoot"]), ("none", "none"))
                self.assertNotIn("stop_reason", release[0])  # Do not mislabel this as a blocked route.
                self.assertFalse(any(.1 <= t < .7 for t, _, _ in transport.sent))
                self.assertAlmostEqual(transport.now, 1., delta=.011)
                self.assertFalse(any("do-not-log-sensitive-text" in line for line in logs))

    def test_connection_error_after_permitted_new_floor_rearms_on_that_floor(self):
        import jev_isaac.exploration as exploration
        count = 0
        def policy(data):
            nonlocal count
            count += 1
            exploration.FloorNavigator.descent_requested = True
            exploration.FloorNavigator.adventure_stats = {}
            if count == 2:
                raise JevTransportError("private")
            return goal()
        def second(frame, **changes):
            data = observed(frame, clear=False, floor_advance_permitted=True, **changes)
            data["floor"]["id"] = "second-floor"
            data["room_id"] = "second-floor-room"
            return data
        events = [(0, observed(30, clear=False), OLD),
                  (.2, second(36), OLD), (.5, second(45), OLD),
                  (.6, second(48, enabled=False), OLD),
                  (.8, second(54, session="new-floor-rearm"), NEW),
                  (.9, second(57, session="new-floor-rearm"), NEW),
                  (1.1, second(63, session="new-floor-rearm"), NEW)]
        for _, data, _ in events:
            data["capabilities"]["interaction_control"] = 1
        _, calls, _, report, _ = run(events, policy=policy, duration=1.3, max_hz=2,
                                     continue_floors=True, adventure_mode=True)
        self.assertEqual(report["floors_advanced"], 1, report["stop_reason"])
        self.assertEqual(report["connection_rearms"], 1)
        self.assertEqual(report["stop_reason"], "duration reached")
        self.assertEqual(calls[-1][1]["floor"]["id"], "second-floor")
        self.assertEqual(calls[-1][1]["session"], "new-floor-rearm")

    def test_repeated_failure_stays_idle_without_automatic_api_retries(self):
        def policy(_):
            raise JevTransportError("private")
        events = [(0, observed(30, clear=False), OLD),
                  (.2, observed(36, clear=False, session="fresh-arm"), NEW),
                  (.3, observed(39, clear=False, session="fresh-arm"), NEW)]
        _, calls, _, report, _ = run(events, policy=policy, duration=.7)
        self.assertEqual(len(calls), 2)
        self.assertEqual(report["connection_stops"], 2)
        self.assertEqual(report["stop_reason"], "duration reached")

    def test_connection_recovery_keeps_request_cap_and_opt_out(self):
        def policy(_):
            raise JevTransportError("private")
        events = [(0, observed(30, clear=False), OLD),
                  (.2, observed(36, clear=False, session="fresh-arm"), NEW)]
        for options, reason in (({"max_calls": 1}, "request cap reached"),
                                ({"stay_ready": False}, "decision failed")):
            _, calls, _, report, _ = run(events, policy=policy, **options)
            self.assertEqual(len(calls), 1)
            self.assertEqual(report["stop_reason"], reason)


if __name__ == "__main__":
    unittest.main()
