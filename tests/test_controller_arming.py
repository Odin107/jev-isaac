"""Deterministic timer and endpoint handover tests; no live sockets or API."""
import concurrent.futures
import io
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jev_isaac.controller import Action, Controller
from jev_isaac.demo import sample
from jev_isaac.cli import main

OLD = ("127.0.0.1", 45001)
NEW = ("127.0.0.1", 45002)


def state(frame=30, **changes):
    data = sample(frame)
    data.update(run_id="same-run", session="arm0")
    data.update(changes)
    return data


class ImmediatePool:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, fn, *args):
        future = concurrent.futures.Future()
        future.set_result(fn(*args))
        return future


class ScheduledSocket:
    def __init__(self, events):
        self.events = list(events)
        self.now = 0.0
        self.sent = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def bind(self, address):
        pass

    def settimeout(self, timeout):
        self.timeout = timeout

    def getsockname(self):
        return ("127.0.0.1", 42421)

    def recvfrom(self, limit):
        if self.now > 20:
            raise AssertionError("Controller outlived the bounded fake scenario")
        if self.events and self.events[0][0] <= self.now+self.timeout:
            self.now, data, address = self.events.pop(0)
            return json.dumps(data).encode(), address
        self.now += self.timeout
        raise socket.timeout()

    def sendto(self, raw, address):
        self.sent.append((self.now, json.loads(raw), address))


def run_scenario(events, **options):
    transport = ScheduledSocket(events)
    calls = []
    def policy(observed):
        calls.append((transport.now, observed))
        return Action("right", "none")
    with patch("jev_isaac.controller.socket.socket", return_value=transport), \
         patch("jev_isaac.controller.time.monotonic", side_effect=lambda: transport.now), \
         patch("jev_isaac.controller.concurrent.futures.ThreadPoolExecutor", ImmediatePool):
        result = Controller(policy, duration=options.pop("duration", .3),
                            logger=lambda _: None, **options).run()
    return transport, calls, result


class ControllerArmingTests(unittest.TestCase):
    def test_waits_without_calls_then_gives_the_first_arm_its_full_budget(self):
        transport, calls, result = run_scenario([
            (0, state(enabled=False), OLD),
            (5, state(31, session="arm1"), OLD),
        ], wait_for_arm=True)
        self.assertAlmostEqual(transport.now, 5.3, delta=.011)
        self.assertTrue(calls)
        self.assertTrue(all(when >= 5 for when, _ in calls))
        self.assertTrue(result["wait_for_arm"])
        self.assertEqual(result["stop_reason"], "duration reached")

    def test_pause_disarm_and_rearm_do_not_restart_the_active_budget(self):
        transport, calls, result = run_scenario([
            (0, state(enabled=False), OLD),
            (5, state(31, session="arm1"), OLD),
            (5.1, state(32, session="arm1", paused=True), OLD),
            (5.15, state(33, session="arm1", enabled=False), OLD),
            (5.2, state(34, session="arm2"), OLD),
        ], wait_for_arm=True)
        self.assertAlmostEqual(transport.now, 5.3, delta=.011)
        self.assertEqual(result["stop_reason"], "duration reached")
        self.assertTrue(calls)

    def test_default_timer_remains_bounded_even_if_never_armed(self):
        transport, calls, result = run_scenario([(0, state(enabled=False), OLD)])
        self.assertAlmostEqual(transport.now, .3, delta=.011)
        self.assertEqual(calls, [])
        self.assertFalse(result["wait_for_arm"])

    def test_fresh_arm_on_replaced_endpoint_is_accepted_without_one_second_delay(self):
        transport, calls, result = run_scenario([
            (0, state(enabled=False), OLD),
            (.05, state(30, session="arm1"), NEW),
        ])
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(calls[0][0], .05)
        self.assertTrue(any(packet["move"] == "right" and address == NEW
                            for _, packet, address in transport.sent))

    def test_unrelated_or_unfresh_endpoints_cannot_use_the_rearm_exception(self):
        candidates = [state(31, session="arm1", run_id="other-run"),
                      state(31, session="arm1", room_id="other-room"),
                      state(31, session="arm0"), state(29, session="arm1"),
                      state(31, session="arm1", enabled=False),
                      state(31, session="arm1", paused=True)]
        missing_run = state(31, session="arm1")
        missing_run.pop("run_id")
        candidates.append(missing_run)
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                _, calls, result = run_scenario([(0, state(enabled=False), OLD),
                                                 (.05, candidate, NEW)])
                self.assertEqual(calls, [])
                self.assertEqual(result["observations"], 1)
        # An enabled-but-paused previous owner is not a disarmed owner.
        _, calls, result = run_scenario([(0, state(paused=True), OLD),
                                         (.05, state(31, session="arm1"), NEW)])
        self.assertEqual(calls, [])
        self.assertEqual(result["observations"], 1)

    def test_cli_option_is_explicit_and_forwarded(self):
        for options, expected in (([], False), (["--wait-for-arm"], True)):
            with self.subTest(options=options), patch("jev_isaac.cli.Controller") as controller, \
                 patch("sys.stdout", new_callable=io.StringIO):
                controller.return_value.run.return_value = {}
                self.assertEqual(main(["run", "--policy", "baseline", *options]), 0)
                self.assertEqual(controller.call_args.kwargs["wait_for_arm"], expected)


if __name__ == "__main__":
    unittest.main()
