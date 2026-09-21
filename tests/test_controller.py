"""Local contract and transport checks. No Isaac, credentials, or paid calls."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jev_isaac.controller import Action, Controller, Stats, baseline, result_is_fresh
from jev_isaac.demo import sample
from jev_isaac.protocol import MAX_DATAGRAM, MAX_FRAME_AGE, Observation, encode_action


def observation(frame=30, **changes):
    data = sample(frame)
    data.update(changes)
    return Observation.decode(json.dumps(data).encode())


class LocalGame:
    """Drive the actual UDP listener, retaining exceptions raised in its thread."""

    def __init__(self, policy, **options):
        self.ready = threading.Event()
        self.result = None
        self.error = None
        self.controller = Controller(policy, port=0, duration=options.pop("duration", .32),
                                     logger=lambda _: None, **options)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", 0))
        self.socket.settimeout(.02)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        if not self.ready.wait(1):
            self.socket.close()
            raise AssertionError(f"Listener failed to start: {self.error!r}")
        self.address = ("127.0.0.1", self.controller.bound_port)

    def _run(self):
        try:
            self.result = self.controller.run(self.ready)
        except BaseException as exc:
            self.error = exc

    def send(self, frame=30, **changes):
        data = sample(frame)
        data.update(changes)
        self.socket.sendto(json.dumps(data).encode(), self.address)

    def collect(self):
        actions = []
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                actions.append(json.loads(self.socket.recvfrom(MAX_DATAGRAM)[0]))
            except socket.timeout:
                if not self.thread.is_alive():
                    break
        self.thread.join(.2)
        self.socket.close()
        if self.thread.is_alive():
            raise AssertionError("Controller did not stop within its bounded test lifetime")
        if self.error is not None:
            raise self.error
        return actions


class ProtocolTests(unittest.TestCase):
    def test_short_movement_packet_retains_longer_shooting_hold(self):
        action = json.loads(encode_action(observation(), "left", "up", 15,
                                          move_frames=6, move_distance=20))
        self.assertEqual((action["hold_frames"], action["move_frames"], action["move_distance"]),
                         (15, 6, 20))
        for frames, distance in ((None, 20), (6, None), (True, 20), (0, 20), (7, 20),
                                 (6, True), (6, 0), (6, 25), (6, float("nan"))):
            with self.subTest(frames=frames, distance=distance), self.assertRaises(ValueError):
                encode_action(observation(), "left", "up", 15,
                              move_frames=frames, move_distance=distance)
        with self.assertRaises(ValueError):
            encode_action(observation(), "left", "up", 3, move_frames=6, move_distance=20)

    def test_valid_observation_and_action_roundtrip(self):
        obs = observation()
        self.assertTrue(obs.active)
        action = json.loads(encode_action(obs, "up_left", "right", 9))
        self.assertEqual(action, {
            "protocol": 1, "type": "action", "session": "synthetic-demo",
            "room_id": "demo-room-1", "frame": 30, "move": "up_left",
            "shoot": "right", "hold_frames": 9,
        })

    def test_each_stop_condition_deactivates_observation(self):
        for key in ("disabled", "paused", "dead", "clear"):
            with self.subTest(key=key):
                data = sample()
                if key == "disabled":
                    data["enabled"] = False
                elif key == "paused":
                    data["paused"] = True
                elif key == "dead":
                    data["player"]["dead"] = True
                else:
                    data["room"]["clear"] = True
                self.assertFalse(Observation.decode(json.dumps(data).encode()).active)

    def test_invalid_shapes_and_numbers_are_rejected(self):
        cases = []
        for key, value in (("protocol", True), ("frame", True), ("frame", -1), ("frame", 2**53),
                           ("enabled", 1), ("paused", "false"), ("session", ""),
                           ("room_id", "r" * 129), ("enemies", {}),
                           ("hazards", [{"x": 0, "y": 0}] * 257)):
            data = sample()
            data[key] = value
            cases.append(data)
        for coordinate in (float("nan"), float("inf"), True, "320"):
            data = sample()
            data["player"]["x"] = coordinate
            cases.append(data)
        data = sample()
        data["enemies"][0]["hp"] = float("inf")
        cases.append(data)
        data = sample()
        data["room"]["bottom_right"]["x"] = 80
        cases.append(data)
        for index, data in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(ValueError):
                    Observation.decode(json.dumps(data).encode())

    def test_extremely_large_coordinate_is_a_validation_error(self):
        data = sample()
        data["player"]["x"] = 10**400
        with self.assertRaises(ValueError):
            Observation.decode(json.dumps(data).encode())

    def test_invalid_json_and_oversized_packets_are_rejected(self):
        for raw in (b"", b"[]", b"null", b"\xff", b"{" * 2000,
                    b" " * (MAX_DATAGRAM + 1)):
            with self.subTest(length=len(raw)):
                with self.assertRaises(ValueError):
                    Observation.decode(raw)

    def test_actions_allow_only_known_directions_and_bounded_durations(self):
        for move, shoot, duration in (("console", "none", 6),
                                     ("none", "up_left", 6),
                                     ("none", "none", 0),
                                     ("none", "none", 16),
                                     ("none", "none", True)):
            with self.subTest(move=move, shoot=shoot, duration=duration):
                with self.assertRaises(ValueError):
                    encode_action(observation(), move, shoot, duration)


class FreshnessTests(unittest.TestCase):
    def test_freshness_requires_current_epoch_identity_time_and_frame(self):
        source = observation()
        self.assertTrue(result_is_fresh(source, observation(30 + MAX_FRAME_AGE), .25, .25, 1, 1))
        cases = [
            (None, .01, 1, 1),
            (observation(31 + MAX_FRAME_AGE), .01, 1, 1),
            (observation(29), .01, 1, 1),
            (observation(33, room_id="new-room"), .01, 1, 1),
            (observation(33, session="new-session"), .01, 1, 1),
            (observation(33, enabled=False), .01, 1, 1),
            (observation(33), .251, 1, 1),
            (observation(33), -.01, 1, 1),
            (observation(33), .01, 1, 3),
        ]
        for current, elapsed, source_epoch, epoch in cases:
            with self.subTest(current=current, elapsed=elapsed, epoch=epoch):
                self.assertFalse(result_is_fresh(source, current, elapsed, .25,
                                                source_epoch, epoch))

    def test_baseline_fires_at_target_and_respects_room_edge(self):
        data = sample()
        self.assertEqual(baseline(data), Action("left", "right"))
        data["player"]["x"] = 90
        data["enemies"][0]["x"] = 100
        self.assertEqual(baseline(data), Action("none", "right"))
        data["enemies"] = []
        self.assertEqual(baseline(data), Action("none", "none"))


class UsageTests(unittest.TestCase):
    def test_error_report_keeps_only_category_and_http_status(self):
        from jev_isaac.jev import JevHTTPError
        stats = Stats()
        stats.record_error(RuntimeError("never-log-this-secret-or-provider-body"))
        self.assertEqual(stats.last_error_type, "RuntimeError")
        self.assertIsNone(stats.last_http_status)
        self.assertNotIn("never-log", json.dumps(stats.summary()))
        stats.record_error(JevHTTPError(429))
        self.assertEqual(stats.errors, 2)
        self.assertEqual(stats.last_error_type, "JevHTTPError")
        self.assertEqual(stats.last_http_status, 429)

    def test_usage_accumulates_and_zero_tokens_are_reported(self):
        stats = Stats()
        for usage in ({"input_tokens": 110, "output_tokens": 4},
                      {"input_tokens": 250, "output_tokens": 9},
                      {"input_tokens": 0, "output_tokens": 0}):
            stats.record_usage(SimpleNamespace(usage=usage))
        stats.record_usage(Action("none", "none"))
        self.assertEqual(stats.responses_with_usage, 3)
        self.assertEqual(stats.reported_input_tokens, 360)
        self.assertEqual(stats.reported_output_tokens, 13)

    def test_invalid_usage_does_not_change_totals(self):
        stats = Stats()
        cases = [[], {}, {"input_tokens": 2}]
        for key in ("input_tokens", "output_tokens"):
            for value in (-1, True, 1.5, "2", None):
                usage = {"input_tokens": 10, "output_tokens": 2}
                usage[key] = value
                cases.append(usage)
        for usage in cases:
            with self.subTest(usage=usage):
                with self.assertRaises(ValueError):
                    stats.record_usage(SimpleNamespace(usage=usage))
        self.assertEqual(stats.responses_with_usage, 0)
        self.assertEqual(stats.reported_input_tokens, 0)
        self.assertEqual(stats.reported_output_tokens, 0)


class TransportTests(unittest.TestCase):
    def test_local_brakes_are_counted_once_per_command_and_ignore_bad_optional_metadata(self):
        game = LocalGame(baseline)
        for frame, control in ((30, {"source_frame": 20, "move_stop_reason": "time_limit"}),
                               (33, {"source_frame": 20, "move_stop_reason": "time_limit"}),
                               (36, {"source_frame": 23, "move_stop_reason": []}),
                               (39, {"source_frame": 25, "move_stop_reason": "distance_limit"})):
            game.send(frame, enabled=False, control=control)
        game.collect()
        self.assertEqual(game.result["observed_movement_brakes"], {"time_limit": 1, "distance_limit": 1})
        self.assertEqual(game.result["decisions"], 0)

    def test_old_mod_cannot_silently_ignore_short_movement_limits(self):
        policy = MagicMock(return_value=Action("left", "up"))
        game = LocalGame(policy, max_calls=1, hold_frames=15, move_frames=6, move_distance=20)
        game.send()
        self.assert_neutral_only(game.collect())
        policy.assert_not_called()
        self.assertEqual(game.result["stop_reason"], "mod update required")

    def test_updated_mod_receives_short_move_and_independent_shooting_lease(self):
        game = LocalGame(lambda _: Action("left", "up"), max_calls=1,
                         hold_frames=15, move_frames=6, move_distance=20)
        game.send(capabilities={"movement_pulses": 1})
        actions = game.collect()
        commands = [a for a in actions if a["hold_frames"] > 1]
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0]["move_frames"], 6)
        self.assertEqual(commands[0]["move_distance"], 20)
        self.assertEqual(commands[0]["hold_frames"], 15)
        self.assertNotIn("move_frames", actions[-1])

    def assert_neutral_only(self, actions):
        self.assertTrue(actions, "Expected a best-effort neutral release")
        self.assertTrue(all(a["move"] == "none" and a["shoot"] == "none"
                            for a in actions), actions)

    def test_local_game_receives_correlated_action_and_neutral_release(self):
        game = LocalGame(baseline, max_calls=1)
        game.send()
        actions = game.collect()
        moving = [a for a in actions if a["hold_frames"] > 1]
        self.assertEqual(len(moving), 1)
        self.assertEqual(moving[0]["frame"], 30)
        self.assertEqual(moving[0]["room_id"], "demo-room-1")
        self.assertEqual((moving[0]["move"], moving[0]["shoot"]), ("left", "right"))
        self.assertEqual(game.result["actions_sent"], 1)
        self.assertEqual(actions[-1]["hold_frames"], 1)
        self.assertEqual(game.result["responses_with_usage"], 0)
        self.assertEqual(game.result["reported_input_tokens"], 0)
        self.assertEqual(game.result["reported_output_tokens"], 0)

    def test_measured_jev_latency_profile_sends_bounded_action(self):
        def policy(_):
            time.sleep(.36)
            return Action("right", "down")
        game = LocalGame(policy, max_calls=1, duration=.7, max_latency=.5, hold_frames=15)
        game.send()
        time.sleep(.20)
        game.send(36)
        actions = game.collect()
        moving = [a for a in actions if a["hold_frames"] > 1]
        self.assertEqual(len(moving), 1)
        self.assertEqual(moving[0]["hold_frames"], 15)
        self.assertEqual(moving[0]["frame"], 30)
        self.assertEqual(game.result["actions_sent"], 1)
        self.assertEqual(actions[-1]["hold_frames"], 1)

    def test_jev_profile_discards_response_beyond_500ms(self):
        def policy(_):
            time.sleep(.54)
            return Action("right", "down")
        game = LocalGame(policy, max_calls=1, duration=.8, max_latency=.5, hold_frames=15)
        game.send()
        time.sleep(.30)
        game.send(39)
        self.assert_neutral_only(game.collect())
        self.assertEqual(game.result["stale_discarded"], 1)

    def test_holds_cannot_exceed_mod_limit(self):
        for hold in (0, 16, True):
            with self.subTest(hold=hold), self.assertRaises(ValueError):
                Controller(baseline, hold_frames=hold)

    def test_one_room_trial_ends_without_more_requests_on_clear_death_or_exit(self):
        for end in ("clear", "dead", "exit", "new_run"):
            with self.subTest(end=end):
                game = LocalGame(baseline, duration=5, max_calls=120, one_room=True)
                game.send(run_id="original-run")
                deadline = time.monotonic() + .5
                while game.controller.stats.actions_sent == 0 and time.monotonic() < deadline:
                    time.sleep(.002)
                self.assertEqual(game.controller.stats.actions_sent, 1)
                state = sample(33)
                state["run_id"] = "original-run"
                if end == "clear":
                    state["room"]["clear"] = True
                elif end == "dead":
                    state["player"]["dead"] = True
                elif end == "exit":
                    state["room_id"] = "next-room"
                else:
                    state["run_id"] = "different-run"
                game.socket.sendto(json.dumps(state).encode(), game.address)
                game.collect()
                self.assertEqual(game.result["decisions"], 1)
                self.assertEqual(game.result["stop_reason"], {
                    "clear":"room cleared", "dead":"player died", "exit":"room changed",
                    "new_run":"run changed"}[end])

    def test_fresh_decision_usage_is_recorded(self):
        decision = SimpleNamespace(move="right", shoot="down",
                                   usage={"input_tokens": 124, "output_tokens": 8})
        game = LocalGame(lambda _: decision, max_calls=1)
        game.send()
        game.collect()
        self.assertEqual(game.result["actions_sent"], 1)
        self.assertEqual(game.result["responses_with_usage"], 1)
        self.assertEqual(game.result["reported_input_tokens"], 124)
        self.assertEqual(game.result["reported_output_tokens"], 8)

    def test_choice_corrections_and_combat_progress_are_reported(self):
        correction = {"question": "move", "reported_choice": "left",
                      "effective_choice": "right", "selected_probability": .16,
                      "max_probability": .17, "gap": .01}
        decision = SimpleNamespace(move="right", shoot="down",
                                   usage={"input_tokens": 124, "output_tokens": 8},
                                   choice_corrections=(correction,))
        game = LocalGame(lambda _: decision, duration=1, one_room=True)
        game.send()
        deadline = time.monotonic() + .5
        while game.controller.stats.actions_sent == 0 and time.monotonic() < deadline:
            time.sleep(.002)
        state = sample(33)
        state["enemies"] = []
        state["room"]["clear"] = True
        game.socket.sendto(json.dumps(state).encode(), game.address)
        game.collect()
        self.assertEqual(game.result["choice_corrections"], 1)
        self.assertEqual(game.result["corrected_responses"], 1)
        self.assertEqual(game.result["last_choice_corrections"], [correction])
        self.assertEqual(game.result["errors"], 0)
        self.assertEqual(game.result["reported_input_tokens"], 124)
        self.assertEqual(game.result["combat_start"]["enemy_count"], 1)
        self.assertEqual(game.result["combat_end"]["enemy_count"], 0)
        self.assertTrue(game.result["combat_end"]["room_clear"])

    def test_stale_decision_usage_is_recorded_without_acting(self):
        def slow(_):
            time.sleep(.09)
            return SimpleNamespace(move="right", shoot="down",
                                   usage={"input_tokens": 210, "output_tokens": 9})
        game = LocalGame(slow, max_calls=1, max_latency=.04)
        game.send()
        self.assert_neutral_only(game.collect())
        self.assertEqual(game.result["stale_discarded"], 1)
        self.assertEqual(game.result["actions_sent"], 0)
        self.assertEqual(game.result["responses_with_usage"], 1)
        self.assertEqual(game.result["reported_input_tokens"], 210)
        self.assertEqual(game.result["reported_output_tokens"], 9)

    def test_response_after_duration_counts_usage_without_acting(self):
        entered, release = threading.Event(), threading.Event()
        def delayed(_):
            entered.set()
            if not release.wait(1):
                raise AssertionError("Test policy was never released")
            return SimpleNamespace(move="right", shoot="down",
                                   usage={"input_tokens": 330, "output_tokens": 12})
        game = LocalGame(delayed, max_calls=1, duration=.04)
        game.send()
        try:
            self.assertTrue(entered.wait(.5))
            # Wait for the neutral release, which occurs before executor shutdown.
            game.socket.settimeout(.5)
            neutral = json.loads(game.socket.recvfrom(MAX_DATAGRAM)[0])
            self.assertEqual(neutral["hold_frames"], 1)
        finally:
            release.set()
        self.assertEqual(game.collect(), [])
        self.assertEqual(game.result["actions_sent"], 0)
        self.assertEqual(game.result["stale_discarded"], 1)
        self.assertEqual(game.result["responses_with_usage"], 1)
        self.assertEqual(game.result["reported_input_tokens"], 330)
        self.assertEqual(game.result["reported_output_tokens"], 12)

    def test_bad_datagrams_do_not_prevent_next_valid_observation(self):
        game = LocalGame(baseline, max_calls=1)
        game.socket.sendto(b"not json", game.address)
        game.socket.sendto(b" " * (MAX_DATAGRAM + 1), game.address)
        game.send()
        game.collect()
        self.assertEqual(game.result["invalid_packets"], 2)
        self.assertEqual(game.result["actions_sent"], 1)

    def test_large_numeric_packet_does_not_crash_listener(self):
        game = LocalGame(baseline, max_calls=1)
        data = sample()
        data["player"]["x"] = 10**400
        game.socket.sendto(json.dumps(data).encode(), game.address)
        game.send()
        game.collect()
        self.assertEqual(game.result["invalid_packets"], 1)
        self.assertEqual(game.result["actions_sent"], 1)

    def test_slow_policy_result_is_discarded(self):
        def slow(_):
            time.sleep(.09)
            return Action("right", "down")
        game = LocalGame(slow, max_calls=1, max_latency=.04)
        game.send()
        self.assert_neutral_only(game.collect())
        self.assertEqual(game.result["stale_discarded"], 1)
        self.assertEqual(game.result["actions_sent"], 0)

    def test_disabled_game_never_dispatches_policy(self):
        called = threading.Event()
        def policy(_):
            called.set()
            return Action("right", "down")
        game = LocalGame(policy)
        game.send(enabled=False)
        self.assert_neutral_only(game.collect())
        self.assertFalse(called.is_set())
        self.assertEqual(game.result["decisions"], 0)

    def run_inflight_transition(self, updates):
        entered, release = threading.Event(), threading.Event()
        def policy(_):
            entered.set()
            if not release.wait(1):
                raise AssertionError("Test policy was never released")
            return Action("right", "down")
        game = LocalGame(policy, max_calls=1)
        game.send()
        try:
            self.assertTrue(entered.wait(.5))
            for index, (frame, changes) in enumerate(updates):
                game.send(frame, **changes)
                # Let the listener observe each control transition before completion.
                deadline = time.monotonic() + .3
                expected = index + 2
                while game.controller.stats.observations < expected and time.monotonic() < deadline:
                    time.sleep(.002)
                self.assertGreaterEqual(game.controller.stats.observations, expected)
        finally:
            release.set()
        return game, game.collect()

    def test_switching_off_invalidates_inflight_result(self):
        game, actions = self.run_inflight_transition([(33, {"enabled": False})])
        self.assert_neutral_only(actions)
        self.assertEqual(game.result["stale_discarded"], 1)

    def test_off_then_on_invalidates_previous_enabled_epoch(self):
        game, actions = self.run_inflight_transition([
            (30, {"enabled": False}), (30, {"enabled": True}),
        ])
        self.assert_neutral_only(actions)
        self.assertEqual(game.result["stale_discarded"], 1)

    def test_room_change_invalidates_inflight_result(self):
        game, actions = self.run_inflight_transition([(33, {"room_id": "room-two"})])
        self.assert_neutral_only(actions)
        self.assertEqual(game.result["stale_discarded"], 1)

    def test_advanced_game_frame_invalidates_inflight_result(self):
        game, actions = self.run_inflight_transition([(31 + MAX_FRAME_AGE, {})])
        self.assert_neutral_only(actions)
        self.assertEqual(game.result["stale_discarded"], 1)

    def test_invalid_policy_action_fails_closed(self):
        game = LocalGame(lambda _: Action("invalid", "right"), max_calls=1)
        game.send()
        self.assert_neutral_only(game.collect())
        self.assertEqual(game.result["errors"], 1)
        self.assertEqual(game.result["actions_sent"], 0)

    def test_invalid_reply_is_skipped_then_only_new_frame_is_dispatched(self):
        from jev_isaac.jev import JevResponseError
        seen = []
        def policy(state):
            seen.append(state["frame"])
            if len(seen) == 1:
                raise JevResponseError("Untrusted reply must never act")
            return Action("right", "down")
        game = LocalGame(policy, max_calls=2, duration=1)
        game.send(30)
        deadline = time.monotonic() + .5
        while game.controller.stats.errors == 0 and time.monotonic() < deadline:
            time.sleep(.002)
        self.assertEqual(game.controller.stats.errors, 1)
        game.send(30)
        # Let the rate limit elapse; a replay still cannot trigger another call.
        time.sleep(.21)
        self.assertEqual(seen, [30])
        game.send(33)
        actions = game.collect()
        moving = [a for a in actions if a["hold_frames"] > 1]
        self.assertEqual(seen, [30,33])
        self.assertEqual(len(moving), 1)
        self.assertEqual(moving[0]["frame"], 33)
        self.assertEqual(game.result["last_error_type"], "JevResponseError")
        self.assertEqual(game.result["last_failed_observation"]["frame"], 30)

    def test_three_consecutive_invalid_replies_stop_without_acting(self):
        from jev_isaac.jev import JevResponseError
        def invalid(_):
            raise JevResponseError("Never apply malformed decisions")
        game = LocalGame(invalid, max_calls=10, max_hz=10, duration=2)
        for index, frame in enumerate((30,33,36), 1):
            game.send(frame)
            deadline = time.monotonic() + .5
            while game.controller.stats.errors < index and time.monotonic() < deadline:
                time.sleep(.002)
            self.assertEqual(game.controller.stats.errors, index)
        self.assert_neutral_only(game.collect())
        self.assertEqual(game.result["decisions"], 3)
        self.assertEqual(game.result["stop_reason"], "three consecutive invalid replies")

    def test_replayed_frame_does_not_trigger_repeated_decisions(self):
        game = LocalGame(baseline, max_hz=10, max_calls=3)
        for _ in range(10):
            game.send()
            time.sleep(.018)
        game.collect()
        self.assertEqual(game.result["decisions"], 1)
        self.assertEqual(game.result["actions_sent"], 1)

    def test_older_frame_does_not_replace_newer_state(self):
        game = LocalGame(baseline, max_calls=1)
        game.send(60, enabled=False)
        game.send(30, enabled=True)
        self.assert_neutral_only(game.collect())
        self.assertEqual(game.result["decisions"], 0)
        self.assertEqual(game.result["observations"], 1)

    def test_other_local_sender_cannot_take_over_active_peer(self):
        game = LocalGame(baseline, max_calls=1)
        game.send(60, enabled=False)
        deadline = time.monotonic() + .2
        while game.controller.stats.observations == 0 and time.monotonic() < deadline:
            time.sleep(.002)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as other:
            other.sendto(json.dumps(sample(63)).encode(), game.address)
        self.assert_neutral_only(game.collect())
        self.assertEqual(game.result["decisions"], 0)
        self.assertEqual(game.result["observations"], 1)

    def test_receive_error_and_failed_release_return_summary(self):
        fake_socket = MagicMock()
        fake_socket.__enter__.return_value = fake_socket
        fake_socket.getsockname.return_value = ("127.0.0.1", 42421)
        fake_socket.recvfrom.side_effect = [
            (json.dumps(sample(enabled=False)).encode(), ("127.0.0.1", 12345)),
            ConnectionResetError("synthetic peer closure"),
        ]
        fake_socket.sendto.side_effect = OSError("synthetic failed release")
        with patch("jev_isaac.controller.socket.socket", return_value=fake_socket):
            result = Controller(baseline, logger=lambda _: None).run()
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["decisions"], 0)
        fake_socket.sendto.assert_called_once()


if __name__ == "__main__":
    unittest.main()
