"""Queued UDP observations retain safety boundaries while planning newest frames."""
import copy
import json
from pathlib import Path
import socket
import sys
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_controller import LocalGame
from test_controller_arming import OLD, NEW, run_scenario, state
from test_controller_pause_ready import run_floor
from test_exploration import state as floor_state, door
from test_floor_controller import CAPS
from test_goal_controller import goal
from jev_isaac.controller import (Action, Stats, _OBSERVATION_BATCH_LIMIT,
                                  _RECEIVE_TIMEOUT, _receive_observations)


# Actual control echo shape from the preserved room-83 stop observation.
ECHO = {"interaction_status": "none", "interaction": "none", "move_stop_reason": "none",
        "shoot": "none", "source_frame": 27811, "applied_move": "left", "requested_move": "left"}


def queued_packets(packets):
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(_RECEIVE_TIMEOUT)
    for packet in packets:
        sender.sendto(packet if isinstance(packet, bytes) else json.dumps(packet).encode(), receiver.getsockname())
    return receiver, sender


class ObservationQueueTests(unittest.TestCase):
    def test_real_udp_echo_burst_keeps_newest_frame_before_any_planning(self):
        packets = [state(frame, control=dict(ECHO, source_frame=frame-5)) for frame in range(100, 124)]
        receiver, sender = queued_packets(packets)
        with receiver, sender:
            stats = Stats()
            pending, saturated = _receive_observations(receiver, stats)
            self.assertFalse(saturated)
            self.assertEqual([item[0].frame for item in pending], [123])
            self.assertEqual(stats.observations_coalesced, 23)
            self.assertEqual(stats.max_observation_batch, 24)
            self.assertGreater(stats.observation_drain_max_ms, 0)
            self.assertEqual(receiver.gettimeout(), _RECEIVE_TIMEOUT)

    def test_real_udp_preserves_ordered_control_room_run_floor_and_echo_boundaries(self):
        packets = [state(30, control=ECHO)]
        for change in (
            {"enabled": False}, {"enabled": True, "paused": True}, {"paused": False},
            {"player": dict(packets[0]["player"], dead=True)},
            {"player": dict(packets[0]["player"], dead=False)}, {"session": "new-arm"},
            {"room_id": "new-room"}, {"run_id": "new-run"},
            {"floor": {"id": "new-floor", "room_index": 1}},
            {"floor": {"id": "new-floor", "room_index": 1, "dimension": 1}},
            {"room": dict(packets[0]["room"], clear=True)},
            {"floor_advance_permitted": True},
            {"control": dict(ECHO, move_stop_reason="distance_limit")},
            {"control": dict(ECHO, interaction="bomb", interaction_id="once", interaction_status="applied")},
        ):
            incoming = copy.deepcopy(packets[-1])
            incoming.update(change)
            packets.append(incoming)
        receiver, sender = queued_packets(packets)
        with receiver, sender:
            pending, saturated = _receive_observations(receiver, Stats())
            self.assertFalse(saturated)
            self.assertEqual([item[0].data for item in pending], packets)

    def test_real_udp_drain_is_bounded_and_does_not_drop_remainder(self):
        packets = [state(frame) for frame in range(_OBSERVATION_BATCH_LIMIT+7)]
        receiver, sender = queued_packets(packets)
        with receiver, sender:
            stats = Stats()
            first, saturated = _receive_observations(receiver, stats)
            self.assertTrue(saturated)
            self.assertEqual(first[-1][0].frame, _OBSERVATION_BATCH_LIMIT-1)
            second, saturated = _receive_observations(receiver, stats)
            self.assertFalse(saturated)
            self.assertEqual(second[-1][0].frame, _OBSERVATION_BATCH_LIMIT+6)
            self.assertEqual(stats.observation_drain_limit_hits, 1)

    def test_invalid_and_out_of_order_datagrams_cannot_hide_newest_valid_frame(self):
        receiver, sender = queued_packets([state(50), b"not-json", state(49), state(51), state(51)])
        with receiver, sender:
            stats = Stats()
            pending, _ = _receive_observations(receiver, stats)
            self.assertEqual([item[0].frame for item in pending], [51])
            self.assertEqual(stats.invalid_packets, 1)
            self.assertEqual(stats.observations_coalesced, 3)

    def test_endpoint_changes_are_not_coalesced(self):
        receiver, first = queued_packets([state(30)])
        with receiver, first, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as second:
            second.sendto(json.dumps(state(31)).encode(), receiver.getsockname())
            pending, _ = _receive_observations(receiver, Stats())
            self.assertEqual([item[0].frame for item in pending], [30, 31])
            self.assertNotEqual(pending[0][1], pending[1][1])


class ControllerBurstTests(unittest.TestCase):
    def test_saturated_batches_catch_up_before_dispatching_one_request(self):
        events = [(0, state(frame), OLD) for frame in range(200)]
        _, calls, result = run_scenario(events, duration=.05)
        self.assertEqual([observed["frame"] for _, observed in calls], [199])
        self.assertEqual(result["observation_drain_limit_hits"], 3)
        self.assertEqual(result["observations_coalesced"], 196)

    def test_pause_and_resume_in_one_burst_revoke_inflight_goal_even_at_same_frame(self):
        from test_controller_recovery import run, observed
        transport, calls, _, result, _ = run([
            (0, observed(30, clear=False), OLD),
            (.03, observed(31, clear=False, paused=True), OLD),
            (.03, observed(31, clear=False), OLD),
        ], delay=.05, duration=.09, max_hz=1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["stale_discarded"], 1)
        self.assertFalse(any(packet["move"] != "none" for _, packet, _ in transport.sent))

    def test_terminal_transition_in_burst_is_not_overwritten_by_later_live_packet(self):
        from test_controller_recovery import run, observed
        for kind, reason in (("death", "player died"), ("run", "run changed"), ("floor", "floor changed")):
            with self.subTest(kind=kind):
                changed = observed(32, clear=False)
                if kind == "death": changed["player"]["dead"] = True
                elif kind == "run": changed["run_id"] = "other-run"
                else: changed["floor"]["id"] = "other-floor"
                _, calls, _, result, _ = run([
                    (0, observed(30, clear=False), OLD),
                    (.02, observed(31, clear=False), OLD),
                    (.02, changed, OLD),
                    (.02, observed(33, clear=False), OLD),
                ], delay=.05)
                self.assertEqual(result["stop_reason"], reason)
                self.assertEqual(result["last_observation"]["frame"], 32)
                self.assertEqual(len(calls), 1)

    def test_real_burst_during_local_work_plans_only_newest_queued_frame_next(self):
        entered, release = threading.Event(), threading.Event()
        planned = []
        def local(observed, *_):
            planned.append(observed["frame"])
            if len(planned) == 1:
                entered.set()
                release.wait(.4)
            return Action("right", "none")
        with patch("jev_isaac.navigation.compute_action", side_effect=local):
            game = LocalGame(lambda _: goal(), goal_mode=True, startup_guard=True,
                             duration=.22, max_hz=1)
            try:
                game.send(30, capabilities=CAPS)
                self.assertTrue(entered.wait(.2))
                for frame in range(31, 51):
                    game.send(frame, capabilities=CAPS, control=dict(ECHO, source_frame=frame-5))
            finally:
                release.set()
                game.collect()
        self.assertEqual(planned, [30, 50])
        self.assertEqual(game.result["observations_coalesced"], 19)


class ManualTravelRearmTests(unittest.TestCase):
    @staticmethod
    def observation(index=84, frame=1, **changes):
        data = floor_state(index, frame=frame, doors=[door(2, 85)] if index == 84 else [], clear=index == 84)
        data.update(protocol=1, type="observation", capabilities=CAPS)
        if not data["room"]["clear"]:
            data["enemies"] = [{"id": "live", "x": 400, "y": 280, "vx": 0, "vy": 0,
                                "radius": 10, "hp": 10, "vulnerable": True}]
        data.update(changes)
        return data

    def test_manual_different_room_rearm_clears_pending_door_but_keeps_map(self):
        _, calls, _, result = run_floor([
            (0, self.observation(), OLD),
            (.05, self.observation(frame=2, enabled=False), OLD),
            (.1, self.observation(99, frame=3, enabled=False), OLD),
            (.2, self.observation(99, frame=4, session="fresh-arm"), NEW),
        ], stay_ready=True)
        self.assertEqual(result["stop_reason"], "duration reached")
        self.assertEqual(result["floor_progress"]["rooms_visited"], 2)
        self.assertEqual(result["floor_progress"]["rooms_cleared"], 1)
        self.assertEqual(result["floor_progress"]["doors_traversed"], 0)
        self.assertEqual([data["session"] for _, data in calls], ["fresh-arm"])

    def test_disabled_then_old_enabled_replay_cannot_resume_navigation(self):
        transport, calls, _, result = run_floor([
            (0, self.observation(), OLD),
            (.05, self.observation(frame=2, enabled=False), OLD),
            (.1, self.observation(99, frame=3), OLD),
        ], stay_ready=True)
        self.assertEqual(calls, [])
        self.assertEqual(result["floor_progress"]["rooms_visited"], 1)
        self.assertFalse(any(when >= .05 and packet["floor_mode"] for when, packet, _ in transport.sent))


if __name__ == "__main__":
    unittest.main()
