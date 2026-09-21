"""Controller-level objective handling with real local planners and fake time."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_controller_arming import ImmediatePool, OLD, NEW, ScheduledSocket
from test_goal_controller import goal
from test_pickups import observed
from test_switches import plate, required
from jev_isaac.combat_stall import NO_COMBAT_OBJECTIVE
from jev_isaac.controller import Controller


def empty(frame=30, **changes):
    value = observed(clear=False)
    value.update(frame=frame, **changes)
    return value


def combat(frame=30, **changes):
    value = empty(frame, **changes)
    value["enemies"] = [{"id": "target", "x": 400, "y": 280, "vx": 0, "vy": 0,
                         "hp": 10, "radius": 12, "vulnerable": True,
                         "keeps_doors_closed": True}]
    return value


def switch(frame=30, **changes):
    value = required()
    value.update(frame=frame, **changes)
    return value


def run(events, *, duration=.4, delay=0, stay_ready=True, **controller_options):
    transport, calls, checkpoints, messages = ScheduledSocket(events), [], [], []
    def policy(value):
        calls.append((transport.now, copy.deepcopy(value)))
        return goal()
    class Pool(ImmediatePool):
        def submit(self, function, *args):
            value, ready = function(*args), transport.now+delay
            return SimpleNamespace(done=lambda: transport.now >= ready, result=lambda: value)
    with patch("jev_isaac.controller.socket.socket", return_value=transport) as sockets, \
         patch("jev_isaac.controller.time.monotonic", side_effect=lambda: transport.now), \
         patch("jev_isaac.controller.concurrent.futures.ThreadPoolExecutor", Pool):
        result = Controller(policy, duration=duration, max_hz=2, max_latency=.5,
                            goal_mode=True, floor_mode=True, stay_ready=stay_ready,
                            logger=messages.append, on_navigation_stop=checkpoints.append,
                            **controller_options).run()
        sockets.assert_called_once()
    return transport, calls, checkpoints, messages, result


class ControllerRoomObjectiveTests(unittest.TestCase):
    def test_dead_flag_overrides_positive_hp_and_never_authorizes_a_shot(self):
        data = combat()
        data["enemies"][0]["dead"] = True
        transport, calls, checkpoints, _, _ = run([(0, data, OLD)])
        self.assertEqual(calls, [])
        self.assertEqual(checkpoints, [])
        self.assertTrue(transport.sent)
        self.assertTrue(all(packet["shoot"] == "none" for _, packet, _ in transport.sent))

    def test_complete_empty_uncleared_room_makes_zero_paid_requests_during_grace(self):
        transport, calls, checkpoints, _, result = run([
            (0, empty(30), OLD), (.1, empty(33), OLD),
            (1, empty(60), OLD), (2, empty(90), OLD),
        ], duration=2.1)
        self.assertEqual(calls, [])
        self.assertEqual(checkpoints, [])
        self.assertEqual(result["decisions"], 0)
        self.assertFalse(result["last_observation"]["room"]["clear"])
        self.assertTrue(all(packet["shoot"] == "none" for _, packet, _ in transport.sent))
        self.assertEqual(result["stop_reason"], "duration reached")

    def test_targetless_grace_writes_one_checkpoint_and_releases_once_while_ready(self):
        events = [(second, empty(30+30*second), OLD) for second in range(8)]
        transport, calls, checkpoints, messages, result = run(events, duration=7.1)
        self.assertEqual(calls, [])
        self.assertEqual(len(checkpoints), 1)
        checkpoint = checkpoints[0]
        self.assertEqual(checkpoint["reason"], NO_COMBAT_OBJECTIVE)
        self.assertTrue(checkpoint["recoverable"])
        self.assertFalse(checkpoint["observation"]["room"]["clear"])
        self.assertEqual(checkpoint["observation"]["enemies"], [])
        self.assertTrue(checkpoint["recent_local_controls"])
        releases = [(when, packet) for when, packet, _ in transport.sent if packet["floor_mode"] is False]
        self.assertEqual(len(releases), 1)
        self.assertAlmostEqual(releases[0][0], 5)
        self.assertEqual(releases[0][1]["stop_reason"], "navigation")
        self.assertFalse(any(when > 5 for when, _, _ in transport.sent))
        self.assertEqual(result["navigation_stops"], 1)
        self.assertEqual(result["stop_reason"], "duration reached")
        self.assertTrue(any("listener and key remain ready" in message for message in messages))

    def test_reachable_supported_switch_moves_locally_without_shooting_or_model_calls(self):
        first, second = switch(), switch(33)
        second["player"]["x"] += 12
        transport, calls, checkpoints, messages, result = run([
            (0, first, OLD), (.1, second, OLD),
        ])
        self.assertEqual(calls, [])
        self.assertEqual(checkpoints, [])
        controls = [packet for _, packet, _ in transport.sent if packet["floor_mode"]]
        self.assertTrue(controls)
        self.assertTrue(all(packet["move"] == "right" and packet["shoot"] == "none" for packet in controls))
        self.assertTrue(any("pressing required room switch" in message for message in messages))
        self.assertFalse(result["last_observation"]["room"]["clear"])

    def test_living_enemy_arrival_resumes_model_decisions(self):
        transport, calls, checkpoints, _, result = run([
            (0, empty(), OLD), (.1, empty(33), OLD), (.2, combat(36), OLD),
        ])
        self.assertEqual([value["frame"] for _, value in calls], [36])
        self.assertEqual(checkpoints, [])
        self.assertEqual(result["goal_updates"], 1)
        self.assertFalse(any(when < .2 and packet["shoot"] != "none" for when, packet, _ in transport.sent))
        self.assertTrue(any(when >= .2 and packet["shoot"] == "right" for when, packet, _ in transport.sent))

    def test_pending_combat_reply_is_counted_but_discarded_after_room_becomes_enemy_free(self):
        transport, calls, checkpoints, _, result = run([
            (0, combat(), OLD), (.1, empty(33), OLD), (.2, empty(36), OLD),
        ], delay=.15)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["responses_with_usage"], 1)
        self.assertEqual(result["stale_discarded"], 1)
        self.assertEqual(result["goal_updates"], 0)
        self.assertEqual(checkpoints, [])
        self.assertTrue(all(packet["shoot"] == "none" for _, packet, _ in transport.sent))

    def test_empty_transition_in_a_burst_invalidates_reply_even_if_enemy_returns(self):
        transport, calls, checkpoints, _, result = run([
            (0, combat(), OLD), (.05, empty(32), OLD), (.05, combat(33), OLD),
        ], delay=.1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["stale_discarded"], 1)
        self.assertEqual(result["goal_updates"], 0)
        self.assertEqual(checkpoints, [])
        self.assertTrue(all(packet["shoot"] == "none" for _, packet, _ in transport.sent))

    def test_exact_tnt_layout_fires_short_pulse_then_retreats_without_paid_calls(self):
        from test_tnt_regressions import recorded, approach_until_fire
        from jev_isaac.tnt import TntDemolition
        data, demolition = recorded(), TntDemolition()
        fire, fired_at, trace = approach_until_fire(demolition, data)
        self.assertNotEqual(fire.shoot, "none")
        events = [(when, observation, OLD) for when, observation, _ in trace]
        data["hazards"] = [h for h in data["hazards"]
                           if not (h.get("kind") == "grid" and h.get("index") == demolition.plan.target_index)]
        data["frame"] += 1
        events.append((fired_at+.04, data, OLD))
        transport, calls, checkpoints, messages, result = run(
            events, duration=fired_at+.2, hold_frames=15, move_frames=6, move_distance=18)
        self.assertEqual(calls, [])
        self.assertEqual(checkpoints, [])
        self.assertEqual(result["navigation_stops"], 0)
        firing = [(when, packet) for when, packet, _ in transport.sent if packet["shoot"] != "none"]
        self.assertEqual(len(firing), 1)
        self.assertEqual(firing[0][1]["hold_frames"], 3)
        self.assertEqual(firing[0][1]["move_frames"], 3)
        self.assertLessEqual(firing[0][1]["move_frames"], firing[0][1]["hold_frames"])
        self.assertAlmostEqual(firing[0][0], fired_at)
        self.assertTrue(any(when > fired_at and packet["move"] != "none" and packet["shoot"] == "none"
                            for when, packet, _ in transport.sent))
        self.assertTrue(any("retreating from TNT" in message for message in messages))
        self.assertFalse(result["last_observation"]["room"]["clear"])

    def test_unbreakable_wall_still_releases_for_no_safe_switch_route(self):
        from test_exploration import grid
        data = switch()
        data["hazards"] = [grid(300, y, kind=15, collision=4) for y in range(140, 441, 40)]
        later = copy.deepcopy(data)
        later["frame"] += 3
        transport, calls, checkpoints, _, result = run([(0, data, OLD), (.1, later, OLD)])
        self.assertEqual(calls, [])
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0]["reason"], "no safe route to required room switch")
        self.assertTrue(checkpoints[0]["recoverable"])
        self.assertEqual(len(transport.sent), 1)
        packet = transport.sent[0][1]
        self.assertEqual((packet["move"], packet["shoot"], packet["floor_mode"]), ("none", "none", False))
        self.assertEqual(result["navigation_stops"], 1)
        self.assertEqual(result["stop_reason"], "duration reached")

    def test_pause_and_fresh_f8_cannot_skip_the_tnt_shot_game_frame_window(self):
        from test_tnt_regressions import recorded, approach_until_fire
        from jev_isaac.tnt import TntDemolition
        data, demolition = recorded(), TntDemolition()
        fire, fired_at, trace = approach_until_fire(demolition, data)
        self.assertNotEqual(fire.shoot, "none")
        fire_frame = data["frame"]
        events = [(when, observation, OLD) for when, observation, _ in trace]
        paused = copy.deepcopy(data)
        paused.update(frame=fire_frame+1, paused=True)
        disabled = copy.deepcopy(paused)
        disabled["enabled"] = False
        events.extend(((fired_at+.04, paused, OLD), (fired_at+.08, disabled, OLD)))
        for frame, offset in ((fire_frame+2, 3), (fire_frame+59, 3.1), (fire_frame+60, 3.2)):
            resumed = copy.deepcopy(data)
            resumed.update(frame=frame, session="fresh-arm-after-shot", enabled=True, paused=False)
            events.append((fired_at+offset, resumed, NEW))
        transport, calls, checkpoints, messages, result = run(
            events, duration=fired_at+3.4, hold_frames=15, move_frames=6, move_distance=18)
        self.assertEqual(calls, [])
        self.assertEqual(checkpoints, [])
        self.assertEqual(result["navigation_stops"], 0)
        self.assertEqual(result["stop_reason"], "duration reached")
        waiting = [packet for when, packet, peer in transport.sent
                   if fired_at+3 <= when < fired_at+3.2 and peer == NEW]
        self.assertEqual(len(waiting), 2)
        self.assertTrue(all((packet["move"], packet["shoot"]) == ("none", "none") for packet in waiting))
        self.assertTrue(any("earlier TNT shots" in message for message in messages))
        resumed = [packet for when, packet, peer in transport.sent
                   if when == fired_at+3.2 and peer == NEW]
        self.assertTrue(resumed)
        self.assertTrue(any(packet["move"] != "none" or packet["shoot"] != "none" for packet in resumed))

    def test_fresh_rearm_after_stall_resumes_local_objective_under_same_deadline(self):
        events = [(second, empty(30+30*second), OLD) for second in range(6)]
        incoming = switch(195, session="fresh-arm")
        events.append((5.5, incoming, NEW))
        transport, calls, checkpoints, _, result = run(events, duration=5.8)
        self.assertEqual(calls, [])
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(result["navigation_rearms"], 1)
        self.assertTrue(any(when == 5.5 and address == NEW and packet["move"] == "right"
                            for when, packet, address in transport.sent))
        self.assertAlmostEqual(transport.now, 5.8, delta=.011)
        self.assertEqual(result["floor_progress"]["rooms_visited"], 1)


if __name__ == "__main__":
    unittest.main()
