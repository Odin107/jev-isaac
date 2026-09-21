"""Full player-policy/controller/executor tests with simulated UDP and no API."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).resolve().parents[1]/"src")]
from test_controller_arming import ImmediatePool, OLD, NEW, ScheduledSocket
from test_adventure import ready
from test_adventure_controller import boss_ready, sequence
from test_exploration import door
from test_pickups import pickup
from test_tnt_regressions import recorded
from test_player_policy import reply
from jev_isaac.controller import Controller
from jev_isaac.player_policy import PlayerClient
from jev_isaac.jev import HttpResponse


def frame(data, number, **changes):
    data = copy.deepcopy(data)
    data.update(frame=number, **changes)
    return data


def combat():
    data = ready(clear=False)
    data["enemies"] = [{"id": "target", "x": 480, "y": 280, "vx": 0, "vy": 0,
                        "hp": 10, "radius": 12, "vulnerable": True, "keeps_doors_closed": True}]
    return data


def run(events, choices, *, delay=0, duration=1.5, max_calls=10, checkpoint=None,
        continue_floors=False):
    transport = ScheduledSocket(events)
    requests, logs = [], []
    def post(request, timeout):
        payload = json.loads(request.data)
        requests.append(copy.deepcopy(payload))
        chosen = choices(payload) if callable(choices) else choices
        return HttpResponse(200, json.dumps(reply(payload, chosen)).encode())
    client = PlayerClient("offline-test", transport=post)
    class Pool(ImmediatePool):
        def submit(self, function, *args):
            result = function(*args)
            due = transport.now+(delay(len(requests)) if callable(delay) else delay)
            return SimpleNamespace(done=lambda: transport.now >= due, result=lambda: result)
    try:
        with patch("jev_isaac.controller.socket.socket", return_value=transport), \
             patch("jev_isaac.controller.time.monotonic", side_effect=lambda: transport.now), \
             patch("jev_isaac.controller.concurrent.futures.ThreadPoolExecutor", Pool):
            result = Controller(client.decide, duration=duration, max_calls=max_calls, max_hz=2,
                max_latency=.5, goal_mode=True, floor_mode=True, adventure_mode=True,
                jev_player=True, stay_ready=True, startup_guard=True, logger=logs.append,
                continue_floors=continue_floors,
                on_navigation_stop=checkpoint).run()
    finally:
        client.close()
    return transport, requests, logs, result


class PlayerControllerTests(unittest.TestCase):
    def test_item_animation_resumes_same_attempt_without_f8_or_repeating_item(self):
        for clear in (False, True):
            with self.subTest(clear=clear):
                data = ready() if clear else combat()
                data["player"].update(active_item=85 if clear else 41, active_charge=2, active_max_charge=2)
                after = copy.deepcopy(data)
                after["player"]["active_charge"] = 0
                events = sequence(data, end=.8333333333)
                events += [(when, frame(after, 28, paused=True, item_animation=True), OLD)
                           for when in (.9, 1.15, 1.4, 1.65, 1.9, 2.15, 2.4, 2.65, 2.9)]
                events += sequence(after, start=3, end=3.8, first_frame=29)
                def choose(payload):
                    result = {"goal": "hold", "fire": "right", "activity": "wait", "ability": "none"}
                    for candidate in payload["state"].get("activity_candidates", []):
                        if candidate["kind"] == "use_active":
                            result["activity"] = candidate["option"]
                    for candidate in payload["state"].get("ability_candidates", []):
                        if candidate["kind"] == "use_active":
                            result["ability"] = candidate["option"]
                    return result
                transport, requests, logs, result = run(events, choose, delay=.05, duration=3.9)
                self.assertEqual(result["errors"], 0)
                self.assertEqual(result["stop_reason"], "duration reached")
                self.assertEqual(result["interaction_pulses"], 1, logs)
                self.assertFalse(any(.9 <= when < 3 for when, _, _ in transport.sent))
                resumed = [packet for when, packet, _ in transport.sent if 3 <= when < 3.8]
                self.assertTrue(any(packet["shoot"] == "right" for packet in resumed))
                self.assertTrue(all(packet.get("interaction", "none") == "none" for packet in resumed))
                self.assertFalse(any("Control stopped" in line for line in logs))
                self.assertTrue(all(not request["state"]["observation"]["paused"] for request in requests))
                self.assertLessEqual(result["decisions"], 6)

    def test_rearm_after_descent_before_first_playable_frame_keeps_attempt_alive(self):
        for arrival_enabled in (True, False):
            with self.subTest(arrival_enabled=arrival_enabled):
                first = boss_ready()
                second = ready()
                second.update(room_id="floor2-room", floor_advance_permitted=True)
                second["floor"].update(id="floor-2")
                events = sequence(first, end=.7333333333)
                events += [(.75, frame(second, 24, paused=True, enabled=arrival_enabled), OLD),
                           (.8, frame(second, 24, enabled=False, floor_advance_permitted=False), OLD)]
                second.update(session="fresh-arm", floor_advance_permitted=False)
                events += [(when, data, NEW) for when, data, _ in
                           sequence(second, start=.9, end=1.9, first_frame=24)]
                def choose(payload):
                    descent = next((c for c in payload["state"].get("activity_candidates", [])
                                    if c["kind"] == "descend"), None)
                    return {"activity": descent["option"] if descent else "wait"}
                transport, requests, logs, result = run(events, choose, duration=2,
                                                        continue_floors=True)
                self.assertEqual(result["stop_reason"], "duration reached")
                self.assertEqual(result["floors_advanced"], 1)
                self.assertEqual(result["floor_progress"]["rooms_visited"], 1)
                self.assertFalse(result["floor_progress"]["boss_cleared"])
                self.assertTrue(result["floor_history"][0]["progress"]["boss_cleared"])
                self.assertEqual(result["decisions"], len(requests))
                self.assertGreaterEqual(len(requests), 2)
                self.assertAlmostEqual(transport.now, 2, delta=.011)
                self.assertEqual(result["errors"], 0)
                self.assertTrue(any(p.get("transition") == "floor" for _, p, _ in transport.sent))
                resumed = [p for when, p, address in transport.sent if address == NEW and .9 <= when < 2]
                self.assertTrue(resumed)
                self.assertTrue(all(p["session"] == "fresh-arm" and p["room_id"] == "floor2-room"
                                    and p.get("transition") != "floor" for p in resumed))
                self.assertFalse(any(.75 <= when < .9 for when, _, _ in transport.sent))

    def test_hold_without_fire_never_fires_at_the_aligned_enemy(self):
        data = combat()
        transport, requests, logs, result = run([(i/10, frame(data, 30+i*3), OLD) for i in range(8)],
                                               {"goal": "hold", "fire": "none"}, duration=.8)
        self.assertTrue(requests)
        self.assertTrue(all(packet["shoot"] == "none" for _, packet, _ in transport.sent))
        self.assertEqual(result["errors"], 0)
        self.assertTrue(any("Jev chose: hold" in line for line in logs))

    def test_jev_firing_direction_is_not_replaced_by_local_aim(self):
        data = combat()
        data["player"]["vx"] = 1.5
        transport, requests, _, result = run([(i/10, frame(data, 30+i*3), OLD) for i in range(6)],
                                             {"goal": "hold", "fire": "up"}, duration=.6)
        self.assertTrue(any(packet["shoot"] == "up" for _, packet, _ in transport.sent))
        self.assertFalse(any(packet["shoot"] in ("left", "right", "down") for _, packet, _ in transport.sent))
        self.assertEqual(requests[0]["state"]["game_context"]["shooting_motion"]["player_velocity"]["vx"], 1.5)
        self.assertTrue(result["jev_player"])
        decision = result["player_decisions"][0]
        self.assertEqual(decision["combat_judgment"]["reported_choice"], "hold__up")
        self.assertEqual(decision["combat_judgment"]["probabilities"]["hold__up"], 1.)
        for moment in ("aim_at_request", "aim_at_reply"):
            self.assertIn("right", decision[moment]["firing_now"]["directions"])

    def test_startup_waits_for_jev_before_shooting(self):
        data = combat()
        transport, _, _, result = run([(i/10, frame(data, 30+i*3), OLD) for i in range(7)],
                                       {"goal": "hold", "fire": "right"}, delay=.35, duration=.7)
        self.assertFalse(any(packet["shoot"] != "none" for when, packet, _ in transport.sent if when < .35))
        self.assertTrue(any(packet["shoot"] == "right" for when, packet, _ in transport.sent if when >= .35))
        self.assertEqual(result["errors"], 0)

    def test_free_supplies_and_open_doors_wait_for_model_choice(self):
        data = ready(pickup())
        transport, requests, _, result = run([(i/10, frame(data, 30+i*3), OLD) for i in range(14)],
                                             {"activity": "wait"})
        self.assertEqual(len(requests), 1)
        self.assertEqual(set(requests[0]["questions"]), {"activity", "fire"})
        self.assertTrue(all((p["move"], p["shoot"]) == ("none", "none") for _, p, _ in transport.sent))
        self.assertEqual(result["pickup_progress"]["attempts"], 0)
        self.assertEqual(result["player_decisions"][0]["selected"], "wait")
        self.assertEqual(result["player_decisions"][0]["fire_judgment"]["probabilities"]["none"], 1.)

    def test_boss_door_choice_beats_old_local_room_order(self):
        data = ready()
        data["doors"] = [door(0, 83), door(2, 85, kind=5)]
        def choose(payload):
            if "activity" not in payload["questions"]:
                return {"fire": "none"}
            selected = next(c for c in payload["state"]["activity_candidates"] if c["key"] == "enter:2:85")
            return {"activity": selected["option"]}
        transport, _, _, result = run([(i/10, frame(data, 30+i*3), OLD) for i in range(11)], choose)
        moves = [p["move"] for _, p, _ in transport.sent if p["move"] != "none"]
        self.assertTrue(moves)
        self.assertEqual(set(moves), {"right"})
        self.assertEqual(result["player_decisions"][0]["selected"], "enter:2:85")

    def test_puzzle_offers_reach_jev_without_local_demolition_or_old_watchdog_stop(self):
        data = recorded()
        number = data["frame"]
        transport, requests, _, result = run([(i/10, frame(data, number+i*3), OLD) for i in range(14)],
                                             {"activity": "wait"})
        self.assertEqual(len(requests), 1)
        kinds = {c["kind"] for c in requests[0]["state"]["activity_candidates"]}
        self.assertTrue({"press_switch", "demolish_tnt"} <= kinds)
        self.assertTrue(all((p["move"], p["shoot"]) == ("none", "none") for _, p, _ in transport.sent))
        self.assertEqual(result["navigation_stops"], 0)
        self.assertEqual(result["errors"], 0)

    def test_inflight_combat_shot_is_discarded_when_room_becomes_a_puzzle(self):
        data = combat()
        empty = copy.deepcopy(data)
        empty["enemies"] = []
        events = [(0, frame(data, 30), OLD), (.1, frame(data, 33), OLD)]
        events += [(i/10, frame(empty, 30+i*3), OLD) for i in range(2, 14)]
        transport, requests, _, result = run(events, {"goal": "hold", "fire": "right", "activity": "wait"}, delay=.35)
        self.assertGreaterEqual(result["stale_discarded"], 1)
        self.assertFalse(any(p["shoot"] != "none" for _, p, _ in transport.sent))
        self.assertTrue(any("activity" in r["questions"] for r in requests))

    def test_pause_and_fresh_f8_cannot_execute_an_old_activity_reply(self):
        data = ready()
        def choose(payload):
            selected = next(c for c in payload["state"]["activity_candidates"] if c["kind"] == "enter_door")
            return {"activity": selected["option"]}
        events = [(0, frame(data, 30), OLD), (.6, frame(data, 48), OLD),
                  (.7, frame(data, 51, enabled=False), OLD),
                  (.8, frame(data, 54, session="fresh-arm"), NEW),
                  (.9, frame(data, 57, session="fresh-arm"), NEW)]
        transport, requests, _, result = run(events, choose, delay=.3, duration=1.1)
        self.assertEqual(len(requests), 1)
        self.assertTrue(all(p["move"] == "none" for _, p, _ in transport.sent))
        self.assertGreaterEqual(result["stale_discarded"], 1)

    def test_shared_request_cap_is_not_renewed_by_activity_waits(self):
        data = ready()
        _, requests, _, result = run([(i/10, frame(data, 30+i*3), OLD) for i in range(30)],
                                     {"activity": "wait"}, duration=3, max_calls=1)
        self.assertEqual(len(requests), 1)
        self.assertEqual(result["decisions"], 1)
        self.assertEqual(result["stop_reason"], "request cap reached")


if __name__ == "__main__":
    unittest.main()
