"""Offline controller orchestration with real exploration and fake time/UDP."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_adventure import ready, pickup, blocked_chest
from test_controller_arming import ImmediatePool, OLD, ScheduledSocket
from test_exploration import door, grid
from jev_isaac.controller import Action, Controller
from jev_isaac.exploration import FloorNavigator
from jev_isaac.strategy import EquippedGoalDecision, StrategyDecision


def sequence(data, start=0, end=1.0, *, first_frame=1):
    events = []
    for tick in range(round((end-start)*30)+1):
        snapshot = copy.deepcopy(data)
        snapshot["frame"] = first_frame + tick
        events.append((start + tick/30, snapshot, OLD))
    return events


def strategic(data):
    options = data.get("_adventure_options")
    if options:
        return StrategyDecision("adventure", options[0]["key"], 10, "offline",
                                {"input_tokens": 10, "output_tokens": 2})
    abilities = data.get("_ability_options", [])
    return EquippedGoalDecision("engage", "target", 10, "offline",
                                {"input_tokens": 10, "output_tokens": 2}, (),
                                abilities[0]["key"] if abilities else None)


def run(events, *, delay=0, policy=strategic, **options):
    transport, calls, messages, navigators = ScheduledSocket(events), [], [], []
    def call(data):
        calls.append((transport.now, data))
        return policy(data)
    class Pool(ImmediatePool):
        def submit(self, function, *args):
            value = function(*args)
            completed_at = transport.now + delay
            return SimpleNamespace(done=lambda: transport.now >= completed_at,
                                   result=lambda: value)
    def navigator(*args, **kwargs):
        instance = FloorNavigator(*args, **kwargs)
        navigators.append(instance)
        return instance
    with patch("jev_isaac.controller.socket.socket", return_value=transport), \
         patch("jev_isaac.controller.time.monotonic", side_effect=lambda: transport.now), \
         patch("jev_isaac.controller.concurrent.futures.ThreadPoolExecutor", Pool), \
         patch("jev_isaac.navigation.compute_action", return_value=Action("right", "up")), \
         patch("jev_isaac.exploration.FloorNavigator", side_effect=navigator):
        result = Controller(call, duration=options.pop("duration", 1.1),
                            max_hz=options.pop("max_hz", 2), max_latency=.5,
                            goal_mode=True, floor_mode=True,
                            adventure_mode=options.pop("adventure_mode", True),
                            logger=messages.append, **options).run()
    return transport, calls, result, navigators, messages


def combat_ready():
    data = ready(clear=False)
    data["player"].update(active_item=34, active_charge=3, active_max_charge=3)
    data["enemies"] = [{"id": "target", "x": 500, "y": 280, "vx": 0, "vy": 0,
                        "radius": 16, "hp": 20, "vulnerable": True}]
    return data


def boss_ready():
    data = ready(kind=5)
    data["doors"] = []
    data["hazards"] = [grid(370, 280, kind=17, collision=0)]
    return data


class AdventureControllerTests(unittest.TestCase):
    def test_clear_room_holds_position_until_fresh_strategic_reply_then_collects(self):
        data = ready(pickup("pill", variant=70, x=240))
        transport, calls, result, _, _ = run(sequence(data), delay=.1)
        self.assertEqual(len(calls), 1)
        dispatched, source = calls[0]
        self.assertTrue(source["room"]["clear"])
        self.assertEqual(source["_adventure_options"][0]["target_id"], "pill")
        self.assertTrue(all(packet["move"] == "none" and packet["shoot"] == "none"
                            and packet.get("interaction", "none") == "none"
                            for when, packet, _ in transport.sent if when < dispatched+.1))
        self.assertTrue(any(packet["move"] == "left" for when, packet, _ in transport.sent
                            if when >= dispatched+.1))
        self.assertEqual(result["strategic_choices"], 1)
        self.assertEqual(result["adventure_progress"]["selected"], 1)
        self.assertEqual(result["errors"], 0)

    def test_strategy_reply_received_after_pause_is_counted_but_not_executed(self):
        data = ready(pickup("pill", variant=70, x=240))
        paused = copy.deepcopy(data)
        paused.update(paused=True, enabled=False)
        events = sequence(data, end=.6333333333)
        events += sequence(paused, start=.65, end=.95, first_frame=21)
        transport, calls, result, _, _ = run(events, delay=.12, stay_ready=True, duration=1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["responses_with_usage"], 1)
        self.assertEqual(result["stale_discarded"], 1)
        self.assertEqual(result["strategic_choices"], 0)
        self.assertEqual(result["adventure_progress"]["selected"], 0)
        self.assertFalse(any(packet["move"] != "none" or packet.get("interaction", "none") != "none"
                             for _, packet, _ in transport.sent))

    def test_strategy_rejects_repriced_offer_before_it_can_spend_coins(self):
        data = ready(pickup("shop", variant=100, subtype=1, collectible_kind=1,
                            shop_item=True, price=15, x=240))
        data["player"]["coins"] = 20
        changed = copy.deepcopy(data)
        changed["pickups"][0]["price"] = 16
        events = sequence(data, end=.6333333333)
        events += sequence(changed, start=.65, end=.95, first_frame=21)
        transport, _, result, _, _ = run(events, delay=.12, duration=1)
        self.assertEqual(result["adventure_progress"]["selected"], 0)
        self.assertFalse(any(packet["move"] == "left" for _, packet, _ in transport.sent))

    def test_combat_goal_and_ability_share_requests_and_emit_only_one_pulse(self):
        transport, calls, result, _, _ = run(sequence(combat_ready(), end=.5),
                                            max_calls=3, max_hz=10, duration=.6)
        self.assertEqual(result["stop_reason"], "request cap reached")
        self.assertEqual(result["decisions"], 3)
        self.assertEqual(len(calls), 3)
        self.assertIn("_ability_options", calls[0][1])
        self.assertTrue(all("_ability_options" not in data for _, data in calls[1:]))
        pulses = [packet for _, packet, _ in transport.sent if packet.get("interaction", "none") != "none"]
        self.assertEqual(len(pulses), 1)
        self.assertEqual(pulses[0]["interaction"], "active")
        self.assertTrue(pulses[0]["interaction_id"])
        self.assertEqual(pulses[0]["move"], "right")
        self.assertEqual(pulses[0]["shoot"], "up")
        self.assertEqual(result["interaction_pulses"], 1)
        self.assertEqual(result["responses_with_usage"], 3)

    def test_combat_ability_reply_after_pause_never_pulses(self):
        data = combat_ready()
        paused = copy.deepcopy(data)
        paused.update(paused=True, enabled=False)
        events = sequence(data, end=.0333333333)
        events += sequence(paused, start=.05, end=.18, first_frame=3)
        transport, calls, result, _, _ = run(events, delay=.1, stay_ready=True, duration=.2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["stale_discarded"], 1)
        self.assertEqual(result["interaction_pulses"], 0)
        self.assertFalse(any(packet.get("interaction", "none") != "none" for _, packet, _ in transport.sent))

    def test_accepted_but_unsent_ability_does_not_survive_a_same_session_pause(self):
        data = combat_ready()
        paused = copy.deepcopy(data)
        paused.update(paused=True, frame=2)
        resumed = copy.deepcopy(data)
        resumed["frame"] = 3
        # Startup reflex already uses source frame1. The reply at.01 queues an
        # ability for the next frame; a pause arrives before that fresh frame.
        events = [(0, data, OLD), (.02, paused, OLD), (.04, resumed, OLD)]
        transport, calls, result, _, _ = run(events, delay=.01, duration=.15,
                                            startup_guard=True, stay_ready=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["goal_updates"], 1)
        self.assertEqual(result["interaction_pulses"], 0)
        self.assertFalse(any(packet.get("interaction", "none") != "none" for _, packet, _ in transport.sent))

    def test_requested_descent_with_game_permit_continues_and_resets_only_map(self):
        first = boss_ready()
        second = combat_ready()
        second.update(room_id="floor2-room")
        second["floor"].update(id="floor-2", room_index=70)
        second["floor_advance_permitted"] = True
        events = sequence(first, end=.7333333333)
        events += sequence(second, start=.75, end=1.05, first_frame=24)
        transport, calls, result, navigators, messages = run(events, duration=1.1,
                                                            continue_floors=True, wait_for_arm=True)
        self.assertEqual(result["floors_advanced"], 1)
        self.assertEqual(len(navigators), 2)
        self.assertEqual(len(result["floor_history"]), 1)
        self.assertTrue(result["floor_history"][0]["progress"]["boss_cleared"])
        self.assertEqual(result["floor_progress"]["rooms_visited"], 1)
        self.assertFalse(result["floor_progress"]["boss_cleared"])
        self.assertAlmostEqual(transport.now, 1.1, delta=.011)
        self.assertEqual(result["stop_reason"], "duration reached")
        self.assertEqual(result["decisions"], len(calls))
        self.assertTrue(any(packet.get("transition") == "floor" for _, packet, _ in transport.sent))
        self.assertTrue(any("same remaining limits" in message for message in messages))

    def test_descent_does_not_reset_the_request_budget(self):
        first, second = boss_ready(), combat_ready()
        second.update(room_id="floor2-room", floor_advance_permitted=True)
        second["floor"].update(id="floor-2", room_index=70)
        events = sequence(first, end=.7333333333)
        events += sequence(second, start=.75, end=1.3, first_frame=24)
        _, calls, result, _, _ = run(events, duration=2, continue_floors=True, max_calls=2)
        self.assertEqual(result["floors_advanced"], 1)
        self.assertEqual(result["decisions"], 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["stop_reason"], "request cap reached")

    def test_floor_change_without_both_local_descent_and_game_permission_stops(self):
        for own_descent, permit in ((False, True), (True, False), (False, False)):
            with self.subTest(own_descent=own_descent, permit=permit):
                first = boss_ready() if own_descent else ready()
                second = combat_ready()
                second.update(room_id="floor2-room", floor_advance_permitted=permit)
                second["floor"].update(id="floor-2", room_index=70)
                events = sequence(first, end=.7333333333)
                events += sequence(second, start=.75, end=.9, first_frame=24)
                _, _, result, navigators, _ = run(events, continue_floors=True)
                self.assertEqual(result["stop_reason"], "floor changed")
                self.assertEqual(result["floors_advanced"], 0)
                self.assertEqual(len(navigators), 1)

    def test_new_run_cannot_reuse_a_descent_permit(self):
        first, second = boss_ready(), combat_ready()
        second.update(room_id="floor2-room", run_id="new-run", floor_advance_permitted=True)
        second["floor"].update(id="floor-2", room_index=70)
        events = sequence(first, end=.7333333333)
        events += sequence(second, start=.75, end=.9, first_frame=24)
        _, _, result, _, _ = run(events, continue_floors=True)
        self.assertEqual(result["stop_reason"], "run changed")
        self.assertEqual(result["floors_advanced"], 0)

    def test_default_floor_mode_does_not_make_strategic_calls_or_enter_exit(self):
        data = boss_ready()
        data["capabilities"].pop("interaction_control")
        transport, calls, result, _, _ = run(sequence(data), adventure_mode=False)
        self.assertEqual(calls, [])
        self.assertEqual(result["strategic_choices"], 0)
        self.assertEqual(result["stop_reason"], "floor cleared")
        self.assertFalse(any(packet.get("transition") for _, packet, _ in transport.sent))

    def test_pending_boss_reward_is_offered_before_floor_descent(self):
        data = boss_ready()
        data["pickups"] = [pickup("reward", variant=100, subtype=1, collectible_kind=1, x=240)]
        nav = FloorNavigator(adventure_mode=True, continue_floors=True)
        nav.step(data, 0)
        data["frame"] = 20
        nav.step(data, .64)
        self.assertTrue(nav.adventure_options)
        self.assertFalse(any(option.kind == "descend" for option in nav.adventure_options))
        self.assertTrue(any(option.target_id == "reward" for option in nav.adventure_options))

    def test_selected_locked_door_remains_viable_inside_its_crossing_corridor(self):
        data = ready(x=560)
        data["player"]["keys"] = 1
        data["doors"] = [door(2, 85, kind=4, opened=False, locked=True)]
        nav = FloorNavigator(adventure_mode=True)
        nav.step(data, 0)
        data["frame"] = 20
        nav.step(data, .64)
        choice = next(option for option in nav.adventure_options if option.kind == "unlock_door")
        self.assertTrue(nav.accept_adventure(choice.key, data, .65))
        # The selected portal explicitly extends beyond the normal room inset.
        # A lock may still be observed until the next contact/animation update.
        data["player"]["x"] = 575
        data["frame"] = 21
        action = nav.step(data, .67)
        self.assertEqual(action.move, "right")
        self.assertEqual(nav.adventure_stats["abandoned"], 0)

    def test_trial_deadline_blocks_bomb_pulse_without_extending_the_budget(self):
        data = blocked_chest()
        data["player"]["x"] = 320  # Already at the verified placement.
        transport, calls, result, _, _ = run(sequence(data), duration=1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["adventure_progress"]["selected"], 1)
        self.assertEqual(result["stop_reason"], "insufficient trial time for bomb retreat")
        self.assertLess(transport.now, 1)
        self.assertEqual(result["interaction_pulses"], 0)
        self.assertFalse(any(packet.get("interaction") == "bomb" for _, packet, _ in transport.sent))
        self.assertEqual(transport.sent[-1][1]["move"], "none")
        self.assertIs(transport.sent[-1][1]["floor_mode"], False)

    def test_same_bomb_plan_can_pulse_when_trial_has_enough_retreat_time(self):
        data = blocked_chest()
        data["player"]["x"] = 320
        transport, _, result, _, _ = run(sequence(data), duration=5)
        pulses = [(when, packet) for when, packet, _ in transport.sent
                  if packet.get("interaction") == "bomb"]
        self.assertEqual(len(pulses), 1)
        self.assertGreaterEqual(5 - pulses[0][0], 4)
        self.assertEqual(result["interaction_pulses"], 1)


if __name__ == "__main__":
    unittest.main()
