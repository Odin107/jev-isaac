"""Pickup collection and floor integration with synthetic observations only."""
import copy
from collections import deque
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_controller_arming import ImmediatePool, OLD, ScheduledSocket
from test_exploration import door, grid, state
from test_floor_drift_regression import circle_grid_clearance, physics_substep
from jev_isaac.controller import Controller
from jev_isaac.exploration import FloorNavigator
from jev_isaac.navigation import _VECTORS, _clear
from jev_isaac.pickups import PickupCollector, exclusion_boxes, priority, valid_pickups


def pickup(identity="coin", *, variant=20, subtype=1, x=240, y=280, **extra):
    result = {"id": identity, "type": 5, "variant": variant, "subtype": subtype,
              "x": x, "y": y, "vx": 0, "vy": 0, "radius": 10,
              "price": 0, "shop_item": False, "options_index": 0, "wait": 0,
              "collectible_kind": 0}
    result.update(extra)
    return result


def observed(*items, **extra):
    result = state(doors=[door(2, 85)], **extra)
    result.update(protocol=1, type="observation",
                  capabilities={"movement_pulses": 1, "local_goal_control": 1,
                                "floor_control": 1, "pickup_collection": 1},
                  pickups=list(items))
    result["player"].update(coins=0, bombs=0, keys=0, active_item=0,
                            can_pick_red_hearts=True, can_pick_soul_hearts=True,
                            can_pick_black_hearts=True, can_pickup_items=True)
    return result


def advance(nav, data, now):
    data["frame"] = max(data["frame"]+1, int(now*30)+1)
    return nav.step(data, now)


def collect(collector, data, now, move_to):
    data["frame"] = max(data["frame"], int(now*30)+1)
    return collector.step(data, now, move_to)


def ready(nav, data):
    first = nav.step(data, 0)
    assert first.move == "none", first
    return advance(nav, data, .61)


class PickupEligibilityTests(unittest.TestCase):
    def test_useful_health_precedes_items_and_supplies(self):
        player = observed()["player"]
        heart = pickup(variant=10)
        item = pickup(variant=100, subtype=1, collectible_kind=1)
        coin = pickup()
        self.assertLess(priority(heart, player), priority(item, player))
        self.assertLess(priority(item, player), priority(coin, player))
        for subtype, capacity in ((1, "can_pick_red_hearts"),
                                  (3, "can_pick_soul_hearts"),
                                  (6, "can_pick_black_hearts")):
            with self.subTest(subtype=subtype):
                changed = dict(player, **{capacity: False})
                self.assertIsNone(priority(pickup(variant=10, subtype=subtype), changed))
        player.update(can_pick_red_hearts=False, can_pick_soul_hearts=False)
        self.assertIsNone(priority(pickup(variant=10, subtype=10), player))

    def test_resource_capacities_and_unknown_variants_are_not_goals(self):
        player = observed()["player"]
        for variant, field in ((20, "coins"), (30, "keys"), (40, "bombs")):
            with self.subTest(variant=variant):
                self.assertIsNotNone(priority(pickup(variant=variant), dict(player, **{field: 98})))
                self.assertIsNone(priority(pickup(variant=variant), dict(player, **{field: 99})))
        for variant, subtype in ((50, 1), (60, 1), (70, 1), (90, 1), (300, 1),
                                 (20, 6), (30, 2), (40, 3), (10, 4), (100, 0)):
            self.assertIsNone(priority(pickup(variant=variant, subtype=subtype), player))

    def test_free_passives_and_familiars_allowed_active_replacement_excluded(self):
        player = observed()["player"]
        for kind in (1, 4):
            self.assertIsNotNone(priority(pickup(variant=100, collectible_kind=kind), player))
        active = pickup(variant=100, collectible_kind=3)
        self.assertIsNotNone(priority(active, player))
        self.assertIsNone(priority(active, dict(player, active_item=105)))
        self.assertIsNone(priority(pickup(variant=100, collectible_kind=0), player))

    def test_prices_and_shop_flags_never_become_collection_goals(self):
        player = observed()["player"]
        for extra in ({"price": 15}, {"price": -1}, {"price": -2},
                      {"price": -3}, {"shop_item": True}):
            for variant in (10, 20, 30, 40, 100):
                with self.subTest(extra=extra, variant=variant):
                    self.assertIsNone(priority(pickup(variant=variant, collectible_kind=1, **extra), player))

    def test_choice_siblings_and_unwanted_pedestals_are_obstacles(self):
        first = pickup("option-a", variant=100, x=220, collectible_kind=1, options_index=7)
        sibling = pickup("option-b", variant=100, x=420, collectible_kind=1, options_index=7)
        empty = pickup("empty", variant=100, subtype=0, x=500)
        data = observed(first, sibling, empty)
        boxes = exclusion_boxes(data, 10, (first["id"], first["variant"], first["subtype"]))
        self.assertEqual(len(boxes), 1)
        self.assertFalse(_clear((380, 280), (460, 280), boxes))
        self.assertTrue(_clear((180, 280), (260, 280), boxes))

    def test_advertised_pickup_data_requires_complete_safe_records(self):
        data = observed(pickup())
        self.assertTrue(valid_pickups(data))
        changes = [lambda s: s.pop("pickups"),
                   lambda s: s.update(pickups={}),
                   lambda s: s["pickups"].append(copy.deepcopy(s["pickups"][0])),
                   lambda s: s["pickups"][0].update(x=float("nan")),
                   lambda s: s["pickups"][0].update(price=True),
                   lambda s: s["pickups"][0].update(shop_item=0),
                   lambda s: s["pickups"][0].update(options_index=-1),
                   lambda s: s["pickups"][0].pop("wait"),
                   lambda s: s["player"].pop("active_item"),
                   lambda s: s["player"].update(can_pickup_items=1),
                   lambda s: s["player"].update(coins=-1)]
        for change in changes:
            with self.subTest(change=change):
                bad = copy.deepcopy(data)
                change(bad)
                self.assertFalse(valid_pickups(bad))


class PickupCollectorTests(unittest.TestCase):
    def test_nearest_equal_priority_option_is_stable_when_observation_order_changes(self):
        data = observed(pickup("far", x=100, options_index=4),
                        pickup("near", x=280, options_index=4))
        collector, targets = PickupCollector(), []
        move_to = lambda item: targets.append(item["id"]) or "left"
        collect(collector, data, 0, move_to)
        collect(collector, data, .61, move_to)
        data["pickups"].reverse()
        collect(collector, data, .7, move_to)
        self.assertEqual(targets, ["near", "near"])
        self.assertEqual(collector.attempts, 1)

    def test_disappearance_and_morph_invalidate_old_target_immediately(self):
        for morph in (False, True):
            with self.subTest(morph=morph):
                data = observed(pickup("target", variant=100, collectible_kind=1),
                                pickup("next", x=400))
                collector, targets = PickupCollector(), []
                move_to = lambda item: targets.append(item["id"]) or "left"
                collect(collector, data, 0, move_to)
                collect(collector, data, .61, move_to)
                if morph:
                    data["pickups"][0]["subtype"] = 0
                else:
                    data["pickups"].pop(0)
                self.assertEqual(collect(collector, data, .7, move_to).move, "none")
                collect(collector, data, 1.11, move_to)
                self.assertEqual(targets, ["target", "next"])
                self.assertEqual(collector.resolved, 1)

    def test_unreachable_targets_are_skipped_once_with_bounded_work_per_frame(self):
        data = observed(*(pickup(str(i), x=80+i*6) for i in range(20)))
        collector, targets = PickupCollector(), []
        def unreachable(item):
            targets.append(item["id"])
            return None
        collect(collector, data, 0, unreachable)
        for frame in range(6):
            before = len(targets)
            result = collect(collector, data, .61+frame*.1, unreachable)
            self.assertLessEqual(len(targets)-before, 4)
        self.assertIsNone(result)
        self.assertEqual(len(targets), 20)
        self.assertEqual(len(set(targets)), 20)
        self.assertEqual(collector.skipped, 20)

    def test_stalled_target_is_skipped_after_three_seconds(self):
        data = observed(pickup())
        collector = PickupCollector()
        collect(collector, data, 0, lambda _: "left")
        self.assertEqual(collect(collector, data, .61, lambda _: "left").move, "left")
        self.assertIsNone(collect(collector, data, 3.62, lambda _: "left"))
        self.assertEqual(collector.skipped, 1)
        self.assertIsNone(collect(collector, data, 4, lambda _: self.fail("Skipped target was retried")))

    def test_slowly_progressing_target_still_has_a_total_ten_second_limit(self):
        data = observed(pickup(x=80), x=520)
        collector = PickupCollector()
        collect(collector, data, 0, lambda _: "left")
        collect(collector, data, .61, lambda _: "left")
        for now in (2, 4, 6, 8, 10):
            data["player"]["x"] -= 10
            self.assertEqual(collect(collector, data, now, lambda _: "left").move, "left")
        self.assertIsNone(collect(collector, data, 10.62, lambda _: "left"))
        self.assertEqual(collector.skipped, 1)

    def test_pickup_wait_animation_sends_no_movement_and_is_bounded(self):
        data = observed(pickup(wait=30))
        collector = PickupCollector()
        blocked = lambda _: self.fail("Moved into an unavailable pickup")
        collect(collector, data, 0, blocked)
        self.assertEqual(collect(collector, data, .61, blocked).move, "none")
        self.assertIsNone(collect(collector, data, 3.62, blocked))

    def test_resuming_cannot_shorten_the_initial_room_reward_wait(self):
        data, collector, targets = observed(pickup()), PickupCollector(), []
        collector.pause()
        self.assertEqual(collect(collector, data, 0, lambda item: targets.append(item)).move, "none")
        self.assertEqual(collect(collector, data, .31, lambda item: targets.append(item)).move, "none")
        self.assertEqual(targets, [])
        self.assertEqual(collect(collector, data, .61, lambda _: "left").move, "left")

    def test_unavailable_player_animation_never_moves_and_eventually_stops(self):
        data, collector = observed(pickup()), PickupCollector()
        data["player"]["can_pickup_items"] = False
        blocked = lambda _: self.fail("Moved during player pickup animation")
        collect(collector, data, 0, blocked)
        self.assertEqual(collect(collector, data, .61, blocked).move, "none")
        result = collect(collector, data, 10.62, blocked)
        self.assertEqual(result.move, "none")
        self.assertEqual(result.stop_reason, "pickup animation timed out")


class PickupFloorTests(unittest.TestCase):
    def test_collects_before_leaving_then_resumes_door_route(self):
        data, nav = observed(pickup()), FloorNavigator()
        self.assertEqual(ready(nav, data).move, "left")
        self.assertEqual(nav.stats, {"rooms_visited": 1, "rooms_cleared": 1,
                                     "doors_traversed": 0, "boss_cleared": False})
        data["pickups"].clear()
        self.assertEqual(advance(nav, data, .7).move, "none")
        self.assertEqual(advance(nav, data, 1.11).move, "right")
        self.assertIsNone(nav.stop_reason)
        self.assertEqual(nav.pickup_stats["targets_disappeared_or_changed"], 1)

    def test_boss_reward_is_collected_before_floor_completion(self):
        data = observed(pickup("boss-reward", variant=100, collectible_kind=1), kind=5)
        data["doors"] = []
        nav = FloorNavigator()
        self.assertEqual(ready(nav, data).move, "left")
        self.assertIsNone(nav.stop_reason)
        data["pickups"].clear()
        self.assertIsNone(advance(nav, data, .7).stop_reason)
        self.assertEqual(advance(nav, data, 1.11).stop_reason, "floor cleared")

    def test_waits_for_delayed_room_clear_reward_before_selecting_door(self):
        data, nav = observed(), FloorNavigator()
        self.assertEqual(nav.step(data, 0).move, "none")
        data["pickups"].append(pickup())
        self.assertEqual(advance(nav, data, .3).move, "none")
        self.assertEqual(advance(nav, data, .61).move, "left")

    def test_boss_completion_waits_for_simulation_frames_not_only_wall_time(self):
        data, nav = observed(kind=5), FloorNavigator()
        data["doors"] = []
        self.assertEqual(nav.step(data, 0).move, "none")
        data["frame"] += 1
        result = nav.step(data, 8)
        self.assertIsNone(result.stop_reason)
        self.assertEqual(result.move, "none")
        data["frame"] = 19
        self.assertEqual(nav.step(data, 8.1).stop_reason, "floor cleared")

    def test_missing_capability_preserves_existing_door_navigation(self):
        data, nav = state(doors=[door(2, 85)]), FloorNavigator()
        self.assertEqual(nav.step(data, 0).move, "right")

    def test_disabled_paused_combat_or_dead_never_collects(self):
        for key in ("disabled", "paused", "combat", "dead"):
            with self.subTest(state=key):
                data, nav = observed(pickup()), FloorNavigator()
                ready(nav, data)
                if key == "disabled": data["enabled"] = False
                elif key == "paused": data["paused"] = True
                elif key == "combat": data["room"]["clear"] = False
                else: data["player"]["dead"] = True
                self.assertEqual(advance(nav, data, .7).move, "none")

    def test_truncated_or_malformed_pickup_observation_stops_before_map_update(self):
        for change in (lambda s: s.update(truncated_arrays={"pickups": True}),
                       lambda s: s.pop("pickups"),
                       lambda s: s["pickups"][0].update(price="free"),
                       lambda s: s["player"].pop("can_pick_red_hearts")):
            data, nav = observed(pickup()), FloorNavigator()
            change(data)
            result = nav.step(data, 0)
            self.assertEqual(result.move, "none")
            self.assertEqual(result.stop_reason, "incomplete floor observation")
            self.assertEqual(nav.stats["rooms_visited"], 0)

    def test_unreachable_pickup_does_not_prevent_accessible_door(self):
        data, nav = observed(pickup(x=120), x=400), FloorNavigator()
        data["hazards"] = [grid(240, y) for y in range(120, 441, 40)]
        result = ready(nav, data)
        self.assertIsNone(result.stop_reason)
        self.assertEqual(result.move, "right")
        self.assertEqual(nav.pickup_stats["skipped_unreachable_or_stalled"], 1)

    def test_pickup_routes_avoid_rocks_pits_and_paid_pickups(self):
        for obstacle in ("rock", "pit", "paid"):
            with self.subTest(obstacle=obstacle):
                data, nav = observed(pickup("target", x=440), x=200), FloorNavigator()
                if obstacle == "paid":
                    data["pickups"].append(pickup("paid", x=320, price=15, shop_item=True))
                    boxes = [(292, 252, 348, 308)]
                else:
                    data["hazards"] = [grid(320, 280, kind=2 if obstacle == "rock" else 7,
                                            collision=3 if obstacle == "rock" else 1)]
                    boxes = [(290, 250, 350, 310)]
                nav.step(data, 0)
                for frame in range(1, 240):
                    action = advance(nav, data, .61+frame/30)
                    self.assertIsNone(action.stop_reason)
                    vector = _VECTORS[action.move]
                    start = data["player"]["x"], data["player"]["y"]
                    end = start[0]+4*vector[0], start[1]+4*vector[1]
                    self.assertTrue(_clear(start, end, boxes), (obstacle, start, end, action))
                    data["player"].update(x=end[0], y=end[1])
                    if math.dist(end, (440, 280)) <= 18:
                        break
                else:
                    self.fail(f"Did not reach pickup safely around {obstacle}")

    def test_door_route_also_avoids_paid_pickups(self):
        data, nav = observed(pickup("shop", x=400, price=15, shop_item=True), x=240), FloorNavigator()
        nav.step(data, 0)
        boxes = [(372, 252, 428, 308)]
        for frame in range(1, 180):
            action = advance(nav, data, .61+frame/30)
            self.assertIsNone(action.stop_reason)
            vector = _VECTORS[action.move]
            start = data["player"]["x"], data["player"]["y"]
            end = start[0]+4*vector[0], start[1]+4*vector[1]
            self.assertTrue(_clear(start, end, boxes), (start, end, action))
            data["player"].update(x=end[0], y=end[1])
            if end[0] > 608:
                break
        else:
            self.fail("Did not reach door without crossing a paid pickup")

    def test_pedestal_collision_then_door_with_inertia_and_delayed_input(self):
        # Two motion substeps and one observation of input delay follow the
        # recorded-floor stress model, not the planner's idealized direction.
        item = pickup("pedestal", variant=100, subtype=17, x=440, collectible_kind=1)
        data, nav = observed(item, x=200), FloorNavigator()
        obstacle = grid(320, 280)
        data["hazards"] = [obstacle]
        pending, collected_at, door_resumed = deque(["none"]), None, False
        player = data["player"]
        for tick in range(360):
            data["frame"] = tick+1
            action = nav.step(data, tick/30)
            self.assertIsNone(action.stop_reason, (tick, action, player))
            self.assertEqual(action.shoot, "none")
            if collected_at is not None and action.status == "entering room 85":
                door_resumed = True
            pending.append(action.move)
            applied = _VECTORS[pending.popleft()]
            for substep in range(2):
                physics_substep(player, applied)
                self.assertGreaterEqual(circle_grid_clearance(player, obstacle), -1e-6,
                                        (tick, substep, player, action))
                if collected_at is None and math.dist((player["x"], player["y"]), (440, 280)) <= 20:
                    collected_at = tick
                    data["pickups"].clear()
            if player["x"] > 608:
                self.assertIsNotNone(collected_at, "Door crossed before reaching the pedestal")
                self.assertTrue(door_resumed)
                self.assertEqual(nav.pickup_stats["targets_disappeared_or_changed"], 1)
                self.assertLessEqual(abs(player["y"]-280), 10)
                return
        self.fail(f"Did not collect pedestal and reach door; collected at {collected_at}")

    def test_pickup_phase_makes_no_jev_requests(self):
        before, targeting = observed(pickup(), frame=30), observed(pickup(), frame=48)
        gone, exiting = observed(frame=51), observed(frame=63)
        transport = ScheduledSocket([(0, before, OLD), (.61, targeting, OLD),
                                     (.7, gone, OLD), (1.11, exiting, OLD)])
        calls = []
        with patch("jev_isaac.controller.socket.socket", return_value=transport), \
             patch("jev_isaac.controller.time.monotonic", side_effect=lambda: transport.now), \
             patch("jev_isaac.controller.concurrent.futures.ThreadPoolExecutor", ImmediatePool):
            result = Controller(lambda data: calls.append(data), duration=1.2,
                                goal_mode=True, floor_mode=True, logger=lambda _: None).run()
        self.assertEqual(calls, [])
        self.assertEqual(result["decisions"], 0)
        packets = [packet for _, packet, _ in transport.sent]
        self.assertTrue(any(p["frame"] == 48 and p["move"] == "left" for p in packets))
        self.assertTrue(any(p["frame"] == 63 and p["move"] == "right" for p in packets))


if __name__ == "__main__":
    unittest.main()
