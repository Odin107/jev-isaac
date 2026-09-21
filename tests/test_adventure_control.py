"""Observed-state interaction execution with real eligibility/geometry checks."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_adventure import ready, blocked_chest
from test_exploration import door, grid
from test_pickups import pickup
from jev_isaac.adventure_control import AdventureControl
from jev_isaac.adventure import _grid_id
from jev_isaac.protocol import Observation, encode_action


def select(control, state, kind, now=0, **options):
    selected = next(item for item in control.choose_offers(state, **options) if item.kind == kind)
    assert control.accept(selected.key, state, now, **options)
    return selected


def charged():
    data = ready()
    data["player"].update(active_item=45, active_charge=4, active_max_charge=4)
    return data


class AdventureControlTests(unittest.TestCase):
    def test_shop_cost_or_resources_changed_cancels_before_movement(self):
        for change in (lambda d: d["pickups"][0].update(price=16),
                       lambda d: d["player"].update(coins=14)):
            data = ready(pickup("shop", variant=100, collectible_kind=1, price=15, shop_item=True))
            data["player"]["coins"] = 20
            control = AdventureControl()
            select(control, data, "buy")
            change(data)
            action = control.step(data, .1)
            self.assertEqual((action.move, action.interaction), ("none", "none"))
            self.assertIsNone(control.plan)
            self.assertEqual(control.abandoned, 1)

    def test_selected_black_heart_tracks_its_observed_position_after_blast(self):
        data = ready(pickup("heart", variant=10, subtype=6,
                            x=304.875, y=287.97387695313), x=400)
        control = AdventureControl()
        selected = select(control, data, "collect")
        # The live failure occurred after only 8.23 units of post-blast drift.
        data["pickups"][0].update(x=298.93710327148, y=282.28036499023)
        action = control.step(data, .1)
        self.assertNotEqual(action.move, "none")
        self.assertIs(control.plan, selected)
        self.assertEqual(control.abandoned, 0)
        # Push the same heart across Isaac: steering must use its new location,
        # not merely remove the old-position validation and chase that point.
        data["pickups"][0].update(x=480, y=data["player"]["y"])
        self.assertEqual(control.step(data, .2).move, "right")
        self.assertIs(control.plan, selected)
        data["pickups"] = []
        control.step(data, .3)
        self.assertIsNone(control.plan)
        self.assertEqual(control.completed, 1)

    def test_pickup_motion_while_jev_decides_preserves_the_bound_choice(self):
        data = ready(pickup("heart", variant=10, subtype=6), x=400)
        control = AdventureControl()
        chosen = next(c for c in control.choose_offers(data) if c.kind == "collect")
        data["pickups"][0].update(x=480, y=data["player"]["y"])
        self.assertTrue(control.accept(chosen.key, data, .4))
        self.assertIs(control.plan, chosen)
        self.assertEqual(control.step(data, .5).move, "right")

    def test_moving_selected_pickup_still_requires_fresh_eligibility_and_route(self):
        changes = (lambda d: d["pickups"][0].update(options_index=2),
                   lambda d: d["pickups"][0].update(price=3, shop_item=True),
                   lambda d: d["player"].update(can_pick_black_hearts=False),
                   lambda d: d["pickups"][0].update(wait=5),
                   lambda d: d["hazards"].append(grid(480, 280)))
        for change in changes:
            with self.subTest(change=change):
                data = ready(pickup("heart", variant=10, subtype=6), x=400)
                control = AdventureControl()
                select(control, data, "collect")
                data["pickups"][0].update(x=480, y=280)
                change(data)
                action = control.step(data, .1)
                self.assertEqual((action.move, action.interaction), ("none", "none"))
                self.assertIsNone(control.plan)
                self.assertEqual(control.abandoned, 1)

    def test_missing_moving_target_never_selects_another_heart(self):
        data = ready(pickup("chosen", variant=10, subtype=6), x=400)
        control = AdventureControl()
        select(control, data, "collect")
        data["pickups"] = [pickup("different", variant=10, subtype=6, x=480)]
        action = control.step(data, .1)
        self.assertEqual((action.move, action.interaction), ("none", "none"))
        self.assertIsNone(control.plan)

    def test_moving_purchase_tracks_same_item_without_changing_price(self):
        data = ready(pickup("shop", variant=100, collectible_kind=1,
                            price=15, shop_item=True), x=400)
        data["player"]["coins"] = 20
        control = AdventureControl()
        chosen = select(control, data, "buy")
        data["pickups"][0].update(x=480, y=data["player"]["y"])
        self.assertEqual(control.step(data, .1).move, "right")
        self.assertIs(control.plan, chosen)
        data["pickups"][0]["price"] = 16
        self.assertEqual(control.step(data, .2).move, "none")
        self.assertIsNone(control.plan)

    def test_target_morph_never_walks_toward_old_purchase(self):
        data = ready(pickup("item", variant=100, collectible_kind=1))
        control = AdventureControl()
        select(control, data, "collect")
        data["pickups"][0]["subtype"] = 100
        action = control.step(data, .1)
        self.assertEqual((action.move, action.interaction), ("none", "none"))
        self.assertIsNone(control.plan)
        self.assertIn("changed or disappeared", action.status)

    def test_active_use_is_a_single_unique_pulse_with_no_automatic_repeat(self):
        data, control = charged(), AdventureControl()
        select(control, data, "use_active")
        first = control.step(data, .1)
        self.assertEqual(first.interaction, "active")
        self.assertTrue(first.interaction_id)
        for now in (.11, .2, .3, .5):
            self.assertEqual(control.step(data, now).interaction, "none")
        self.assertEqual(control.step(data, .61).interaction, "none")
        self.assertIsNone(control.plan)
        self.assertFalse(control.choose_offers(data))
        data["room_id"] = "next-room"
        select(control, data, "use_active", 1)
        second = control.step(data, 1.1)
        self.assertNotEqual(first.interaction_id, second.interaction_id)

    def test_charge_or_inventory_change_before_pulse_cancels(self):
        for change in (lambda d: d["player"].update(active_charge=3),
                       lambda d: d["player"].update(active_item=78),
                       lambda d: d["player"].update(can_pick_red_hearts=False)):
            data, control = charged(), AdventureControl()
            select(control, data, "use_active")
            change(data)
            self.assertEqual(control.step(data, .1).interaction, "none")
            self.assertIsNone(control.plan)

    def test_pocket_pulse_is_not_held_for_the_movement_lease(self):
        data, control = ready(), AdventureControl()
        data["player"]["pocket_card"] = 23
        select(control, data, "use_pocket")
        first = control.step(data, .1)
        later = control.step(data, .11)
        packets = [json.loads(encode_action(Observation(data), action.move, action.shoot,
                         15, move_frames=6, move_distance=20, floor_mode=True,
                         interaction=action.interaction, interaction_id=action.interaction_id))
                   for action in (first, later)]
        self.assertEqual(packets[0]["interaction"], "pocket")
        self.assertEqual(packets[0]["interaction_id"], first.interaction_id)
        self.assertEqual(packets[0]["hold_frames"], 15)
        self.assertNotIn("interaction", packets[1])
        self.assertNotIn("interaction_id", packets[1])

    def test_interaction_packet_requires_identity_and_floor_transition_requires_floor_mode(self):
        observation = Observation(ready())
        for options in ({"interaction": "bomb"}, {"interaction": "active", "interaction_id": ""},
                        {"interaction": "pocket", "interaction_id": "bad\nid"},
                        {"interaction": "unknown", "interaction_id": "id"},
                        {"interaction_id": "unused"}, {"transition": "floor"},
                        {"transition": "invented", "floor_mode": True}):
            with self.assertRaises(ValueError):
                encode_action(observation, "none", "none", **options)
        packet = json.loads(encode_action(observation, "left", "none", floor_mode=True,
                                         transition="floor"))
        self.assertEqual(packet["transition"], "floor")

    def test_pause_disable_and_death_never_pulse_or_move(self):
        for change in (lambda d: d.update(paused=True), lambda d: d.update(enabled=False),
                       lambda d: d["player"].update(dead=True)):
            data, control = charged(), AdventureControl()
            select(control, data, "use_active")
            change(data)
            action = control.step(data, .1)
            self.assertEqual((action.move, action.interaction), ("none", "none"))
            self.assertIsNone(control.used_at)

    def test_room_run_and_floor_changes_invalidate_old_plan(self):
        for change in (lambda d: d.update(room_id="other"), lambda d: d.update(run_id="other"),
                       lambda d: d["floor"].update(id="other")):
            data, control = charged(), AdventureControl()
            select(control, data, "use_active")
            change(data)
            action = control.step(data, .1)
            self.assertTrue(action is None or (action.move, action.interaction) == ("none", "none"))
            self.assertIsNone(control.plan)

    def test_skipping_current_offers_does_not_blacklist_future_new_pickups(self):
        data = ready(pickup("pill", variant=70))
        control = AdventureControl()
        old = control.choose_offers(data)
        self.assertTrue(old)
        self.assertFalse(control.accept(None, data, 0))
        self.assertFalse(control.choose_offers(data))
        data["pickups"].append(pickup("new-pill", variant=70, x=400))
        offers = control.choose_offers(data)
        self.assertEqual(len(offers), 1)
        self.assertEqual(offers[0].target_id, "new-pill")

    def test_unknown_selection_never_becomes_a_plan(self):
        data, control = charged(), AdventureControl()
        control.choose_offers(data)
        self.assertFalse(control.accept("invented:bomb", data, 0))
        self.assertIsNone(control.step(data, .1))

    def test_chest_spends_no_explicit_button_and_stops_after_observed_open(self):
        data = ready(pickup("locked", variant=60))
        data["player"]["keys"] = 1
        control = AdventureControl()
        select(control, data, "open_chest")
        action = control.step(data, .1)
        self.assertEqual(action.move, "left")
        self.assertEqual(action.interaction, "none")
        data["pickups"][0]["subtype"] = 0
        self.assertEqual(control.step(data, .2).move, "none")
        self.assertIsNone(control.plan)

    def test_unlocked_door_completion_stops_old_approach(self):
        data = ready()
        data["player"]["keys"] = 1
        data["doors"] = [door(2, 85, 4, opened=False, locked=True)]
        control = AdventureControl()
        select(control, data, "unlock_door")
        self.assertEqual(control.step(data, .1).move, "right")
        data["doors"][0].update(open=True, locked=False)
        action = control.step(data, .2)
        self.assertEqual(action.move, "none")
        self.assertEqual(action.status, "door unlocked")
        self.assertIsNone(control.plan)

    def test_bomb_pulses_only_at_placement_then_retreats_and_waits_for_observed_blast(self):
        data, control = blocked_chest(), AdventureControl()
        plan = select(control, data, "bomb_rock")
        approaching = control.step(data, .1)
        self.assertEqual(approaching.interaction, "none")
        self.assertNotEqual(approaching.move, "none")
        data["player"].update(x=plan.point[0], y=plan.point[1])
        pulse = control.step(data, .2)
        self.assertEqual(pulse.interaction, "bomb")
        self.assertTrue(pulse.interaction_id)
        self.assertNotEqual(pulse.move, "none")
        data["player"]["bombs"] = 0
        data["hazards"].append({"kind": "bomb", "x": plan.point[0], "y": plan.point[1],
                                "vx": 0, "vy": 0, "radius": 10})
        retreat = control.step(data, .3)
        self.assertEqual(retreat.interaction, "none")
        self.assertNotEqual(retreat.move, "none")
        data["player"].update(x=plan.escape_point[0], y=plan.escape_point[1])
        waiting = control.step(data, .5)
        self.assertEqual((waiting.move, waiting.interaction), ("none", "none"))
        self.assertIsNotNone(control.plan)
        data["hazards"] = [h for h in data["hazards"] if h.get("kind") != "bomb"
                           and _grid_id(h) != plan.rock_id]
        finished = control.step(data, 2)
        self.assertEqual(finished.interaction, "none")
        self.assertIsNone(control.plan)
        self.assertEqual(control.completed, 1)

    def test_bomb_inventory_target_and_retreat_changes_cancel_before_placement(self):
        for change in (lambda d, p: d["player"].update(bombs=0),
                       lambda d, p: d["player"].update(keys=0),
                       lambda d, p: d["pickups"].clear(),
                       lambda d, p: d["hazards"].append(grid(p.escape_point[0], p.escape_point[1]))):
            data, control = blocked_chest(), AdventureControl()
            plan = select(control, data, "bomb_rock")
            data["player"].update(x=plan.point[0], y=plan.point[1])
            change(data, plan)
            action = control.step(data, .1)
            self.assertEqual((action.move, action.interaction), ("none", "none"))
            self.assertIsNone(control.plan)

    def test_bomb_waits_for_actual_placement_velocity_to_settle_before_one_pulse(self):
        data, control = blocked_chest(), AdventureControl()
        plan = select(control, data, "bomb_rock")
        data["player"].update(x=plan.point[0]+4, y=plan.point[1], vx=1.2, vy=.2)
        settling = control.step(data, .1)
        self.assertEqual((settling.move, settling.interaction), ("none", "none"))
        self.assertEqual(control.phase, "approach")
        self.assertIsNone(control.used_at)
        data["player"].update(vx=.2, vy=.1)
        pulse = control.step(data, .2)
        self.assertEqual(pulse.interaction, "bomb")
        self.assertNotEqual(pulse.move, "none")
        data["player"]["bombs"] = 0
        self.assertEqual(control.step(data, .21).interaction, "none")

    def test_pausing_during_bomb_retreat_preserves_escape_after_long_pause(self):
        data, control = blocked_chest(), AdventureControl()
        plan = select(control, data, "bomb_rock")
        data["player"].update(x=plan.point[0], y=plan.point[1])
        self.assertEqual(control.step(data, .1).interaction, "bomb")
        data["player"]["bombs"] = 0
        data["hazards"].append({"kind": "bomb", "x": plan.point[0], "y": plan.point[1],
                                "vx": 0, "vy": 0, "radius": 10})
        data["paused"] = True
        self.assertEqual(control.step(data, .2).interaction, "none")
        self.assertEqual(control.step(data, 30).move, "none")
        data["paused"] = False
        resumed = control.step(data, 30.1)
        self.assertNotEqual(resumed.move, "none")
        self.assertEqual(resumed.interaction, "none")
        self.assertIsNotNone(control.plan)

    def test_late_bomb_placement_cannot_drop_retreat_at_the_approach_timeout(self):
        data, control = blocked_chest(), AdventureControl()
        plan = select(control, data, "bomb_rock")
        data["player"].update(x=plan.point[0], y=plan.point[1])
        # A long but progressing route can arrive near the approach deadline.
        # Either refuse the bomb here or commit to completing its retreat.
        pulse = control.step(data, 14.9)
        if pulse.interaction == "none":
            self.assertIsNone(control.plan)
            return
        self.assertEqual(pulse.interaction, "bomb")
        data["player"]["bombs"] = 0
        data["hazards"].append({"kind": "bomb", "x": plan.point[0], "y": plan.point[1],
                                "vx": 0, "vy": 0, "radius": 10})
        retreat = control.step(data, 15.01)
        self.assertNotEqual(retreat.move, "none")
        self.assertIsNotNone(control.plan)

    def test_descent_requires_opt_in_and_marks_transition_only_near_exit(self):
        data = ready(kind=5, x=100)
        data["hazards"] = [grid(400, 280, kind=17, collision=0)]
        control = AdventureControl()
        self.assertFalse(control.choose_offers(data))
        select(control, data, "descend", allow_descend=True)
        self.assertIsNone(control.step(data, .1, allow_descend=True).transition)
        data["player"]["x"] = 350
        action = control.step(data, .2, allow_descend=True)
        self.assertEqual(action.transition, "floor")
        self.assertTrue(control.descent_requested)

    def test_timed_out_plan_releases_control(self):
        data, control = charged(), AdventureControl()
        select(control, data, "use_active")
        action = control.step(data, 15)
        self.assertEqual((action.move, action.interaction), ("none", "none"))
        self.assertIsNone(control.plan)


if __name__ == "__main__":
    unittest.main()
