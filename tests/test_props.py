"""Grounded prop targeting and movement/tear geometry, without game/API use."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_adventure import ready
from test_exploration import grid
from test_pickups import pickup
from jev_isaac.navigation import _VECTORS, _clear
from jev_isaac.props import prop_action, prop_candidates, prop_valid


def poop(index=10, x=400, y=280, **extra):
    result = dict(grid(x, y, kind=14), index=index, variant=0, state=0,
                  tear_destructible=True)
    result.update(extra)
    return result


def fire(identity="fire1", x=400, y=280, **extra):
    result = {"kind": "fire", "id": identity, "type": 33, "variant": 0,
              "subtype": 0, "x": x, "y": y, "vx": 0, "vy": 0,
              "radius": 12, "hp": 15, "max_hp": 15, "tear_destructible": True}
    result.update(extra)
    return result


def props(*hazards, **extra):
    data = ready(**extra)
    data["player"].update(weapon_type=1, weapon_types=[1])
    data["hazards"] = list(hazards)
    return data


class PropTests(unittest.TestCase):
    def test_normal_poop_and_regular_or_red_fire_have_bound_free_choices(self):
        for target in (poop(), fire(), fire(variant=1)):
            with self.subTest(target=target):
                data = props(target, x=200)
                choices = prop_candidates(data)
                self.assertEqual(len(choices), 1)
                choice = choices[0]
                self.assertEqual(choice.kind, "shoot_prop")
                self.assertEqual(choice.cost, {})
                self.assertEqual(choice.interaction, "none")
                self.assertIn("actual drops", choice.description)
                self.assertTrue(prop_valid(data, choice))
                action = prop_action(data, choice)
                self.assertEqual((action.move, action.shoot), ("none", "right"))

    def test_missing_flag_and_unknown_or_regrowing_variants_are_never_targets(self):
        for target in (poop(tear_destructible=False), poop(tear_destructible=1),
                       poop(variant=1), poop(variant=5), poop(variant=-1),
                       fire(variant=2), fire(variant=3), fire(variant=4),
                       fire(variant=10), fire(type=32)):
            self.assertEqual(prop_candidates(props(target, x=200)), ())
        data = props(poop(), x=200)
        del data["hazards"][0]["tear_destructible"]
        self.assertEqual(prop_candidates(data), ())

    def test_destroyed_poop_and_extinguished_fire_cancel_shooting(self):
        for target in (poop(), fire()):
            data = props(target, x=200)
            choice = prop_candidates(data)[0]
            if target["kind"] == "grid":
                data["hazards"][0].update(state=1000, collision=0, tear_destructible=False)
            else:
                data["hazards"][0].update(hp=0, tear_destructible=False)
            self.assertFalse(prop_valid(data, choice))
            self.assertIsNone(prop_action(data, choice))
            self.assertEqual(prop_candidates(data), ())

    def test_observed_damage_does_not_change_target_identity(self):
        for target in (poop(), fire()):
            data = props(target, x=200)
            choice = prop_candidates(data)[0]
            if target["kind"] == "grid":
                data["hazards"][0]["state"] = 500
            else:
                data["hazards"][0]["hp"] = 3
            self.assertTrue(prop_valid(data, choice))
            self.assertEqual(prop_candidates(data)[0].as_dict(), choice.as_dict())

    def test_disappearance_and_variant_change_never_keep_old_shots(self):
        for change in (lambda d: d["hazards"].clear(),
                       lambda d: d["hazards"][0].update(variant=1),
                       lambda d: d["hazards"][0].update(index=11),
                       lambda d: d["hazards"][0].update(x=440)):
            data = props(poop(), x=200)
            choice = prop_candidates(data)[0]
            change(data)
            self.assertFalse(prop_valid(data, choice))
            self.assertIsNone(prop_action(data, choice))

    def test_run_room_floor_pause_and_combat_invalidate_the_old_prop_choice(self):
        for change in (lambda d: d.update(run_id="other"), lambda d: d.update(room_id="other"),
                       lambda d: d["floor"].update(id="other"), lambda d: d.update(paused=True),
                       lambda d: d.update(enabled=False), lambda d: d["room"].update(clear=False),
                       lambda d: d.update(truncated=True)):
            data = props(poop(), x=200)
            choice = prop_candidates(data)[0]
            change(data)
            self.assertIsNone(prop_action(data, choice))

    def test_incomplete_or_explosive_weapons_do_not_use_ordinary_tear_lane(self):
        for change in (lambda d: d["player"].pop("weapon_type"),
                       lambda d: d["player"].update(weapon_type=2),
                       lambda d: d["player"].update(weapon_types=[1, 4]),
                       lambda d: d["player"].update(inventory_truncated=True),
                       lambda d: d["player"].update(inventory=[{"id": 149, "count": 1}]),
                       lambda d: d["player"].update(inventory=[{"id": 257, "count": 1}]),
                       lambda d: d["player"].update(inventory=[{"id": 418, "count": 1}])):
            data = props(poop(), x=200)
            change(data)
            self.assertEqual(prop_candidates(data), ())

    def test_never_walks_into_a_prop_to_shoot_at_point_blank_range(self):
        data = props(poop(), x=355)
        choice = prop_candidates(data)[0]
        action = prop_action(data, choice)
        self.assertEqual(action.shoot, "none")
        vector = _VECTORS[action.move]
        endpoint = (355+vector[0]*12, 280+vector[1]*12)
        self.assertTrue(_clear((355, 280), endpoint, [(370, 250, 430, 310)]))
        self.assertNotEqual(action.move, "right")

    def test_routes_around_a_rock_to_a_reachable_firing_lane(self):
        data = props(poop(x=440), grid(320, 280), x=200)
        choice = prop_candidates(data)[0]
        solids = [(290, 250, 350, 310), (410, 250, 470, 310)]
        for frame in range(100):
            data["frame"] += 1
            action = prop_action(data, choice)
            self.assertIsNotNone(action)
            if action.shoot != "none":
                break
            start = data["player"]["x"], data["player"]["y"]
            vector = _VECTORS[action.move]
            end = start[0]+vector[0]*4, start[1]+vector[1]*4
            self.assertTrue(_clear(start, end, solids))
            data["player"].update(x=end[0], y=end[1])
        else:
            self.fail("Prop navigator never reached a firing lane")

    def test_paid_items_and_free_supplies_both_block_a_prop_route(self):
        for item in (pickup("coin", x=300), pickup("shop", variant=100, x=300,
                                                 collectible_kind=1, price=15, shop_item=True),
                     pickup("option", variant=100, x=300, collectible_kind=1, options_index=7)):
            data = props(poop(), x=200)
            data["pickups"] = [item]
            choice = prop_candidates(data)[0]
            action = prop_action(data, choice)
            self.assertEqual(action.shoot, "none")
            self.assertNotEqual(action.move, "right")

    def test_tnt_or_unknown_poop_behind_target_forces_a_different_firing_lane(self):
        for danger in (grid(500, 280, kind=12), poop(index=11, x=500, variant=5),
                       dict(grid(500, 280), kind="tnt")):
            data = props(poop(), danger, x=200)
            choice = prop_candidates(data)[0]
            action = prop_action(data, choice)
            self.assertEqual(action.shoot, "none")
            self.assertNotEqual(action.move, "none")

    def test_no_shot_can_continue_into_a_pickup_behind_the_target(self):
        data = props(fire(), x=200)
        data["pickups"] = [pickup("paid", variant=100, collectible_kind=1, price=15,
                                  shop_item=True, x=500)]
        choice = prop_candidates(data)[0]
        self.assertEqual(prop_action(data, choice).shoot, "none")

    def test_wall_enclosed_prop_has_no_candidate_and_does_not_route_through_exit(self):
        data = props(poop(x=440), x=200)
        data["hazards"].extend(grid(360, y, kind=15, collision=4) for y in range(160, 401, 40))
        self.assertEqual(prop_candidates(data), ())
        data = props(poop(), x=200)
        data["hazards"].extend(grid(320, y, kind=17, collision=0) for y in range(160, 401, 40))
        # Tears can cross exits, but movement cannot. This left-side firing
        # lane is still legal and must not require entering the exit cells.
        choice = prop_candidates(data)[0]
        self.assertEqual(prop_action(data, choice).move, "none")

    def test_new_bomb_or_laser_interrupts_prop_shooting(self):
        for kind in ("bomb", "laser"):
            data = props(poop(), x=200)
            choice = prop_candidates(data)[0]
            data["hazards"].append({"kind": kind, "x": 250, "y": 280, "radius": 10})
            self.assertIsNone(prop_action(data, choice))

    def test_fire_projectile_can_trigger_local_dodge_instead_of_standing_still(self):
        data = props(fire(variant=1), x=200)
        choice = prop_candidates(data)[0]
        data["projectiles"] = [{"x": 260, "y": 280, "vx": -5, "vy": 0, "radius": 5}]
        action = prop_action(data, choice)
        self.assertNotEqual(action.move, "none")
        self.assertEqual(action.shoot, "none")

    def test_firing_waits_until_alignment_velocity_settles(self):
        data = props(poop(), x=200)
        choice = prop_candidates(data)[0]
        data["player"].update(vx=1, vy=1)
        self.assertEqual(prop_action(data, choice).shoot, "none")
        data["player"].update(vx=0, vy=0)
        self.assertEqual(prop_action(data, choice).shoot, "right")

    def test_duplicate_prop_identity_is_rejected(self):
        data = props(poop(), poop(), x=200)
        self.assertEqual(prop_candidates(data), ())

    def test_candidate_count_is_bounded_and_planning_never_mutates_observations(self):
        data = props(*(poop(i, x=x, y=y) for i, (x, y) in enumerate(
            ((160, 200), (240, 200), (400, 200), (480, 200), (160, 360), (480, 360)))), x=320)
        before = copy.deepcopy(data)
        choices = prop_candidates(data)
        self.assertEqual(len(choices), 4)
        self.assertEqual(len({c.key for c in choices}), 4)
        for choice in choices:
            prop_action(data, choice)
        self.assertEqual(before, data)


if __name__ == "__main__":
    unittest.main()
