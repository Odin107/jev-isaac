"""Real prop choices, executor, reward collection and floor orchestration."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_props import fire, poop, props
from test_pickups import pickup
from jev_isaac.adventure import candidate_valid, candidates
from jev_isaac.adventure_control import AdventureControl
from jev_isaac.exploration import FloorNavigator


def tick(nav, state, now):
    state["frame"] = max(state["frame"]+1, int(now*30)+1)
    return nav.step(state, now)


def begin(nav, state):
    nav.step(state, 0)
    tick(nav, state, .65)
    choice = next(item for item in nav.adventure_options if item.kind == "shoot_prop")
    assert nav.accept_adventure(choice.key, state, .66)
    return choice


def executor(state):
    control = AdventureControl()
    choice = next(item for item in control.choose_offers(state) if item.kind == "shoot_prop")
    assert control.accept(choice.key, state, 0)
    return control, choice


class PropIntegrationTests(unittest.TestCase):
    def test_normal_poop_choice_shoots_then_collects_only_a_new_observed_drop(self):
        data = props(poop(), x=200)
        nav = FloorNavigator(adventure_mode=True)
        choice = begin(nav, data)
        self.assertTrue(candidate_valid(data, choice))
        self.assertIn(choice.key, {c.key for c in candidates(data)})
        shooting = tick(nav, data, .7)
        self.assertEqual((shooting.move, shooting.shoot), ("none", "right"))
        self.assertEqual(nav.pickup_stats["attempts"], 0)
        data["hazards"][0].update(state=1000, collision=0, tear_destructible=False)
        data["pickups"] = [pickup("actual-drop", x=400)]
        done = tick(nav, data, .75)
        self.assertEqual(done.shoot, "none")
        self.assertFalse(candidate_valid(data, choice))
        self.assertEqual(tick(nav, data, .8).move, "none")
        collect = tick(nav, data, 1.45)
        self.assertEqual((collect.move, collect.shoot), ("right", "none"))
        self.assertIn("collecting pickup", collect.status)
        data["player"].update(x=400, coins=1)
        data["pickups"] = []
        tick(nav, data, 1.5)
        self.assertEqual(nav.pickup_stats["targets_disappeared_or_changed"], 1)

    def test_extinguished_fire_without_a_drop_resumes_exploration_without_inventing_one(self):
        data = props(fire(), x=200)
        nav = FloorNavigator(adventure_mode=True)
        begin(nav, data)
        self.assertEqual(tick(nav, data, .7).shoot, "right")
        data["hazards"] = []
        self.assertEqual(tick(nav, data, .75).shoot, "none")
        tick(nav, data, .8)
        action = tick(nav, data, 1.45)
        self.assertEqual(action.move, "right")
        self.assertIn("entering room", action.status)
        self.assertEqual(nav.pickup_stats["attempts"], 0)
        self.assertEqual(nav.pickup_stats["targets_disappeared_or_changed"], 0)

    def test_four_seconds_without_prop_damage_or_movement_ends_the_attempt(self):
        data = props(fire(), x=200)
        control, choice = executor(data)
        self.assertEqual(control.step(data, .1).shoot, "right")
        self.assertEqual(control.step(data, 4.09).shoot, "right")
        stopped = control.step(data, 4.11)
        self.assertEqual((stopped.move, stopped.shoot), ("none", "none"))
        self.assertIn("no observed progress", stopped.status)
        self.assertIsNone(control.plan)
        self.assertNotIn(choice.key, {item.key for item in control.choose_offers(data)})

    def test_observed_prop_damage_renews_progress_but_never_the_total_attempt_budget(self):
        data = props(fire(), x=200)
        control, _ = executor(data)
        self.assertEqual(control.step(data, .1).shoot, "right")
        for now, hp in ((3.9, 14), (7.8, 13), (11.7, 12), (14.9, 11)):
            data["hazards"][0]["hp"] = hp
            self.assertEqual(control.step(data, now).shoot, "right")
        stopped = control.step(data, 15.01)
        self.assertEqual(stopped.shoot, "none")
        self.assertIsNone(control.plan)
        self.assertEqual(stopped.status, "interaction timed out")

    def test_same_position_new_fire_identity_is_not_the_selected_target(self):
        data = props(fire(), x=200)
        control, _ = executor(data)
        data["hazards"][0]["id"] = "replacement-fire"
        self.assertEqual(control.step(data, .1).shoot, "none")
        self.assertIsNone(control.plan)

    def test_context_change_never_keeps_old_prop_shooting(self):
        for change in (lambda d: d.update(room_id="next-room"),
                       lambda d: d.update(run_id="next-run"),
                       lambda d: d["floor"].update(id="next-floor")):
            data = props(poop(), x=200)
            control, _ = executor(data)
            change(data)
            action = control.step(data, .1)
            self.assertTrue(action is None or action.shoot == "none")
            self.assertIsNone(control.plan)

    def test_lingering_projectile_preempts_the_selected_fire_shots_then_allows_resume(self):
        data = props(fire(variant=1), x=200)
        nav = FloorNavigator(adventure_mode=True)
        begin(nav, data)
        self.assertEqual(tick(nav, data, .7).shoot, "right")
        data["projectiles"] = [{"x": 260, "y": 280, "vx": -5, "vy": 0, "radius": 5}]
        dodge = tick(nav, data, .75)
        self.assertEqual(dodge.shoot, "none")
        self.assertNotEqual(dodge.move, "none")
        self.assertIn("projectiles", dodge.status)
        data["projectiles"] = []
        self.assertEqual(tick(nav, data, .8).shoot, "right")

    def test_projectile_preemption_never_picks_up_an_unselected_paid_item(self):
        data = props(fire(variant=1), x=200)
        nav = FloorNavigator(adventure_mode=True)
        begin(nav, data)
        data["pickups"] = [pickup("paid", variant=100, x=200, y=240,
                                  price=15, shop_item=True, collectible_kind=1)]
        data["projectiles"] = [{"x": 260, "y": 280, "vx": -5, "vy": 0, "radius": 5}]
        dodge = tick(nav, data, .7)
        self.assertEqual(dodge.shoot, "none")
        self.assertNotEqual(dodge.move, "up")
        self.assertEqual(nav.pickup_stats["attempts"], 0)

    def test_stale_fire_choice_cannot_be_accepted_after_extinguishing(self):
        data = props(fire(), x=200)
        control = AdventureControl()
        choice = next(item for item in control.choose_offers(data) if item.kind == "shoot_prop")
        data["hazards"][0].update(hp=0, tear_destructible=False)
        self.assertFalse(control.accept(choice.key, data, .1))
        self.assertIsNone(control.step(data, .2))


if __name__ == "__main__":
    unittest.main()
