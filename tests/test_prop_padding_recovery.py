"""Synthetic inertia/push cases, not a replay of the unsaved live poop stop."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_exploration import grid
from test_pickups import pickup
from test_props import fire, poop, props
from jev_isaac.adventure_control import AdventureControl
from jev_isaac.navigation import _VECTORS, _clear, _free
from jev_isaac.props import prop_action, prop_candidates, prop_valid


def selected(obstacle):
    state = props(poop(index=20, x=440), obstacle, x=270)
    control = AdventureControl()
    choice = next(item for item in control.choose_offers(state)
                  if item.target_id == "poop:20:14:0")
    assert control.accept(choice.key, state, 0)
    return state, control, choice


class PropPaddingRecoveryTests(unittest.TestCase):
    def test_shallow_rock_or_normal_poop_overlap_keeps_the_selected_plan(self):
        for obstacle in (grid(320, 280), poop(index=10, x=320)):
            with self.subTest(obstacle=obstacle):
                state, control, choice = selected(obstacle)
                state["player"].update(x=291, vx=1)
                before = copy.deepcopy(state)
                self.assertTrue(prop_valid(state, choice))
                action = control.step(state, .1)
                self.assertIs(control.plan, choice)
                self.assertEqual(action.shoot, "none")
                self.assertNotEqual(action.move, "none")
                start = (291, 280)
                vector = _VECTORS[action.move]
                self.assertLess(vector[0], 0)
                end = (start[0]+vector[0]*24, start[1]+vector[1]*24)
                self.assertTrue(_free(end, [(290, 250, 350, 310)]))
                self.assertTrue(_clear(start, end, [(300, 260, 340, 300)]))
                self.assertEqual(before, state)

    def test_recovery_returns_to_a_firing_lane_without_crossing_the_obstacle(self):
        state, control, choice = selected(poop(index=10, x=320))
        state["player"].update(x=291)
        for frame in range(1, 150):
            state["frame"] += 1
            action = control.step(state, frame/30)
            self.assertIs(control.plan, choice)
            if action.shoot != "none":
                break
            start = state["player"]["x"], state["player"]["y"]
            vector = _VECTORS[action.move]
            end = start[0]+vector[0]*4, start[1]+vector[1]*4
            self.assertTrue(_clear(start, end, [(300, 260, 340, 300), (420, 260, 460, 300)]))
            state["player"].update(x=end[0], y=end[1])
        else:
            self.fail("Recovering prop controller did not regain a firing lane")

    def test_deep_overlap_or_actual_body_interior_does_not_authorize_an_escape(self):
        for x in (296, 305, 320):
            state, _, choice = selected(poop(index=10, x=320))
            state["player"]["x"] = x
            self.assertFalse(prop_valid(state, choice))
            self.assertIsNone(prop_action(state, choice))

    def test_unknown_or_nonordinary_poop_padding_has_no_exception(self):
        for changes in ({"variant": 1}, {"variant": 5}, {"variant": -1},
                        {"tear_destructible": False}, {"state": 1000}):
            state, _, choice = selected(poop(index=10, x=320))
            state["hazards"][1].update(changes)
            state["player"]["x"] = 291
            self.assertIsNone(prop_action(state, choice))

    def test_identical_fire_box_cannot_inherit_a_poop_exception(self):
        state, _, choice = selected(poop(index=10, x=320))
        # Fire radius 12 + player 10 + margin 8 exactly matches poop's box.
        state["hazards"].insert(0, fire(identity="overlapping-fire", x=320))
        state["player"]["x"] = 291
        self.assertFalse(prop_valid(state, choice))
        self.assertIsNone(prop_action(state, choice))

    def test_pickups_and_trap_npcs_remain_blocked_during_padding_recovery(self):
        additions = (
            ("pickups", pickup("paid", x=291, price=5, shop_item=True)),
            ("pickups", pickup("free", x=291)),
            ("enemies", {"id": "trap", "x": 291, "y": 280, "radius": 10,
                         "vx": 0, "vy": 0, "hp": 100, "vulnerable": False}),
            ("hazards", grid(291, 280, kind=8, collision=0)),
            ("hazards", grid(291, 280, kind=17, collision=0)),
        )
        for collection, item in additions:
            with self.subTest(collection=collection, item=item):
                state, _, choice = selected(poop(index=10, x=320))
                state[collection].append(item)
                state["player"]["x"] = 291
                self.assertIsNone(prop_action(state, choice))

    def test_door_padding_is_not_an_allowed_escape(self):
        state = props(poop(x=400), x=500)
        choice = prop_candidates(state)[0]
        state["hazards"].append(grid(591, 280))
        state["player"]["x"] = 563
        self.assertIsNone(prop_action(state, choice))

    def test_recoverable_padding_does_not_make_a_walled_off_target_reachable(self):
        state, _, choice = selected(poop(index=10, x=320))
        state["hazards"].extend(grid(360, y, kind=15, collision=4)
                                 for y in range(140, 421, 40))
        state["player"]["x"] = 291
        self.assertFalse(prop_valid(state, choice))
        self.assertIsNone(prop_action(state, choice))
        self.assertNotIn(choice.key, {item.key for item in prop_candidates(state)})

    def test_destroyed_poop_remnant_does_not_block_a_different_prop(self):
        state, _, choice = selected(poop(index=10, x=320))
        state["hazards"][1].update(state=1000, collision=0, tear_destructible=False)
        state["player"]["x"] = 320
        action = prop_action(state, choice)
        self.assertIsNotNone(action)
        self.assertEqual((action.move, action.shoot), ("none", "right"))


if __name__ == "__main__":
    unittest.main()
