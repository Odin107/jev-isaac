"""Offline curse-door budgets, fresh binding and complete navigation visits."""
import copy
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_adventure import ready
from test_exploration import door, grid
from test_pickups import pickup
from jev_isaac.adventure import candidate_valid
from jev_isaac.adventure_control import AdventureControl
from jev_isaac.curse_doors import curse_candidates, curse_valid
from jev_isaac.exploration import FloorNavigator, _validated


def curse_door(slot=2, target=85, *, leaving=False, **extra):
    result = door(slot, target, kind=1 if leaving else 10)
    result.update(current_type=10 if leaving else 1, curse_room_door=True)
    result.update(extra)
    return result


def observed(*items, leaving=False, health=8, index=None, **extra):
    result = ready(*items, kind=10 if leaving else 1, **extra)
    index = (85 if leaving else 84) if index is None else index
    result["floor"]["room_index"] = index
    result["room_id"] = f"curse-visit:{index}"
    result["player"].update(player_type=0, hearts=health, soul_hearts=0)
    result["doors"] = [curse_door(0, 84, leaving=True)] if leaving else [curse_door()]
    return result


def advance(nav, data, now):
    data["frame"] = max(data["frame"] + 1, int(now * 30) + 2)
    return nav.step(data, now)


def selected_control(data):
    control = AdventureControl()
    candidate = next(choice for choice in control.choose_offers(data) if choice.kind == "enter_curse")
    assert control.accept(candidate.key, data, 0)
    return control, candidate


class CurseBudgetTests(unittest.TestCase):
    def test_entry_reserves_two_crossings_and_one_heart_margin(self):
        for health in range(7):
            data = observed(health=health)
            choices = curse_candidates(data)
            self.assertEqual(bool(choices), health >= 6, health)
        choice = curse_candidates(observed(health=6))[0]
        self.assertEqual(choice.cost, {"health_half_hearts_budget": 4})
        self.assertEqual(choice.details["crossing_budget"], 2)
        self.assertEqual(choice.details["return_budget"], 2)
        self.assertEqual(choice.details["survival_margin"], 2)
        self.assertFalse(choice.details["reward_known"])
        self.assertIn("unknown", choice.description.lower())

    def test_exit_reserves_one_crossing_and_one_heart_margin(self):
        for health in range(6):
            choices = curse_candidates(observed(leaving=True, health=health), leaving=True)
            self.assertEqual(bool(choices), health >= 4, health)
        choice = curse_candidates(observed(leaving=True, health=4), leaving=True)[0]
        self.assertEqual(choice.cost, {"health_half_hearts_budget": 2})
        self.assertEqual(choice.details["return_budget"], 0)

    def test_red_and_soul_health_count_but_flight_and_items_give_no_discount(self):
        data = observed(health=3)
        data["player"]["soul_hearts"] = 3
        self.assertTrue(curse_candidates(data))
        data["player"].update(soul_hearts=2, can_fly=True,
                              inventory=[{"id": 313, "count": 1}, {"id": 375, "count": 1}])
        self.assertFalse(curse_candidates(data))

    def test_unknown_character_or_incomplete_health_never_offered(self):
        for field, value in (("player_type", None), ("player_type", 1), ("player_type", True),
                             ("player_type", 0.0), ("player_type", -1), ("hearts", None),
                             ("hearts", True), ("hearts", 6.0), ("hearts", -1),
                             ("soul_hearts", None), ("soul_hearts", 49)):
            with self.subTest(field=field, value=value):
                data = observed()
                if value is None:
                    data["player"].pop(field)
                else:
                    data["player"][field] = value
                self.assertFalse(curse_candidates(data))

    def test_unknown_locked_closed_or_unsupported_door_is_not_a_candidate(self):
        for change in ({"curse_room_door": None}, {"curse_room_door": False},
                       {"current_type": None}, {"current_type": 2}, {"locked": True},
                       {"open": False}, {"target_type": 5}, {"target_index": -1},
                       {"target_index": 169}, {"target_index": 84}):
            with self.subTest(change=change):
                data = observed()
                data["doors"][0].update(change)
                self.assertFalse(curse_candidates(data))
        data = observed()
        data["doors"].append(copy.deepcopy(data["doors"][0]))
        self.assertFalse(curse_candidates(data))

    def test_only_clear_active_complete_state_may_offer_a_damaging_crossing(self):
        changes = (lambda d: d.update(enabled=False), lambda d: d.update(paused=True),
                   lambda d: d.update(truncated=True), lambda d: d["player"].update(dead=True),
                   lambda d: d["room"].update(clear=False),
                   lambda d: d["capabilities"].pop("interaction_control"))
        for change in changes:
            data = observed()
            change(data)
            self.assertFalse(curse_candidates(data))

    def test_candidate_requires_a_route_and_never_ignores_a_blocking_wall(self):
        data = observed()
        data["hazards"] = [grid(520, y, kind=15, collision=4) for y in range(160, 401, 40)]
        self.assertFalse(curse_candidates(data))

    def test_return_binds_origin_and_supports_ordinary_shop_and_treasure_only(self):
        data = observed(leaving=True)
        data["doors"].append(curse_door(2, 86, leaving=True))
        self.assertEqual([c.details["target_index"] for c in
                          curse_candidates(data, leaving=True, return_index=84)], [84])
        self.assertFalse(curse_candidates(data, leaving=True, return_index=87))
        for kind in (1, 2, 4, 5, 10, 11):
            data["doors"] = [curse_door(0, 84, leaving=True, target_type=kind)]
            self.assertEqual(bool(curse_candidates(data, leaving=True)), kind in (1, 2, 4))


class CurseExecutionTests(unittest.TestCase):
    def test_cost_context_and_door_binding_are_rechecked_before_motion(self):
        source = observed()
        candidate = curse_candidates(source)[0]
        self.assertTrue(candidate_valid(source, candidate))
        self.assertFalse(candidate_valid(source, replace(candidate, cost={"health_half_hearts_budget": 2})))
        changes = (lambda d: d["player"].update(hearts=5),
                   lambda d: d.update(room_id="new-visit"), lambda d: d.update(run_id="new-run"),
                   lambda d: d["floor"].update(id="new-floor"),
                   lambda d: d["doors"][0].update(target_index=86),
                   lambda d: d["doors"][0].update(target_type=1),
                   lambda d: d["doors"][0].update(x=596),
                   lambda d: d["doors"][0].update(locked=True),
                   lambda d: d["doors"][0].update(curse_room_door=False))
        for change in changes:
            data = copy.deepcopy(source)
            change(data)
            self.assertFalse(candidate_valid(data, candidate))
        data = observed()
        control, _ = selected_control(data)
        data["player"]["hearts"] = 5
        action = control.step(data, .1)
        self.assertEqual((action.move, action.interaction), ("none", "none"))
        self.assertIsNone(control.plan)

    def test_approach_is_cancelled_if_health_drops_before_commit(self):
        data = observed(health=6, x=554)
        control, _ = selected_control(data)
        self.assertEqual(control.step(data, .1).move, "right")
        self.assertEqual(control.phase, "approach")
        data["player"].update(x=556, hearts=4)
        self.assertEqual(control.step(data, .2).move, "none")
        self.assertIsNone(control.plan)

    def test_committed_crossing_continues_forward_after_first_damage(self):
        data = observed(health=6, x=555)
        control, candidate = selected_control(data)
        self.assertEqual(control.step(data, .1).move, "right")
        self.assertEqual(control.phase, "crossing")
        for x, health in ((560, 4), (578, 2), (599, 1), (611, 1)):
            data["player"].update(x=x, hearts=health)
            self.assertFalse(curse_valid(data, candidate))
            self.assertTrue(curse_valid(data, candidate, committed=True))
            action = control.step(data, .2 + x / 1000)
            self.assertEqual((action.move, action.interaction), ("right", "none"))
            self.assertIsNotNone(control.plan)

    def test_commit_does_not_override_identity_lock_route_or_zero_health(self):
        changes = (lambda d: d["player"].update(hearts=0),
                   lambda d: d["player"].update(player_type=1),
                   lambda d: d["doors"][0].update(locked=True),
                   lambda d: d["doors"][0].update(target_index=86),
                   lambda d: d["doors"][0].update(x=601),
                   lambda d: d["doors"][0].update(curse_room_door=False),
                   lambda d: d["hazards"].append(grid(580, 280, kind=15, collision=4)))
        for change in changes:
            data = observed(x=560)
            control, _ = selected_control(data)
            control.step(data, .1)
            change(data)
            self.assertEqual(control.step(data, .2).move, "none")
            self.assertIsNone(control.plan)

    def test_pause_and_dead_states_send_no_input_and_timeout_remains_bounded(self):
        data = observed(x=560)
        control, _ = selected_control(data)
        self.assertEqual(control.step(data, .1).move, "right")
        data["paused"] = True
        action = control.step(data, .2)
        self.assertEqual((action.move, action.shoot, action.interaction), ("none", "none", "none"))
        data["paused"] = False
        self.assertEqual(control.step(data, 20.2).move, "right")
        data["player"]["dead"] = True
        self.assertEqual(control.step(data, 20.3).move, "none")
        data["player"]["dead"] = False
        self.assertEqual(control.step(data, 35).move, "none")
        self.assertIsNone(control.plan)


class CurseFloorIntegrationTests(unittest.TestCase):
    def test_generic_graph_never_enters_or_leaves_curse_doors(self):
        for adventure in (False, True):
            data = observed()
            self.assertEqual(_validated(data, allow_shop=adventure)[-1], ())
            nav = FloorNavigator(adventure_mode=adventure)
            nav.observe(data)
            self.assertIsNone(nav._next_door())
            data["doors"].append(door(0, 83))
            nav.observe(data)
            self.assertEqual(nav._next_door().target_index, 83)
        data = observed(leaving=True)
        self.assertEqual(_validated(data, allow_shop=True)[-1], ())
        nav = FloorNavigator(adventure_mode=True)
        nav.observe(data)
        self.assertIsNone(nav._next_door())

    def test_entry_waits_for_explicit_selection_and_skip_never_crosses(self):
        data, nav = observed(), FloorNavigator(adventure_mode=True)
        advance(nav, data, 0)
        action = advance(nav, data, .61)
        self.assertEqual(action.move, "none")
        self.assertEqual([c.kind for c in nav.adventure_options], ["enter_curse"])
        self.assertIsNone(nav._pending)
        self.assertFalse(nav.accept_adventure(None, data, .62))
        self.assertEqual(advance(nav, data, .7).move, "none")
        self.assertIsNone(nav._pending)

    def test_complete_entry_reward_and_origin_return_never_reoffers_visited_room(self):
        data, nav = observed(), FloorNavigator(adventure_mode=True)
        advance(nav, data, 0)
        advance(nav, data, .61)
        self.assertTrue(nav.accept_adventure(nav.adventure_options[0].key, data, .62))
        self.assertEqual(advance(nav, data, .7).move, "right")
        data["player"].update(x=560)
        self.assertEqual(advance(nav, data, .8).move, "right")
        data["player"].update(x=608, hearts=6)
        self.assertEqual(advance(nav, data, .9).move, "right")
        inside = observed(pickup("reward", variant=100, collectible_kind=1),
                          leaving=True, health=6, frame=40)
        inside["doors"].append(curse_door(2, 86, leaving=True))
        advance(nav, inside, 1.4)
        advance(nav, inside, 2.01)
        self.assertEqual([c.kind for c in nav.adventure_options], ["collect"])
        self.assertTrue(nav.accept_adventure(nav.adventure_options[0].key, inside, 2.02))
        self.assertEqual(advance(nav, inside, 2.1).move, "left")
        inside["pickups"] = []
        advance(nav, inside, 2.2)
        advance(nav, inside, 2.3)
        action = advance(nav, inside, 2.91)
        self.assertEqual(action.status, "returning from the selected curse-room visit")
        self.assertEqual(nav._adventure.plan.kind, "leave_curse")
        self.assertEqual(nav._adventure.plan.details["target_index"], 84)
        self.assertEqual(advance(nav, inside, 3).move, "left")
        inside["player"].update(x=80)
        self.assertEqual(advance(nav, inside, 3.1).move, "left")
        inside["player"].update(x=30, hearts=4)
        self.assertEqual(advance(nav, inside, 3.2).move, "left")
        back = observed(health=8, frame=100)
        back["room_id"] = "returned-to-origin"
        back["doors"].append(door(0, 83))
        advance(nav, back, 3.4)
        action = advance(nav, back, 4.01)
        self.assertEqual(action.move, "left")
        self.assertFalse(any(c.kind == "enter_curse" for c in nav.adventure_options))
        self.assertEqual(nav.stats["doors_traversed"], 2)
        self.assertEqual(nav.stats["rooms_visited"], 2)
        self.assertEqual(nav._pending.target_index, 83)

    def test_armed_inside_can_return_without_entry_authorization_or_model_call(self):
        data, nav = observed(leaving=True), FloorNavigator(adventure_mode=True)
        advance(nav, data, 0)
        action = advance(nav, data, .61)
        self.assertEqual(action.status, "returning from the selected curse-room visit")
        self.assertEqual(nav._adventure.plan.kind, "leave_curse")
        self.assertEqual(advance(nav, data, .7).move, "left")
        event = nav.adventure_stats["events"][-1]
        self.assertIn("Local return", event["reason"])
        self.assertNotIn("Jev selected", event["reason"])

    def test_armed_inside_return_does_not_offer_reentry_to_that_visited_curse_room(self):
        data, nav = observed(leaving=True), FloorNavigator(adventure_mode=True)
        advance(nav, data, 0)
        advance(nav, data, .61)
        back = observed(health=8, frame=30)
        back["doors"].append(door(0, 83))
        advance(nav, back, 1)
        action = advance(nav, back, 1.61)
        self.assertFalse(any(c.kind == "enter_curse" for c in nav.adventure_options))
        self.assertEqual(action.move, "left")

    def test_low_health_inside_stops_before_spending_unbudgeted_health(self):
        data, nav = observed(leaving=True, health=3), FloorNavigator(adventure_mode=True)
        advance(nav, data, 0)
        action = advance(nav, data, .61)
        self.assertEqual(action.move, "none")
        self.assertIn("health budget", action.stop_reason)
        self.assertIsNone(nav._adventure.plan)


if __name__ == "__main__":
    unittest.main()
