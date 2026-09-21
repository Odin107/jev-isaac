"""Visited game facts guide routing without manufacturing room connections."""
import copy
from types import SimpleNamespace
import unittest

from test_exploration import FloorNavigator, door, state


def visited(index, list_index, *, clear=True, kind=1, aliases=None):
    row = {"room_index": index, "list_index": list_index, "type": kind,
           "clear": clear, "visited_count": 1}
    if aliases is not None:
        row["room_indices"] = aliases
    return row


def observed(index=84, *, list_index=1, **kwargs):
    value = state(index, **kwargs)
    value["floor"].update(room_list_index=list_index, dimension=0)
    return value


class FloorMemoryTests(unittest.TestCase):
    def test_fresh_controller_prefers_unvisited_door_to_known_clear_room(self):
        value = observed(doors=[door(0, 83), door(2, 85)])
        value["visited_rooms"] = [visited(83, 2)]
        nav = FloorNavigator()
        self.assertEqual(nav.step(value, 0).status, "entering room 85")
        self.assertEqual(nav.stats["rooms_visited"], 2)
        self.assertNotIn(("list", 2), nav._graph)
        self.assertNotIn(("list", 2), nav._inspected)

    def test_known_uncleared_room_is_a_frontier(self):
        value = observed(doors=[door(0, 83), door(2, 85)])
        value["visited_rooms"] = [visited(83, 2), visited(85, 3, clear=False)]
        nav = FloorNavigator()
        self.assertEqual(nav.step(value, 0).status, "entering room 85")

    def test_boss_frontier_precedes_clear_uninspected_transit(self):
        value = observed(doors=[door(0, 83), door(2, 85, 5)])
        value["visited_rooms"] = [visited(83, 2)]
        self.assertEqual(FloorNavigator().step(value, 0).status, "entering room 85")

    def test_clear_import_still_requires_inspecting_accessible_connections(self):
        value = observed(kind=5, doors=[door(0, 83)])
        value["visited_rooms"] = [visited(83, 2), visited(84, 1, kind=5)]
        nav = FloorNavigator()
        result = nav.step(value, 0)
        self.assertIsNone(result.stop_reason)
        self.assertEqual(result.status, "entering room 83")
        next_room = observed(83, list_index=2, frame=2, doors=[door(2, 84, 5)])
        self.assertEqual(nav.step(next_room, .1).stop_reason, "floor cleared")

    def test_disconnected_summary_does_not_create_a_route(self):
        value = observed()
        value["visited_rooms"] = [visited(12, 2, clear=False)]
        nav = FloorNavigator()
        nav.observe(value)
        self.assertIsNone(nav._next_door())
        self.assertEqual(set(nav._graph), {("list", 1)})

    def test_imported_cleared_boss_does_not_complete_during_door_opening(self):
        value = observed(doors=[door(2, 85, opened=False)])
        value["visited_rooms"] = [visited(12, 2, kind=5)]
        nav = FloorNavigator(adventure_mode=True, continue_floors=True)
        result = nav.step(value, 0)
        self.assertIsNone(result.stop_reason)
        self.assertFalse(nav._allow_descend)
        value["frame"] += 1
        value["doors"][0]["open"] = True
        self.assertEqual(nav.step(value, .1).status, "entering room 85")

    def test_locked_and_special_closed_doors_do_not_block_completion(self):
        value = observed(doors=[door(0, 83, opened=False, locked=True), door(2, 85, 6, opened=False)])
        value["visited_rooms"] = [visited(12, 2, kind=5)]
        self.assertEqual(FloorNavigator().step(value, 0).stop_reason, "floor cleared")

    def test_large_room_aliases_share_clear_fact_and_do_not_include_gap(self):
        value = observed(doors=[door(0, 83), door(2, 85)])
        value["visited_rooms"] = [visited(70, 2, aliases=[70, 71, 83])]
        nav = FloorNavigator()
        self.assertEqual(nav.step(value, 0).status, "entering room 85")
        self.assertEqual(nav._aliases[83], ("list", 2))
        self.assertNotIn(84-2, nav._aliases)
        self.assertEqual(nav.stats["rooms_visited"], 2)

    def test_current_room_clear_and_type_override_its_summary(self):
        for clear in (False, True):
            value = observed(clear=clear)
            value["visited_rooms"] = [visited(84, 1, clear=not clear, kind=5, aliases=[84, 85])]
            nav = FloorNavigator()
            nav.observe(value)
            self.assertIsNone(nav.stop_reason)
            self.assertEqual(nav._rooms[("list", 1)], (1, clear))
            self.assertEqual(nav._aliases[85], ("list", 1))
            self.assertEqual(nav.stats["boss_cleared"], False)

    def test_malformed_optional_summary_fails_before_changing_map(self):
        rows = [None, {}, [visited(83, 2), visited(83, 3)],
                [visited(83, 2, aliases=[83, True])], [visited(83, 2, clear=1)]]
        for summary in rows:
            with self.subTest(summary=summary):
                value = observed()
                value["visited_rooms"] = summary
                nav = FloorNavigator()
                nav.observe(value)
                self.assertEqual(nav.stop_reason, "incomplete floor observation")
                self.assertEqual(nav.stats["rooms_visited"], 0)

    def test_paused_disarmed_dead_or_changed_dimension_do_not_import(self):
        for field, change in (("paused", True), ("enabled", False), ("dead", True), ("dimension", 1)):
            nav = FloorNavigator()
            value = observed()
            nav.observe(value)
            value["frame"] += 1
            value["visited_rooms"] = [visited(83, 2)]
            target = value["player"] if field == "dead" else value["floor"] if field == "dimension" else value
            target[field] = change
            nav.observe(value)
            self.assertNotIn(("list", 2), nav._rooms)

    def test_remote_unfinished_branch_beats_local_uninspected_clear_room(self):
        nav = FloorNavigator()
        first = observed(doors=[door(0, 83), door(2, 85)])
        self.assertEqual(nav.step(first, 0).status, "entering room 83")
        second = observed(83, list_index=2, frame=2, doors=[door(2, 84), door(0, 82)])
        second["visited_rooms"] = [visited(82, 3)]
        self.assertEqual(nav.step(second, .1).status, "entering room 84")

    def test_imported_visited_curse_room_is_not_offered_after_controller_restart(self):
        from test_curse_doors import advance, observed as curse_observed
        for clear in (False, True):
            with self.subTest(clear=clear):
                value = curse_observed()
                value["floor"].update(room_list_index=1, dimension=0)
                value["visited_rooms"] = [visited(85, 2, kind=10, clear=clear)]
                nav = FloorNavigator(adventure_mode=True)
                advance(nav, value, 0)
                advance(nav, value, .61)
                self.assertIn(85, nav._curse_attempted)
                self.assertFalse(any(c.kind == "enter_curse" for c in nav.adventure_options))

    def test_manual_rearm_imports_visited_curse_aliases_without_reoffering_entry(self):
        from test_curse_doors import advance, observed as curse_observed
        prior = curse_observed()
        prior["floor"].update(room_list_index=1, dimension=0)
        old = FloorNavigator(adventure_mode=True)
        old.observe(prior)
        for target in (85, 98):
            with self.subTest(target=target):
                value = curse_observed(index=86, frame=5)
                value["floor"].update(room_list_index=3, dimension=0)
                value["session"] = "new-arm"
                value["doors"][0]["target_index"] = target
                value["visited_rooms"] = [visited(85, 2, kind=10, aliases=[85, 98])]
                new = old.rearmed(value)
                advance(new, value, 0)
                advance(new, value, .61)
                self.assertTrue({85, 98}.issubset(new._curse_attempted))
                self.assertFalse(any(c.kind == "enter_curse" for c in new.adventure_options))


class RearmedMemoryTests(unittest.TestCase):
    def make_nav(self):
        nav = FloorNavigator(adventure_mode=True, continue_floors=True)
        value = observed(doors=[door(2, 85)])
        nav.observe(value)
        nav._pending = nav._graph[("list", 1)][0]
        nav._pending_started = nav._progress_at = 2
        nav._progress_point = (320, 280)
        nav._stop = "stuck while approaching door"
        nav._allow_descend = True
        nav._doors_traversed = 4
        nav._pickups.attempts, nav._pickups.resolved, nav._pickups.skipped = 3, 2, 1
        nav._adventure.selected, nav._adventure.completed, nav._adventure.abandoned = 4, 2, 1
        nav._adventure.events = [{"reason": "history"}]
        nav._adventure.plan = SimpleNamespace(key="interrupted")
        nav._adventure.skipped.add("already-tried")
        nav._adventure.pulse_id = "old-pulse"
        nav._adventure.descent_requested = True
        value.update(session="run:arm2", frame=2)
        return nav, value

    def test_same_floor_rearm_preserves_map_and_counters_but_resets_controls(self):
        old, value = self.make_nav()
        new = old.rearmed(value)
        self.assertIsNone(new.stop_reason)
        self.assertEqual(new.stats, old.stats)
        self.assertEqual(new._graph, old._graph)
        self.assertEqual(new._inspected, old._inspected)
        self.assertEqual(new.pickup_stats, old.pickup_stats)
        self.assertEqual(new.adventure_stats, old.adventure_stats)
        self.assertEqual(new._adventure.skipped, {"already-tried", "interrupted"})
        self.assertIsNone(new._pending)
        self.assertIsNone(new._adventure.plan)
        self.assertIsNone(new._adventure.pulse_id)
        self.assertFalse(new._allow_descend)
        self.assertFalse(new.descent_requested)
        new._adventure.events[0]["reason"] = "changed"
        self.assertEqual(old._adventure.events[0]["reason"], "history")
        new.observe(value)
        self.assertIsNone(new.stop_reason)

    def test_manual_rearm_in_another_room_rebinds_without_guessing_an_edge(self):
        old, value = self.make_nav()
        value = observed(83, list_index=2, frame=3)
        value["session"] = "run:arm2"
        new = old.rearmed(value)
        new.observe(value)
        self.assertIsNone(new.stop_reason)
        self.assertEqual(new.stats["doors_traversed"], 4)
        self.assertEqual(new.stats["rooms_visited"], 2)
        self.assertEqual(new._adventure.skipped, set())

    def test_rearm_can_begin_on_same_frame_when_new_session_is_authenticated(self):
        old, value = self.make_nav()
        value["frame"] = 1
        new = old.rearmed(value)
        new.observe(value)
        self.assertIsNone(new.stop_reason)

    def test_ineligible_rearm_never_reuses_memory(self):
        for mutation in (lambda s: s.update(run_id="other"), lambda s: s["floor"].update(id="other"),
                         lambda s: s["floor"].update(dimension=1), lambda s: s.update(frame=0),
                         lambda s: s.update(enabled=False), lambda s: s.update(paused=True),
                         lambda s: s["player"].update(dead=True), lambda s: s.pop("doors")):
            old, value = self.make_nav()
            mutation(value)
            new = old.rearmed(value)
            self.assertIsNotNone(new.stop_reason)
            self.assertEqual(new.stats["rooms_visited"], 0)

    def test_curse_return_only_survives_the_same_actual_curse_visit(self):
        for same_visit in (False, True):
            nav = FloorNavigator(adventure_mode=True)
            value = observed(kind=10)
            value["visited_rooms"] = [visited(84, 1, kind=10, aliases=[84, 85])]
            nav.observe(value)
            nav._curse_return_index = 83
            value.update(frame=2, session="run:arm2")
            if not same_visit:
                value["room_id"] = "another-curse-visit"
            new = nav.rearmed(value)
            self.assertEqual(new._curse_return_index, 83 if same_visit else None)
            self.assertTrue({84, 85}.issubset(new._curse_attempted))


if __name__ == "__main__":
    unittest.main()
