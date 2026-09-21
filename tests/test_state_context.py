"""State enrichment preserves facts, ownership, freshness and legal choices."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).resolve().parents[1]/"src")]
from test_goals import state
import test_strategy as strategy_tests
from test_controller_recovery import run, observed, OLD, NEW
from test_post_tnt_navigation import recorded_state
from jev_isaac.exploration import FloorNavigator
from jev_isaac.goals import build_goal_request
from jev_isaac.state_context import build_game_context


class StateContextTests(unittest.TestCase):
    def test_recorded_snapshot_preserves_raw_facts_and_explains_types(self):
        data = recorded_state()
        before = copy.deepcopy(data)
        request = build_goal_request(data)
        context = request["state"]["game_context"]
        self.assertEqual(context["type_names"]["room_and_door_target_type"]["5"], "boss")
        self.assertEqual(context["type_names"]["room_and_door_target_type"]["8"], "supersecret")
        for key in ("doors", "player", "visited_rooms", "switches", "room", "floor"):
            self.assertEqual(request["state"]["observation"][key], before[key])
        self.assertEqual(data, before)
        self.assertEqual(context["units"]["hearts_soul_hearts_max_hearts"], "half-hearts")

    def test_unknown_types_and_missing_arrays_are_not_filled_in(self):
        data = state()
        data["room"]["type"] = 999
        data["pickups"] = [{"variant": 991}]
        data["player"]["weapon_types"] = [1, 999]
        context = build_game_context(data)
        self.assertEqual(context["type_names"]["room_and_door_target_type"], {"999": "unknown"})
        self.assertEqual(context["type_names"]["pickup_variant"], {"991": "unknown"})
        self.assertEqual(context["type_names"]["weapon_type"], {"1": "tears", "999": "unknown"})
        self.assertEqual(context["coverage"]["switches"], "missing")
        self.assertEqual(context["coverage"]["visited_rooms"], "missing")
        self.assertNotIn("switches", data)
        data["truncated_arrays"] = {"pickups": True}
        data["player"].update(inventory=[], inventory_truncated=True)
        context = build_game_context(data)
        self.assertEqual(context["coverage"]["pickups"], "partial")
        self.assertEqual(context["coverage"]["inventory"], "partial")

    def test_controller_memory_is_separate_deep_copied_context_not_game_fact(self):
        data = state()
        data["_controller_context"] = {"previous_goal": {"kind": "hold"}}
        request = build_goal_request(data)
        data["_controller_context"]["previous_goal"]["kind"] = "evade"
        self.assertEqual(request["state"]["controller_context"]["previous_goal"]["kind"], "hold")
        self.assertNotIn("_controller_context", request["state"]["observation"])
        self.assertNotIn("local-only", json.dumps(request))

    def test_adventure_context_does_not_carry_conflicting_combat_only_authority(self):
        data = state()
        data["_adventure_options"] = [strategy_tests.choice()]
        _, request = strategy_tests.StrategyTests().call(data, strategy_tests.result())
        contract = request["state"]["game_context"]["control_contract"]
        self.assertEqual(contract["decision_kind"], "adventure")
        self.assertIn("interaction", contract["model"])
        self.assertNotIn("Choose engage", contract["model"])
        self.assertIn("game_context", request["questions"]["goal"]["instructions"])
        self.assertEqual(set(request["questions"]["goal"]["criteria"]), {"hold", "evade", "enemy_0"})

    def test_memory_keeps_aliases_and_observed_edges_without_mutating_navigator(self):
        data = recorded_state()
        navigator = FloorNavigator(adventure_mode=True)
        navigator.observe(data)
        before = copy.deepcopy(navigator.__dict__)
        context = navigator.decision_context()
        self.assertEqual(navigator._graph, before["_graph"])
        self.assertEqual(navigator._rooms, before["_rooms"])
        self.assertIsNone(navigator._pending)
        columns = context["map_columns"]
        rows = [dict(zip(columns, row)) for row in context["map_rows"]]
        self.assertEqual(len(rows), navigator.stats["rooms_visited"])
        self.assertTrue(any(set(row["grid_aliases"]) >= {69, 70, 83} for row in rows))
        current = next(row for row in rows if row["room_key"] == context["current_room_key"])
        self.assertIn([3, 126, 5], current[columns[-1]])
        self.assertTrue(all(not row[columns[-1]] for row in rows if row is not current))
        context["map_rows"].clear()
        self.assertEqual(navigator._rooms, before["_rooms"])

    def test_memory_survives_authenticated_rearm_but_new_floor_starts_empty(self):
        data = recorded_state()
        navigator = FloorNavigator(adventure_mode=True)
        navigator.observe(data)
        before = navigator.decision_context()
        data.update(session="new-arm", frame=data["frame"]+1)
        fresh = navigator.rearmed(data)
        fresh.observe(data)
        self.assertEqual(fresh.decision_context()["map_rows"], before["map_rows"])
        self.assertEqual(FloorNavigator().decision_context()["map_rows"], [])

    def test_dispatch_includes_original_budget_and_previous_goal_without_extra_requests(self):
        first, second = observed(30, clear=False), observed(34, clear=False)
        before = copy.deepcopy(first)
        _, calls, _, summary, _ = run([(0, first, OLD), (.15, second, OLD)], max_calls=2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(summary["decisions"], 2)
        a, b = [row[1]["_controller_context"] for row in calls]
        self.assertEqual(a["remaining_requests_after_this"], 1)
        self.assertEqual(b["remaining_requests_after_this"], 0)
        self.assertLess(b["remaining_seconds"], a["remaining_seconds"])
        self.assertIsNone(a["previous_goal"])
        self.assertEqual(b["previous_goal"]["kind"], "engage")
        self.assertEqual(b["last_local_command"]["frame"], 34)
        self.assertEqual(first, before)

    def test_rearm_does_not_expose_old_goal_or_old_session_command_as_current(self):
        _, calls, _, _, _ = run([
            (0, observed(30, clear=False), OLD),
            (.05, observed(31, clear=False, enabled=False), OLD),
            (.2, observed(36, clear=False, session="fresh-arm"), NEW)], max_calls=2)
        self.assertEqual(len(calls), 2)
        context = calls[-1][1]["_controller_context"]
        self.assertIsNone(context["previous_goal"])
        self.assertIsNone(context["last_local_command"])


if __name__ == "__main__":
    unittest.main()
