"""Observed room contents and repeat visits inform Jev without selecting its route."""
import copy
import unittest

from test_adventure import ready
from test_exploration import door
from test_floor_memory import visited
from test_pickups import pickup
from test_player_policy import call
from jev_isaac.player_navigation import PlayerNavigator


def room(index, list_index, *, kind=1, frame=1, items=(), doors=()):
    data = ready(*items, index=index, kind=kind, frame=frame)
    data["floor"].update(room_list_index=list_index, dimension=0)
    data["doors"] = list(doors)
    return data


def settled(nav, data):
    nav.step(data, data["frame"]/30)
    data["frame"] += 18
    nav.step(data, data["frame"]/30)
    assert nav.stop_reason is None, nav.stop_reason
    return {candidate.key: candidate for candidate in nav.adventure_options}


def return_from_treasure(items=(), treasure_exits=None):
    treasure = room(85, 2, kind=4, items=items,
                    doors=treasure_exits if treasure_exits is not None else [door(0, 84)])
    nav = PlayerNavigator()
    nav.observe(treasure)
    source = room(84, 1, frame=treasure["frame"]+1,
                  doors=[door(0, 83), door(2, 85, 4)])
    source["session"] = "run:arm2"
    nav = nav.rearmed(source)
    return nav, source


class DoorKnowledgeTests(unittest.TestCase):
    def test_imported_visit_does_not_claim_empty_contents_or_known_return_route(self):
        data = room(84, 1, doors=[door(0, 83), door(2, 85, 4)])
        data["visited_rooms"] = [visited(85, 2, kind=4)]
        nav = PlayerNavigator()
        options = settled(nav, data)
        destination = options["enter:2:85"].details
        expected = {"status": "unknown", "last_observed_frame": None, "groups": []}
        self.assertEqual(destination["destination_pickups_last_observed"], expected)
        self.assertIsNone(destination["destination_is_known_return_only"])
        self.assertEqual(nav.decision_context()["item_rooms"]["rooms"][0]["pickups_last_observed"], expected)
        self.assertIn("enter:0:83", options)

    def test_observed_empty_pedestal_is_explicit_but_reentry_remains_selectable(self):
        nav, data = return_from_treasure([pickup("empty", variant=100, subtype=0)])
        options = settled(nav, data)
        candidate = options["enter:2:85"]
        memory = candidate.details["destination_pickups_last_observed"]
        self.assertEqual(memory, {"status": "none_observed", "last_observed_frame": 1, "groups": []})
        self.assertTrue(candidate.details["destination_is_known_return_only"])
        self.assertEqual(nav.decision_context()["item_rooms"]["rooms"][0]["pickups_last_observed"], memory)
        self.assertIn("enter:0:83", options)
        self.assertIsNone(nav._pending)
        self.assertTrue(nav.accept_adventure(candidate.key, data, data["frame"]/30+.01))
        self.assertEqual(nav._pending.target_index, 85)

    def test_swapped_active_item_left_on_pedestal_is_still_present(self):
        nav, data = return_from_treasure([pickup("swapped", variant=100, subtype=34, collectible_kind=3)])
        data["player"].update(active_item=66, inventory=[{"id": 66, "count": 1}])
        candidate = settled(nav, data)["enter:2:85"]
        memory = candidate.details["destination_pickups_last_observed"]
        self.assertEqual(memory["status"], "present")
        self.assertEqual([(g["variant"], g["subtype"], g["count"]) for g in memory["groups"]], [(100, 34, 1)])
        memory["groups"].clear()
        self.assertEqual(nav.decision_context()["item_rooms"]["rooms"][0]["pickups_last_observed"]["groups"][0]["subtype"], 34)

    def test_return_only_uses_actual_room_aliases_and_preserves_other_branches(self):
        for branch in (False, True):
            with self.subTest(branch=branch):
                exits = [door(0, 83)] + ([door(2, 86)] if branch else [])
                treasure = room(85, 2, kind=4, doors=exits)
                treasure["visited_rooms"] = [visited(84, 1, aliases=[83, 84])]
                nav = PlayerNavigator()
                nav.observe(treasure)
                source = room(84, 1, frame=2, doors=[door(2, 85, 4)])
                source["session"] = "run:arm2"
                nav = nav.rearmed(source)
                candidate = settled(nav, source)["enter:2:85"]
                self.assertEqual(candidate.details["destination_is_known_return_only"], not branch)
                self.assertEqual(candidate.details["destination_pickups_last_observed"]["status"], "none_observed")

    def test_missing_snapshot_stays_unknown_and_rearm_keeps_only_same_floor_memory(self):
        treasure = room(85, 2, kind=4, doors=[door(0, 84)])
        treasure.pop("pickups")
        treasure["capabilities"].pop("pickup_collection")
        nav = PlayerNavigator()
        nav.observe(treasure)
        self.assertEqual(nav.decision_context()["item_rooms"]["rooms"][0]["pickups_last_observed"]["status"], "unknown")
        nav, data = return_from_treasure()
        self.assertEqual(settled(nav, data)["enter:2:85"].details["destination_pickups_last_observed"]["status"], "none_observed")
        new_floor = copy.deepcopy(data)
        new_floor["floor"]["id"] = "next-floor"
        new_floor["visited_rooms"] = [visited(85, 2, kind=4)]
        fresh = PlayerNavigator()
        options = settled(fresh, new_floor)
        self.assertEqual(options["enter:2:85"].details["destination_pickups_last_observed"]["status"], "unknown")

    def test_activity_criteria_explain_bound_door_and_retain_free_choice(self):
        nav, data = return_from_treasure()
        options = settled(nav, data)
        data["_adventure_options"] = [candidate.as_dict() for candidate in options.values()]
        data["_controller_context"] = {"exploration": nav.decision_context()}
        chosen_index = list(options).index("enter:2:85")
        decision, payload = call(data, {"activity": f"action_{chosen_index}", "fire": "none"})
        candidate = options["enter:2:85"]
        criterion = payload["questions"]["activity"]["criteria"][f"action_{chosen_index}"]
        self.assertIn(candidate.key, criterion)
        self.assertIn(candidate.description, criterion)
        self.assertIn("activity_candidates", criterion)
        self.assertEqual(decision.target_id, candidate.key)
        self.assertTrue(any(c["key"] == "enter:0:83" for c in payload["state"]["activity_candidates"]))

    def test_transition_history_counts_arrivals_not_selected_or_failed_attempts(self):
        nav = PlayerNavigator()
        data = room(84, 1, doors=[door(2, 85, 4)])
        arrivals = []
        for i in range(14):
            options = settled(nav, data)
            chosen = next(c for c in options.values() if c.kind == "enter_door")
            origin, target = data["floor"]["room_index"], chosen.details["target_index"]
            self.assertTrue(nav.accept_adventure(chosen.key, data, data["frame"]/30+.01))
            arrival = room(target, 2 if target == 85 else 1, kind=4 if target == 85 else 1,
                           frame=data["frame"]+1,
                           doors=[door(0, 84)] if target == 85 else [door(2, 85, 4)])
            arrival["room_id"] = f"arrival:{i}"
            nav.observe(arrival)
            arrivals.append({"frame": arrival["frame"], "from_room_index": origin, "to_room_index": target})
            data = arrival
        options = settled(nav, data)
        chosen = options["enter:2:85"]
        self.assertEqual(chosen.details["recent_times_entered_from_here"], 7)
        self.assertEqual(nav.decision_context()["recent_room_transitions"], arrivals[-12:])
        nav._event(data, "selected", "Jev selected this activity", chosen)
        nav._event(data, "failed", "selected door route blocked", chosen)
        nav._offers = ()
        options = settled(nav, data)
        self.assertEqual(options[chosen.key].details["recent_times_entered_from_here"], 7)
        self.assertEqual(nav.decision_context()["recent_room_transitions"], arrivals[-12:])
        # Once evidence ages out of the bounded event buffer it is not invented.
        for _ in range(80):
            nav._event(data, "failed", "selected door route blocked", chosen)
        nav._offers = ()
        options = settled(nav, data)
        self.assertEqual(options[chosen.key].details["recent_times_entered_from_here"], 0)
        self.assertEqual(nav.decision_context()["recent_room_transitions"], [])


if __name__ == "__main__":
    unittest.main()
