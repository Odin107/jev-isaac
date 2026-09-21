"""Observed bomb choices and floor resource memory; no live game or API."""
import copy
import unittest

from test_adventure import ready
from test_exploration import door, grid
from test_floor_memory import visited
from test_pickups import pickup
from test_player_navigation import offers
from test_player_policy import call
from jev_isaac.adventure import candidates, candidate_valid
from jev_isaac.player_navigation import PlayerNavigator


def rock_room(kind=4):
    data = ready(x=200)
    data["player"].update(bombs=1)
    data["hazards"] = [grid(360, 280, kind=kind)]
    return data


def rock_offer(data):
    return next((c for c in candidates(data, include_rock_targets=True, limit=160)
                 if c.details.get("purpose") == "destroy_rock"), None)


class RockBombTests(unittest.TestCase):
    def test_ordinary_and_tinted_rocks_are_explicit_choices_without_a_pickup(self):
        for kind in (2, 4):
            with self.subTest(kind=kind):
                data = rock_room(kind)
                self.assertEqual(data["pickups"], [])
                candidate = rock_offer(data)
                self.assertIsNotNone(candidate)
                self.assertEqual(candidate.cost, {"bombs": 1})
                self.assertIsNone(candidate.pickup_signature)
                self.assertEqual(candidate.details["rock_type"], kind)
                self.assertTrue(candidate_valid(data, candidate))
                self.assertIn("unknown", candidate.description)

    def test_unsupported_bombs_obstacles_combat_and_incomplete_state_are_not_offered(self):
        changes = [lambda d: d["player"].update(bombs=0),
                   lambda d: d["player"].update(bomb_flags=4),
                   lambda d: d["player"].update(inventory_truncated=True),
                   lambda d: d["player"].update(move_speed=.7),
                   lambda d: d["room"].update(clear=False),
                   lambda d: d.update(truncated=True),
                   lambda d: d["hazards"].append(grid(500, 350, kind=12)),
                   lambda d: d["hazards"][0].update(type=15, collision=4),
                   lambda d: d["hazards"][0].update(type=7, collision=1),
                   lambda d: d["hazards"][0].update(type=22),
                   lambda d: d["hazards"][0].update(collision=0)]
        for change in changes:
            data = rock_room()
            change(data)
            self.assertIsNone(rock_offer(data))

    def test_rock_and_fixed_retreat_are_revalidated_before_spending(self):
        changes = [lambda d, c: d["player"].update(bombs=0),
                   lambda d, c: d["hazards"][0].update(type=2),
                   lambda d, c: d["hazards"][0].update(x=400),
                   lambda d, c: d["hazards"].clear(),
                   lambda d, c: d["hazards"].append(grid(*c.escape_point)),
                   lambda d, c: d.update(room_id="other")]
        for change in changes:
            data = rock_room()
            candidate = rock_offer(data)
            change(data, candidate)
            self.assertFalse(candidate_valid(data, candidate))

    def test_bomb_is_never_automatic_and_one_pulse_finishes_after_observed_blast(self):
        data, nav = rock_room(), PlayerNavigator()
        choices = offers(nav, data)
        candidate = next(c for c in choices if c.details.get("purpose") == "destroy_rock")
        self.assertIsNone(nav._intent)
        self.assertEqual(nav._adventure.selected, 0)
        self.assertTrue(nav.accept_adventure(candidate.key, data, .7))
        self.assertFalse(nav.allows_independent_fire)
        data["player"].update(x=candidate.point[0], y=candidate.point[1], vx=0, vy=0)
        data["frame"] += 1
        self.assertEqual(nav.step(data, .8).interaction, "bomb")
        data["player"]["bombs"] = 0
        data["hazards"].append({"kind": "bomb", "x": candidate.point[0], "y": candidate.point[1],
                                "vx": 0, "vy": 0, "radius": 10})
        data["frame"] += 1
        retreat = nav.step(data, .9)
        self.assertEqual(retreat.interaction, "none")
        self.assertNotEqual(retreat.move, "none")
        data["player"].update(x=candidate.escape_point[0], y=candidate.escape_point[1])
        data["frame"] += 1
        self.assertEqual(nav.step(data, 1).interaction, "none")
        self.assertIsNotNone(nav._intent)
        data["hazards"] = []
        data["pickups"] = [pickup("drop", variant=10, subtype=3, x=360)]
        data["frame"] += 1
        finished = nav.step(data, 2.5)
        self.assertEqual((finished.move, getattr(finished, "interaction", "none")), ("none", "none"))
        self.assertIsNone(nav._intent)
        self.assertEqual(nav.pickup_stats["attempts"], 0)


class ResourceMemoryTests(unittest.TestCase):
    def test_item_room_is_found_even_if_locked_and_never_becomes_an_implied_route(self):
        data, nav = ready(), PlayerNavigator()
        data["doors"] = [door(2, 85, 4, opened=False, locked=True)]
        nav.observe(data)
        info = nav.decision_context()["item_rooms"]
        self.assertEqual(info["status"], "found_unvisited")
        self.assertEqual((info["visited_count"], info["known_unvisited_count"]), (0, 1))
        self.assertTrue(info["rooms"][0]["observed_entrances"][0]["locked_last_observed"])
        self.assertEqual(nav._graph[nav._current], ())
        self.assertIsNone(nav._pending)
        data["visited_rooms"] = [visited(85, 2, kind=4)]
        data["frame"] += 1
        nav.observe(data)
        info = nav.decision_context()["item_rooms"]
        self.assertEqual((info["status"], info["visited_count"], info["known_unvisited_count"]), ("visited", 1, 0))

    def test_multiple_item_rooms_and_large_room_aliases_are_distinguished(self):
        data, nav = ready(), PlayerNavigator()
        data["doors"] = [door(0, 83, 4), door(2, 85, 4)]
        data["visited_rooms"] = [visited(70, 2, kind=4, aliases=[70, 71, 83])]
        nav.observe(data)
        info = nav.decision_context()["item_rooms"]
        self.assertEqual((info["visited_count"], info["known_unvisited_count"]), (1, 1))
        self.assertEqual(info["status"], "found_unvisited")
        self.assertEqual(len(info["rooms"]), 2)
        self.assertEqual(next(r for r in info["rooms"] if r["visited"])["room_indices"], [70, 71, 83])

    def test_pickup_memory_distinguishes_heart_types_counts_and_shop_costs(self):
        data = ready(pickup("red1", variant=10), pickup("red2", variant=10),
                     pickup("soul", variant=10, subtype=3),
                     pickup("paid", variant=10, price=3, shop_item=True),
                     pickup("empty-pedestal", variant=100, subtype=0))
        nav = PlayerNavigator()
        nav.observe(data)
        remembered = nav.decision_context()["remembered_pickups"][0]
        groups = remembered["groups"]
        self.assertEqual(len(groups), 3)
        self.assertEqual(next(g for g in groups if g["price"] == 0 and g["subtype"] == 1)["count"], 2)
        self.assertEqual(next(g for g in groups if g["subtype"] == 3)["heart_type"], "soul")
        self.assertTrue(next(g for g in groups if g["price"] == 3)["shop_item"])
        remembered["groups"].clear()
        self.assertEqual(len(nav.decision_context()["remembered_pickups"][0]["groups"]), 3)

    def test_offscreen_memory_survives_rearm_and_refreshes_only_when_room_is_observed(self):
        data = ready(pickup("heart", variant=10))
        data["doors"] = [door(2, 85, 4, opened=False, locked=True)]
        nav = PlayerNavigator()
        nav.observe(data)
        other = copy.deepcopy(data)
        other.update(session="new-arm", room_id="other", frame=data["frame"]+1, pickups=[], doors=[])
        other["floor"].update(room_index=83)
        fresh = nav.rearmed(other)
        fresh.observe(other)
        self.assertIsNone(fresh.stop_reason)
        self.assertEqual(fresh.decision_context()["item_rooms"]["status"], "found_unvisited")
        remembered = fresh.decision_context()["remembered_pickups"]
        self.assertEqual(len(remembered), 1)
        self.assertEqual(remembered[0]["room_indices"], [84])
        self.assertEqual(remembered[0]["last_observed_frame"], data["frame"])
        returned = copy.deepcopy(data)
        returned.update(session="third-arm", frame=other["frame"]+1, pickups=[])
        refreshed = fresh.rearmed(returned)
        refreshed.observe(returned)
        self.assertEqual(refreshed.decision_context()["remembered_pickups"], [])
        self.assertEqual(len(nav.decision_context()["remembered_pickups"]), 1)

    def test_missing_pickups_and_imported_visits_do_not_invent_or_erase_supplies(self):
        data = ready(pickup("heart", variant=10))
        nav = PlayerNavigator()
        nav.observe(data)
        data.pop("pickups")
        data["capabilities"].pop("pickup_collection")
        data["visited_rooms"] = [visited(12, 2)]
        data["frame"] += 1
        nav.observe(data)
        remembered = nav.decision_context()["remembered_pickups"]
        self.assertEqual(len(remembered), 1)
        self.assertEqual(remembered[0]["groups"][0]["count"], 1)
        self.assertEqual(nav.decision_context()["item_rooms"]["status"], "not_observed")

    def test_new_floor_starts_without_old_item_room_or_pickup_memory(self):
        old = ready(pickup("heart", variant=10))
        old["doors"] = [door(2, 85, 4)]
        nav = PlayerNavigator()
        nav.observe(old)
        new = ready()
        new["floor"]["id"] = "another-floor"
        rejected = nav.rearmed(new)
        self.assertIsNotNone(rejected.stop_reason)
        self.assertEqual(rejected.decision_context()["remembered_pickups"], [])
        fresh = PlayerNavigator()
        fresh.observe(new)
        self.assertEqual(fresh.decision_context()["item_rooms"]["status"], "not_observed")

    def test_player_request_contains_memory_and_bomb_mechanics_without_hidden_pill_effect(self):
        data = ready(pickup("pill", variant=70, subtype=3, pill_known=False, pill_effect=7))
        nav = PlayerNavigator()
        options = offers(nav, data)
        data["_adventure_options"] = [c.as_dict() for c in options]
        data["_controller_context"] = {"exploration": nav.decision_context()}
        _, payload = call(data, {"activity": "wait", "fire": "none"})
        memory = payload["state"]["controller_context"]["exploration"]["remembered_pickups"]
        self.assertNotIn("pill_effect", str(memory))
        self.assertEqual(memory[0]["groups"][0]["kind"], "pill")
        self.assertIn("bombs_and_rocks", payload["state"]["game_context"])
        self.assertIn("item_rooms", payload["state"]["game_context"])


if __name__ == "__main__":
    unittest.main()
