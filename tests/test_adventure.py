"""Strategic eligibility and real grid geometry; no provider or live controls."""
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_exploration import door, grid
from test_pickups import observed, pickup
from jev_isaac.adventure import (BLAST_MARGIN, BLAST_RADIUS, AdventureCandidate,
                                candidate_geometry, candidate_valid, candidates)
from jev_isaac.navigation import _clear


def ready(*items, **extra):
    data = observed(*items, **extra)
    data["capabilities"]["interaction_control"] = 1
    data["player"].update(pocket_card=0, pocket_pill=0, trinket=0, trinket_1=0,
                          inventory=[], inventory_truncated=False,
                          giga_bombs=0, bomb_flags=0, unsafe_bomb_trinket=False, move_speed=1.0,
                          active_charge=0, active_max_charge=0,
                          active_charge_type=0)
    return data


def selected(data, kind, **kwargs):
    return next((choice for choice in candidates(data, **kwargs) if choice.kind == kind), None)


def blocked_chest():
    data = ready(pickup("locked", variant=60, x=480), x=200)
    data["player"].update(bombs=1, keys=1)
    data["hazards"] = [grid(360, y) for y in range(160, 401, 40)]
    return data


class AdventureTests(unittest.TestCase):
    def test_unknown_pill_collection_and_consumption_are_separate_choices(self):
        data = ready(pickup("pill", variant=70, subtype=3, pill_known=False))
        choice = selected(data, "collect")
        self.assertIsNotNone(choice)
        self.assertIn("unidentified", choice.description)
        self.assertTrue(candidate_valid(data, choice))
        data["pickups"] = []
        data["player"].update(pocket_pill=3, pill_known=False, pill_effect=7)
        use = selected(data, "use_pocket")
        self.assertIsNotNone(use)
        self.assertEqual(use.key, "pocket:pill:3:unidentified")
        self.assertNotIn("effect", use.details)
        self.assertFalse(use.details["identified"])

    def test_cards_and_trinkets_need_an_observed_empty_slot(self):
        for variant, field in ((300, "pocket_card"), (70, "pocket_pill"), (350, "trinket")):
            with self.subTest(variant=variant):
                data = ready(pickup(variant=variant))
                self.assertIsNotNone(selected(data, "collect"))
                data["player"][field] = 5
                self.assertIsNone(selected(data, "collect"))
                data["player"].pop(field)
                self.assertIsNone(selected(data, "collect"))

    def test_ordinary_and_locked_chests_require_closed_and_affordable(self):
        data = ready(pickup("chest", variant=60))
        self.assertIsNone(selected(data, "open_chest"))
        data["player"]["keys"] = 1
        choice = selected(data, "open_chest")
        self.assertEqual(choice.cost, {"keys": 1})
        self.assertTrue(candidate_valid(data, choice))
        data["player"]["keys"] = 0
        self.assertFalse(candidate_valid(data, choice))
        data["pickups"][0].update(variant=50)
        self.assertEqual(selected(data, "open_chest").cost, {})
        data["pickups"][0]["subtype"] = 0
        self.assertIsNone(selected(data, "open_chest"))
        for variant in (51, 52, 53, 54, 55, 56, 57, 58, 360):
            data["pickups"][0].update(variant=variant, subtype=1)
            self.assertIsNone(selected(data, "open_chest"))

    def test_shop_purchase_binds_exact_price_and_never_spends_health(self):
        data = ready(pickup("shop", variant=100, subtype=1, collectible_kind=1,
                            shop_item=True, price=15, name="An item"))
        data["player"]["coins"] = 14
        self.assertIsNone(selected(data, "buy"))
        data["player"]["coins"] = 20
        choice = selected(data, "buy")
        self.assertEqual(choice.cost, {"coins": 15})
        self.assertTrue(candidate_valid(data, choice))
        data["pickups"][0]["price"] = 16
        self.assertFalse(candidate_valid(data, choice))
        for price in (-1, -2, -3, -4, -1000):
            data["pickups"][0]["price"] = price
            self.assertIsNone(selected(data, "buy"))

    def test_active_item_replacement_is_an_explicit_choice(self):
        data = ready(pickup("new-active", variant=100, subtype=105, collectible_kind=3))
        data["player"]["active_item"] = 45
        choice = selected(data, "collect")
        self.assertTrue(choice.details["replaces_active"])
        self.assertIn("replaces held active", choice.description)
        self.assertTrue(candidate_valid(data, choice))
        data["player"]["active_item"] = 85
        self.assertFalse(candidate_valid(data, choice))
        data["player"]["active_item"] = 45
        data["pickups"][0]["subtype"] = 97
        self.assertFalse(candidate_valid(data, choice))

    def test_reroll_choice_is_invalid_when_the_observed_pedestal_changes(self):
        data = ready(pickup("item", variant=100, subtype=1, collectible_kind=1))
        data["player"].update(active_item=105, active_charge=6, active_max_charge=6)
        choice = selected(data, "use_active")
        self.assertIsNotNone(choice)
        self.assertTrue(candidate_valid(data, choice))
        data["pickups"][0]["subtype"] = 2
        self.assertFalse(candidate_valid(data, choice))

    def test_context_and_paused_or_truncated_state_invalidate_choices(self):
        original = ready(pickup("pill", variant=70))
        choice = selected(original, "collect")
        for change in (lambda d: d.update(enabled=False), lambda d: d.update(paused=True),
                       lambda d: d.update(room_id="different"), lambda d: d.update(run_id="new"),
                       lambda d: d["floor"].update(id="next"), lambda d: d.update(truncated=True),
                       lambda d: d["capabilities"].pop("interaction_control")):
            data = copy.deepcopy(original)
            change(data)
            self.assertFalse(candidate_valid(data, choice))

    def test_locked_doors_only_use_keys_for_observed_supported_room_types(self):
        for kind in (1, 2, 4):
            data = ready()
            data["player"]["keys"] = 1
            data["doors"] = [door(2, 85, kind, opened=False, locked=True)]
            choice = selected(data, "unlock_door")
            self.assertEqual(choice.details["target_index"], 85)
            self.assertTrue(candidate_valid(data, choice))
            data["doors"][0]["target_index"] = 86
            self.assertFalse(candidate_valid(data, choice))
        for kind in (5, 6, 9, 14, 16):
            data["doors"] = [door(2, 85, kind, opened=False, locked=True)]
            self.assertIsNone(selected(data, "unlock_door"))

    def test_bomb_opens_proven_rock_route_and_has_safe_retreat(self):
        data = blocked_chest()
        choice = selected(data, "bomb_rock")
        self.assertIsNotNone(choice)
        self.assertEqual(choice.cost, {"keys": 1, "bombs": 1})
        self.assertTrue(candidate_valid(data, choice))
        bounds, boxes, phase = candidate_geometry(data, choice)
        self.assertTrue(_clear(choice.point, choice.escape_point, boxes))
        self.assertGreaterEqual(math.dist(choice.point, choice.escape_point),
                                BLAST_RADIUS + data["player"]["radius"] + BLAST_MARGIN)
        self.assertIsNotNone(choice.rock_id)
        self.assertIn("retreat", choice.description)
        # Advancing to the placement keeps the fixed plan valid.
        data["player"].update(x=choice.point[0], y=choice.point[1])
        self.assertTrue(candidate_valid(data, choice))

    def test_bomb_never_targets_a_walkable_chest(self):
        data = blocked_chest()
        data["hazards"] = [grid(360, 280)]
        self.assertIsNotNone(selected(data, "open_chest"))
        self.assertIsNone(selected(data, "bomb_rock"))

    def test_bomb_requires_both_resources_and_complete_standard_inventory(self):
        original = blocked_chest()
        for change in (lambda d: d["player"].update(keys=0),
                       lambda d: d["player"].update(bombs=0),
                       lambda d: d["player"].update(inventory_truncated=True),
                       lambda d: d["player"].pop("inventory"),
                       lambda d: d["player"].pop("move_speed"),
                       lambda d: d["player"].update(move_speed=.7),
                       lambda d: d["player"].pop("trinket_1"),
                       lambda d: d["player"].update(trinket_1=133),
                       lambda d: d["player"].update(trinket=73 | 32768),
                       lambda d: d["player"].update(giga_bombs=1),
                       lambda d: d["player"].pop("bomb_flags"),
                       lambda d: d["player"].update(bomb_flags=4),
                       lambda d: d["player"].update(unsafe_bomb_trinket=True),
                       lambda d: d["player"].update(inventory=[{"id": 106, "count": 1}]),
                       lambda d: d["player"].update(inventory=[{"id": 353, "count": 1}]),
                       lambda d: d["player"].update(inventory=[{"id": 563, "count": 1}]),
                       lambda d: d["player"].update(inventory=[{"id": 583, "count": 1}]),
                       lambda d: d["player"].update(inventory=[{"id": 9999, "count": 1}])):
            data = copy.deepcopy(original)
            change(data)
            self.assertIsNone(selected(data, "bomb_rock"))

    def test_bomb_rejects_pits_walls_and_chain_reactions(self):
        for kind, collision in ((7, 1), (15, 4), (8, 0), (5, 3)):
            data = blocked_chest()
            data["hazards"] = [grid(360, y, kind=kind, collision=collision) for y in range(160, 401, 40)]
            self.assertIsNone(selected(data, "bomb_rock"))
        data = blocked_chest()
        data["hazards"].append(dict(grid(120, 360), kind="bomb", type=4))
        self.assertIsNone(selected(data, "bomb_rock"))

    def test_bomb_fixed_retreat_is_rechecked_when_a_hazard_arrives(self):
        data = blocked_chest()
        choice = selected(data, "bomb_rock")
        x, y = choice.escape_point
        data["hazards"].append(grid(x, y, kind=8, collision=0))
        self.assertFalse(candidate_valid(data, choice))

    def test_bomb_rejects_an_enclosed_retreat_even_if_target_could_be_opened(self):
        data = blocked_chest()
        data["player"]["x"] = 280
        data["hazards"].extend(grid(240, y, kind=15, collision=4) for y in range(160, 401, 40))
        self.assertIsNone(selected(data, "bomb_rock"))

    def test_bomb_choice_does_not_silently_upgrade_the_spend_after_price_change(self):
        data = blocked_chest()
        choice = selected(data, "bomb_rock")
        data["pickups"][0].update(variant=100, subtype=1, collectible_kind=1, price=15, shop_item=True)
        data["player"]["coins"] = 15
        self.assertFalse(candidate_valid(data, choice))

    def test_charged_active_effect_is_contextual_and_revalidated(self):
        data = ready(clear=False)
        data["player"].update(active_item=34, active_charge=3, active_max_charge=3)
        choice = selected(data, "use_active")
        self.assertEqual(choice.interaction, "active")
        self.assertTrue(candidate_valid(data, choice))
        data["player"]["active_charge"] = 2
        self.assertFalse(candidate_valid(data, choice))
        data["player"]["active_charge"] = 3
        data["room"]["clear"] = True
        self.assertFalse(candidate_valid(data, choice))
        data["player"].update(active_item=45, can_pick_red_hearts=False)
        self.assertIsNone(selected(data, "use_active"))
        data["player"]["can_pick_red_hearts"] = True
        self.assertIsNotNone(selected(data, "use_active"))

    def test_unsafe_or_unknown_active_and_pocket_effects_are_not_options(self):
        data = ready(clear=False)
        data["player"].update(active_charge=6, active_max_charge=6)
        for item in (0, 65, 126, 127, 135, 283, 284, 475, 9999):
            data["player"]["active_item"] = item
            self.assertIsNone(selected(data, "use_active"))
        for card in (1, 5, 17, 18, 19, 46, 9999):
            data["player"].update(pocket_card=card, pocket_pill=0)
            self.assertIsNone(selected(data, "use_pocket"))
        for effect in (1, 4, 6, 11, 15, 19, 21, 22, 29, 42, 44, 49):
            data["player"].update(pocket_card=0, pocket_pill=2, pill_known=True, pill_effect=effect)
            self.assertIsNone(selected(data, "use_pocket"))

    def test_known_pill_and_card_action_binds_exact_held_effect(self):
        data = ready()
        data["player"].update(pocket_pill=2, pill_known=True, pill_effect=7)
        choice = selected(data, "use_pocket")
        self.assertEqual(choice.interaction, "pocket")
        self.assertTrue(candidate_valid(data, choice))
        data["player"]["pill_effect"] = 6
        self.assertFalse(candidate_valid(data, choice))
        data["player"].update(pocket_card=4, pocket_pill=0)
        self.assertIsNone(selected(data, "use_pocket"))
        data["room"]["clear"] = False
        self.assertIsNotNone(selected(data, "use_pocket"))

    def test_verified_numeric_ids_never_offer_unbound_reroll_or_self_damage(self):
        # Repentance API CollectibleType/Card enum values, checked 2026-09-20.
        # D20 changes pickups, D4/D100 change the build, Razor Blade hurts Isaac.
        for clear in (False, True):
            data = ready(clear=clear)
            data["player"].update(active_charge=6, active_max_charge=6)
            for item in (126, 166, 283, 284):
                data["player"]["active_item"] = item
                self.assertIsNone(selected(data, "use_active"), (clear, item))
            for card in (3, 37, 40, 41, 46, 49, 55):
                data["player"].update(pocket_card=card, pocket_pill=0)
                self.assertIsNone(selected(data, "use_pocket"), (clear, card))
        data = ready(clear=False)
        data["player"].update(active_charge=6, active_max_charge=6)
        for item in (41, 291):  # Mom's Pad and Flush! have valid combat effects.
            data["player"]["active_item"] = item
            self.assertIsNotNone(selected(data, "use_active"))
        data = ready()
        data["player"]["pocket_card"] = 36  # Ansuz reveals the map; Perthro is 37.
        self.assertIsNotNone(selected(data, "use_pocket"))

    def test_documented_pill_effect_ids_do_not_admit_harmful_or_random_effects(self):
        # Names anchor the numeric exclusions to actual API effects instead of
        # relying on adjacency to similarly numbered beneficial effects.
        excluded = {1: "Bad Trip", 3: "Bombs Are Key", 4: "Explosive Diarrhea",
                    6: "Health Down", 11: "Range Down", 13: "Speed Down",
                    15: "Tears Down", 17: "Luck Down", 19: "Telepills",
                    21: "Hematemesis", 22: "Paralysis", 25: "Amnesia",
                    27: "Wizard", 29: "Addicted", 31: "Question Mark",
                    37: "Retro Vision", 42: "I'm Excited", 44: "Horf",
                    47: "Shot Speed Down", 49: "Experimental"}
        for clear in (False, True):
            data = ready(clear=clear)
            data["player"].update(pocket_card=0, pocket_pill=2, pill_known=True,
                                  active_item=45, active_charge=1, active_max_charge=6)
            for effect, name in excluded.items():
                data["player"]["pill_effect"] = effect
                self.assertIsNone(selected(data, "use_pocket"), (clear, name))
        data = ready(clear=False)
        data["player"].update(pocket_card=0, pocket_pill=2, pill_known=True)
        for effect in (24, 26, 28, 41):  # Pheromones, Lemon Party, Percs, Drowsy.
            data["player"]["pill_effect"] = effect
            self.assertIsNotNone(selected(data, "use_pocket"))

    def test_descent_requires_boss_clear_reward_completion_and_explicit_mode(self):
        data = ready(kind=5)
        exit_grid = grid(400, 280, kind=17, collision=0)
        data["hazards"] = [exit_grid, grid(320, 200, kind=18, collision=0)]
        self.assertIsNone(selected(data, "descend"))
        self.assertIsNone(selected(data, "descend", allow_descend=True))
        choice = selected(data, "descend", allow_descend=True, rewards_done=True)
        self.assertIsNotNone(choice)
        self.assertTrue(candidate_valid(data, choice, allow_descend=True, rewards_done=True))
        _, boxes, _ = candidate_geometry(data, choice)
        self.assertFalse(any(box[0] < 400 < box[2] and box[1] < 280 < box[3] for box in boxes))
        self.assertTrue(any(box[0] < 320 < box[2] and box[1] < 200 < box[3] for box in boxes))
        data["room"]["clear"] = False
        self.assertFalse(candidate_valid(data, choice, allow_descend=True, rewards_done=True))

    def test_no_descent_from_unrelated_room_or_unknown_exit(self):
        data = ready()
        data["hazards"] = [grid(400, 280, kind=17, collision=0)]
        self.assertIsNone(selected(data, "descend", allow_descend=True, rewards_done=True))
        data["room"]["type"] = 5
        data["hazards"][0]["type"] = 23
        self.assertIsNone(selected(data, "descend", allow_descend=True, rewards_done=True))

    def test_candidate_count_and_serialized_choice_are_bounded_and_grounded(self):
        data = ready(*(pickup(f"coin-{i}", x=100+i*20, y=200) for i in range(12)))
        options = candidates(data)
        self.assertEqual(len(options), 8)
        self.assertEqual(len({c.key for c in options}), 8)
        for choice in options:
            self.assertIn(choice.target_id, {p["id"] for p in data["pickups"]})
            self.assertEqual(json.loads(json.dumps(choice.as_dict()))["context"], list(choice.context))


if __name__ == "__main__":
    unittest.main()
