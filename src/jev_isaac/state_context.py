"""Readable observed-state meanings and explicit limits of local control.

Enum names checked against the installed Repentance+ resources/scripts/enums.lua.
This adds context, never grants actions or infers unseen room contents/effects.
"""
from collections.abc import Mapping


ROOM_TYPES = dict(enumerate((
    "null", "ordinary", "shop", "error", "treasure/item", "boss", "miniboss",
    "secret", "supersecret", "arcade", "curse", "challenge", "library",
    "sacrifice", "devil", "angel", "dungeon", "boss rush", "Isaacs", "barren",
    "chest", "dice", "black market", "greed exit", "planetarium", "teleporter",
    "teleporter exit", "secret exit", "blue", "ultrasecret", "deathmatch")))
GRID_TYPES = dict(enumerate((
    "null", "decoration", "rock", "rock B", "tinted rock", "bomb rock", "alternate rock",
    "pit", "spikes", "toggling spikes", "spiderweb", "lock", "TNT", "fireplace",
    "poop", "wall", "door", "trapdoor", "stairs", "gravity", "pressure plate",
    "statue", "super special rock", "teleporter", "pillar", "spiked rock",
    "alternate rock 2", "gold rock")))
WEAPON_TYPES = dict(enumerate((
    "tears", "brimstone", "laser", "knife", "bombs", "rockets", "Monstros lungs",
    "Ludovico technique", "Tech X", "bone", "notched axe", "urn of souls",
    "spirit sword", "fetus", "umbilical whip"), start=1))
PICKUP_TYPES = {
    0: "null", 10: "heart", 20: "coin", 30: "key", 40: "bomb", 41: "throwable bomb",
    42: "poop", 50: "chest", 51: "bomb chest", 52: "spiked chest", 53: "eternal chest",
    54: "mimic chest", 55: "old chest", 56: "wooden chest", 57: "mega chest",
    58: "haunted chest", 60: "locked chest", 69: "grab bag", 70: "pill",
    90: "battery", 100: "collectible", 110: "broken shovel", 150: "shop item",
    300: "card", 340: "big chest", 350: "trinket", 360: "red chest",
    370: "trophy", 380: "bed", 390: "Moms chest",
}
HEART_TYPES = dict(enumerate(("full red", "half red", "soul", "eternal", "double red",
    "black", "golden", "half soul", "scared red", "blended", "bone", "rotten"), start=1))


def room_name(value):
    return ROOM_TYPES.get(value, "unknown") if type(value) is int else "unknown"


def _rows(state, key):
    value = state.get(key)
    return [row for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []


def _legend(values, names):
    return {str(value): names.get(value, "unknown")
            for value in sorted({v for v in values if type(v) is int})}


CONTEXT_INSTRUCTIONS = (
    " `game_context` explains observed type codes, units, missing information and "
    "the division of control. `controller_context`, when present, records local "
    "memory and recent intent; it is not new game observation. A remembered door "
    "is not proof that it is open now. Unknown or missing information is not absence. "
    "More context does not create choices beyond the listed options."
)


def build_game_context(state, *, decision_kind="combat"):
    player = state.get("player", {})
    player = player if isinstance(player, Mapping) else {}
    room = state.get("room", {})
    room = room if isinstance(room, Mapping) else {}
    doors, visited = _rows(state, "doors"), _rows(state, "visited_rooms")
    weapons = player.get("weapon_types", [player.get("weapon_type")])
    weapons = weapons if isinstance(weapons, list) else []
    truncated = state.get("truncated_arrays", {})
    truncated = truncated if isinstance(truncated, Mapping) else {}
    coverage = {}
    for key in ("enemies", "projectiles", "hazards", "pickups", "doors", "switches", "visited_rooms"):
        coverage[key] = ("missing" if not isinstance(state.get(key), list) else
                         "partial" if truncated.get(key) or state.get("truncated") else "exported")
    coverage["inventory"] = ("missing" if not isinstance(player.get("inventory"), list) else
                             "partial" if player.get("inventory_truncated") else "exported")
    return {
        "source": "Deterministic labels and coverage for observation; no hidden-map lookup.",
        "units": {"hearts_soul_hearts_max_hearts": "half-hearts",
                  "positions": "world units; x right, y down",
                  "velocity": "observed world displacement per physics frame"},
        "type_names": {
            "room_and_door_target_type": _legend([room.get("type"),
                *(d.get("target_type") for d in doors), *(r.get("type") for r in visited)], ROOM_TYPES),
            "grid_type": _legend((h.get("type") for h in _rows(state, "hazards") if h.get("kind") == "grid"), GRID_TYPES),
            "pickup_variant": _legend((p.get("variant") for p in _rows(state, "pickups")), PICKUP_TYPES),
            "weapon_type": _legend(weapons, WEAPON_TYPES),
        },
        "coverage": coverage,
        "floor_surfaces": {
            "coverage": "known non-player creep variants exported" if state.get("capabilities", {}).get("ground_creep") == 1 else "creep exporter unavailable; absence is not evidence of a safe floor",
            "meaning": "hazards with kind=creep are observed enemy floor effects, potentially damaging or movement-affecting. Player/friendly creep is excluded when identified. These are not walls and do not block tears.",
            "size": "radius is observed Entity.Size, an approximate footprint. scale and timeout are raw engine fields when available; exact damage boundary and immunity are not calibrated.",
        },
        "shooting_motion": {
            "player_velocity": {key: player[key] for key in ("vx", "vy") if key in player},
            "effect": "Player momentum can deflect tears diagonally when movement and firing axes differ. Input release does not instantly remove momentum.",
            "calibration": "Exact tear velocity and inherited-momentum multiplier are not measured; do not treat ShotSpeed as world velocity.",
        },
        "bombs_and_rocks": {
            "bombs_available": player.get("bombs"),
            "mechanics": "Placed bombs spend a bomb and explode after a fuse; the blast can hurt Isaac. Explosions can destroy ordinary and tinted rocks, opening space or exposing drops. Not every obstacle is bomb-destructible.",
            "tinted_rocks": "Grid type 4 is an observed tinted rock: a potential source of useful drops such as soul hearts, supplies or an item. Actual contents remain unknown until exposed.",
            "choices": "Only offered bomb_rock activities authorize spending. An offered rock-destruction activity places one bomb, retreats, waits for the observed blast and asks again; it does not automatically collect drops or choose a door.",
            "limits": "Bomb plans require complete ordinary-bomb inventory data and a checked placement/retreat. Modified bombs, chain explosives, walls, pits and special rock mechanics are not covered by this planner. Missing choices do not mean bombs can never work there.",
        },
        "item_rooms": {
            "value": "Treasure/item rooms are important sources of items that can strengthen the rest of the run. Exploring for an unvisited one can improve later combat; travel, keys, health and leaving the floor remain decisions to weigh.",
            "doors": "The first floor's normal treasure room is generally unlocked; later normal treasure-room doors usually cost a key. Use the observed door lock and available actions, not an assumed cost.",
            "knowledge": "controller_context.exploration.item_rooms distinguishes not_observed, found_unvisited and visited on this floor. No observed entrance does not prove no item room exists; some floors/modes lack normal item rooms. A visit does not prove its reward was collected.",
        },
        "limitations": [
            "Exported does not mean every game mechanic is represented. Laser beams, unrecognized floor effects, enemy attack phases and tear effects are incomplete.",
            "Unseen room contents, future drops and unidentified pill effects are unknown. Item text may be untranslated localization tokens.",
            "Local geometry estimates assume grounded movement and ordinary tears even if observed equipment supports more. Raw player stats remain in observation.player.",
        ],
        "control_contract": {
            "decision_kind": decision_kind,
            "model": ("Select or skip one offered interaction." if decision_kind == "adventure" else
                      "Choose engage for a listed target, evade or hold; choose an item only if separately offered."),
            "local_execution": "Pathfinding, aiming, movement pulses, immediate dodging and fresh target/resource checks.",
            "combat_overrides": "Dodging can change movement. Hold/evade can still shoot an aligned enemy; engage prefers its target but can fire at another clear target without changing pursuit.",
            "local_strategy": "Free supplies, ordinary open-door order, required switches and TNT puzzles are local. Exploration prefers unfinished non-boss branches, then bosses, before inspecting known cleared transit rooms.",
            "candidate_limits": "Combat offers at most eight nearby vulnerable enemies. Interaction offers are filtered by implemented mechanics, resources and estimated routes; missing choices may be a controller limitation.",
            "freshness": "Changed room/session, pause, expired goal or stale response cancels old intent; replanning uses the newest observation.",
        },
    }
