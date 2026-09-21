"""Finite, observed choices for Jev; this module never sends game controls.

Execution must revalidate the selected choice against a fresh observation. A
bomb plan deliberately proves a route both before and after one ordinary rock
is removed, including a straight retreat outside a conservative blast margin.
The planner does not infer hidden pickups, map cells, pill effects or doors.

Numeric constants follow the Repentance Lua API PickupVariant, ChestSubType,
GridEntityType, CollectibleType, Card and PillEffect documentation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import math

from .combat import _inside, _number, _point
from .navigation import _clear, _free, _radius
from .pickups import priority, signature

MAX_CANDIDATES = 8
BLAST_RADIUS = 100.0
BLAST_MARGIN = 25.0
_DOOR_DIRECTIONS = ((-1., 0.), (0., -1.), (1., 0.), (0., 1.))


@dataclass(frozen=True)
class AdventureCandidate:
    key: str
    kind: str
    target_id: str
    point: tuple[float, float] | None
    cost: dict[str, int]
    description: str
    interaction: str = "none"
    escape_point: tuple[float, float] | None = None
    pickup_signature: tuple[str, int, int] | None = None
    rock_id: str | None = None
    context: tuple[str, str, str] = ("", "", "")
    details: dict = field(default_factory=dict)

    def as_dict(self):
        return asdict(self)


def _integer(value, lo=0, hi=2**31-1):
    return type(value) is int and lo <= value <= hi


def _context(state):
    return state["run_id"], state["room_id"], state["floor"]["id"]


def _parsed(state):
    # Local import keeps this independent of the explorer's integration hook.
    from .exploration import _validated
    if (not isinstance(state, Mapping) or not isinstance(state.get("capabilities"), Mapping)
            or type(state["capabilities"].get("interaction_control")) is not int
            or state["capabilities"].get("interaction_control") != 1):
        return None
    parsed = _validated(state)
    if (parsed is None or not state["enabled"] or state["paused"]
            or state["player"]["dead"]):
        return None
    return parsed


def _label(item):
    name = item.get("name")
    return name[:100] if isinstance(name, str) and name else f"pickup {item['variant']}:{item['subtype']}"


def _pickup_option(item, player):
    """Logical eligibility, independent of whether a walking route exists."""
    if item["wait"] > 0 or not player["can_pickup_items"]:
        return None
    variant, subtype, price = item["variant"], item["subtype"], item["price"]
    if price < 0 or (price == 0 and item["shop_item"]):
        return None  # No health deals or ambiguous free shop records.
    if price > 0 and (not item["shop_item"] or price > player["coins"]):
        return None
    cost = {"coins": price} if price > 0 else {}
    if variant in (50, 60):
        if price or subtype != 1 or (variant == 60 and player["keys"] < 1):
            return None
        return "open_chest", ({"keys": 1} if variant == 60 else {}), "Open " + ("locked chest" if variant == 60 else "ordinary chest")
    supported = False
    if variant in (70, 300):
        supported = (subtype > 0 and type(player.get("pocket_card")) is int and player["pocket_card"] == 0
                     and type(player.get("pocket_pill")) is int and player["pocket_pill"] == 0)
    elif variant == 350:
        supported = subtype > 0 and type(player.get("trinket")) is int and player["trinket"] == 0
    elif variant == 100:
        supported = subtype > 0 and item["collectible_kind"] in (1, 3, 4)
    else:
        # Reuse the established health/capacity rules without treating a shop
        # price as a free collection goal. The original item is never changed.
        supported = priority(dict(item, price=0, shop_item=False), player) is not None
    if not supported:
        return None
    kind = "buy" if price > 0 else "collect"
    text = f"Buy {_label(item)} for {price} coins" if price > 0 else f"Collect {_label(item)}"
    if variant == 100 and item["collectible_kind"] == 3 and player["active_item"]:
        text += f"; replaces held active item {player['active_item']}"
    if variant == 70 and not item.get("pill_known", False):
        text += "; carry the unidentified pill without consuming it"
    return kind, cost, text


def _grid_id(item):
    value = item.get("id")
    if isinstance(value, str) and value:
        return value
    # Older grid reports have no entity identity; type plus exact position is
    # stable within the context-bound room. No rounded or guessed coordinate.
    point = _point(item)
    return f"grid:{item['type']}:{point[0]:g}:{point[1]:g}"


def _geometry(state, radius, *, target=None, remove_grid=None, door=None):
    from .exploration import _room_geometry
    selected = state
    if remove_grid is not None:
        selected = dict(state, hazards=[item for item in state["hazards"]
                                       if not (item.get("kind") == "grid" and _grid_id(item) == remove_grid)])
    return _room_geometry(selected, radius, door=door, pickup_target=target)


def _reachable(start, point, bounds, geometry):
    from .exploration import _point_waypoint
    boxes, roomy, _, phase = geometry
    if not _inside(start, bounds) or not _free(start, boxes):
        return False
    return (_point_waypoint(start, point, bounds, roomy, phase) is not None
            or _point_waypoint(start, point, bounds, boxes, phase) is not None)


def _inventory_ids(player):
    values = player.get("inventory")
    if not isinstance(values, list) or player.get("inventory_truncated") is not False:
        return None
    if any(not isinstance(item, Mapping) or not _integer(item.get("id"), 1, 732)
           or not _integer(item.get("count"), 1) for item in values):
        return None
    return {item["id"] for item in values}


def _ordinary_bombs(state):
    player = state["player"]
    inventory = _inventory_ids(player)
    # Changed radius, fuse, trajectory, chaining or remote detonation needs its
    # own simulation. The single normal-bomb plan must not silently assume it.
    changed = {106, 125, 137, 140, 149, 209, 220, 256, 353, 366, 367, 432,
               517, 563, 583, 614, 646, 727}
    trinkets = player.get("trinket"), player.get("trinket_1")
    known_trinkets = all(_integer(value, 0, 131071) for value in trinkets)
    speed = player.get("move_speed")
    return (player["bombs"] > 0 and inventory is not None and not inventory & changed
            and _number(speed) and speed >= .8
            and type(player.get("giga_bombs")) is int and player["giga_bombs"] == 0
            and type(player.get("bomb_flags")) is int and player["bomb_flags"] == 0
            and player.get("unsafe_bomb_trinket") is False
            and known_trinkets and not {value & 32767 for value in trinkets} & {73, 133}
            and not any(item.get("kind") in ("bomb", "tnt") or
                        (item.get("kind") == "grid" and item.get("type") in (5, 12))
                        for item in state["hazards"]))


def _bomb_option(state, parsed, item, option):
    if not _ordinary_bombs(state):
        return None
    _, _, _, start, radius, _, bounds, _ = parsed
    sig, target = signature(item), _point(item)
    geometry = _geometry(state, radius, target=sig)
    boxes = geometry[0]
    if _reachable(start, target, bounds, geometry):
        return None  # Never spend a bomb when a walking route exists.
    rocks = [rock for rock in state["hazards"] if rock.get("kind") == "grid"
             and rock.get("type") == 2 and rock.get("collision") == 3]
    rocks.sort(key=lambda rock: (math.dist(start, _point(rock)) + math.dist(_point(rock), target), _grid_id(rock)))
    for rock in rocks[:8]:
        rock_id, center = _grid_id(rock), _point(rock)
        after = _geometry(state, radius, target=sig, remove_grid=rock_id)
        if not _reachable(start, target, bounds, after):
            continue
        stand_off = _radius(rock, 20) + radius + 10
        for dx, dy in sorted(_DOOR_DIRECTIONS, key=lambda vector:
                             math.dist(start, (center[0] + vector[0]*stand_off, center[1] + vector[1]*stand_off))):
            point = center[0]+dx*stand_off, center[1]+dy*stand_off
            if not _reachable(start, point, bounds, geometry):
                continue
            retreat_distance = BLAST_RADIUS + radius + BLAST_MARGIN + 20
            escape = point[0]+dx*retreat_distance, point[1]+dy*retreat_distance
            # A straight outward route makes time-to-safety bounded; a long
            # maze path being reachable would not prove the bomb fuse is safe.
            if (not _inside(escape, bounds) or not _free(escape, boxes)
                    or not _clear(point, escape, boxes)):
                continue
            kind, reserved, description = option
            cost = dict(reserved, bombs=1)
            return AdventureCandidate(f"bomb:{rock_id}:{item['id']}", "bomb_rock", item["id"],
                point, cost, f"Spend one bomb to open the rock route, retreat, then {description.lower()}",
                interaction="bomb", escape_point=escape, pickup_signature=sig,
                rock_id=rock_id, context=_context(state),
                details={"after_kind": kind, "rock_point": center, "target_point": target,
                         "blast_radius": BLAST_RADIUS, "escape_distance": retreat_distance})
    return None


def _active_choices(state):
    player, combat = state["player"], not state["room"]["clear"]
    active = player.get("active_item")
    maximum, charge = player.get("active_max_charge"), player.get("active_charge")
    if (not _integer(active, 1) or not _integer(maximum, 1, 10000)
            or not _integer(charge, maximum, 10000)):
        return []
    reason = None
    if active == 45 and player["can_pick_red_hearts"]:
        reason = "restore red health with Yum Heart"
    elif active == 78 and player["can_pick_soul_hearts"]:
        reason = "gain a soul heart with Book of Revelations; this may affect the floor's boss"
    elif active == 292 and player["can_pick_black_hearts"]:
        reason = "gain a black heart with Satanic Bible; this can change the boss reward into a devil deal"
    elif active in (85, 97, 102, 145, 288) and not combat:
        reason = "generate a card, pill, supply, or friendly helper with the held active item"
    # Enum 41 is Mom's Pad and 291 is Flush!, not Razor Blade/D4. D20 (166)
    # rerolls pickups and must never be offered as a generic combat effect.
    elif active in (34, 35, 39, 41, 58, 77, 93, 107, 145, 160, 164, 171,
                   192, 288, 291, 293, 298) and combat:
        reason = "use the held active item's known combat damage, protection, or helper effect"
    elif active == 105 and not combat and any(item["variant"] == 100 and item["subtype"] > 0
                                             and item["price"] == 0 for item in state["pickups"]):
        reason = "reroll the free item pedestal(s) with the D6; existing items will be replaced"
    if reason is None:
        return []
    details = {"name": player.get("active_name", ""),
               "description": player.get("active_description", "")}
    if active == 105:
        details["pedestals"] = sorted(signature(item) for item in state["pickups"]
                                       if item["variant"] == 100 and item["subtype"] > 0)
    return [AdventureCandidate(f"active:{active}", "use_active", str(active), None,
              {"active_charge": maximum}, reason, interaction="active", context=_context(state),
              details=details)]


def _pocket_choices(state):
    player, combat = state["player"], not state["room"]["clear"]
    card, pill = player.get("pocket_card"), player.get("pocket_pill")
    reason, target = None, None
    if _integer(card, 1) and pill == 0:
        # High Priestess (3) can stomp Isaac if no enemy remains. A room's
        # uncleared flag is not a bound stomp target, so exclude it here.
        if card in (2, 4, 8, 12, 13, 14, 16, 39, 50, 52, 54) and combat:
            reason = "use the known combat card for damage, protection, or a combat buff"
        elif not combat and card in (6, 9, 22, 35, 36, 38, 51, 53):
            reason = "use the known card for supplies, protection, helpers, or floor information"
        elif card in (7, 20, 26) and player["can_pick_red_hearts"]:
            reason = "use the known healing card while red health is missing"
        elif not combat and card in (23, 24, 25):
            resource = {23: "bombs", 24: "coins", 25: "keys"}[card]
            if player[resource] < 99:
                reason = f"use the known card to increase {resource}"
        target = f"card:{card}"
    elif _integer(pill, 1) and card == 0 and player.get("pill_known") is False:
        return [AdventureCandidate(f"pocket:pill:{pill}:unidentified", "use_pocket", f"pill:{pill}:unidentified", None,
            {"pocket": 1}, "Try this unidentified pill to learn its effect. It may help or harm, change stats, teleport, or alter the room; the effect is unknown until identified by the game.",
            interaction="pocket", context=_context(state),
            details={"name": "Unidentified pill", "identified": False, "color": pill})]
    elif _integer(pill, 1) and card == 0 and player.get("pill_known") is True:
        effect = player.get("pill_effect")
        if effect in (7, 10, 12, 14, 16, 18, 23, 33, 38, 48) and not combat:
            reason = "consume the identified beneficial stat, helper, or exploration pill"
        elif effect == 2 and player["can_pick_soul_hearts"]:
            reason = "consume identified Balls of Steel while soul-heart capacity is available"
        elif effect == 5 and player["can_pick_red_hearts"]:
            reason = "consume identified Full Health while red health is missing"
        elif effect in (24, 26, 28, 41) and combat:
            reason = "consume the identified combat-control or protection pill"
        elif effect == 20 and _integer(player.get("active_max_charge"), 1) and _integer(player.get("active_charge")) and player["active_charge"] < player["active_max_charge"]:
            reason = "consume identified 48 Hour Energy to recharge the active item"
        target = f"pill:{pill}:{effect}"
    if reason is None:
        return []
    return [AdventureCandidate(f"pocket:{target}", "use_pocket", target, None,
              {"pocket": 1}, reason, interaction="pocket", context=_context(state),
              details={"name": player.get("pocket_name", ""),
                       "description": player.get("pocket_description", "")})]


def candidates(state, *, rewards_done=False, allow_descend=False, limit=MAX_CANDIDATES):
    """Return up to eight currently eligible, context-bound model choices."""
    parsed = _parsed(state)
    if parsed is None:
        return ()
    player, room, _, start, radius, _, bounds, _ = parsed
    result = _active_choices(state) + _pocket_choices(state)
    if not room["clear"]:
        return tuple(result[:limit])
    if any(item.get("kind") == "bomb" for item in state["hazards"]):
        return tuple(result[:limit])
    from .props import prop_candidates
    from .curse_doors import curse_candidates
    items = sorted(state.get("pickups", []), key=lambda item: (math.dist(start, _point(item)), item["id"]))
    blocked = []
    for item in items:
        option = _pickup_option(item, player)
        if option is None:
            continue
        kind, cost, description = option
        sig = signature(item)
        if _reachable(start, _point(item), bounds, _geometry(state, radius, target=sig)):
            result.append(AdventureCandidate(f"{kind}:{item['id']}:{item['variant']}:{item['subtype']}",
                kind, item["id"], _point(item), cost, description, pickup_signature=sig,
                context=_context(state), details={"name": item.get("name", ""),
                "description": item.get("description", ""), "options_index": item["options_index"],
                "held_active": player["active_item"] if item["variant"] == 100 and item["collectible_kind"] == 3 else None,
                "replaces_active": item["variant"] == 100 and item["collectible_kind"] == 3 and player["active_item"] > 0}))
        else:
            blocked.append((item, option))
        if len(result) >= limit:
            return tuple(result[:limit])
    if player["keys"] > 0:
        from .exploration import _Door
        for door in state["doors"]:
            if (not door["locked"] or door["target_type"] not in (1, 2, 4)
                    or not 0 <= door["target_index"] <= 168
                    or door["target_index"] == state["floor"]["room_index"]):
                continue
            vector, point = _DOOR_DIRECTIONS[door["slot"] % 4], _point(door)
            approach = point[0]-vector[0]*40, point[1]-vector[1]*40
            chosen = _Door(door["slot"], point, door["target_index"], door["target_type"])
            if _reachable(start, approach, bounds, _geometry(state, radius, door=chosen)):
                target = f"door:{door['slot']}:{door['target_index']}"
                result.append(AdventureCandidate(f"unlock:{target}", "unlock_door", target,
                    point, {"keys": 1}, "Spend one key to unlock the observed " +
                    {1: "ordinary", 2: "shop", 4: "treasure"}[door["target_type"]] + " room door",
                    context=_context(state), details={"slot": door["slot"],
                    "target_index": door["target_index"], "target_type": door["target_type"]}))
    for item, option in blocked[:4]:
        if len(result) >= limit:
            break
        plan = _bomb_option(state, parsed, item, option)
        if plan is not None:
            result.append(plan)
    result.extend(prop_candidates(state))
    result.extend(curse_candidates(state))
    if allow_descend and rewards_done and room["type"] == 5:
        for grid in state["hazards"]:
            if grid.get("kind") != "grid" or grid.get("type") not in (17, 18):
                continue
            target = _grid_id(grid)
            if _reachable(start, _point(grid), bounds, _geometry(state, radius, remove_grid=target)):
                result.append(AdventureCandidate(f"descend:{target}", "descend", target, _point(grid), {},
                    "Enter the observed floor exit after collecting boss-room rewards", context=_context(state),
                    details={"grid_type": grid["type"]}))
    return tuple(result[:limit])


def candidate_geometry(state, candidate):
    """Return (bounds, boxes, phase) for this choice's local movement only.

    This does not grant permission to execute; call candidate_valid first.
    Door crossing uses the explorer's dedicated portal steering, not these
    interior bounds. Bomb routes retain the rock until it is observed gone.
    """
    parsed = _parsed(state)
    if parsed is None or candidate.context != _context(state):
        return None
    radius, bounds = parsed[4], parsed[6]
    door = None
    if candidate.kind == "unlock_door":
        from .exploration import _Door
        info = candidate.details
        door = _Door(info["slot"], candidate.point, info["target_index"], info["target_type"])
    geometry = _geometry(state, radius, target=candidate.pickup_signature,
                         remove_grid=candidate.target_id if candidate.kind == "descend" else None,
                         door=door)
    return bounds, geometry[0], geometry[3]


def candidate_valid(state, candidate, *, rewards_done=False, allow_descend=False):
    """Recheck fresh identity, inventory, target, cost and route before acting."""
    parsed = _parsed(state)
    if parsed is None or not isinstance(candidate, AdventureCandidate) or candidate.context != _context(state):
        return False
    player, room, _, start, radius, _, bounds, _ = parsed
    if candidate.kind == "shoot_prop":
        from .props import prop_valid
        return prop_valid(state, candidate)
    if candidate.kind in ("enter_curse", "leave_curse"):
        from .curse_doors import curse_valid
        return curse_valid(state, candidate)
    if candidate.kind in ("use_active", "use_pocket"):
        options = _active_choices(state) if candidate.kind == "use_active" else _pocket_choices(state)
        return any(option.key == candidate.key and option.cost == candidate.cost
                   and option.details.get("pedestals") == candidate.details.get("pedestals") for option in options)
    if not room["clear"] or not player["can_pickup_items"]:
        return False
    if candidate.kind in ("collect", "open_chest", "buy", "bomb_rock"):
        item = next((item for item in state["pickups"] if signature(item) == candidate.pickup_signature), None)
        if item is None:
            return False
        option = _pickup_option(item, player)
        if option is None:
            return False
        kind, cost, _ = option
        if candidate.kind != "bomb_rock":
            return (candidate.kind == kind and candidate.cost == cost
                    and (candidate.details.get("held_active") is None
                         or player["active_item"] == candidate.details["held_active"])
                    and item["options_index"] == candidate.details.get("options_index", item["options_index"])
                    and math.dist(candidate.point, _point(item)) <= 8
                    and _reachable(start, _point(item), bounds, _geometry(state, radius, target=signature(item))))
        if not _ordinary_bombs(state) or candidate.cost != dict(cost, bombs=1):
            return False
        rock = next((grid for grid in state["hazards"] if grid.get("kind") == "grid"
                     and grid.get("type") == 2 and grid.get("collision") == 3
                     and _grid_id(grid) == candidate.rock_id), None)
        if rock is None or candidate.escape_point is None:
            return False
        geometry = _geometry(state, radius, target=signature(item))
        boxes = geometry[0]
        after = _geometry(state, radius, target=signature(item), remove_grid=candidate.rock_id)
        return (math.dist(candidate.point, _point(rock)) < BLAST_RADIUS - 20
                and _reachable(start, candidate.point, bounds, geometry)
                and _reachable(start, _point(item), bounds, after)
                and not _reachable(start, _point(item), bounds, geometry)
                and _inside(candidate.escape_point, bounds) and _free(candidate.escape_point, boxes)
                and math.dist(candidate.point, candidate.escape_point) >= BLAST_RADIUS + radius + BLAST_MARGIN
                and _clear(candidate.point, candidate.escape_point, boxes))
    if candidate.kind == "unlock_door":
        from .exploration import _Door, _door_move
        info = candidate.details
        observed = next((door for door in state["doors"]
                         if door["slot"] == info["slot"] and door["target_index"] == info["target_index"]
                         and door["target_type"] == info["target_type"]), None)
        return (candidate.cost == {"keys": 1} and player["keys"] >= 1
                and observed is not None and observed["locked"] is True
                and _point(observed) == candidate.point
                and _door_move(state, parsed, _Door(info["slot"], candidate.point,
                               info["target_index"], info["target_type"])) is not None)
    if candidate.kind == "descend":
        if not allow_descend or not rewards_done or room["type"] != 5:
            return False
        grid = next((grid for grid in state["hazards"] if grid.get("kind") == "grid"
                     and grid.get("type") in (17, 18) and _grid_id(grid) == candidate.target_id), None)
        return (grid is not None and candidate.point == _point(grid)
                and _reachable(start, candidate.point, bounds, _geometry(state, radius, remove_grid=candidate.target_id)))
    return False
