"""Fresh, bounded tear lanes for observed ordinary poop and fireplaces.

PoopGridEntityVariant.NORMAL=0, PoopState.DESTROYED=1000, and
FireplaceVariant.NORMAL/RED=0/1 follow the IsaacScript SDK's engine enums.
Only the mod's explicit tear_destructible flag enables a target. Drops are
never inferred: after disappearance/destruction, callers inspect new pickups.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .combat import _inside, _number, _point
from .navigation import NavigationAction, _VECTORS, _avoid, _clear, _free, _radius, _recovery_options, _shot, _steer, _velocity, _waypoint

MAX_PROPS = 4
_DIRECTIONS = {"left": (-1., 0.), "up": (0., -1.), "right": (1., 0.), "down": (0., 1.)}


@dataclass
class PropApproach:
    """One interaction's fixed firing point and currently verified segment."""
    destination: tuple[float, float] | None = None
    waypoint: tuple[float, float] | None = None
    pulse_frame: int | None = None


def _settle_move(state, start, waypoint, velocity, bounds, boxes, approach):
    # Recorded release drift is about 1.88 * velocity each observation while
    # velocity decays by .775, a remaining coast of about 8.4 * velocity.
    # Near the final point, use isolated correction pulses and observe their
    # decay. Reversing every fresh frame otherwise amplifies input latency.
    speed = math.hypot(*velocity)
    coast = (start[0]+velocity[0]*8.4, start[1]+velocity[1]*8.4)
    coast_safe = _inside(coast, bounds) and _clear(start, coast, boxes)
    if (approach.pulse_frame is not None and coast_safe
            and (speed > .5 or state["frame"]-approach.pulse_frame < 6)):
        return "none"
    if math.dist(start, waypoint) <= 6 and math.dist(coast, waypoint) <= 6 and speed <= .5 and coast_safe:
        return "none"
    move = _steer(start, waypoint, velocity, bounds, boxes, deadband=3, drift_frames=8.4)
    if move != "none":
        # A reverse input cannot erase current momentum. Check the initial
        # blended trajectory as well as the commanded direction's segment.
        vector = _VECTORS[move]
        end = (start[0]+velocity[0]*2+vector[0]*2,
               start[1]+velocity[1]*2+vector[1]*2)
        if not _inside(end, bounds) or not _clear(start, end, boxes):
            return None
        approach.pulse_frame = state["frame"]
    return move


def _int(value, lo=0, hi=2**31-1):
    return type(value) is int and lo <= value <= hi


def _identity(item):
    if item.get("kind") == "grid":
        if item.get("type") == 14 and _int(item.get("index"), 0, 100000) and _int(item.get("variant"), 0, 100):
            return f"poop:{item['index']}:14:{item['variant']}"
    elif item.get("kind") == "fire":
        ident = item.get("id")
        if (item.get("type") == 33 and isinstance(ident, str) and 0 < len(ident) <= 100
                and _int(item.get("variant"), 0, 100)):
            return f"fire:{ident}:33:{item['variant']}"
    return None


def _alive(item):
    if item.get("tear_destructible") is not True or _identity(item) is None:
        return False
    if item["kind"] == "grid":
        return (item["variant"] == 0 and item.get("collision") == 3
                and _int(item.get("state"), 0, 999))
    return (item["variant"] in (0, 1) and _number(item.get("hp")) and item["hp"] > 0
            and _number(item.get("max_hp")) and item["max_hp"] >= item["hp"])


def _parsed(state):
    from .adventure import _inventory_ids, _parsed as adventure_parsed
    parsed = adventure_parsed(state)
    if parsed is None or not state["room"]["clear"]:
        return None
    player = state["player"]
    if player.get("weapon_type") != 1 or type(player.get("weapon_type")) is not int:
        return None
    weapons = player.get("weapon_types")
    if weapons is not None and (not isinstance(weapons, list) or not weapons
                                or any(type(weapon) is not int or weapon != 1 for weapon in weapons)):
        return None
    inventory = _inventory_ids(player)
    # These known effects invalidate ordinary straight tears or can explode.
    # Other weapon classes need a separate aiming/prop-breaking implementation.
    unusual = {5, 52, 149, 168, 222, 233, 257, 329, 394, 401, 418, 561, 570, 572}
    if inventory is None or inventory & unusual:
        return None
    if any(h.get("kind") in ("bomb", "laser") for h in state["hazards"]):
        return None
    return parsed


def _dangerous(item):
    return (item.get("kind") in ("bomb", "tnt")
            or item.get("kind") == "grid" and
            (item.get("type") in (5, 12) or item.get("type") == 14 and item.get("variant") != 0))


def _box(item, padding=0):
    x, y = _point(item)
    radius = _radius(item, 20) + padding
    return x-radius, y-radius, x+radius, y+radius


def _geometry(state, parsed, selected):
    from .exploration import _DANGEROUS_GRIDS, _room_geometry
    radius, bounds = parsed[4], parsed[6]
    movement, _, recoverable, phase = _room_geometry(state, radius)
    # The clear-room geometry already identifies ordinary rock padding. Also
    # allow a strictly outward recovery from a confirmed normal poop's padding.
    # Retain the hazard-order mapping: matching boxes by position alone could
    # accidentally grant the same exemption to an overlapping fire or pickup.
    index = 0
    for item in state["hazards"]:
        if (item["kind"] == "grid" and not item["collision"]
                and item.get("type") not in _DANGEROUS_GRIDS):
            continue
        if (item["kind"] == "grid" and _alive(item)
                and index < len(movement) and movement[index] == _box(item, radius)):
            recoverable[index] = _box(item)
        index += 1
    for door in state["doors"]:
        x, y = _point(door)
        pad = radius + 28
        movement.append((x-pad, y-pad, x+pad, y+pad))
    shot_boxes, dangers = [], []
    for item in state["hazards"]:
        if _identity(item) == _identity(selected):
            continue
        kind = item.get("kind")
        if (kind == "grid" and item.get("collision") in (3, 4, 5)) or kind in ("fire", "tnt"):
            shot_boxes.append(_box(item, 3))
        if _dangerous(item):
            dangers.append(_box(item, 12))
    # Prop clearing does not authorize walking over any unselected pickup,
    # including option siblings, paid goods, cards, and even free supplies.
    for pickup in state.get("pickups", []):
        movement.append(_box(pickup, radius+8))
        shot_boxes.append(_box(pickup, 8))
        dangers.append(_box(pickup, 12))
    for enemy in state["enemies"]:
        if enemy["hp"] > 0:
            shot_boxes.append(_box(enemy, 6))
    # A tear may continue after destroying the prop. Block a firing side if
    # its continuation could hit an explosive/unknown poop or another pickup.
    # Virtual blockers sit immediately before the target from that side; this
    # also makes _waypoint search for a different safe cardinal firing lane.
    tx, ty = _point(selected)
    half = _radius(selected, 20)
    for name, (dx, dy) in _DIRECTIONS.items():
        edge = ((bounds[0]-radius if dx < 0 else bounds[2]+radius) if dx else tx,
                (bounds[1]-radius if dy < 0 else bounds[3]+radius) if dy else ty)
        if _clear((tx, ty), edge, dangers):
            continue
        before = tx-dx*(half+6), ty-dy*(half+6)
        if dx:
            shot_boxes.append((before[0]-2, ty-16, before[0]+2, ty+16))
        else:
            shot_boxes.append((tx-16, before[1]-2, tx+16, before[1]+2))
    return bounds, movement, shot_boxes, phase, recoverable


def _action(state, parsed, selected, approach=None):
    from .exploration import _point_waypoint
    approach = approach if approach is not None else PropApproach()
    start, radius, velocity = parsed[3], parsed[4], parsed[5]
    bounds, movement, shot_boxes, phase, recoverable = _geometry(state, parsed, selected)
    if not _inside(start, bounds):
        return None
    target = (*_point(selected), _radius(selected, 20))
    threats = [(*_point(item), _radius(item, 10), *_velocity(item))
               for item in state["projectiles"]]
    threats.extend((*_point(item), _radius(item, 20), *_velocity(item))
                   for item in state["enemies"] if item["hp"] > 0)
    if not _free(start, movement):
        recovery = _recovery_options(start, velocity, radius, bounds, movement, recoverable)
        if not recovery:
            return None
        # A recoverable overlap does not establish that this prop is reachable.
        # Retain only outward steps that lead to a freshly verified firing route.
        reachable = {}
        for name, score in recovery.items():
            vector = _VECTORS[name]
            end = start[0]+vector[0]*24, start[1]+vector[1]*24
            if (_shot(end, target, shot_boxes) != "none"
                    or _waypoint(end, target, bounds, movement, shot_boxes, phase) is not None):
                reachable[name] = score
        if not reachable:
            return None
        move = _avoid(start, velocity, radius, threats, bounds, movement,
                      max(reachable, key=reachable.get), force=True, legal_moves=reachable)
        return NavigationAction(move, "none")
    if approach.destination is None:
        route = _waypoint(start, target, bounds, movement, shot_boxes, phase, with_destination=True)
        if route is None:
            return None
        approach.waypoint, approach.destination = route
    destination = approach.destination
    # Do not silently switch firing sides as the player moves. Fresh obstacles,
    # pickups or a dangerous tear continuation invalidate this fixed position.
    if (not _inside(destination, bounds) or not _free(destination, movement)
            or _shot(destination, target, shot_boxes) == "none"):
        return None
    waypoint = approach.waypoint
    if waypoint is None or math.dist(start, waypoint) <= 4:
        waypoint = _point_waypoint(start, destination, bounds, movement, phase)
        if waypoint is None:
            return None
        approach.waypoint = waypoint
    if not _free(waypoint, movement) or not _clear(start, waypoint, movement):
        return None
    final_segment = waypoint == destination and math.dist(start, destination) <= 36
    if final_segment:
        move = _settle_move(state, start, destination, velocity, bounds, movement, approach)
        if move is None:
            return None
    else:
        move = _steer(start, waypoint, velocity, bounds, movement, deadband=2, drift_frames=4)
    shoot = _shot(start, target, shot_boxes)
    if (move != "none" or math.dist(start, destination) > 6
            or math.hypot(*velocity) > .5
            or approach.pulse_frame is not None and state["frame"]-approach.pulse_frame < 6
            or _shot((start[0]+velocity[0]*8.4, start[1]+velocity[1]*8.4), target, shot_boxes) != shoot):
        shoot = "none"
    if threats:
        move = _avoid(start, velocity, radius, threats, bounds, movement, move)
        if move != "none":
            shoot = "none"
    return NavigationAction(move, shoot)


def prop_candidates(state):
    """Up to four reachable live props; selecting one never promises a drop."""
    from .adventure import AdventureCandidate, _context
    parsed = _parsed(state)
    if parsed is None:
        return ()
    targets = [item for item in state["hazards"] if _alive(item)]
    identities = [_identity(item) for item in targets]
    targets = [item for item in targets if identities.count(_identity(item)) == 1]
    targets.sort(key=lambda item: (math.dist(parsed[3], _point(item)), _identity(item)))
    result = []
    for item in targets[:8]:
        if _action(state, parsed, item) is None:
            continue
        identity = _identity(item)
        label = "ordinary poop" if item["kind"] == "grid" else "red fireplace" if item["variant"] == 1 else "ordinary fireplace"
        result.append(AdventureCandidate(f"shoot:{identity}", "shoot_prop", identity,
            _point(item), {}, f"Shoot the observed {label} from a clear lane, then check for any actual drops",
            context=_context(state), details={"prop_kind": item["kind"], "type": item["type"],
            "variant": item["variant"], "grid_index": item.get("index"), "entity_id": item.get("id")}))
        if len(result) >= MAX_PROPS:
            break
    return tuple(result)


def _selected(state, candidate):
    from .adventure import AdventureCandidate, _context
    if (not isinstance(candidate, AdventureCandidate) or candidate.kind != "shoot_prop"
            or candidate.context != _context(state) or candidate.interaction != "none"
            or candidate.cost != {}):
        return None
    matches = [item for item in state["hazards"] if _identity(item) == candidate.target_id]
    if len(matches) != 1 or not _alive(matches[0]) or _point(matches[0]) != candidate.point:
        return None
    return matches[0]


def prop_valid(state, candidate):
    parsed = _parsed(state)
    if parsed is None:
        return False
    selected = _selected(state, candidate)
    return selected is not None and _action(state, parsed, selected) is not None


def prop_action(state, candidate, approach=None):
    """One fresh direction/shot pair, or None if gone, unsupported or unreachable."""
    parsed = _parsed(state)
    if parsed is None:
        return None
    selected = _selected(state, candidate)
    return _action(state, parsed, selected, approach) if selected is not None else None


def prop_damage(state, candidate):
    """Only the selected identity's monotone destruction/HP evidence counts."""
    selected = _selected(state, candidate)
    if selected is None:
        return None
    return selected["state"] if selected["kind"] == "grid" else -selected["hp"]


def prop_finished(state, candidate):
    """Complete fresh observations may prove disappearance or destruction."""
    from .adventure import _context
    if _parsed(state) is None or candidate.context != _context(state):
        return False
    matches = [item for item in state["hazards"] if _identity(item) == candidate.target_id]
    if not matches:
        # A malformed/changed identity at the same point is ambiguous, not
        # evidence that the original prop was successfully cleared.
        return not any(_point(item) == candidate.point for item in state["hazards"])
    if len(matches) != 1 or _point(matches[0]) != candidate.point:
        return False
    selected = matches[0]
    if selected["kind"] == "grid":
        return (_int(selected.get("state"), 1000, 1000) and selected.get("collision") == 0
                and selected.get("tear_destructible") is False)
    return (_number(selected.get("hp")) and selected["hp"] == 0
            and selected.get("tear_destructible") is False)
