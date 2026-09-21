"""Bounded observed geometry for tear-triggered TNT puzzle demolition.

This is an engineering risk policy, not an explosion simulation. A 105-unit
chain estimate plus observed body sizes links possible secondary explosions.
Retreat favors the widest reachable clearance even when a crowded room offers
no point beyond that estimate. No fuse timing, cover immunity or damage-free
outcome is inferred. Actual movement retains every current obstacle.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

from .combat import _inside, _number, _point
from .exploration import _DANGEROUS_GRIDS, _point_waypoint, _validated
from .navigation import MAX_NODES, _axis, _clear, _free, _radius, _velocity
from .protocol import valid_switches


MIN_STANDOFF = 140.0
MAX_TEAR_DISTANCE = 220.0
CHAIN_ESTIMATE = 105.0
MAX_EXPLOSIVES = 32
FIRING_RESERVE = 8.0
RETREAT_RESERVE = 8.0


@dataclass(frozen=True)
class TntPlan:
    target_index: int
    target_point: tuple[float, float]
    firing_point: tuple[float, float]
    retreat_point: tuple[float, float]
    shoot: str
    chain_points: tuple[tuple[float, float], ...]


def _ordinary(item):
    return (isinstance(item, Mapping) and item.get("kind") == "grid"
            and type(item.get("type")) is int and item["type"] == 12
            and type(item.get("variant")) is int and item["variant"] == 0
            and type(item.get("collision")) is int and item["collision"] == 2
            and type(item.get("state")) is int and 0 <= item["state"] <= 3
            and type(item.get("index")) is int and 0 <= item["index"] <= 4095
            and _point(item) is not None and _radius(item, 20) is not None)


def _explosive(item):
    return (item.get("kind") == "tnt" or item.get("kind") == "grid"
            and item.get("type") in (5, 12) and item.get("collision") != 0)


def _range(state):
    value = state.get("player", {}).get("tear_range")
    return min(MAX_TEAR_DISTANCE, float(value)) if _number(value) and value >= MIN_STANDOFF else None


def _selected(state, target):
    if isinstance(target, TntPlan):
        index, point = target.target_index, target.target_point
    elif type(target) is int:
        index, point = target, None
    elif isinstance(target, Mapping):
        index, point = target.get("index"), _point(target)
    else:
        return None
    matches = [item for item in state.get("hazards", [])
               if item.get("kind") == "grid" and item.get("index") == index]
    if len(matches) != 1 or not _ordinary(matches[0]):
        return None
    return matches[0] if point is None or _point(matches[0]) == point else None


def shot_direction(state, target, start):
    """A fresh ordinary-TNT target, cardinal line, range and blocker check."""
    selected = _selected(state, target)
    maximum = _range(state)
    if selected is None or maximum is None or not all(_number(v) for v in start):
        return "none"
    tx, ty = _point(selected)
    dx, dy = tx-start[0], ty-start[1]
    if abs(dy) <= 8 and MIN_STANDOFF <= abs(dx) <= maximum:
        direction, end = ("right" if dx > 0 else "left"), (tx, start[1])
    elif abs(dx) <= 8 and MIN_STANDOFF <= abs(dy) <= maximum:
        direction, end = ("down" if dy > 0 else "up"), (start[0], ty)
    else:
        return "none"
    boxes = []
    for item in state["hazards"]:
        if item is selected:
            continue
        kind = item.get("kind")
        if kind in ("bomb", "laser"):
            return "none"
        if (kind == "grid" and item.get("collision") in (2, 3, 4, 5)
                or kind in ("tnt", "fire")):
            x, y = _point(item)
            half = _radius(item, 20)+3
            boxes.append((x-half, y-half, x+half, y+half))
    for item in state.get("pickups", []):
        x, y = _point(item)
        half = _radius(item, 10)+8
        boxes.append((x-half, y-half, x+half, y+half))
    if any(item.get("hp", 0) > 0 for item in state.get("enemies", [])):
        return "none"
    return direction if _clear(start, end, boxes) else "none"


def _target_boxes(state, boxes, radius):
    """Map hazard identities to their exact entries, retaining coincident boxes."""
    result, index = {}, 0
    for item in state["hazards"]:
        if (item["kind"] == "grid" and not item["collision"]
                and item.get("type") not in _DANGEROUS_GRIDS):
            continue
        if _ordinary(item):
            x, y = _point(item)
            half = _radius(item, 20)+radius
            expected = (x-half, y-half, x+half, y+half)
            if index < len(boxes) and boxes[index] == expected:
                result[item["index"]] = index
        index += 1
    return result


def _chain(selected, explosives):
    points = {_point(selected)}
    remaining = list(explosives)
    while remaining:
        linked = [item for item in remaining
                  if any(math.dist(_point(item), point) <= CHAIN_ESTIMATE+_radius(item, 20)
                         for point in points)]
        if not linked:
            break
        points.update(_point(item) for item in linked)
        remaining = [item for item in remaining if item not in linked]
    return tuple(sorted(points))


def _retreat_points(start, bounds, boxes, phase, explosives):
    # The settling controller tolerates six units of residual position error.
    # Targeting the exact player-inset boundary caused a live retreat to stop
    # on harmless residual drift; reserve space for settling on every side.
    retreat_bounds = (bounds[0]+RETREAT_RESERVE, bounds[1]+RETREAT_RESERVE,
                      bounds[2]-RETREAT_RESERVE, bounds[3]-RETREAT_RESERVE)
    if retreat_bounds[0] > retreat_bounds[2] or retreat_bounds[1] > retreat_bounds[3]:
        return []
    xs = _axis(retreat_bounds[0], retreat_bounds[2], phase[0], (start[0],))
    ys = _axis(retreat_bounds[1], retreat_bounds[3], phase[1], (start[1],))
    if not xs or not ys or len(xs)*len(ys) > MAX_NODES:
        return []
    points = [(x, y) for x in xs for y in ys
              if _inside((x, y), retreat_bounds) and _free((x, y), boxes)]
    points.sort(key=lambda point: (-min(math.dist(point, _point(item)) for item in explosives),
                                  math.dist(start, point), point))
    return points


def plan_demolition(state, switch_point, boxes, phase, *, target_index=None):
    """Find a needed, reachable firing position and widest available retreat.

    A single-target removal must first open the switch route. Only if none do,
    permit a conservative linked-component removal model. Neither model grants
    movement through a barrel before a later observation shows it disappeared.
    """
    parsed = _validated(state)
    if (parsed is None or not state["enabled"] or state["paused"] or state["player"]["dead"]
            or state["room"]["clear"] or state["room"].get("has_trigger_pressure_plates") is not True
            or not valid_switches(state.get("switches")) or _range(state) is None
            or state["projectiles"] or any(h["kind"] in ("bomb", "laser") for h in state["hazards"])
            or any(enemy["hp"] > 0 for enemy in state["enemies"])):
        return None
    if not any(row.get("variant") == 0 and type(row.get("variant")) is int
               and row.get("state") == 0 and type(row.get("state")) is int
               and row["collision"] == 0 and _point(row) == tuple(switch_point)
               for row in state["switches"]):
        return None
    start, radius, _, bounds = parsed[3:7]
    if (not _inside(start, bounds) or not _inside(switch_point, bounds) or not _free(start, boxes)
            or _point_waypoint(start, switch_point, bounds, boxes, phase) is not None):
        return None
    explosives = [item for item in state["hazards"] if _explosive(item)]
    targets = [item for item in explosives if _ordinary(item)]
    indices = [item["index"] for item in targets]
    if not targets or len(explosives) > MAX_EXPLOSIVES or len(indices) != len(set(indices)):
        return None
    if any(math.hypot(*_velocity(item)) > .5 for item in explosives):
        return None
    mapping = _target_boxes(state, boxes, radius)
    if any(item["index"] not in mapping for item in targets):
        return None
    chains = {item["index"]: _chain(item, explosives) for item in targets}

    def opens_route(removed):
        hypothetical = [box for index, box in enumerate(boxes) if index not in removed]
        return _point_waypoint(start, switch_point, bounds, hypothetical, phase) is not None

    needed = [item for item in targets if opens_route({mapping[item["index"]]})]
    if not needed:
        # This fallback models only destruction of observed ordinary grid TNT.
        # Movable TNT and other explosive types still block hypothetical walks.
        needed = [item for item in targets if opens_route({mapping[other["index"]] for other in targets
                                                          if _point(other) in chains[item["index"]]})]
    if not needed:
        return None
    if target_index is not None:
        needed = [item for item in needed if item["index"] == target_index]
        if not needed:
            return None
    retreat_points = _retreat_points(start, bounds, boxes, phase, explosives)
    retreats = []
    for point in retreat_points:
        if _point_waypoint(start, point, bounds, boxes, phase) is not None:
            retreats.append(point)
            if len(retreats) == 4:
                break
    if not retreats:
        return None
    maximum = _range(state)
    # The shared settling controller accepts a six-unit residual error. Keep
    # extra room on both range boundaries so that settling cannot invalidate
    # an otherwise clear firing lane or violate the hard minimum stand-off.
    minimum_firing = MIN_STANDOFF+FIRING_RESERVE
    maximum_firing = maximum-FIRING_RESERVE
    if minimum_firing > maximum_firing:
        return None
    distances = sorted({minimum_firing, maximum_firing,
                        *(float(d) for d in range(160, 221, 20) if minimum_firing <= d <= maximum_firing)},
                       reverse=True)
    choices = []
    for target in needed:
        tx, ty = _point(target)
        for distance in distances:
            for vx, vy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                firing = tx+vx*distance, ty+vy*distance
                if not _inside(firing, bounds) or not _free(firing, boxes):
                    continue
                shoot = shot_direction(state, target, firing)
                if (shoot == "none" or _point_waypoint(start, firing, bounds, boxes, phase) is None):
                    continue
                # Retain widest available clearance, with short travel as a
                # tie-breaker; no guessed fuse or guaranteed immunity is used.
                for retreat in retreats:
                    if _point_waypoint(firing, retreat, bounds, boxes, phase) is None:
                        continue
                    clearance = min(math.dist(retreat, _point(item)) for item in explosives)
                    score = (-clearance, -distance, math.dist(start, firing)+math.dist(firing, retreat),
                             target["index"], firing, retreat)
                    choices.append((score, TntPlan(target["index"], (tx, ty), firing, retreat,
                                                  shoot, chains[target["index"]])))
                    break
    return min(choices, key=lambda item: item[0])[1] if choices else None
