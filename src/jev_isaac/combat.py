"""Bounded grid-geometry hints; these describe options and never select actions.

The engine-shipped enums.lua defines PIT=1, SOLID=3, WALL=4 and
WALL_EXCEPT_PLAYER=5; its bullet collision mode excludes pits. We conservatively
treat every nonzero grid collision as blocking ground movement. Square cells,
grounded movement and ordinary tears are estimates, not the engine's physics.
Actual tear range, flight, spectral tears and future enemy motion are unknown.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from .protocol import MAX_HAZARDS


_EPS = 1e-6
_PROBE = 36.0
_DIRECTIONS = {
    "none": (0, 0), "left": (-1, 0), "right": (1, 0),
    "up": (0, -1), "down": (0, 1), "up_left": (-1, -1),
    "up_right": (1, -1), "down_left": (-1, 1), "down_right": (1, 1),
}


def _number(value, *, limit=1_000_000):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and -limit <= value <= limit and math.isfinite(value))


def _point(value):
    if isinstance(value, Mapping) and _number(value.get("x")) and _number(value.get("y")):
        return float(value["x"]), float(value["y"])
    return None


def _inside(point, bounds):
    x, y = point
    left, top, right, bottom = bounds
    return left - _EPS <= x <= right + _EPS and top - _EPS <= y <= bottom + _EPS


def _entry(start, direction, box, length):
    """Distance to the box interior, excluding a tangential boundary contact."""
    enter, leave = -math.inf, math.inf
    for pos, delta, lower, upper in zip(start, direction, box[:2], box[2:]):
        if abs(delta) < _EPS:
            if pos <= lower + _EPS or pos >= upper - _EPS:
                return None
            continue
        a, b = (lower - pos) / delta, (upper - pos) / delta
        enter, leave = max(enter, min(a, b)), min(leave, max(a, b))
    if leave <= max(0.0, enter) + _EPS or enter >= length - _EPS:
        return None
    return max(0.0, enter)


def _path(start, end, obstacles, padding=0.0):
    distance = math.dist(start, end)
    if distance < _EPS:
        return []
    direction = ((end[0] - start[0]) / distance, (end[1] - start[1]) / distance)
    hits = []
    for item in obstacles:
        x, y, radius = item["x"], item["y"], item["radius"] + padding
        entry = _entry(start, direction, (x-radius, y-radius, x+radius, y+radius), distance)
        if entry is not None:
            hits.append((entry, item["index"]))
    return sorted(hits)


def build_combat_context(state, *, target_limit=8):
    """Return compact, finite facts from bounded enemies and complete geometry.

    Waypoints are at most four direct, unobstructed alignments with the nearest
    eight vulnerable enemies. They are not routes or commands. Empty/malformed
    required geometry returns no hints; raw observations remain the caller's job.
    """
    if not isinstance(state, Mapping):
        return {}
    player, room = state.get("player"), state.get("room")
    if not isinstance(player, Mapping) or not isinstance(room, Mapping):
        return {}
    start = _point(player)
    first, last = _point(room.get("top_left")), _point(room.get("bottom_right"))
    radius = player.get("radius", 10.0)
    if start is None or first is None or last is None or not _number(radius, limit=10_000) or radius < 0:
        return {}
    bounds = (first[0]+radius, first[1]+radius, last[0]-radius, last[1]-radius)
    if bounds[0] > bounds[2] or bounds[1] > bounds[3] or not _inside(start, bounds):
        return {}

    raw_hazards = state.get("hazards")
    complete = isinstance(raw_hazards, list) and len(raw_hazards) <= MAX_HAZARDS and not state.get("truncated", False)
    obstacles = []
    for offset, item in enumerate(raw_hazards[:MAX_HAZARDS] if isinstance(raw_hazards, list) else []):
        if not isinstance(item, Mapping) or item.get("kind") != "grid":
            continue
        point, collision, half = _point(item), item.get("collision"), item.get("radius", 20)
        if point is None or not isinstance(collision, int) or isinstance(collision, bool) or not _number(half, limit=10_000) or half <= 0:
            complete = False
            continue
        if collision:
            index = item.get("index", offset)
            if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index <= 1_000_000:
                index = offset
            obstacles.append({"x": point[0], "y": point[1], "radius": half,
                              "collision": collision, "index": index})
    shot_obstacles = [item for item in obstacles if item["collision"] in (3, 4, 5)]
    moves = {}
    for name, vector in _DIRECTIONS.items():
        if name == "none":
            moves[name] = {"clearance": 0.0, "probe_clear": True}
            continue
        size = math.hypot(*vector)
        direction = vector[0]/size, vector[1]/size
        clearance = _PROBE
        for pos, delta, lower, upper in zip(start, direction, bounds[:2], bounds[2:]):
            if abs(delta) > _EPS:
                clearance = min(clearance, max(0.0, ((upper if delta > 0 else lower)-pos)/delta))
        end = start[0]+direction[0]*_PROBE, start[1]+direction[1]*_PROBE
        hits = _path(start, end, obstacles, radius)
        if hits:
            clearance = min(clearance, hits[0][0])
        moves[name] = {"clearance": round(clearance, 2), "probe_clear": clearance >= _PROBE-_EPS}

    enemies = []
    raw_enemies = state.get("enemies", [])
    for offset, enemy in enumerate(raw_enemies[:64] if isinstance(raw_enemies, list) else []):
        if (not isinstance(enemy, Mapping) or enemy.get("vulnerable") is not True
                or enemy.get("dead") is True):
            continue
        pos, hp = _point(enemy), enemy.get("hp")
        if pos is None or not _number(hp) or hp <= 0:
            continue
        target_radius = enemy.get("radius", 0)
        if not _number(target_radius, limit=10_000) or target_radius < 0:
            target_radius = 0
        ident = enemy.get("id")
        ident = ident[:128] if isinstance(ident, str) else str(offset)
        enemies.append((math.dist(start, pos), offset, ident, pos, target_radius))
    enemies.sort()
    targets, waypoints = [], []
    for distance, _, ident, pos, target_radius in enemies[:target_limit]:
        dx, dy = pos[0]-start[0], pos[1]-start[1]
        lanes = {}
        for name, offset, ahead, end in (
                ("horizontal", dy, dx, (pos[0], start[1])),
                ("vertical", dx, dy, (start[0], pos[1]))):
            direction = (("right" if ahead > 0 else "left") if name == "horizontal"
                         else ("down" if ahead > 0 else "up")) if abs(ahead) > _EPS else "none"
            lanes[name] = {"direction": direction, "offset": round(abs(offset), 2),
                           "aligned": direction != "none" and abs(offset) <= target_radius,
                           "blockers": [index for _, index in _path(start, end, shot_obstacles)][:8]}
        targets.append({"id": ident, "dx": round(dx, 2), "dy": round(dy, 2),
                        "distance": round(distance, 2), "range_unknown": True, "lanes": lanes})
        if not complete:
            continue
        for point, shoot in (((pos[0], start[1]), "down" if dy > 0 else "up"),
                             ((start[0], pos[1]), "right" if dx > 0 else "left")):
            separation, travel = math.dist(point, pos), math.dist(start, point)
            if separation <= 60 or travel < _EPS or not _inside(point, bounds):
                continue
            if _path(start, point, obstacles, radius) or _path(point, pos, shot_obstacles):
                continue
            # A zero-length/tangent endpoint must not lie inside a padded obstacle.
            if any(abs(point[0]-cell["x"]) < cell["radius"]+radius-_EPS
                   and abs(point[1]-cell["y"]) < cell["radius"]+radius-_EPS for cell in obstacles):
                continue
            # Keep candidate coordinates exact: rounding could cross a boundary.
            waypoints.append({"x": point[0], "y": point[1],
                              "target_id": ident, "shoot": shoot,
                              "move_distance": round(travel, 2),
                              "target_distance": round(separation, 2), "range_unknown": True})
    waypoints.sort(key=lambda item: (item["move_distance"], item["target_distance"], item["target_id"]))
    return {"estimated_grid_geometry": True, "grid_complete": complete,
            "assumptions": "Conservative grounded movement; ordinary straight tears; range unknown. Grid-only estimates exclude moving threats.",
            "move_probe_distance": _PROBE, "moves": moves, "targets": targets,
            "firing_positions": waypoints[:4]}


def current_firing_view(context):
    """Group current enemy lanes by button, without ranking or choosing one.

    Future waypoints and previous controls cannot contribute to this view.
    Empty clear-lane lists are meaningful only with complete grid geometry.
    These are straight-line estimates, not predictions of weapon hits.
    """
    if not context:
        return {"geometry_available": False, "directions": {}}
    complete = context["grid_complete"]
    directions = {direction: {"enemies_on_side": [], "aligned_enemies": [],
                              "grid_clear_aligned_enemies": [] if complete else None}
                  for direction in ("left", "right", "up", "down")}
    for target in context["targets"]:
        for lane in target["lanes"].values():
            row = directions.get(lane["direction"])
            if row is None:
                continue
            row["enemies_on_side"].append(target["id"])
            if lane["aligned"]:
                row["aligned_enemies"].append(target["id"])
                if complete and not lane["blockers"]:
                    row["grid_clear_aligned_enemies"].append(target["id"])
    return {"geometry_available": True, "grid_complete": complete,
            "directions": directions}
