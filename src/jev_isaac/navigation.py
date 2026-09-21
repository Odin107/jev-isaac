"""Small, local grounded navigator for ordinary tears in one observed room.

Jev chooses a target or an evade/hold goal. This module chooses only the next
direction from the current observation, with no action queue or model calls.
Grid cells are conservative squares; flight, spectral tears, beam shapes and
unobserved room geometry are deliberately outside this starter controller.
"""
from __future__ import annotations

import heapq
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from .combat import _entry, _inside, _number, _point


MAX_NODES = 1200
_STEP = 40.0
_DEADBAND = 7.0
_VECTORS = {"none": (0., 0.), "left": (-1., 0.), "right": (1., 0.),
            "up": (0., -1.), "down": (0., 1.),
            "up_left": (-2**-.5, -2**-.5), "up_right": (2**-.5, -2**-.5),
            "down_left": (-2**-.5, 2**-.5), "down_right": (2**-.5, 2**-.5)}


@dataclass(frozen=True)
class NavigationAction:
    move: str
    shoot: str
    override: str | None = None


_IDLE = NavigationAction("none", "none")


def _valid_enemy_id(value):
    return (isinstance(value, str) and bool(value.strip()) and len(value) <= 128
            and all(ord(char) >= 32 and ord(char) != 127 for char in value))


def _radius(item, default):
    value = item.get("radius", default)
    return float(value) if _number(value, limit=200) and value >= 0 else None


def _velocity(item):
    values = item.get("vx", 0), item.get("vy", 0)
    return tuple(float(v) for v in values) if all(_number(v, limit=1000) for v in values) else None


def _clear(start, end, boxes):
    """A tangential cell contact is legal, but cutting a corner is not."""
    distance = math.dist(start, end)
    if distance < 1e-6:
        return True
    direction = ((end[0]-start[0])/distance, (end[1]-start[1])/distance)
    xmin, xmax = sorted((start[0], end[0]))
    ymin, ymax = sorted((start[1], end[1]))
    for box in boxes:
        if xmax <= box[0] or xmin >= box[2] or ymax <= box[1] or ymin >= box[3]:
            continue
        if _entry(start, direction, box, distance) is not None:
            return False
    return True


def _free(point, boxes):
    x, y = point
    return not any(left+1e-6 < x < right-1e-6 and top+1e-6 < y < bottom-1e-6
                   for left, top, right, bottom in boxes)


def _shot(start, target, boxes):
    """Cardinal aim only when an ordinary tear has a clear, plausible lane."""
    dx, dy = target[0]-start[0], target[1]-start[1]
    # Leave margin inside the observed enemy radius; no inferred tear upgrades.
    tolerance = min(8.0, max(2.0, target[2]*.6))
    if abs(dx) <= tolerance and 60 <= abs(dy) <= 220:
        if _clear(start, (start[0], target[1]), boxes):
            return "down" if dy > 0 else "up"
    if abs(dy) <= tolerance and 60 <= abs(dx) <= 220:
        if _clear(start, (target[0], start[1]), boxes):
            return "right" if dx > 0 else "left"
    return "none"


def _axis(lower, upper, phase, extras):
    count = math.floor((upper-phase)/_STEP)-math.ceil((lower-phase)/_STEP)+1
    if count > MAX_NODES:
        return []
    first = math.ceil((lower-phase)/_STEP)
    return sorted({lower, upper, *extras,
                   *(phase+(first+i)*_STEP for i in range(max(0, count)))})


def _waypoint(start, target, bounds, movement_boxes, shot_boxes, phase, *, with_destination=False):
    """Shortest cardinal grid route to any reachable clear firing position.

    Current and target axes augment the 40-unit lattice, so a short alignment
    does not require walking to the center of a tile. At most 1200 vertices are
    considered, and every edge is collision checked with the player's radius.
    """
    xs = _axis(bounds[0], bounds[2], phase[0], (start[0], target[0]))
    ys = _axis(bounds[1], bounds[3], phase[1], (start[1], target[1]))
    # Enemy centers may be outside the inset player bounds.
    xs = [x for x in xs if bounds[0] <= x <= bounds[2]]
    ys = [y for y in ys if bounds[1] <= y <= bounds[3]]
    if not xs or not ys or len(xs)*len(ys) > MAX_NODES:
        return None
    width = len(xs)
    points = [(x, y) for y in ys for x in xs]
    available = [_free(p, movement_boxes) for p in points]
    origin = ys.index(start[1])*width+xs.index(start[0])
    if not available[origin]:
        return None
    distance, previous = {origin: 0.0}, {}
    queue = [(0.0, origin)]
    destination = None
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distance[node]:
            continue
        shot = _shot(points[node], target, shot_boxes)
        centered = (points[node][0] == target[0] if shot in ("up", "down")
                    else points[node][1] == target[1])
        if shot != "none" and (not with_destination or centered):
            destination = node
            break
        row, column = divmod(node, width)
        neighbors = []
        if column:
            neighbors.append(node-1)
        if column+1 < width:
            neighbors.append(node+1)
        if row:
            neighbors.append(node-width)
        if row+1 < len(ys):
            neighbors.append(node+width)
        for other in neighbors:
            if not available[other] or not _clear(points[node], points[other], movement_boxes):
                continue
            candidate = cost+math.dist(points[node], points[other])
            if candidate < distance.get(other, math.inf):
                distance[other], previous[other] = candidate, node
                heapq.heappush(queue, (candidate, other))
    if destination is None:
        return None
    if destination == origin:
        return (start, start) if with_destination else None
    route = [destination]
    while previous[route[-1]] != origin:
        route.append(previous[route[-1]])
    # Look ahead along a verified straight segment to avoid unnecessary turns.
    for node in route:
        if _clear(start, points[node], movement_boxes):
            return (points[node], points[destination]) if with_destination else points[node]
    return (points[route[-1]], points[destination]) if with_destination else points[route[-1]]


def _steer(start, waypoint, velocity, bounds, boxes, deadband=_DEADBAND, drift_frames=1.5):
    if waypoint is None:
        return "none"
    # A brief drift estimate releases the input before inertia carries it past
    # an alignment. This is deliberately much shorter than an inference cycle.
    predicted = (start[0]+velocity[0]*drift_frames, start[1]+velocity[1]*drift_frames)
    dx, dy = waypoint[0]-predicted[0], waypoint[1]-predicted[1]
    horizontal = ("right" if dx > 0 else "left") if abs(dx) > deadband else ""
    vertical = ("down" if dy > 0 else "up") if abs(dy) > deadband else ""
    name = f"{vertical}_{horizontal}" if horizontal and vertical else horizontal or vertical or "none"
    if name == "none":
        return name
    vector = _VECTORS[name]
    # Check the next short physical step, even if diagonal quantization differs
    # from the full line to the waypoint. Never squeeze across a blocked corner.
    length = min(12.0, math.dist(start, waypoint))
    end = start[0]+vector[0]*length, start[1]+vector[1]*length
    if _inside(end, bounds) and _clear(start, end, boxes):
        return name
    # A blocked diagonal can still have a safe cardinal component.
    for option in sorted((horizontal, vertical), key=lambda n: abs(dx) if n == horizontal else abs(dy), reverse=True):
        if option:
            vector = _VECTORS[option]
            end = start[0]+vector[0]*length, start[1]+vector[1]*length
            if _inside(end, bounds) and _clear(start, end, boxes):
                return option
    return "none"


def _recovery_options(start, velocity, radius, bounds, boxes, recoverable):
    """Legal exits from shallow conservative rock/poop padding overlap.

    A real observation can start inside a padded square after inertia or an
    enemy push. Permit only an outward ray, never a general obstacle exemption.
    Near a square's corner, the observed player center may even be just inside
    its conservative core; at most half a unit is allowed on either axis.
    """
    overlaps = [i for i, box in enumerate(boxes) if not _free(start, [box])]
    if not overlaps:
        return None
    constraints = []
    for index in overlaps:
        solid = recoverable.get(index)
        if solid is None:
            return None
        box = boxes[index]
        shallow = [(start[0]-box[0], (-1., 0.)), (box[2]-start[0], (1., 0.)),
                   (start[1]-box[1], (0., -1.)), (box[3]-start[1], (0., 1.))]
        shallow = [normal for depth, normal in shallow if depth <= min(4., radius*.5)]
        center = ((solid[0]+solid[2])/2, (solid[1]+solid[3])/2)
        corner_x = (start[0]-solid[2], 1.) if start[0] >= center[0] else (solid[0]-start[0], -1.)
        corner_y = (start[1]-solid[3], 1.) if start[1] >= center[1] else (solid[1]-start[1], -1.)
        corner = (corner_x[1], corner_y[1]) if min(corner_x[0], corner_y[0]) >= -min(.5, radius*.05) else None
        if not shallow and corner is None:
            return None
        constraints.append((solid, shallow, corner))
    others = [box for index, box in enumerate(boxes) if index not in overlaps]
    candidates = {}
    for name, vector in _VECTORS.items():
        if name == "none":
            continue
        for solid, shallow, corner in constraints:
            through_face = any(vector[0]*n[0]+vector[1]*n[1] > 0 for n in shallow)
            from_corner = corner is not None and vector[0]*corner[0] >= 0 and vector[1]*corner[1] >= 0
            if not (through_face or from_corner):
                break
            # Distance to a convex rectangle is nondecreasing along a ray
            # whose initial separation vector has a nonnegative dot product.
            separation = (start[0]-max(solid[0], min(start[0], solid[2])),
                          start[1]-max(solid[1], min(start[1], solid[3])))
            if separation[0]*vector[0]+separation[1]*vector[1] < 0:
                break
        else:
            end = start[0]+vector[0]*24, start[1]+vector[1]*24
            if _inside(end, bounds) and _free(end, boxes) and _clear(start, end, others):
                # Prefer a definite outward step when no threat chooses the
                # direction. Existing momentum is only a small tie-breaker.
                clearance = min(math.hypot(max(s[0]-end[0], 0, end[0]-s[2]),
                                          max(s[1]-end[1], 0, end[1]-s[3]))
                                for s, _, _ in constraints)
                candidates[name] = clearance+.05*(vector[0]*velocity[0]+vector[1]*velocity[1])
    return candidates


def _padding_escape(start, velocity, radius, bounds, boxes, recoverable):
    """Choose one outward recovery shared by clear-room and prop navigation.

    The caller identifies which exact obstacle entries are ordinary solid
    rocks or confirmed ordinary poop. Every other overlapping box remains an
    absolute blocker, including pickups, traps and identical-position hazards.
    """
    options = _recovery_options(start, velocity, radius, bounds, boxes, recoverable)
    return max(options, key=options.get) if options else None


def _avoid(start, velocity, radius, threats, bounds, boxes, intended, force=False, legal_moves=None):
    """Predict a few local frames, then pick a short legal escaping direction."""
    if not threats:
        return intended
    nearby = [t for t in threats if math.dist(start, t[:2]) < 120+math.hypot(t[3], t[4])*12]
    if not nearby:
        return intended

    def clearance(vector, drift=False):
        worst = math.inf
        final = math.inf
        # After the proposed 24-unit step, assume a stop while the threat keeps
        # moving. This favors leaving a bullet's line over briefly outrunning it.
        duration = 1.5 if drift else 6.0
        pvx, pvy = velocity if drift else (vector[0]*4, vector[1]*4)
        for x, y, size, vx, vy in nearby:
            rx, ry = start[0]-x, start[1]-y
            for dx, dy, length in ((pvx-vx, pvy-vy, duration), (-vx, -vy, 12-duration)):
                speed2 = dx*dx+dy*dy
                closest = max(0.0, min(length, -(rx*dx+ry*dy)/speed2)) if speed2 else 0.0
                worst = min(worst, math.hypot(rx+dx*closest, ry+dy*closest)-radius-size)
                rx, ry = rx+dx*length, ry+dy*length
            final = min(final, math.hypot(rx, ry)-radius-size)
        return worst, final

    if not force and clearance(_VECTORS[intended], intended == "none")[0] > 12:
        return intended
    candidates = []
    for name, vector in _VECTORS.items():
        end = start[0]+vector[0]*24, start[1]+vector[1]*24
        if legal_moves is not None and name not in legal_moves:
            continue
        if legal_moves is None and (not _inside(end, bounds) or not _clear(start, end, boxes)):
            continue
        worst, final = clearance(vector, name == "none")
        # Future minimum separation dominates; tie-break for opening distance.
        score = worst+final*.15+(1 if name == intended else 0)
        candidates.append((score, name))
    return max(candidates)[1] if candidates else "none"


def compute_action(state, goal_kind, target_id=None, *, fire_direction=None):
    """Return a fresh bounded direction pair, or neutral on incomplete data.

    Movement and shooting are independent. Evade/hold and invalidated engage
    targets may shoot at a freshly observed clear cardinal target, without
    pursuing it or altering their defensive movement. An engage target still
    determines pursuit and gets first preference when its firing lane is clear.
    The caller owns goal expiry, observation age, room identity and output timing.
    This function never mutates observations or keeps an obsolete action queue.
    """
    if not isinstance(state, Mapping) or goal_kind not in ("engage", "evade", "hold"):
        return _IDLE
    if fire_direction is not None and fire_direction not in ("none", "left", "right", "up", "down"):
        return _IDLE
    if goal_kind == "engage" and not _valid_enemy_id(target_id):
        return _IDLE
    player, room = state.get("player"), state.get("room")
    if (not isinstance(player, Mapping) or not isinstance(room, Mapping)
        or state.get("enabled") is not True or state.get("paused") is not False
        or player.get("dead") is not False or room.get("clear") is not False
        or state.get("truncated", False)):
        return _IDLE
    truncated = state.get("truncated_arrays", {})
    if not isinstance(truncated, Mapping) or any(truncated.get(k, False) for k in ("hazards", "enemies", "projectiles")):
        return _IDLE
    start, first, last = _point(player), _point(room.get("top_left")), _point(room.get("bottom_right"))
    radius, velocity = _radius(player, 10), _velocity(player)
    if None in (start, first, last, radius, velocity):
        return _IDLE
    bounds = (first[0]+radius, first[1]+radius, last[0]-radius, last[1]-radius)
    if not (0 < bounds[2]-bounds[0] <= 4000 and 0 < bounds[3]-bounds[1] <= 4000) or not _inside(start, bounds):
        return _IDLE
    hazards, enemies, projectiles = (state.get(k) for k in ("hazards", "enemies", "projectiles"))
    if any(not isinstance(items, list) or len(items) > limit
           for items, limit in ((hazards, 160), (enemies, 64), (projectiles, 96))):
        return _IDLE
    movement_boxes, shot_boxes, threats, recoverable = [], [], [], {}
    phase = (0., 0.)
    for item in hazards:
        if not isinstance(item, Mapping) or _point(item) is None or _radius(item, 20) is None:
            return _IDLE
        x, y = _point(item)
        half = _radius(item, 20)
        if item.get("kind") == "grid":
            collision = item.get("collision")
            if type(collision) is not int or not 0 <= collision <= 5:
                return _IDLE
            phase = x % _STEP, y % _STEP
            # Grid spikes are reported even when their collision class is zero.
            if collision or item.get("type") in (8, 9, 17, 18, 23):
                padding = half+radius
                if collision == 3 and item.get("type") in (2, 14):
                    recoverable[len(movement_boxes)] = (x-half, y-half, x+half, y+half)
                movement_boxes.append((x-padding, y-padding, x+padding, y+padding))
            if collision in (3, 4, 5):
                shot_boxes.append((x-half, y-half, x+half, y+half))
        else:
            motion = _velocity(item)
            if motion is None:
                return _IDLE
            threats.append((x, y, half, *motion))
            if item.get("kind") in ("tnt", "bomb"):
                shot_boxes.append((x-half-3, y-half-3, x+half+3, y+half+3))
            if item.get("kind") == "fire":
                padding = half+radius+4
                movement_boxes.append((x-padding, y-padding, x+padding, y+padding))
    target, opportunistic_shots = None, []
    enemy_ids = Counter(item.get("id") for item in enemies
                        if isinstance(item, Mapping) and _valid_enemy_id(item.get("id")))
    for is_enemy, item in [(True, item) for item in enemies]+[(False, item) for item in projectiles]:
        if not isinstance(item, Mapping):
            return _IDLE
        point, size, motion = _point(item), _radius(item, 10), _velocity(item)
        if None in (point, size, motion):
            return _IDLE
        if is_enemy:
            hp = item.get("hp")
            if not _number(hp) or hp <= 0:
                continue
            ident = item.get("id")
            if (_valid_enemy_id(ident) and enemy_ids[ident] == 1
                    and item.get("dead") is not True and item.get("vulnerable") is True):
                if goal_kind == "engage" and ident == target_id:
                    target = (*point, size)
                direction = _shot(start, (*point, size), shot_boxes)
                if direction != "none":
                    opportunistic_shots.append((math.dist(start, point), ident, direction))
        threats.append((*point, size, *motion))
    shoot = _shot(start, target, shot_boxes) if target is not None else "none"
    move = "none"
    if goal_kind == "engage" and target is not None and shoot == "none":
        waypoint = _waypoint(start, target, bounds, movement_boxes, shot_boxes, phase)
        deadband = min(_DEADBAND, max(2.0, target[2]*.6))
        if waypoint is not None and _shot(waypoint, target, shot_boxes) == "none":
            deadband = 2.0  # Do not stop short of a corner needed by the route.
        move = _steer(start, waypoint, velocity, bounds, movement_boxes, deadband)
    intended, override = move, None
    recovery = _recovery_options(start, velocity, radius, bounds, movement_boxes, recoverable)
    if recovery is not None:
        move = max(recovery, key=recovery.get) if recovery else "none"
        override = "leaving obstacle margin" if move != intended else None
    before_dodge = move
    move = _avoid(start, velocity, radius, threats, bounds, movement_boxes, move,
                  force=goal_kind == "evade" or recovery is not None, legal_moves=recovery)
    if move != before_dodge and goal_kind != "evade":
        override = "emergency collision avoidance"
    # Select fallback fire only after movement is finalized: an available shot
    # cannot cancel a dodge or redirect pursuit toward an unselected enemy.
    if shoot == "none" and opportunistic_shots:
        shoot = min(opportunistic_shots)[2]
    if fire_direction is not None:
        # The player policy owns the firing button, including angled shots
        # from inherited momentum. Do not secretly select or substitute aim.
        return NavigationAction(move, fire_direction, override)
    return NavigationAction(move, shoot)
