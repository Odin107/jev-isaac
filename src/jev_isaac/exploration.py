"""Bounded exploration of observed, open ordinary doors on one floor.

This only steers between cleared rooms. Combat stays with the Jev goal loop.
The map contains doors actually reported by the mod, not an inferred floor
layout. Shops, deals, damage doors, locks and floor exits are never goals.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import copy
from dataclasses import dataclass
import heapq
import math

from .combat import _inside, _number, _point
from .navigation import MAX_NODES, _VECTORS, _axis, _avoid, _clear, _free, _padding_escape, _radius, _steer, _velocity
from .pickups import PickupCollector, exclusion_boxes, signature, valid_pickups
from .protocol import MAX_HAZARDS, valid_visited_rooms


# RoomType and DoorSlot in the Isaac Lua API. The second set of door slots is
# used by larger rooms; its four directions have the same order.
_ALLOWED_TYPES = frozenset((1, 4, 5))  # ordinary, treasure, boss
_BOSS = 5
_DIRECTIONS = ((-1., 0.), (0., -1.), (1., 0.), (0., 1.))
_DIRECTION_NAMES = ("left", "up", "right", "down")
_DANGEROUS_GRIDS = frozenset((8, 9, 17, 18, 23))


@dataclass(frozen=True)
class ExplorationAction:
    move: str = "none"
    shoot: str = "none"
    stop_reason: str | None = None
    status: str | None = None


@dataclass(frozen=True)
class _Door:
    slot: int
    point: tuple[float, float]
    target_index: int
    target_type: int


def _integer(value, lower, upper):
    return type(value) is int and lower <= value <= upper


def _identity(value):
    return isinstance(value, str) and 0 < len(value) <= 256


def _visited_rooms(state):
    """Strict optional facts, independent of protocol decoding in direct callers."""
    rows = state.get("visited_rooms", [])
    if not valid_visited_rooms(rows):
        return None
    return [(row, tuple(row.get("room_indices", [row["room_index"]]))) for row in rows]


def _validated(state, *, allow_shop=False, allow_secret=False):
    """Validate the complete observation subset before changing the map."""
    if not isinstance(state, Mapping):
        return None
    player, room, floor = (state.get(k) for k in ("player", "room", "floor"))
    if not all(isinstance(v, Mapping) for v in (player, room, floor)):
        return None
    if (not all(_identity(state.get(k)) for k in ("session", "run_id", "room_id"))
            or not _identity(floor.get("id"))
            or not _integer(floor.get("room_index"), 0, 168)
            or ("room_list_index" in floor and not _integer(floor["room_list_index"], 0, 511))
            or ("dimension" in floor and not _integer(floor["dimension"], 0, 2))
            or _visited_rooms(state) is None
            or not _integer(state.get("frame"), 0, 2**53)
            or not _integer(room.get("type"), 0, 31)
            or type(room.get("clear")) is not bool
            or any(type(state.get(k)) is not bool for k in ("enabled", "paused"))
            or type(player.get("dead")) is not bool
            or state.get("truncated", False) is not False):
        return None
    truncation = state.get("truncated_arrays", {})
    if not isinstance(truncation, Mapping) or any(truncation.values()):
        return None
    if not valid_pickups(state):
        return None
    start, first, last = _point(player), _point(room.get("top_left")), _point(room.get("bottom_right"))
    radius, velocity = _radius(player, 10), _velocity(player)
    if None in (start, first, last, radius, velocity) or not 0 < radius <= 40:
        return None
    bounds = first[0]+radius, first[1]+radius, last[0]-radius, last[1]-radius
    if not (0 < bounds[2]-bounds[0] <= 4000 and 0 < bounds[3]-bounds[1] <= 4000):
        return None
    hazards, enemies, projectiles, doors = (state.get(k) for k in ("hazards", "enemies", "projectiles", "doors"))
    if any(not isinstance(items, list) or len(items) > limit for items, limit in
           ((hazards, MAX_HAZARDS), (enemies, 64), (projectiles, 96), (doors, 8))):
        return None
    for item in hazards + enemies + projectiles:
        if not isinstance(item, Mapping) or None in (_point(item), _radius(item, 20), _velocity(item)):
            return None
    for item in hazards:
        if item.get("kind") == "grid":
            if not _integer(item.get("collision"), 0, 5) or not _integer(item.get("type", 0), 0, 100):
                return None
        elif item.get("kind") not in ("fire", "bomb", "laser", "tnt", "creep"):
            return None
    for item in enemies:
        if (not _number(item.get("hp")) or ("keeps_doors_closed" in item
                                           and type(item["keeps_doors_closed"]) is not bool)):
            return None
    allowed, seen = [], set()
    for item in doors:
        if (not isinstance(item, Mapping) or not _integer(item.get("slot"), 0, 7)
                or _point(item) is None or item["slot"] in seen
                or type(item.get("open")) is not bool or type(item.get("locked")) is not bool
                or not _integer(item.get("target_index"), -32768, 32767)
                or not _integer(item.get("target_type"), 0, 31)):
            return None
        seen.add(item["slot"])
        if (item["open"] and not item["locked"] and room["type"] != 10
                and item.get("curse_room_door") is not True
                and item["target_type"] in (_ALLOWED_TYPES | ({2} if allow_shop else set())
                                            | ({7, 8} if allow_secret else set()))
                and 0 <= item["target_index"] <= 168 and item["target_index"] != floor["room_index"]):
            allowed.append(_Door(item["slot"], _point(item), item["target_index"], item["target_type"]))
    return player, room, floor, start, radius, velocity, bounds, tuple(sorted(allowed, key=lambda d: d.slot))


def _point_waypoint(start, target, bounds, boxes, phase):
    """Dijkstra over a bounded collision-checked lattice to an exact point."""
    if not _inside(start, bounds) or not _inside(target, bounds) or not _free(target, boxes):
        return None
    if _clear(start, target, boxes):
        return target
    xs = _axis(bounds[0], bounds[2], phase[0], (start[0], target[0]))
    ys = _axis(bounds[1], bounds[3], phase[1], (start[1], target[1]))
    if not xs or not ys or len(xs)*len(ys) > MAX_NODES:
        return None
    width = len(xs)
    points = [(x, y) for y in ys for x in xs]
    available = [_free(p, boxes) for p in points]
    origin = ys.index(start[1])*width+xs.index(start[0])
    destination = ys.index(target[1])*width+xs.index(target[0])
    if not available[origin] or not available[destination]:
        return None
    distances, previous, queue = {origin: 0.0}, {}, [(0.0, origin)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distances[node]:
            continue
        if node == destination:
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
            if not available[other] or not _clear(points[node], points[other], boxes):
                continue
            candidate = cost+math.dist(points[node], points[other])
            if candidate < distances.get(other, math.inf):
                distances[other], previous[other] = candidate, node
                heapq.heappush(queue, (candidate, other))
    if destination not in previous:
        return None
    route = [destination]
    while previous[route[-1]] != origin:
        route.append(previous[route[-1]])
    for node in route:
        if _clear(start, points[node], boxes):
            return points[node]
    return points[route[-1]]


def _stationary_live_fire(item):
    return (item.get("kind") == "fire" and item.get("type") == 33
            and type(item.get("variant")) is int and item["variant"] in (0, 1)
            and _number(item.get("hp")) and item["hp"] > 0
            and _number(item.get("max_hp")) and item["max_hp"] >= item["hp"]
            and _number(item.get("radius")) and item["radius"] > 0
            and _number(item.get("vx")) and _number(item.get("vy"))
            and item.get("tear_destructible") is True
            and _velocity(item) == (0., 0.))


def _room_geometry(state, radius, door=None, pickup_target=None, include_npcs=True,
                   fire_recovery=False):
    """The same hazards and pickup exclusions protect every clear-room route."""
    boxes, roomy_boxes, rocks, phase = [], [], {}, (0., 0.)
    for item in state["hazards"]:
        x, y = _point(item)
        half = _radius(item, 20)
        if item["kind"] == "grid":
            phase = x % 40, y % 40
            # Only the selected, currently open door cell may be crossed. Never
            # carve an opening through an ordinary wall or any unrelated door.
            if door is not None and item.get("type") == 16 and math.dist((x, y), door.point) <= 22:
                continue
            if not item["collision"] and item.get("type") not in _DANGEROUS_GRIDS:
                continue
            padding = half+radius
        else:
            # Only the outward recovery helper uses the inner buffer. Normal
            # routes retain the full eight-unit clearance from every fire.
            padding = half+radius+(2 if fire_recovery and _stationary_live_fire(item) else 8)
        box = (x-padding, y-padding, x+padding, y+padding)
        ordinary_rock = item["kind"] == "grid" and item.get("type") == 2 and item["collision"] == 3
        ordinary_poop = (item["kind"] == "grid" and item.get("type") == 14
                         and type(item.get("variant")) is int and item["variant"] == 0
                         and item["collision"] == 3 and _integer(item.get("state"), 0, 999)
                         and item.get("tear_destructible") is True)
        # The recorded L-room stop is outside a plain wall's physical square,
        # just inside its player-radius margin. Its ordinary solid padding can
        # be left outward under the same checks; doors and damaging grids keep
        # their full exclusions, and no wall-core overlap gains this permission.
        ordinary_wall = (item["kind"] == "grid" and item.get("type") == 15
                         and type(item.get("variant")) is int and item["variant"] == 0
                         and item["collision"] == 4 and _integer(item.get("state"), 0, 0)
                         and _free(_point(state["player"]), [(x-half, y-half, x+half, y+half)]))
        if ordinary_rock or ordinary_poop or ordinary_wall:
            rocks[len(boxes)] = (x-half, y-half, x+half, y+half)
        boxes.append(box)
        margin = 8 if ordinary_rock or ordinary_wall else 0
        roomy_boxes.append((box[0]-margin, box[1]-margin, box[2]+margin, box[3]+margin))
    exclusions = exclusion_boxes(state, radius, pickup_target)
    boxes.extend(exclusions)
    roomy_boxes.extend(exclusions)
    # Some trap NPCs remain alive after the game clears the room. They are
    # still collision threats, not kill targets. Avoid their currently observed
    # body and a short swept velocity envelope on every freshly planned step.
    for enemy in state["enemies"] if include_npcs else ():
        if enemy["hp"] <= 0:
            continue
        x, y = _point(enemy)
        vx, vy = _velocity(enemy)
        pad = radius + _radius(enemy, 20) + 12
        box = (min(x, x + vx * 12) - pad, min(y, y + vy * 12) - pad,
               max(x, x + vx * 12) + pad, max(y, y + vy * 12) + pad)
        boxes.append(box)
        # Prefer enough space that the next fresh moving envelope does not
        # immediately overtake a route along the previous envelope's edge.
        roomy_boxes.append((box[0] - 12, box[1] - 12, box[2] + 12, box[3] + 12))
    return boxes, roomy_boxes, rocks, phase


def _fire_padding_escape(state, parsed, door=None, pickup_target=None):
    """Leave a live fire's extra margin without entering its contact envelope."""
    _, _, _, start, radius, velocity, bounds, _ = parsed
    overlaps = []
    for item in state["hazards"]:
        if not _stationary_live_fire(item):
            continue
        x, y = _point(item)
        half = _radius(item, 20)+radius+8
        if not _free(start, [(x-half, y-half, x+half, y+half)]):
            overlaps.append((x, y))
    if not overlaps or not _inside(start, bounds):
        return None
    boxes, _, _, _ = _room_geometry(state, radius, door, pickup_target)
    inner, _, _, _ = _room_geometry(state, radius, door, pickup_target, fire_recovery=True)
    # Keep the entire player radius plus two units outside fire contact, and
    # retain all walls, pickups, other hazards and live NPCs unchanged.
    if not _free(start, inner):
        return None
    coast = start[0]+velocity[0]*2, start[1]+velocity[1]*2
    if (not _inside(coast, bounds) or not _clear(start, coast, inner)
            or any((start[0]-x)*velocity[0]+(start[1]-y)*velocity[1] < -1e-6
                   for x, y in overlaps)):
        return None
    choices = {}
    for name, vector in _VECTORS.items():
        if name == "none" or any((start[0]-x)*vector[0]+(start[1]-y)*vector[1] <= 0
                                 for x, y in overlaps):
            continue
        end = start[0]+vector[0]*24, start[1]+vector[1]*24
        drift_end = end[0]+velocity[0]*2, end[1]+velocity[1]*2
        if all(_inside(point, bounds) and _free(point, boxes) and _clear(start, point, inner)
               and _clear(coast, point, inner) for point in (end, drift_end)):
            choices[name] = min(math.dist(end, fire) for fire in overlaps)
    return max(choices, key=choices.get) if choices else None


def _npc_escape(state, parsed, door=None, pickup_target=None):
    """A moving trap may enter our predicted padding between observations.

    Retain every fixed obstacle and use the normal short threat prediction to
    step away, rather than treating that moving envelope as a solid wall the
    player is embedded in. This never grants a rock/pit/pickup exception.
    """
    _, _, _, start, radius, velocity, bounds, _ = parsed
    boxes, _, _, _ = _room_geometry(state, radius, door, pickup_target, include_npcs=False)
    if not _inside(start, bounds) or not _free(start, boxes):
        return None
    threats = [(*_point(enemy), _radius(enemy, 20), *_velocity(enemy))
               for enemy in state["enemies"] if enemy["hp"] > 0]
    if not threats:
        return None
    return _avoid(start, velocity, radius, threats, bounds, boxes, "none", force=True)


def _ground_effect_escape(state, parsed):
    """Leave newly spawned creep without treating the patch as a solid wall."""
    _, _, _, start, radius, velocity, bounds, _ = parsed
    creep = [h for h in state["hazards"] if h["kind"] == "creep"]
    if not any(math.dist(start, _point(h)) <= radius+_radius(h, 20)+8 for h in creep):
        return None
    fixed = dict(state, hazards=[h for h in state["hazards"] if h["kind"] != "creep"])
    boxes, _, _, _ = _room_geometry(fixed, radius)
    if not _inside(start, bounds) or not _free(start, boxes):
        return "none"
    threats = [(*_point(h), _radius(h, 20), *_velocity(h)) for h in creep+state["projectiles"]]
    return _avoid(start, velocity, radius, threats, bounds, boxes, "none", force=True)


def _lingering_projectile_move(state, parsed):
    """Dodge remaining bullets without pursuing rewards or authorizing an exit.

    No selected-target exemption is passed to room geometry: paid pickups,
    option siblings, item replacements, floor exits and all doors stay protected.
    Bomb blast sizes and laser segments are not described by our current input,
    so their presence cannot authorize a new movement through this helper.
    """
    if any(item["kind"] in ("laser", "bomb") for item in state["hazards"]):
        return "none"
    _, _, _, start, radius, velocity, bounds, _ = parsed
    boxes, _, _, _ = _room_geometry(state, radius)
    if not _inside(start, bounds) or not _free(start, boxes):
        return "none"
    threats = [(*_point(item), _radius(item, 20), *_velocity(item))
               for item in state["projectiles"]]
    threats.extend((*_point(item), _radius(item, 20), *_velocity(item))
                   for item in state["enemies"] if item["hp"] > 0)
    return _avoid(start, velocity, radius, threats, bounds, boxes, "none")


def _pickup_move(state, parsed, pickup):
    _, _, _, start, radius, velocity, bounds, _ = parsed
    boxes, roomy_boxes, rocks, phase = _room_geometry(state, radius, pickup_target=signature(pickup))
    if not _inside(start, bounds):
        return None
    if not _free(start, boxes):
        escape = _padding_escape(start, velocity, radius, bounds, boxes, rocks)
        if escape is None:
            escape = _fire_padding_escape(state, parsed, pickup_target=signature(pickup))
        return escape if escape is not None else _npc_escape(state, parsed, pickup_target=signature(pickup))
    target = _point(pickup)
    waypoint = _point_waypoint(start, target, bounds, roomy_boxes, phase)
    if waypoint is None:
        waypoint = _point_waypoint(start, target, bounds, boxes, phase)
    if waypoint is None:
        return None
    return _steer(start, waypoint, velocity, bounds, boxes, deadband=1.5, drift_frames=4.0)


def _door_move(state, parsed, door):
    """Approach inside the room, then cross only the selected open portal."""
    _, _, _, start, radius, velocity, bounds, _ = parsed
    vector = _DIRECTIONS[door.slot % 4]
    # Stage near the middle of the space between the last interior grid row
    # and boundary wall, rather than exactly tangent to an expanded rock row.
    approach = door.point[0]-vector[0]*40, door.point[1]-vector[1]*40
    exit_point = door.point[0]+vector[0]*28, door.point[1]+vector[1]*28
    if not _inside(approach, bounds):
        return None
    boxes, roomy_boxes, rocks, phase = _room_geometry(state, radius, door=door)
    if not _free(start, boxes):
        escape = _padding_escape(start, velocity, radius, bounds, boxes, rocks)
        if escape is None:
            escape = _fire_padding_escape(state, parsed, door=door)
        return escape if escape is not None else _npc_escape(state, parsed, door=door)
    along = (start[0]-approach[0])*vector[0]+(start[1]-approach[1])*vector[1]
    across = abs((start[0]-door.point[0])*vector[1]-(start[1]-door.point[1])*vector[0])
    # A doorway is one 40-unit grid cell wide. The player center has only the
    # remaining half-width available once its collision radius is accounted for.
    # Keep the destination beyond the door throughout this corridor: steering
    # back toward `approach` when sideways inertia drifts past a tiny alignment
    # tolerance causes oscillation and can strand the player just outside bounds.
    half_width = max(0.0, 20.0-radius)
    if (-15 <= along <= 73 and across <= half_width and half_width > 0
            and _clear(start, exit_point, boxes)):
        # The expanded boundary applies only to this selected open doorway.
        # Every segment still checks all neighboring wall and obstacle boxes.
        if vector[0]:
            corridor = (min(approach[0]-vector[0]*15, exit_point[0]+vector[0]*5),
                        door.point[1]-half_width,
                        max(approach[0]-vector[0]*15, exit_point[0]+vector[0]*5),
                        door.point[1]+half_width)
        else:
            corridor = (door.point[0]-half_width,
                        min(approach[1]-vector[1]*15, exit_point[1]+vector[1]*5),
                        door.point[0]+half_width,
                        max(approach[1]-vector[1]*15, exit_point[1]+vector[1]*5))
        return _steer(start, exit_point, velocity, corridor, boxes, deadband=1.5, drift_frames=5.0)
    # This corridor is the sole exception to the usual inset boundary. Until
    # aligned at its mouth all movement remains inside the padded room bounds.
    if not _inside(start, bounds):
        return None
    # An obstacle can cover the center staging point without sealing the door.
    # The recorded post-TNT room has exactly this shape: the remaining movable
    # barrel covers (320,400), but the left side of the same open portal is free.
    # Replan across a bounded set of mouth points. Every point, route and final
    # crossing retains all obstacle boxes; no TNT or neighboring wall is carved.
    offsets = (0.0, -.75*half_width, .75*half_width, -half_width, half_width)
    for offset in dict.fromkeys(offsets):
        target = (approach[0]+vector[1]*offset, approach[1]-vector[0]*offset)
        if not _inside(target, bounds) or not _free(target, boxes) or not _clear(target, exit_point, boxes):
            continue
        # Prefer turning room around rocks while allowing verified narrow gaps.
        waypoint = _point_waypoint(start, target, bounds, roomy_boxes, phase)
        if waypoint is None:
            waypoint = _point_waypoint(start, target, bounds, boxes, phase)
        if waypoint is None:
            continue
        # Live traces travel about twice their reported velocity per observation
        # and receive changed input later. The five-frame steering estimate also
        # retains clearance at the recorded TNT turn with two delayed inputs.
        return _steer(start, waypoint, velocity, bounds, boxes, deadband=2.0, drift_frames=5.0)
    return None


def _moving_npc_wait(state, parsed, door):
    """Recognize a transient trap blockage, without using its cleared route.

    Removing only explicitly moving, non-room-blocking NPCs must restore the
    route. This is diagnostic: any returned step still uses every fixed
    obstacle and fresh predictions of every live NPC, including those removed
    for the diagnostic. Stationary NPCs and all pickups remain route blockers.
    """
    moving = [enemy for enemy in state["enemies"]
              if enemy["hp"] > 0 and enemy.get("keeps_doors_closed") is False
              and math.hypot(*_velocity(enemy)) > 1e-6]
    if not moving or not _inside(parsed[3], parsed[6]):
        return None
    reduced = dict(state, enemies=[enemy for enemy in state["enemies"] if enemy not in moving])
    fixed_boxes, _, _, _ = _room_geometry(reduced, parsed[4], door=door)
    if not _free(parsed[3], fixed_boxes) or _door_move(reduced, parsed, door) is None:
        return None
    boxes, _, _, _ = _room_geometry(state, parsed[4], door=door, include_npcs=False)
    threats = [(*_point(enemy), _radius(enemy, 20), *_velocity(enemy))
               for enemy in state["enemies"] if enemy["hp"] > 0]
    return _avoid(parsed[3], parsed[5], parsed[4], threats, parsed[6], boxes, "none")


class FloorNavigator:
    """One-floor map, deterministic frontier search and traversal watchdogs.

    The caller must enforce packet receipt age and send at most once per fresh
    frame. ``observe`` is idempotent; ``step`` also calls it, and additionally
    suppresses duplicate frames. Paused/disarmed observations never navigate.
    """

    def __init__(self, *, stuck_timeout=4.0, transition_timeout=30.0,
                 adventure_mode=False, continue_floors=False, allow_secret=False):
        if not (_number(stuck_timeout) and 0 < stuck_timeout <= 30
                and _number(transition_timeout) and stuck_timeout <= transition_timeout <= 120):
            raise ValueError("Invalid exploration watchdog limits")
        self._stuck_timeout = float(stuck_timeout)
        self._transition_timeout = float(transition_timeout)
        self._identity = None
        self._frame = -1
        self._step_frame = -1
        self._current = None
        self._current_index = None
        self._room_visit = None
        self._rooms = {}
        self._aliases = {}
        self._graph = {}
        self._observed = set()
        self._inspected = set()
        self._pending = None
        self._pending_started = None
        self._progress_at = None
        self._progress_point = None
        self._no_route_since = None
        self._blocked_npc_since = None
        self._doors_traversed = 0
        self._stop = None
        self._last_now = None
        self._paused = False
        self._pickups = PickupCollector()
        self.adventure_mode = adventure_mode
        self.allow_secret = allow_secret
        self.continue_floors = continue_floors
        self._adventure = None
        self._allow_descend = False
        self._curse_attempted = set()
        self._curse_return_index = None
        if adventure_mode:
            from .adventure_control import AdventureControl
            self._adventure = AdventureControl()

    def rearmed(self, state):
        """Keep observed floor knowledge, never outstanding controls, on recovery.

        The controller must authenticate a fresh F8 session, current run/floor
        and packet receipt age before calling this. A user may have moved to
        another room while disarmed, so the new visit is rebound from its fresh
        observation. The previous map is usable only within the exact same run
        and floor.
        """
        fresh = FloorNavigator(stuck_timeout=self._stuck_timeout,
                               transition_timeout=self._transition_timeout,
                               adventure_mode=self.adventure_mode,
                               continue_floors=self.continue_floors, allow_secret=self.allow_secret)
        parsed = _validated(state, allow_shop=self.adventure_mode, allow_secret=self.allow_secret)
        if parsed is None:
            fresh._stop = "incomplete floor observation"
            return fresh
        player, _, floor, *_ = parsed
        identity = state["run_id"], floor["id"], floor.get("dimension")
        if self._identity is not None and identity != self._identity:
            fresh._stop = "floor or run changed"
            return fresh
        if state["frame"] < self._frame:
            fresh._stop = "stale floor observation"
            return fresh
        if not state["enabled"] or state["paused"] or player["dead"]:
            fresh._stop = "recovery requires a live armed observation"
            return fresh
        if self._identity is None:
            # A permitted descent creates a new navigator during the animation.
            # If control stops before its first playable observation, there is
            # no map to restore. Bind this authenticated rearm as its first frame.
            fresh._identity, fresh._frame = identity, state["frame"]
            return fresh
        fresh._identity, fresh._frame = self._identity, self._frame
        fresh._rooms = dict(self._rooms)
        fresh._aliases = dict(self._aliases)
        fresh._graph = dict(self._graph)
        fresh._observed = set(self._observed)
        fresh._inspected = set(self._inspected)
        fresh._doors_traversed = self._doors_traversed
        fresh._curse_attempted = set(self._curse_attempted)
        # The return belongs to one actual curse-room visit. A manual detour
        # while disarmed must never carry it into another curse room/visit.
        if state["room_id"] == self._room_visit and state["room"]["type"] == 10:
            fresh._curse_return_index = self._curse_return_index
        # Historical counters remain cumulative. Pickup targets, timers and
        # animation state restart from current observations after rearming.
        fresh._pickups.attempts = self._pickups.attempts
        fresh._pickups.resolved = self._pickups.resolved
        fresh._pickups.skipped = self._pickups.skipped
        if self._adventure is not None:
            previous, current = self._adventure, fresh._adventure
            current.selected = previous.selected
            current.completed = previous.completed
            current.abandoned = previous.abandoned
            current.events = copy.deepcopy(previous.events)
            if previous.visit == (state["run_id"], state["room_id"]):
                current.visit = previous.visit
                current.skipped = set(previous.skipped)
                current.replaced_active = previous.replaced_active
                if previous.plan is not None:
                    # Interrupted interactions need a new user choice; never
                    # silently replay a bomb/item pulse or retry a failed prop.
                    current.skipped.add(previous.plan.key)
        return fresh

    @property
    def stats(self):
        return {"rooms_visited": len(self._rooms),
                "rooms_cleared": sum(clear for _, clear in self._rooms.values()),
                "doors_traversed": self._doors_traversed,
                "boss_cleared": any(kind == _BOSS and clear for kind, clear in self._rooms.values())}

    @property
    def stop_reason(self):
        return self._stop

    def decision_context(self):
        """Snapshot authenticated local knowledge without changing exploration.

        Edges are the permitted open doors remembered at last inspection, not
        a complete floor map or proof of current reachability. Large-room aliases
        remain attached to one actual room. No hidden descriptor is queried.
        """
        from .state_context import room_name
        def room_key(key):
            return f"{key[0]}:{key[1]}" if key is not None else None
        def door_row(door):
            return {"slot": door.slot, "target_index": door.target_index,
                    "target_type": door.target_type, "target_room": room_name(door.target_type)}
        rooms = []
        for key, (kind, clear) in sorted(self._rooms.items()):
            rooms.append([room_key(key), sorted(i for i, target in self._aliases.items() if target == key),
                          room_name(kind), clear, key in self._inspected,
                          [[d.slot, d.target_index, d.target_type] for d in self._graph.get(key, ())]])
        plan = self._adventure.plan if self._adventure is not None else None
        events = self._adventure.events[-3:] if self._adventure is not None else []
        return {"current_room_key": room_key(self._current), "progress": self.stats,
                "map_columns": ["room_key", "grid_aliases", "room_type", "clear_last_observed",
                                "doors_inspected", "remembered_permitted_doors_slot_target_type"],
                "map_rows": rooms,
                "map_scope": "Visited current-floor rooms only; remembered permitted open doors may have changed. No edges inferred from visited-room summaries.",
                "pending_door": door_row(self._pending) if self._pending is not None else None,
                "interaction": ({"key": plan.key, "kind": plan.kind, "phase": self._adventure.phase}
                                if plan is not None else None),
                "recent_interaction_outcomes": [{"frame": e["frame"], "event": e["event"],
                    "candidate_key": e["candidate"]["key"], "reason": e["reason"]} for e in events]}

    @property
    def pickup_stats(self):
        return {"attempts": self._pickups.attempts,
                "targets_disappeared_or_changed": self._pickups.resolved,
                "skipped_unreachable_or_stalled": self._pickups.skipped}

    @property
    def adventure_options(self):
        return self._adventure.offers if self._adventure is not None else ()

    def observe_interaction_pause(self, state, now):
        if self._adventure is not None:
            self._adventure.observe_pause(state, now)

    @property
    def adventure_stats(self):
        if self._adventure is None:
            return {}
        return {"selected": self._adventure.selected, "completed_or_changed": self._adventure.completed,
                "abandoned": self._adventure.abandoned, "events": self._adventure.events}

    @property
    def descent_requested(self):
        return self._adventure is not None and self._adventure.descent_requested

    def accept_adventure(self, key, state, now, *, decision_reason="Jev selected a verified candidate"):
        if self._adventure is None:
            return False
        accepted = self._adventure.accept(key, state, now, allow_descend=self._allow_descend,
                                          decision_reason=decision_reason)
        if accepted and self._adventure.plan.kind in ("unlock_door", "enter_curse", "leave_curse"):
            plan = self._adventure.plan
            if plan.kind == "enter_curse":
                self._curse_attempted.add(plan.details["target_index"])
                self._curse_return_index = plan.details["origin_index"]
            self._pending = _Door(plan.details["slot"], plan.point,
                                  plan.details["target_index"], plan.details["target_type"])
            self._pending_started = self._progress_at = now
            self._progress_point = _point(state["player"])
        return accepted

    def _reset_traversal(self):
        self._pending = self._pending_started = self._progress_at = self._progress_point = None
        self._no_route_since = None
        self._blocked_npc_since = None

    def observe(self, state):
        if self._stop is not None:
            return
        parsed = _validated(state, allow_shop=self.adventure_mode, allow_secret=self.allow_secret)
        if parsed is None:
            self._stop = "incomplete floor observation"
            return
        player, room, floor, _, _, _, _, doors = parsed
        # Do not bind the pre-F8 session or add paused frames to the room map.
        identity = state["run_id"], floor["id"], floor.get("dimension")
        if self._identity is not None and identity != self._identity:
            self._stop = "floor or run changed"
            return
        if not state["enabled"] or state["paused"]:
            # Room transition animations may pause the simulation. Keep the
            # intended destination so the arrival can still be authenticated.
            self._paused = True
            self._pickups.pause()
            return
        if player["dead"]:
            self._stop = "player died"
            return
        if self._identity is None:
            self._identity = identity
        if state["frame"] < self._frame:
            self._stop = "stale floor observation"
            return
        index = floor["room_index"]
        # A large room can be entered at several grid indices. ListIndex is
        # stable for that actual room; learn aliases only when it is observed.
        key = ("list", floor["room_list_index"]) if "room_list_index" in floor else ("grid", index)
        if index in self._aliases and self._aliases[index] != key:
            self._stop = "room identity changed"
            return
        changed_room = index != self._current_index or key != self._current
        changed_visit = changed_room or state["room_id"] != self._room_visit
        expected_arrival = self._pending is not None and index == self._pending.target_index and changed_room
        if self._current is not None and state["frame"] == self._frame and changed_visit and not expected_arrival:
            self._stop = "conflicting floor observation"
            return
        if room["type"] not in (_ALLOWED_TYPES | ({2, 10} if self.adventure_mode else set())
                                | ({7, 8} if self.allow_secret else set())):
            self._stop = "unsupported room type"
            return
        if self._current is not None and changed_room:
            if self._pending is None or index != self._pending.target_index:
                self._stop = "unexpected room transition"
                return
            self._doors_traversed += 1
            self._reset_traversal()
        elif self._room_visit is not None and state["room_id"] != self._room_visit:
            self._stop = "unexpected room visit"
            return
        if key in self._observed and self._rooms[key][0] != room["type"]:
            self._stop = "room type changed"
            return
        if room["clear"] and any(item["hp"] > 0 and item.get("keeps_doors_closed") is not False
                                 for item in state["enemies"]):
            self._stop = "conflicting room clear state"
            return
        self._frame = state["frame"]
        self._current, self._current_index, self._room_visit = key, index, state["room_id"]
        # Import only game-reported visited/clear facts. No edges can be learned
        # here: the controller still needs the room's actual door observation.
        for row, aliases in _visited_rooms(state):
            imported_key = ("list", row["list_index"])
            if imported_key == key:
                for alias in aliases:
                    if alias not in self._aliases or self._aliases[alias] == key:
                        self._aliases[alias] = key
                continue  # The live current-room observation takes precedence.
            if index in aliases:
                continue
            if any(i in self._aliases and self._aliases[i] != imported_key for i in aliases):
                continue
            if imported_key in self._observed and self._rooms[imported_key][0] != row["type"]:
                continue
            self._rooms[imported_key] = row["type"], row["clear"]
            for alias in aliases:
                self._aliases[alias] = imported_key
            if row["type"] == 10:
                # A real prior visit already spent this optional health budget,
                # even when it occurred before this controller was started.
                self._curse_attempted.update(aliases)
        self._aliases[index] = key
        self._rooms[key] = room["type"], room["clear"]
        self._observed.add(key)
        if room["type"] == 10:
            self._curse_attempted.update(alias for alias, target in self._aliases.items() if target == key)
        elif changed_room:
            self._curse_return_index = None
        # Closed doors from combat are not remembered as traversable. The
        # first clear observation refreshes this room's real open doors.
        self._graph[key] = doors if room["clear"] else ()
        if room["clear"]:
            self._inspected.add(key)
        else:
            self._inspected.discard(key)
        if not room["clear"]:
            self._reset_traversal()
            self._pickups.reset()
        if self._adventure is not None:
            self._adventure.reset_visit(state)

    def _next_door(self, excluded_slots=()):
        """Prefer real unfinished frontiers, then inspect cleared transit rooms.

        Search through cleared visited rooms only. A frontier beyond a boss is
        not considered known until that boss room has actually been visited.
        """
        for inspect_pass, boss_pass in ((False, False), (False, True), (True, False), (True, True)):
            queue, seen = deque([(self._current, None)]), {self._current}
            while queue:
                index, first = queue.popleft()
                for door in self._graph.get(index, ()):
                    if first is None and door.slot in excluded_slots:
                        continue
                    target_key = self._aliases.get(door.target_index)
                    target = self._rooms.get(target_key)
                    first_door = first or door
                    unfinished = target is None or not target[1]
                    uninspected = target is not None and target[1] and target_key not in self._inspected
                    if unfinished or uninspected:
                        if ((unfinished and not inspect_pass) or (uninspected and inspect_pass)) and (
                                door.target_type == _BOSS) == boss_pass:
                            return first_door
                        continue
                    if target[1] and target_key not in seen:
                        seen.add(target_key)
                        queue.append((target_key, first_door))
        return None

    def step(self, state, now_monotonic):
        if not _number(now_monotonic, limit=1e15) or (self._last_now is not None and now_monotonic < self._last_now):
            self._stop = "invalid exploration clock"
        else:
            self._last_now = float(now_monotonic)
        self.observe(state)
        if self._stop:
            return ExplorationAction(stop_reason=self._stop)
        if not state["enabled"] or state["paused"]:
            return ExplorationAction(status="paused")
        if state["frame"] <= self._step_frame:
            return ExplorationAction(status="waiting for fresh observation")
        self._step_frame = state["frame"]
        if not state["room"]["clear"]:
            return ExplorationAction(status="combat")
        parsed = _validated(state, allow_shop=self.adventure_mode, allow_secret=self.allow_secret)
        doors = parsed[-1]
        waiting_for_doors = any(not d["open"] and not d["locked"]
                                and d.get("curse_room_door") is not True
                                and d["target_type"] in (_ALLOWED_TYPES | {2} if self.adventure_mode else _ALLOWED_TYPES)
                                and 0 <= d["target_index"] <= 168
                                for d in state["doors"])
        # Dodge live projectiles before continuing a selected pickup/item plan.
        # A bomb already placed is the exception: finish its verified retreat
        # rather than freezing beside it or deviating into its blast area.
        bomb_retreat = (self._adventure is not None and self._adventure.plan is not None
                        and self._adventure.plan.kind == "bomb_rock" and self._adventure.phase == "retreat")
        if state["projectiles"] and not bomb_retreat:
            move = _lingering_projectile_move(state, parsed)
            if self._pending is not None:
                self._pending_started = self._progress_at = now_monotonic
                self._progress_point = parsed[3]
            self._no_route_since = None
            self._blocked_npc_since = None
            return ExplorationAction(move=move, status="dodging lingering projectiles" if move != "none"
                                     else "waiting for lingering hazards")
        if self._adventure is not None and self._adventure.plan is not None:
            action = self._adventure.step(state, now_monotonic, allow_descend=self._allow_descend)
            if action is not None:
                if self._adventure.plan is None:
                    self._pickups.reset()  # Chest contents/items may spawn after an interaction.
                return action
        # Wait for remaining attacks before touching rewards or choosing an exit.
        if state["projectiles"] or any(h["kind"] in ("laser", "bomb") for h in state["hazards"]):
            return ExplorationAction(status="waiting for lingering hazards")
        # Once inside a doorway corridor, finish that authenticated crossing.
        # All other pickup movement remains within the ordinary room bounds.
        if _inside(parsed[3], parsed[6]):
            pickup = self._pickups.step(state, now_monotonic,
                                        lambda item: _pickup_move(state, parsed, item),
                                        defer_items=self.adventure_mode)
            if pickup is not None:
                # Preserve a doorway already selected before a pause. The
                # item's detour must not spend that door's progress budget.
                if self._pending is not None:
                    self._pending_started = self._progress_at = now_monotonic
                    self._progress_point = parsed[3]
                self._no_route_since = None
                self._blocked_npc_since = None
                if pickup.stop_reason:
                    return self._finish(pickup.stop_reason, pickup.status)
                return ExplorationAction(move=pickup.move, status=pickup.status,
                                         stop_reason=pickup.stop_reason)
        if self._adventure is not None and _inside(parsed[3], parsed[6]):
            offered = self._adventure.choose_offers(state, excluded_curse_targets=self._curse_attempted)
            self._allow_descend = False
            if (not offered and self.continue_floors and not waiting_for_doors
                    and self.stats["boss_cleared"] and self._next_door() is None):
                self._allow_descend = True
                offered = self._adventure.choose_offers(state, allow_descend=True,
                                                        excluded_curse_targets=self._curse_attempted)
            if offered:
                return ExplorationAction(status="waiting for Jev's interaction choice")
            if state["room"]["type"] == 10:
                from .curse_doors import curse_candidates
                exits = curse_candidates(state, leaving=True, return_index=self._curse_return_index)
                if not exits:
                    return self._finish("no supported curse-room exit within the health budget")
                self._adventure.offers = exits[:1]
                self.accept_adventure(exits[0].key, state, now_monotonic,
                                      decision_reason="Local return leg after checking curse-room rewards")
                return ExplorationAction(status="returning from the selected curse-room visit")
        # Revalidate the selected door every frame; a close/lock/target change
        # invalidates it immediately rather than preserving an action queue.
        if self._pending is not None and self._pending not in doors:
            self._reset_traversal()
        chosen = self._pending or self._next_door()
        if chosen is None:
            # Door opening animations can briefly leave no open doors just
            # after the final enemy dies. Avoid claiming completion then.
            all_frontiers_done = all(self._aliases.get(d.target_index) in self._rooms
                                     and self._rooms[self._aliases[d.target_index]][1]
                                     and self._aliases[d.target_index] in self._inspected
                                     for doorset in self._graph.values() for d in doorset)
            if self.stats["boss_cleared"] and all_frontiers_done and not waiting_for_doors:
                return self._finish("floor cleared", "boss defeated; observed accessible rooms cleared")
            if self._no_route_since is None:
                self._no_route_since = now_monotonic
            if now_monotonic-self._no_route_since >= 2.0:
                return self._finish("no accessible unexplored rooms")
            return ExplorationAction(status="waiting for an open ordinary door")
        self._no_route_since = None
        start = parsed[3]
        if self._pending is None:
            self._pending, self._pending_started = chosen, now_monotonic
            self._progress_at, self._progress_point = now_monotonic, start
        if self._paused:
            self._pending_started = self._progress_at = now_monotonic
            self._progress_point = start
            self._blocked_npc_since = None
            self._paused = False
        if math.dist(start, self._progress_point) >= 10:
            self._progress_at, self._progress_point = now_monotonic, start
        if now_monotonic-self._pending_started >= self._transition_timeout:
            return self._finish("door traversal timed out")
        if now_monotonic-self._progress_at >= self._stuck_timeout:
            return self._finish("stuck while approaching door")
        move = _door_move(state, parsed, chosen)
        if move is None:
            waiting_move = _moving_npc_wait(state, parsed, chosen)
            if waiting_move is not None:
                # A temporarily occupied first doorway need not block a safe
                # frontier through another currently observed ordinary door.
                # Check each alternate against the full, unmodified geometry.
                # Retain the original total traversal budget across reroutes.
                excluded = {chosen.slot}
                alternate = self._next_door(excluded)
                while alternate is not None:
                    alternate_move = _door_move(state, parsed, alternate)
                    if alternate_move is not None:
                        self._pending = alternate
                        self._progress_at, self._progress_point = now_monotonic, start
                        self._blocked_npc_since = None
                        return ExplorationAction(move=alternate_move,
                                                 status=f"entering room {alternate.target_index}")
                    excluded.add(alternate.slot)
                    alternate = self._next_door(excluded)
                if self._blocked_npc_since is None:
                    self._blocked_npc_since = now_monotonic
                if now_monotonic-self._blocked_npc_since < 2.0:
                    return ExplorationAction(move=waiting_move, status="waiting for a moving trap to clear the route")
            return self._finish("no safe route to open door")
        self._blocked_npc_since = None
        return ExplorationAction(move=move, status=f"entering room {chosen.target_index}")

    def _finish(self, reason, status=None):
        self._stop = reason
        return ExplorationAction(stop_reason=reason, status=status)
