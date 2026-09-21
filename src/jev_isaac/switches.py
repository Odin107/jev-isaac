"""Reach observed required pressure plates, clearing blocking TNT when needed.

Required-plate flag, variant0, state0 and no grid collision are all explicit
facts. Reward, Greed, rail and unknown plates are never authorized by this
module. TNT stays a movement obstacle until fresh observations show destruction.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

from .combat import _inside, _point
from .exploration import ExplorationAction, _point_waypoint, _room_geometry, _validated
from .navigation import _VECTORS, _clear, _free, _recovery_options, _steer
from .protocol import valid_switches


def _ordinary(row, state):
    return (type(row.get("variant")) is int and row["variant"] == 0
            and type(row.get("state")) is int and row["state"] == state
            and row["collision"] == 0)


def _geometry(state, radius, target):
    boxes, roomy, recoverable, phase = _room_geometry(state, radius)
    # An explosion/push can leave the player a few units inside the artificial
    # margin around an open ordinary doorway. Retain the door box, but permit
    # the same checked outward-from-padding recovery used for adjacent walls.
    # This never authorizes crossing the door: recovery must end inside the
    # room and have a verified route to the selected plate.
    from .exploration import _DANGEROUS_GRIDS
    index = 0
    for hazard in state["hazards"]:
        if (hazard["kind"] == "grid" and not hazard["collision"]
                and hazard.get("type") not in _DANGEROUS_GRIDS):
            continue
        if (hazard["kind"] == "grid" and hazard.get("type") == 16
                and hazard.get("collision") == 5):
            point = _point(hazard)
            matching = [d for d in state["doors"] if _point(d) == point]
            if (len(matching) == 1 and matching[0]["open"] and not matching[0]["locked"]
                    and matching[0].get("curse_room_door") is False
                    and matching[0]["target_type"] in (1, 2, 4, 5)):
                x, y = point
                half = hazard.get("radius", 20)
                core = (x-half, y-half, x+half, y+half)
                expected = (x-half-radius, y-half-radius, x+half+radius, y+half+radius)
                if index < len(boxes) and boxes[index] == expected and _free(_point(state["player"]), [core]):
                    recoverable[index] = core
        index += 1
    # All unselected plates remain protected unless their ordinary pressed
    # state explicitly proves they are inert. No inferred plate hitbox is used
    # to authorize contact: exclusion covers the complete 40-unit grid tile.
    for row in state["switches"]:
        if (row["index"], *_point(row)) == target or _ordinary(row, 3):
            continue
        x, y = _point(row)
        half = 20+radius+8
        box = (x-half, y-half, x+half, y+half)
        boxes.append(box)
        roomy.append(box)
    return boxes, roomy, recoverable, phase


def _recovery_step(start, velocity, radius, bounds, boxes, recoverable, phase, target):
    """An allowed outward step must lead to a verified route to this plate."""
    options = _recovery_options(start, velocity, radius, bounds, boxes, recoverable) or {}
    for move in sorted(options, key=options.get, reverse=True):
        vector = _VECTORS[move]
        end = start[0]+vector[0]*24, start[1]+vector[1]*24
        if _point_waypoint(end, target, bounds, boxes, phase) is not None:
            return move, end
    return None


class SwitchNavigator:
    """A bounded local switch objective, revalidated on each observation."""

    def __init__(self):
        self.context = None
        self.has_objective = False
        self.target = self.waypoint = None
        self.started = self.progress_at = None
        self.distances = {}
        self.paused_at = self.pressed_at = None
        self.demolition = self.demolition_target = None
        self.shot_guard = None

    def reset(self):
        guard = self._remember_shot()
        self.__init__()
        self.shot_guard = guard

    def _remember_shot(self):
        demolition = self.demolition
        if (demolition is not None and not demolition.complete
                and demolition.last_fire_frame is not None):
            context = demolition.context
            self.shot_guard = ((context[0], context[2], context[3], context[4]),
                               demolition.last_fire_frame+60)
        return self.shot_guard

    def _reset_target(self):
        self._remember_shot()
        self.target = self.waypoint = None
        self.started = self.progress_at = None
        self.distances = {}
        self.pressed_at = None
        self.demolition = self.demolition_target = None

    def _stop(self, reason):
        stop_reason = (reason if reason in ("room switch approach stalled", "room switch activation timed out")
                       else "no safe route to required room switch")
        return ExplorationAction(status=reason, stop_reason=stop_reason)

    def step(self, state, now, *, selected_switch=None, allow_demolition=True):
        self.has_objective = False
        parsed = _validated(state)
        caps = state.get("capabilities", {}) if isinstance(state, Mapping) else {}
        if (parsed is None or not isinstance(caps, Mapping)
                or type(caps.get("room_switches")) is not int or caps["room_switches"] != 1
                or not valid_switches(state.get("switches"))
                or state["room"]["clear"] or state["room"]["type"] != 1
                or state["room"].get("has_trigger_pressure_plates") is not True
                or any(enemy["hp"] > 0 for enemy in state["enemies"])
                or state["projectiles"] or state["player"]["dead"]):
            self._reset_target()
            return None
        context = (state["session"], state["run_id"], state["floor"]["id"], state["room_id"])
        if context != self.context:
            self.context = context
            self._reset_target()
            self.paused_at = None
        pending = [row for row in state["switches"] if _ordinary(row, 0)]
        if selected_switch is not None:
            pending = [row for row in pending if row["index"] == selected_switch]
        all_pressed = bool(state["switches"]) and all(_ordinary(row, 3) for row in state["switches"])
        if not pending and not all_pressed:
            self._reset_target()
            return None
        self.has_objective = True
        if not state["enabled"] or state["paused"]:
            if self.paused_at is None:
                self.paused_at = now
            return ExplorationAction(status="room switch paused")
        if self.paused_at is not None:
            duration = max(0, now-self.paused_at)
            for name in ("started", "progress_at", "pressed_at"):
                if getattr(self, name) is not None:
                    setattr(self, name, getattr(self, name)+duration)
            if self.demolition is not None:
                for name in ("started", "progress_at", "changed_at", "fired_at"):
                    if getattr(self.demolition, name) is not None:
                        setattr(self.demolition, name, getattr(self.demolition, name)+duration)
            self.paused_at = None
        if self.shot_guard is not None and self.demolition is None:
            identity, ready_frame = self.shot_guard
            current = (state["run_id"], state["room_id"], state["floor"]["id"],
                       state["floor"].get("dimension"))
            if identity == current and state["frame"] < ready_frame:
                return ExplorationAction(status="waiting for earlier TNT shots to settle after resume")
            self.shot_guard = None
        if all_pressed:
            if self.pressed_at is None:
                self.pressed_at = now
            if now-self.pressed_at >= 2:
                return self._stop("room switch activation timed out")
            return ExplorationAction(status="required switches pressed; waiting for room to clear")
        self.pressed_at = None
        if any(h["kind"] in ("bomb", "laser") for h in state["hazards"]):
            return self._stop("room switch route interrupted by an unsafe hazard")
        start, radius, velocity, bounds = parsed[3:7]
        # Only shallow, slow drift outside the conservative player inset can
        # recover. The route below must still prove an outward obstacle exit;
        # no coordinate clamping, teleport, or blanket boundary expansion.
        shallow_bounds = (bounds[0]-4, bounds[1]-4, bounds[2]+4, bounds[3]+4)
        if not _inside(start, bounds) and (not _inside(start, shallow_bounds) or math.hypot(*velocity) > 1):
            return self._stop("no safe route to required room switch")
        if self.demolition is not None:
            if self.demolition.complete:
                self._reset_target()
            else:
                row = next((row for row in pending
                            if (row["index"], *_point(row)) == self.demolition_target), None)
                if row is None:
                    return self._stop("required room switch changed during TNT clearing")
                boxes, _, _, phase = _geometry(state, radius, self.demolition_target)
                action = self.demolition.step(state, now, self.demolition_target[1:], boxes, phase)
                return action if action is not None else self._stop("TNT clearing plan became unavailable")
        if self.target is not None:
            selected = next((row for row in state["switches"] if row["index"] == self.target[0]), None)
            if selected is not None and _ordinary(selected, 3):
                self._reset_target()
            elif selected is None or (selected["index"], *_point(selected)) != self.target or not _ordinary(selected, 0):
                return self._stop("required room switch changed before activation")
        recovery_step = None
        if self.target is None:
            for row in sorted(pending, key=lambda row: (math.dist(start, _point(row)), row["index"])):
                target = (row["index"], *_point(row))
                boxes, roomy, recoverable, phase = _geometry(state, radius, target)
                waypoint = _point_waypoint(start, target[1:], bounds, roomy, phase)
                if waypoint is None:
                    waypoint = _point_waypoint(start, target[1:], bounds, boxes, phase)
                if waypoint is None and not _free(start, boxes):
                    recovery_step = _recovery_step(start, velocity, radius, bounds, boxes,
                                                   recoverable, phase, target[1:])
                    if recovery_step is not None:
                        waypoint = recovery_step[1]
                if waypoint is not None:
                    self.target, self.waypoint = target, waypoint
                    self.started = self.progress_at = now
                    break
            else:
                if not allow_demolition:
                    return self._stop("Jev-selected switch needs a separate demolition decision")
                from .tnt import TntDemolition
                for row in sorted(pending, key=lambda row: (math.dist(start, _point(row)), row["index"])):
                    target = (row["index"], *_point(row))
                    boxes, _, _, phase = _geometry(state, radius, target)
                    demolition = TntDemolition()
                    action = demolition.step(state, now, target[1:], boxes, phase)
                    if action is not None:
                        self.demolition, self.demolition_target = demolition, target
                        return action
                return self._stop("no safe route to required room switch")
        boxes, roomy, recoverable, phase = _geometry(state, radius, self.target)
        if not _free(start, boxes):
            if now-self.started >= 20 or now-self.progress_at >= 4:
                return self._stop("room switch approach stalled")
            if recovery_step is None:
                recovery_step = _recovery_step(start, velocity, radius, bounds, boxes,
                                               recoverable, phase, self.target[1:])
            if recovery_step is None:
                return self._stop("no safe route to required room switch")
            return ExplorationAction(move=recovery_step[0], status="leaving obstacle margin for room switch")
        old_waypoint = self.waypoint
        if self.waypoint is None or math.dist(start, self.waypoint) <= 4:
            self.waypoint = _point_waypoint(start, self.target[1:], bounds, roomy, phase)
            if self.waypoint is None:
                self.waypoint = _point_waypoint(start, self.target[1:], bounds, boxes, phase)
        if (self.waypoint is None or not _free(self.waypoint, boxes)
                or not _clear(start, self.waypoint, boxes)):
            return self._stop("no safe route to required room switch")
        for point in {old_waypoint, self.waypoint} - {None}:
            distance = math.dist(start, point)
            best = self.distances.get(point)
            if best is None:
                self.distances[point] = distance
            elif best-distance >= 8 or distance <= 4 < best:
                self.distances[point], self.progress_at = distance, now
        if now-self.started >= 20 or now-self.progress_at >= 4:
            return self._stop("room switch approach stalled")
        # Match the observed two physics steps plus input delay used by door
        # steering; four frames can overshoot a plate near the boundary.
        move = _steer(start, self.waypoint, velocity, bounds, boxes, deadband=1.5, drift_frames=5)
        vector = _VECTORS[move]
        predicted = start[0]+velocity[0]*2+vector[0]*2, start[1]+velocity[1]*2+vector[1]*2
        if not _inside(predicted, bounds) or not _clear(start, predicted, boxes):
            return self._stop("room switch approach blocked by current momentum")
        return ExplorationAction(move=move, status="pressing required room switch")
