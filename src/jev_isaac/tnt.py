"""Bounded tear demolition for a required switch, using fresh game feedback.

TNT damage states are observed, never a predicted fuse or fixed tear count.
The planner's blast distances are risk estimates, not engine-certified immunity.
In crowded puzzles the best available retreat may still carry chain-blast risk.
"""
from dataclasses import dataclass
import math

from .adventure import _inventory_ids
from .combat import _inside, _point
from .exploration import _point_waypoint, _validated
from .navigation import _VECTORS, _clear, _free, _steer
from .props import PropApproach, _settle_move
from .protocol import number
from .tnt_geometry import plan_demolition, shot_direction


@dataclass(frozen=True)
class TntAction:
    move: str = "none"
    shoot: str = "none"
    status: str = "clearing TNT for required room switch"
    stop_reason: str | None = None
    hold_frames: int = 6


def ordinary_tears(player):
    inventory = _inventory_ids(player)
    weapons = player.get("weapon_types", [player.get("weapon_type")])
    return (type(player.get("weapon_type")) is int and player["weapon_type"] == 1
            and isinstance(weapons, list) and bool(weapons)
            and all(type(value) is int and value == 1 for value in weapons)
            and inventory is not None
            and not inventory & {5, 52, 149, 168, 222, 233, 257, 329, 394, 401, 418, 561, 570, 572}
            and number(player.get("tear_range")) and player["tear_range"] >= 160)


def _fingerprint(state):
    return tuple(sorted((str(h.get("kind")), str(h.get("index", h.get("id"))),
                         h.get("type", 0), h.get("state", -1), h.get("collision", -1),
                         h["x"], h["y"], h.get("vx", 0), h.get("vy", 0))
                        for h in state["hazards"]))


def _explosives(state):
    return tuple(sorted((str(h.get("kind")), str(h.get("index", h.get("id"))),
                         h.get("type", 0), h.get("variant", -1),
                         h["x"], h["y"], h.get("vx", 0), h.get("vy", 0))
                        for h in state["hazards"] if h["kind"] == "tnt"
                        or h["kind"] == "grid" and h.get("type") in (5, 12) and h.get("collision") != 0))


class TntDemolition:
    def __init__(self, *, target_index=None):
        self.requested_target_index = target_index
        self.plan = None
        self.active = self.complete = False
        self.context = None
        self.phase = "approach"
        self.started = self.progress_at = self.changed_at = self.fired_at = None
        self.last_frame = -1
        self.last_fire_frame = self.changed_frame = None
        self.fingerprint = None
        self.explosives = None
        self.pulses = 0
        self.best = {}
        self.approach = PropApproach()
        self.damage = -1
        self.waypoint = None

    def _stop(self, message):
        return TntAction(status=message, stop_reason="TNT demolition stopped")

    def _selected(self, state):
        rows = [h for h in state["hazards"] if h.get("kind") == "grid"
                and h.get("index") == self.plan.target_index]
        if len(rows) == 1:
            row = rows[0]
            if (type(row.get("type")) is int and row["type"] == 12
                    and type(row.get("variant")) is int and row["variant"] == 0
                    and _point(row) == self.plan.target_point
                    and type(row.get("state")) is int and 0 <= row["state"] <= 4
                    and type(row.get("collision")) is int):
                # The live fourth-hit frame reports EXPLODED while the grid
                # still collides. Keep it as the same, nonshootable target
                # until collision/disappearance confirms that passage opened.
                if row["state"] <= 4 and row["collision"] == 2:
                    return row, False
                if row["state"] == 4 and row["collision"] == 0:
                    return None, True
            return None, False
        gone = not rows and not any(_point(h) == self.plan.target_point for h in state["hazards"])
        return None, gone

    def step(self, state, now, switch_point, boxes, phase):
        parsed = _validated(state)
        if (parsed is None or not state["enabled"] or state["paused"]
                or state["player"]["dead"] or state["room"]["clear"]
                or any(e["hp"] > 0 for e in state["enemies"]) or state["projectiles"]
                or any(h["kind"] in ("bomb", "laser") for h in state["hazards"])
                or not ordinary_tears(state["player"])):
            return self._stop("TNT plan interrupted by changed room or controls") if self.active else None
        context = (state["run_id"], state["session"], state["room_id"],
                   state["floor"]["id"], state["floor"].get("dimension"), tuple(switch_point))
        if self.context is not None and context != self.context:
            return self._stop("TNT plan identity changed")
        if state["frame"] <= self.last_frame:
            return TntAction(status="waiting for fresh TNT observation")
        self.last_frame = state["frame"]
        start, radius, velocity, bounds = parsed[3:7]
        if not _inside(start, bounds) or not _free(start, boxes):
            return self._stop("TNT approach blocked by current geometry") if self.active else None
        if self.plan is None:
            self.plan = plan_demolition(state, switch_point, boxes, phase,
                                        target_index=self.requested_target_index)
            if self.plan is None:
                return None
            self.context, self.active = context, True
            self.explosives = _explosives(state)
            self.started = self.progress_at = self.changed_at = now
        if now-self.started >= 60 or self.pulses >= 12 and self.phase == "approach":
            return self._stop("TNT demolition time or shot limit reached")
        fingerprint = _fingerprint(state)
        if fingerprint != self.fingerprint:
            self.fingerprint, self.changed_at = fingerprint, now
            self.changed_frame = state["frame"]
        selected, gone = self._selected(state)
        if selected is None and not gone:
            return self._stop("TNT identity changed before destruction was confirmed")
        if selected is not None and selected["state"] > self.damage:
            self.damage, self.progress_at = selected["state"], now
        exploding = selected is not None and selected["state"] == 4
        if (gone or exploding) and self.phase == "approach":
            self.phase, self.fired_at = "retreat", now
            self.last_fire_frame = state["frame"]
            self.progress_at, self.waypoint = now, None
            self.approach = PropApproach()
        if self.phase == "approach" and _explosives(state) != self.explosives:
            return self._stop("TNT positions changed; a fresh demolition plan is needed")
        destination = self.plan.retreat_point if self.phase != "approach" else self.plan.firing_point
        if not _inside(destination, bounds) or not _free(destination, boxes):
            return self._stop("TNT firing or retreat position became blocked")
        if (self.waypoint is None or math.dist(start, self.waypoint) <= 4
                or not _clear(start, self.waypoint, boxes)):
            self.waypoint = _point_waypoint(start, destination, bounds, boxes, phase)
        if self.waypoint is None or not _clear(start, self.waypoint, boxes):
            return self._stop("TNT approach or retreat route became blocked")
        for point in (self.waypoint, destination):
            key = (self.phase, point)
            distance = math.dist(start, point)
            if key not in self.best or self.best[key]-distance >= 6:
                self.best[key] = distance
                self.progress_at = now
        final_segment = self.waypoint == destination and math.dist(start, destination) <= 36
        if final_segment:
            move = _settle_move(state, start, destination, velocity, bounds, boxes, self.approach)
            if move is None:
                return self._stop("TNT movement blocked by current momentum")
        else:
            move = _steer(start, self.waypoint, velocity, bounds, boxes, deadband=2, drift_frames=4)
        vector = _VECTORS[move]
        predicted = (start[0]+velocity[0]*2+vector[0]*2, start[1]+velocity[1]*2+vector[1]*2)
        if not _inside(predicted, bounds) or not _clear(start, predicted, boxes):
            # A recorded retreat reached the exact inset boundary with less
            # than half a unit of residual velocity. Neutral input then left
            # that inset, despite a clear inward correction. Permit only this
            # bounded boundary case; obstacle or fast-momentum failures retain
            # their stop. The one-frame pulse never renews the progress timer.
            if (move == "none" and self.phase != "approach"
                    and math.hypot(*velocity) <= .5 and not _inside(predicted, bounds)
                    and now-self.progress_at < 6):
                horizontal = "right" if predicted[0] < bounds[0] else "left" if predicted[0] > bounds[2] else ""
                vertical = "down" if predicted[1] < bounds[1] else "up" if predicted[1] > bounds[3] else ""
                correction = f"{vertical}_{horizontal}" if horizontal and vertical else horizontal or vertical
                vector = _VECTORS[correction]
                end = (start[0]+velocity[0]*2+vector[0]*2,
                       start[1]+velocity[1]*2+vector[1]*2)
                if _inside(end, bounds) and _clear(start, end, boxes):
                    self.approach.pulse_frame = state["frame"]
                    return TntAction(move=correction, hold_frames=1,
                                     status="correcting TNT retreat boundary drift")
            return self._stop("TNT movement blocked by current momentum")
        settled = (move == "none" and math.dist(start, destination) <= 6
                   and math.hypot(*velocity) <= .5
                   and (self.approach.pulse_frame is None
                        or state["frame"]-self.approach.pulse_frame >= 6))
        if not settled:
            if now-self.progress_at >= 6:
                return self._stop("TNT approach or retreat stalled")
            return TntAction(move=move, status="retreating from TNT" if self.phase != "approach"
                             else "lining up a shot to clear blocking TNT")
        if self.phase == "approach":
            shoot = shot_direction(state, selected, start)
            coast = (start[0]+velocity[0]*8.4, start[1]+velocity[1]*8.4)
            if shoot == "none" or shoot != shot_direction(state, selected, coast):
                return self._stop("TNT firing lane changed")
            self.phase, self.fired_at = "retreat", now
            self.last_fire_frame = state["frame"]
            self.progress_at, self.waypoint = now, None
            self.pulses += 1
            self.approach = PropApproach()
            return TntAction(shoot=shoot, hold_frames=3, status="shooting blocking TNT")
        self.phase = "wait"
        if (now-self.fired_at < 2 or now-self.changed_at < .75
                or state["frame"]-self.last_fire_frame < 60
                or state["frame"]-self.changed_frame < 23):
            return TntAction(status="waiting for TNT explosions to settle")
        if gone:
            self.complete, self.active = True, False
            return TntAction(status="TNT cleared; checking the route to the switch")
        if exploding:
            return TntAction(status="waiting for exploding TNT collision to clear")
        self.phase, self.progress_at, self.waypoint = "approach", now, None
        self.best.clear()
        self.approach = PropApproach()
        return TntAction(status="checking TNT damage before the next shot")
