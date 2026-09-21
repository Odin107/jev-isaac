"""Execute one freshly selected interaction with observed-state feedback."""
from dataclasses import dataclass
import math
import uuid

from .adventure import candidate_geometry, candidate_valid, candidates, _grid_id
from .combat import _inside, _point
from .navigation import _clear, _free, _steer, _velocity
from .pickups import signature


@dataclass(frozen=True)
class InteractionAction:
    move: str = "none"
    shoot: str = "none"
    status: str = "considering interactions"
    interaction: str = "none"
    interaction_id: str | None = None
    transition: str | None = None
    stop_reason: str | None = None


class AdventureControl:
    def __init__(self):
        self.visit = None
        self.offers = ()
        self.plan = None
        self.skipped = set()
        self.started = self.progress_at = 0
        self.progress_distance = None
        self.phase = "approach"
        self.pulse_id = None
        self.used_at = None
        self.counter = 0
        self.namespace = uuid.uuid4().hex[:12]
        self.replaced_active = False
        self.selected = self.completed = self.abandoned = 0
        self.descent_requested = False
        self.events = []
        self.paused_at = None
        self.progress_marker = None
        self.prop_approach = None
        self.prop_distances = {}

    def observe_pause(self, state, now):
        paused = state.get("enabled") is not True or state.get("paused") is not False
        if paused:
            if self.paused_at is None:
                self.paused_at = now
        elif self.paused_at is not None:
            duration = max(0, now - self.paused_at)
            self.started += duration
            self.progress_at += duration
            if self.used_at is not None:
                self.used_at += duration
            self.paused_at = None

    def _event(self, state, kind, plan, reason):
        self.events.append({"frame": state["frame"], "room_id": state["room_id"],
                            "event": kind, "candidate": plan.as_dict(), "reason": reason})
        del self.events[:-80]

    def reset_visit(self, state):
        visit = state["run_id"], state["room_id"]
        if self.visit != visit:
            self.visit = visit
            self.offers, self.plan = (), None
            self.skipped.clear()
            self.replaced_active = False
            self.descent_requested = False
            self.paused_at = None
            self.prop_approach = None
            self.prop_distances = {}

    def accept(self, key, state, now, *, allow_descend=False,
               decision_reason="Jev selected a verified candidate"):
        self.reset_visit(state)
        if key is None:
            self.skipped.update(item.key for item in self.offers)
            self.offers = ()
            return False
        plan = next((item for item in self.offers if item.key == key), None)
        if plan is None or not candidate_valid(state, plan, rewards_done=True, allow_descend=allow_descend):
            self.skipped.add(key)
            self.offers = ()
            return False
        self.plan, self.offers = plan, ()
        self.started = self.progress_at = now
        self.progress_distance = None
        self.progress_marker = None
        self.prop_approach = None
        self.prop_distances = {}
        self.phase, self.used_at = "approach", None
        self.counter += 1
        self.pulse_id = f"{self.namespace}:{self.counter}"
        self.selected += 1
        self._event(state, "selected", plan, decision_reason)
        return True

    def _finish(self, state, *, success=False, reason):
        plan = self.plan
        if plan is not None:
            self.skipped.add(plan.key)
            self.completed += int(success)
            self.abandoned += int(not success)
            if plan.details.get("replaces_active"):
                self.replaced_active = True
            self._event(state, "finished" if success else "abandoned", plan, reason)
        self.plan = None
        return InteractionAction(status=reason)

    def choose_offers(self, state, *, allow_descend=False, excluded_curse_targets=()):
        self.reset_visit(state)
        if self.plan is not None:
            return ()
        offered = candidates(state, rewards_done=True, allow_descend=allow_descend)
        self.offers = tuple(item for item in offered if item.key not in self.skipped
                            and not (item.kind == "enter_curse" and item.details["target_index"] in excluded_curse_targets)
                            and not (self.replaced_active and item.details.get("replaces_active")))
        return self.offers

    def _move(self, state, point, *, retreat=False):
        from .exploration import _point_waypoint
        selected = state
        if retreat:
            # The pre-verified retreat starts at our own just-dropped bomb.
            # Its center is not a solid obstacle; escape outward from it while
            # retaining every other hazard/obstacle and the verified corridor.
            selected = dict(state, hazards=[item for item in state["hazards"]
                if not (item.get("kind") == "bomb"
                        and math.dist(_point(item), self.plan.point) <= 30)])
        geometry = candidate_geometry(selected, self.plan)
        if geometry is None:
            return None
        bounds, boxes, phase = geometry
        start, velocity = _point(state["player"]), _velocity(state["player"])
        if not _inside(start, bounds) or not _free(start, boxes):
            return None
        waypoint = _point_waypoint(start, point, bounds, boxes, phase)
        if waypoint is None:
            return None
        return _steer(start, waypoint, velocity, bounds, boxes, deadband=1.5, drift_frames=4)

    def step(self, state, now, *, allow_descend=False):
        self.reset_visit(state)
        plan = self.plan
        if plan is None:
            return None
        self.observe_pause(state, now)
        if state.get("enabled") is not True or state.get("paused") is not False or state["player"]["dead"]:
            return InteractionAction(status="interaction paused")
        if now - self.started >= 15:
            return self._finish(state, reason="interaction timed out")
        if plan.kind == "shoot_prop":
            from .props import PropApproach, prop_action, prop_damage, prop_finished
            if prop_finished(state, plan):
                return self._finish(state, success=True, reason="prop destroyed or disappeared; rechecking rewards")
            if self.prop_approach is None:
                self.prop_approach = PropApproach()
            old_waypoint = self.prop_approach.waypoint
            action = prop_action(state, plan, self.prop_approach)
            if action is None:
                return self._finish(state, reason="no safe firing lane to selected prop")
            marker = prop_damage(state, plan)
            if self.progress_marker is None or marker > self.progress_marker:
                self.progress_marker, self.progress_at = marker, now
            # Compare against the best distance to each fixed approach point.
            # Moving away and returning, or unrelated nearby entity changes,
            # must never renew the interaction's no-progress watchdog.
            for point in {old_waypoint, self.prop_approach.waypoint} - {None}:
                distance = math.dist(_point(state["player"]), point)
                previous = self.prop_distances.get(point)
                if previous is None:
                    self.prop_distances[point] = distance
                elif previous-distance >= 8 or distance <= 4 < previous:
                    self.prop_distances[point], self.progress_at = distance, now
            if now-self.progress_at >= 4:
                return self._finish(state, reason="shooting made no observed progress")
            return InteractionAction(move=action.move, shoot=action.shoot, status=plan.description)
        if plan.kind in ("enter_curse", "leave_curse"):
            from .curse_doors import curse_move, curse_valid
            if not curse_valid(state, plan, committed=self.phase == "crossing"):
                return self._finish(state, reason="curse-door route or health budget changed")
            move = curse_move(state, plan)
            if math.dist(_point(state["player"]), plan.point) <= 45:
                self.phase = "crossing"  # Finish crossing after the first damage; never oscillate at spikes.
            return InteractionAction(move=move, status=plan.description)
        if plan.pickup_signature and plan.kind != "bomb_rock":
            if not any(signature(item) == plan.pickup_signature for item in state["pickups"]):
                return self._finish(state, success=True, reason="target changed or disappeared; rechecking rewards")
        if plan.kind == "unlock_door":
            door = next((d for d in state["doors"] if d["slot"] == plan.details["slot"]
                         and d["target_index"] == plan.details["target_index"]), None)
            if door is not None and not door["locked"] and door["open"]:
                return self._finish(state, success=True, reason="door unlocked")
        if self.phase == "approach" and not candidate_valid(state, plan, rewards_done=True,
                                                            allow_descend=allow_descend):
            return self._finish(state, reason="interaction is no longer viable")
        if plan.kind in ("use_active", "use_pocket"):
            if self.used_at is None:
                self.used_at = now
                self.phase = "used"
                return InteractionAction(status=plan.description, interaction=plan.interaction,
                                         interaction_id=self.pulse_id)
            if now - self.used_at >= .5:
                return self._finish(state, success=True, reason="item-use pulse sent; rechecking state")
            return InteractionAction(status="waiting for item effect")
        if plan.kind == "unlock_door":
            from .exploration import _Door, _door_move, _validated
            door = _Door(plan.details["slot"], plan.point, plan.details["target_index"], plan.details["target_type"])
            move = _door_move(state, _validated(state), door)
        elif plan.kind == "bomb_rock":
            if self.phase == "approach" and math.dist(_point(state["player"]), plan.point) <= 5:
                # Require the already checked straight retreat at placement,
                # not just a route computed when Jev's request was made.
                geometry = candidate_geometry(state, plan)
                if geometry is None or not _clear(_point(state["player"]), plan.escape_point, geometry[1]):
                    return self._finish(state, reason="straight bomb retreat became blocked")
                if math.hypot(*_velocity(state["player"])) > .5:
                    return InteractionAction(status="settling before bomb placement")
                self.phase, self.used_at = "retreat", now
                self.started = now
                move = self._move(state, plan.escape_point, retreat=True)
                if move is None:
                    return self._finish(state, reason="bomb retreat became blocked")
                return InteractionAction(move=move, status="placing one bomb and retreating",
                                         interaction="bomb", interaction_id=self.pulse_id)
            if self.phase == "retreat":
                if math.dist(_point(state["player"]), plan.point) >= 135 + state["player"]["radius"]:
                    self.phase = "wait_blast"
                else:
                    move = self._move(state, plan.escape_point, retreat=True)
                    if move is None:
                        return InteractionAction(status="bomb retreat blocked", stop_reason="bomb retreat blocked")
                    return InteractionAction(move=move, status="retreating from placed bomb")
            if self.phase == "wait_blast":
                still_rock = any(h.get("kind") == "grid" and _grid_id(h) == plan.rock_id
                                 and h.get("collision") == 3 for h in state["hazards"])
                if not still_rock and not any(h.get("kind") == "bomb" for h in state["hazards"]):
                    return self._finish(state, success=True, reason="selected rock destroyed; rechecking routes and rewards")
                return InteractionAction(status="waiting outside bomb range")
            move = self._move(state, plan.point)
        else:
            move = self._move(state, plan.point)
        if move is None:
            return self._finish(state, reason="no walking route to selected interaction")
        distance = math.dist(_point(state["player"]), plan.point)
        if self.progress_distance is None or self.progress_distance - distance >= 8:
            self.progress_distance, self.progress_at = distance, now
        if now - self.progress_at >= 3:
            return self._finish(state, reason="selected interaction made no progress")
        transition = "floor" if plan.kind == "descend" and distance <= 60 else None
        if transition:
            self.descent_requested = True
        return InteractionAction(move=move, status=plan.description, transition=transition)
