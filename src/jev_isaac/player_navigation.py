"""Execute Jev-selected activities; never choose the next room or pickup."""
import copy
import math
from dataclasses import replace

from .adventure import AdventureCandidate, _context, candidates, candidate_valid
from .combat import _number, _point
from .combat_stall import no_living_enemies
from .exploration import FloorNavigator, ExplorationAction, _Door, _door_move, _validated, _lingering_projectile_move, _ground_effect_escape
from .protocol import valid_switches
from .state_context import room_name
from .switches import SwitchNavigator, _ordinary, _geometry
from .tnt import TntDemolition, ordinary_tears
from .tnt_geometry import plan_demolition


class PlayerNavigator(FloorNavigator):
    def __init__(self, *, continue_floors=False):
        super().__init__(adventure_mode=True, continue_floors=continue_floors, allow_secret=True)
        self._offers = ()
        self._intent = None
        self._selected_switch = SwitchNavigator()
        self._tnt = None
        self._tnt_guard = None
        self._activity_started = self._activity_progress = 0
        self._activity_position = None
        self._idle_until = self._idle_frame = 0
        self._player_visit = None
        self._input_start_frame = 0
        self.player_events = []
        self.activity_failures = []
        self._reward_signature = None
        self._reward_was_clear = False
        self._room_choice_revision = 0
        self._reward_wait_started_frame = None
        self._reward_ready = True

    @staticmethod
    def _reward_snapshot(state):
        # Identity/readiness changes matter to the available choices. Bouncing
        # positions, velocity and a still-positive countdown do not.
        if not isinstance(state.get("pickups"), list):
            return None  # Optional missing data does not prove an empty room.
        return (state["player"].get("can_pickup_items"), tuple(sorted(
            (p["id"], p["variant"], p["subtype"], p["price"], p["shop_item"],
             p["options_index"], p["collectible_kind"], p["wait"] == 0)
            for p in state.get("pickups", []))))

    def _waiting_for_rewards(self, state):
        return (not self._reward_ready and self._reward_wait_started_frame is not None
                and state["frame"]-self._reward_wait_started_frame < 90)

    def _event(self, state, event, reason, candidate=None, *, origin_index=None):
        candidate = candidate or self._intent
        self.player_events.append({"frame": state["frame"], "event": event,
            "room_index": state["floor"]["room_index"] if origin_index is None else origin_index,
            "target_index": candidate.details.get("target_index") if candidate else None,
            "key": candidate.key if candidate else None,
            "kind": candidate.kind if candidate else "wait", "reason": reason})
        del self.player_events[:-80]

    def _remember_blast(self):
        if self._tnt is not None and self._tnt.last_fire_frame is not None:
            context = self._tnt.context
            self._tnt_guard = ((context[0], context[2], context[3], context[4]), self._tnt.last_fire_frame+60)

    def cancel_intent(self):
        self._remember_blast()
        self._offers, self._intent, self._tnt = (), None, None
        self._selected_switch.reset()
        self._adventure.plan, self._adventure.offers = None, ()
        self._reset_traversal()

    def rearmed(self, state):
        self._remember_blast()
        remembered = super().rearmed(state)
        fresh = PlayerNavigator(continue_floors=self.continue_floors)
        fresh.__dict__.update(remembered.__dict__)
        fresh._tnt_guard = self._tnt_guard
        fresh.player_events = copy.deepcopy(self.player_events)
        fresh.activity_failures = copy.deepcopy(self.activity_failures)
        return fresh

    def observe(self, state):
        previous = self._intent
        previous_index = self._current_index
        super().observe(state)
        if self._stop or not state["enabled"] or state["paused"] or state["player"]["dead"]:
            return
        visit = (state["run_id"], state["room_id"])
        new_visit = self._player_visit != visit
        if new_visit:
            if previous is not None:
                self._event(state, "finished", "observed room transition", previous, origin_index=previous_index)
            self._offers, self._intent, self._tnt = (), None, None
            self._selected_switch.reset()
            self._player_visit = visit
            self._idle_until = 0
            self._idle_frame = state["frame"]+18
            self._reward_signature = None
            self._reward_wait_started_frame = None
            self._reward_ready = True
        snapshot = self._reward_snapshot(state)
        changed_rewards = (not new_visit and snapshot is not None
                           and snapshot != self._reward_signature)
        just_cleared = not new_visit and state["room"]["clear"] and not self._reward_was_clear
        if new_visit or changed_rewards or just_cleared:
            self._room_choice_revision += 1
            self._offers = ()
        if state["room"]["clear"] and (changed_rewards or just_cleared):
            self._idle_frame = max(self._idle_frame, state["frame"]+(18 if just_cleared else 12))
            selected = self._intent
            departure = selected is not None and selected.kind in (
                "enter_door", "unlock_door", "descend", "enter_curse", "leave_curse")
            committed = (selected is not None and (
                selected.kind == "descend" and self.descent_requested
                or selected.kind in ("enter_curse", "leave_curse")
                and self._adventure.plan is selected and self._adventure.phase == "crossing"))
            if departure and not committed:
                self._event(state, "canceled", "room rewards changed; Jev needs a fresh departure choice")
                self.cancel_intent()
        self._reward_was_clear = state["room"]["clear"]
        if snapshot is not None:
            self._reward_signature = snapshot
            remaining = [p for p in state["pickups"]
                         if not (p["variant"] in (50, 60, 100) and p["subtype"] == 0)]
            self._reward_ready = (state["player"].get("can_pickup_items") is not False
                                  and all(p["wait"] == 0 for p in remaining))
        if self._reward_ready or not state["room"]["clear"]:
            self._reward_wait_started_frame = None
        elif self._reward_wait_started_frame is None:
            self._reward_wait_started_frame = state["frame"]
        if not state["room"]["clear"] and not no_living_enemies(state) and self._intent is not None:
            self._event(state, "canceled", "combat resumed; Jev needs a fresh combat choice")
            self.cancel_intent()

    @property
    def adventure_options(self):
        if self._intent is not None and self.allows_independent_fire:
            return (AdventureCandidate("continue", "continue", self._intent.key, None, {},
                    "Continue the existing bound activity; update only firing", context=self._intent.context,
                    details={"activity": self._intent.as_dict(),
                             "room_choice_revision": self._room_choice_revision}),)
        return self._offers

    @property
    def allows_independent_fire(self):
        return (self._rooms.get(self._current, (None, False))[1]
                and self._tnt_guard is None and self._tnt is None
                and (self._intent is None or self._intent.kind not in ("shoot_prop", "demolish_tnt", "bomb_rock")))

    @property
    def adventure_stats(self):
        return dict(super().adventure_stats, player_events=copy.deepcopy(self.player_events),
                    activity_failures=copy.deepcopy(self.activity_failures))

    def decision_context(self):
        context = super().decision_context()
        context["authority"] = "Jev chooses all activities; no automatic room order, free pickups or puzzle selection."
        context["selected_activity"] = ({"key": self._intent.key, "kind": self._intent.kind}
                                        if self._intent else None)
        context["recent_player_outcomes"] = copy.deepcopy(self.player_events[-6:])
        context["recent_room_transitions"] = [
            {"frame": e["frame"], "from_room_index": e["room_index"], "to_room_index": e["target_index"]}
            for e in self.player_events if e["event"] == "finished"
            and e["reason"] == "observed room transition" and e["target_index"] is not None][-12:]
        context["room_transition_scope"] = "Recent confirmed crossings retained on this floor, oldest first; not planned moves or an instruction to repeat them. Manual moves while disarmed are not recorded."
        context["secret_rooms"] = "Observed open secret and supersecret doors can be selected. Unopened hidden entrances are not search/bomb candidates."
        context["reward_observation"] = {
            "choice_revision": self._room_choice_revision,
            "settling_until_frame": self._idle_frame,
            "pickups_ready": self._reward_ready,
            "readiness_wait_limit_frames": 90,
            "readiness_wait_expired": (not self._reward_ready and self._reward_wait_started_frame is not None
                                        and self._frame-self._reward_wait_started_frame >= 90),
            "scope": "Wait briefly after room clear and observed reward changes so late drops can appear. Unready pickups can still be unavailable after the bounded wait; Jev chooses whether to wait or leave."}
        return context

    def _puzzle_offers(self, state, parsed):
        if (state["room"].get("has_trigger_pressure_plates") is not True
                or not valid_switches(state.get("switches")) or state["room"]["type"] != 1
                or state.get("capabilities", {}).get("room_switches") != 1):
            return []
        result = []
        for row in state["switches"]:
            if not _ordinary(row, 0):
                continue
            point = _point(row)
            result.append(AdventureCandidate(f"switch:{row['index']}", "press_switch", str(row["index"]),
                point, {}, "Press this observed required pressure plate; this does not authorize demolition",
                context=_context(state), details={"switch_index": row["index"]}))
            if not ordinary_tears(state["player"]):
                continue
            boxes, _, _, phase = _geometry(state, parsed[4], (row["index"], *point))
            for hazard in state["hazards"]:
                if (hazard.get("kind") != "grid" or hazard.get("type") != 12
                        or type(hazard.get("index")) is not int):
                    continue
                plan = plan_demolition(state, point, boxes, phase, target_index=hazard["index"])
                if plan is not None:
                    result.append(AdventureCandidate(f"tnt:{hazard['index']}:switch:{row['index']}",
                        "demolish_tnt", str(hazard["index"]), plan.target_point, {},
                        "Shoot this blocking TNT with bounded tear pulses and retreat; possible chain damage. Then ask Jev again",
                        context=_context(state), details={"switch_index": row["index"], "switch_point": point,
                            "tnt_index": hazard["index"], "firing_point": plan.firing_point,
                            "retreat_point": plan.retreat_point}))
        return result

    def _offer(self, state, parsed):
        choices = []
        self._allow_descend = self.continue_floors and state["room"]["clear"] and state["room"]["type"] == 5
        if state["room"]["clear"]:
            from .room_inputs import room_input_candidates
            choices.extend(room_input_candidates(state, parsed))
            choices.extend(candidates(state, rewards_done=True, allow_descend=self._allow_descend,
                                      limit=160, include_rock_targets=True))
            for door in parsed[-1]:
                destination = self._aliases.get(door.target_index)
                known = self._rooms.get(destination)
                recent = [e for e in self.player_events if e.get("room_index") == state["floor"]["room_index"]
                          and e.get("key") == f"enter:{door.slot}:{door.target_index}"]
                known_exits = [{"target_index": d.target_index, "target_type": d.target_type,
                               "target_room": room_name(d.target_type),
                               "visited": self._aliases.get(d.target_index) in self._rooms}
                              for d in self._graph.get(destination, ())]
                pickups = self._pickup_context(destination)
                return_only = (bool(known_exits) and all(
                    self._aliases.get(d["target_index"]) == self._current for d in known_exits)
                    if destination in self._inspected else None)
                description = (f"{'Revisit' if known is not None else 'Enter'} the observed "
                               f"{room_name(door.target_type)} room {door.target_index}")
                if pickups["status"] == "none_observed":
                    description += "; no remaining pickups at its last complete observation"
                elif pickups["status"] == "present":
                    description += f"; {sum(g['count'] for g in pickups['groups'])} pickups last observed"
                else:
                    description += "; remaining pickups unknown"
                if return_only:
                    description += "; its last observed permitted exits only lead back to this room"
                choices.append(AdventureCandidate(f"enter:{door.slot}:{door.target_index}", "enter_door",
                    str(door.target_index), door.point, {}, description,
                    context=_context(state), details={"slot": door.slot, "target_index": door.target_index,
                        "target_type": door.target_type, "visited": known is not None,
                        "clear_last_observed": known[1] if known is not None else None,
                        "remembered_destination_exits": known_exits,
                        "destination_doors_inspected": destination in self._inspected,
                        "destination_pickups_last_observed": pickups,
                        "destination_is_known_return_only": return_only,
                        "recent_times_selected_from_here": sum(e["event"] == "selected" for e in recent),
                        "recent_times_entered_from_here": sum(e["event"] == "finished"
                            and e["reason"] == "observed room transition" for e in recent),
                        "last_outcome_from_here": next((e["reason"] for e in reversed(recent)
                                                         if e["event"] in ("finished", "failed")), None)}))
            if state["room"]["type"] == 10:
                from .curse_doors import curse_candidates
                choices.extend(curse_candidates(state, leaving=True))
        else:
            choices.extend(self._puzzle_offers(state, parsed))
        # Keep model waiting explicit, including a room with no supported action.
        choices.append(AdventureCandidate("wait", "wait", "wait", None, {},
                                          "Wait briefly, then reconsider", context=_context(state)))
        self._offers = tuple(replace(c, details=dict(c.details,
            room_choice_revision=self._room_choice_revision))
            for c in {c.key: c for c in choices}.values())[:192]

    def accept_adventure(self, key, state, now, **kwargs):
        self.observe(state)
        if self._stop:
            return False
        selected = next((c for c in self.adventure_options if c.key == key), None)
        if key is None or key == "wait":
            self._event(state, "selected", "Jev chose to wait")
            self._offers = ()
            self._idle_until = now+1
            return True
        self._offers = ()
        if selected is None or selected.context != _context(state):
            return False
        parsed = _validated(state, allow_shop=True, allow_secret=True)
        if (parsed is None or not state["enabled"] or state["paused"] or state["player"]["dead"]
                or (not state["room"]["clear"] and not no_living_enemies(state))):
            return False
        if selected.kind == "continue":
            return self._intent is not None and self._intent.key == selected.target_id
        if selected.kind == "move_to":
            from .room_inputs import room_input_valid
            if not room_input_valid(state, parsed, selected):
                return False
            self._input_start_frame = state["frame"]
        elif selected.kind == "enter_door":
            d = selected.details
            chosen = _Door(d["slot"], selected.point, d["target_index"], d["target_type"])
            if not state["room"]["clear"] or chosen not in parsed[-1]:
                return False
            self._pending = chosen
        elif selected.kind in ("press_switch", "demolish_tnt"):
            row = next((r for r in state.get("switches", []) if r["index"] == selected.details["switch_index"]), None)
            point = selected.point if selected.kind == "press_switch" else tuple(selected.details["switch_point"])
            if row is None or not _ordinary(row, 0) or _point(row) != point or state["room"]["clear"]:
                return False
            self._selected_switch.reset()
            if selected.kind == "demolish_tnt":
                from .tnt_geometry import _ordinary as ordinary_tnt
                target = next((h for h in state["hazards"] if h.get("kind") == "grid"
                    and h.get("index") == selected.details["tnt_index"]), None)
                if target is None or not ordinary_tnt(target) or _point(target) != selected.point:
                    return False
                self._tnt = TntDemolition(target_index=selected.details["tnt_index"])
        else:
            if not candidate_valid(state, selected, rewards_done=True, allow_descend=self._allow_descend):
                return False
            self._adventure.offers = (selected,)
            if not super().accept_adventure(key, state, now):
                return False
        self._intent = selected
        self._activity_started = self._activity_progress = now
        self._activity_position = _point(state["player"])
        self._event(state, "selected", "Jev selected this activity")
        return True

    def _done(self, state, now, reason, *, failed=False):
        if failed:
            self.activity_failures.append(copy.deepcopy({"reason": reason,
                "candidate": self._intent.as_dict() if self._intent else None, "observation": state}))
            del self.activity_failures[:-8]
        self._event(state, "failed" if failed else "finished", reason)
        self.cancel_intent()
        self._idle_until, self._idle_frame = now+.4, state["frame"]+12
        return ExplorationAction(status=f"Activity {'failed' if failed else 'finished'}: {reason}; asking Jev again")

    def step(self, state, now_monotonic):
        now = now_monotonic
        if not _number(now, limit=1e15) or self._last_now is not None and now < self._last_now:
            return self._finish("invalid exploration clock")
        self._last_now = now
        self.observe(state)
        if self._stop:
            return ExplorationAction(stop_reason=self._stop)
        if not state["enabled"] or state["paused"] or state["player"]["dead"]:
            return ExplorationAction(status="waiting for armed play")
        parsed = _validated(state, allow_shop=True, allow_secret=True)
        if parsed is None:
            return ExplorationAction(stop_reason="incomplete floor observation")
        if not state["room"]["clear"] and not no_living_enemies(state):
            return ExplorationAction(status="combat")
        if self._tnt_guard is not None and self._tnt is None:
            identity, frame = self._tnt_guard
            actual = (state["run_id"], state["room_id"], state["floor"]["id"], state["floor"].get("dimension"))
            if identity == actual and state["frame"] < frame:
                return ExplorationAction(status="Local override: waiting for committed TNT shot after rearm")
            self._tnt_guard = None
        # Once explosives are committed, finish the selected retreat. Otherwise
        # a fresh collision dodge can briefly interrupt movement, never pick loot.
        bomb_retreat = self._adventure.plan is not None and self._adventure.phase in ("retreat", "wait_blast")
        if self._tnt is None and not bomb_retreat:
            escape = _ground_effect_escape(state, parsed)
            if escape is not None:
                return ExplorationAction(move=escape, status="Local override: leaving enemy floor creep")
        if state["projectiles"] and self._tnt is None and not bomb_retreat:
            return ExplorationAction(move=_lingering_projectile_move(state, parsed),
                                     status="Local override: emergency projectile avoidance")
        selected = self._intent
        if selected is not None:
            if selected.kind == "move_to":
                from .room_inputs import room_input_move, room_input_valid
                if not room_input_valid(state, parsed, selected):
                    return self._done(state, now, "selected input no longer valid", failed=True)
                elapsed = now-self._activity_started
                if math.dist(_point(state["player"]), selected.point) <= 5:
                    return self._done(state, now, "selected room position reached")
                if elapsed >= 3 or state["frame"]-self._input_start_frame >= 90:
                    return self._done(state, now, "selected repositioning timed out", failed=True)
                return ExplorationAction(move=room_input_move(state, parsed, selected.point),
                                         status="Jev selected: move to room position")
            elif selected.kind == "press_switch":
                row = next((r for r in state.get("switches", []) if r["index"] == selected.details["switch_index"]), None)
                if state["room"]["clear"] or row is not None and _ordinary(row, 3):
                    return self._done(state, now, "selected switch activated")
                action = self._selected_switch.step(state, now, selected_switch=selected.details["switch_index"], allow_demolition=False)
            elif selected.kind == "demolish_tnt":
                if state["room"]["clear"]:
                    return self._done(state, now, "room cleared during selected demolition")
                target = (selected.details["switch_index"], *selected.details["switch_point"])
                boxes, _, _, phase = _geometry(state, parsed[4], target)
                action = self._tnt.step(state, now, target[1:], boxes, phase)
                if self._tnt.complete:
                    return self._done(state, now, "selected TNT cleared; Jev must choose the next activity")
            elif self._adventure.plan is not None:
                action = self._adventure.step(state, now, allow_descend=self._allow_descend)
                if self._adventure.plan is None and selected.kind != "unlock_door":
                    reason = self._adventure.events[-1]["reason"] if self._adventure.events else "interaction ended"
                    failed = bool(self._adventure.events and self._adventure.events[-1]["event"] == "abandoned")
                    return self._done(state, now, reason, failed=failed)
            elif self._pending is not None:
                if self._pending not in parsed[-1]:
                    return self._done(state, now, "selected door changed", failed=True)
                move = _door_move(state, parsed, self._pending)
                if move is None:
                    return self._done(state, now, "selected door route blocked; no automatic alternate", failed=True)
                position = _point(state["player"])
                if math.dist(position, self._activity_position) >= 10:
                    self._activity_progress, self._activity_position = now, position
                if now-self._activity_started >= self._transition_timeout or now-self._activity_progress >= self._stuck_timeout:
                    return self._done(state, now, "selected door traversal stalled", failed=True)
                action = ExplorationAction(move=move, status=f"Jev selected: entering room {self._pending.target_index}")
            else:
                return self._done(state, now, "selected activity ended")
            if action is None or action.stop_reason:
                return self._done(state, now, getattr(action, "status", None) or "selected activity no longer executable", failed=True)
            return action
        if now < self._idle_until or state["frame"] < self._idle_frame:
            return ExplorationAction(status="waiting for Jev-selected pause or observed effects")
        if state["room"]["clear"] and self._waiting_for_rewards(state):
            return ExplorationAction(status="waiting for observed pickup animation or readiness")
        if any(h["kind"] in ("bomb", "laser") for h in state["hazards"]):
            return ExplorationAction(status="Local override: waiting for an explosive or laser hazard")
        if not self._offers:
            self._offer(state, parsed)
        return ExplorationAction(status="waiting for Jev's next activity")
