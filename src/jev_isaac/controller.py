"""One in-flight decision, newest state only, and bounded control lifetimes."""
from __future__ import annotations

import concurrent.futures
import copy
import json
import math
import socket
import time
import uuid
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Callable

from .protocol import MAX_DATAGRAM, MAX_FRAME_AGE, Observation, encode_action
from .combat_stall import CombatStallWatchdog, NO_COMBAT_OBJECTIVE, no_living_enemies


_RECOVERABLE_NAVIGATION_STOPS = frozenset({
    "no safe route to open door", "stuck while approaching door",
    "door traversal timed out", "pickup animation timed out",
    NO_COMBAT_OBJECTIVE, "no safe route to required room switch",
    "room switch approach stalled", "room switch activation timed out",
    "TNT demolition stopped",
    "room objective navigation failed", "local navigation failed", "floor navigation failed",
    "incomplete floor observation",
})

_OBSERVATION_BATCH_LIMIT = 64
_RECEIVE_TIMEOUT = .01


def _observation_epoch_key(observation: Observation, address) -> tuple:
    """Keep every boundary that can revoke a goal, lease, or floor plan."""
    data = observation.data
    floor = data.get("floor", {})
    return (address, observation.identity, data.get("run_id"), floor.get("id"),
            floor.get("dimension"), floor.get("room_index"), floor.get("room_list_index"),
            data["enabled"], data["paused"], data["player"]["dead"], data["room"]["clear"],
            data.get("floor_advance_permitted"), no_living_enemies(data))


def _coalescing_key(observation: Observation, address) -> tuple:
    # Echo source frames change with every command, but a changed brake or
    # interaction acknowledgement must remain visible to diagnostics/planners.
    control = observation.data.get("control")
    echo = (json.dumps({k: v for k, v in control.items() if k != "source_frame"},
                       sort_keys=True) if isinstance(control, dict) else None)
    player = observation.data["player"]
    return (_observation_epoch_key(observation, address), echo,
            tuple(player.get(k) for k in ("active_item", "active_charge", "pocket_card", "pocket_pill")))


def _receive_observations(sock, stats):
    """Drain a bounded UDP burst without waiting, retaining transitions in order.

    Receipt times are captured here, not when buffered packets are later used.
    Only consecutive same-state frames may replace one another. Invalid data
    cannot terminate the drain; a connection error is delivered after any
    already received observations so they can still be safely released.
    """
    pending = deque()
    count = 0
    started = time.perf_counter()
    previous_key = None
    try:
        while count < _OBSERVATION_BATCH_LIMIT:
            try:
                raw, address = sock.recvfrom(MAX_DATAGRAM+1)
                received_at = time.monotonic()
            except (socket.timeout, BlockingIOError):
                break
            except OSError as exc:
                pending.append(exc)
                break
            count += 1
            if count == 1:
                sock.settimeout(0)
                started = time.perf_counter()
            try:
                observation = Observation.decode(raw)
            except ValueError:
                stats.invalid_packets += 1
                continue
            key = _coalescing_key(observation, address)
            if pending and key == previous_key:
                previous = pending[-1][0]
                if observation.frame > previous.frame:
                    pending[-1] = (observation, address, received_at)
                stats.observations_coalesced += 1
            else:
                pending.append((observation, address, received_at))
                previous_key = key
    finally:
        sock.settimeout(_RECEIVE_TIMEOUT)
    if count:
        stats.observation_batches += 1
        stats.max_observation_batch = max(stats.max_observation_batch, count)
        stats.observation_drain_limit_hits += int(count == _OBSERVATION_BATCH_LIMIT)
        stats.observation_drain_max_ms = max(stats.observation_drain_max_ms,
                                             round((time.perf_counter()-started)*1000, 3))
    return pending, count == _OBSERVATION_BATCH_LIMIT


@dataclass
class _NavigationRecovery:
    """Require a new manual arm, never a replay of an already observed session."""
    run_id: str
    floor_id: str
    retired_sessions: frozenset[str]
    frame: int

    def accepts(self, observation: Observation) -> bool:
        return (observation.controllable
                and observation.data.get("run_id") == self.run_id
                and observation.data.get("floor", {}).get("id") == self.floor_id
                and observation.identity[0] not in self.retired_sessions
                and observation.frame >= self.frame)

    def observe(self, observation: Observation) -> None:
        if (observation.data.get("run_id") == self.run_id
                and observation.data.get("floor", {}).get("id") == self.floor_id):
            self.frame = max(self.frame, observation.frame)


@dataclass(frozen=True)
class Action:
    move: str
    shoot: str
    interaction: str = "none"
    interaction_id: str | None = None


def baseline(state: dict) -> Action:
    """A deliberately simple plumbing check; this is not Jev or a trained policy."""
    enemies = state["enemies"]
    if not enemies:
        return Action("none", "none")
    player = state["player"]
    target = min(enemies, key=lambda e: (e["x"]-player["x"])**2 + (e["y"]-player["y"])**2)
    dx, dy = target["x"]-player["x"], target["y"]-player["y"]
    shoot = ("right" if dx >= 0 else "left") if abs(dx) >= abs(dy) else ("down" if dy >= 0 else "up")
    move = "none"
    if math.hypot(dx, dy) < 120:
        move = {"left":"right", "right":"left", "up":"down", "down":"up"}[shoot]
    # Conservative check against room edges. Obstacles require the real policy.
    bounds = state["room"]
    margin = 35
    if ((move == "left" and player["x"] < bounds["top_left"]["x"] + margin)
        or (move == "right" and player["x"] > bounds["bottom_right"]["x"] - margin)
        or (move == "up" and player["y"] < bounds["top_left"]["y"] + margin)
        or (move == "down" and player["y"] > bounds["bottom_right"]["y"] - margin)):
        move = "none"
    return Action(move, shoot)


@dataclass
class Stats:
    observations: int = 0
    observations_coalesced: int = 0
    observation_batches: int = 0
    max_observation_batch: int = 0
    observation_drain_limit_hits: int = 0
    observation_drain_max_ms: float = 0.0
    decisions: int = 0
    strategic_choices: int = 0
    interaction_pulses: int = 0
    floors_advanced: int = 0
    navigation_stops: int = 0
    navigation_rearms: int = 0
    last_navigation_stop: dict | None = None
    navigation_stop_snapshots: list[dict] = field(default_factory=list)
    actions_sent: int = 0
    nonneutral_actions: int = 0
    observed_control_packets: int = 0
    startup_local_actions: int = 0
    first_goal_ms: float | None = None
    first_local_action_ms: float | None = None
    first_nonneutral_action_ms: float | None = None
    first_control_echo_ms: float | None = None
    goal_updates: int = 0
    expired_goals: int = 0
    stale_discarded: int = 0
    invalid_packets: int = 0
    errors: int = 0
    last_error_type: str | None = None
    last_http_status: int | None = None
    last_validation: dict | None = None
    last_failed_observation: dict | None = None
    corrected_responses: int = 0
    choice_corrections: int = 0
    last_choice_corrections: list[dict] = field(default_factory=list)
    player_decisions: list[dict] = field(default_factory=list)
    local_overrides: int = 0
    combat_start: dict | None = None
    combat_end: dict | None = None
    first_observation: dict | None = None
    last_observation: dict | None = None
    recent_local_controls: list[dict] = field(default_factory=list)
    observed_movement_brakes: dict = field(default_factory=lambda: {"time_limit": 0, "distance_limit": 0})
    responses_with_usage: int = 0
    reported_input_tokens: int = 0
    reported_output_tokens: int = 0
    latency_ms: list[float] = field(default_factory=list)

    def record_navigation_stop(self, reason: str, observation: Observation,
                               navigator, *, recoverable: bool) -> None:
        """Freeze up to eight stopped segments before their navigator is replaced.

        These segment counts are deliberately separate from the current floor's
        progress: rooms visited again after rearming must not be double-counted.
        Only observed game data and controller diagnostics are retained here;
        the policy, API client, headers and credentials are never inspected.
        """
        self.navigation_stop_snapshots.append(copy.deepcopy({
            "reason": reason, "recoverable": recoverable,
            "frame": observation.frame, "room_id": observation.identity[1],
            "run_id": observation.data.get("run_id"),
            "floor_id": observation.data.get("floor", {}).get("id"),
            "decisions": self.decisions, "actions_sent": self.actions_sent,
            "observation": observation.data,
            "floor_progress": getattr(navigator, "stats", {}),
            "pickup_progress": getattr(navigator, "pickup_stats", {}),
            "adventure_progress": getattr(navigator, "adventure_stats", {}),
            "recent_local_controls": self.recent_local_controls[-120:],
        }))
        del self.navigation_stop_snapshots[:-8]

    def record_error(self, error: Exception) -> None:
        """Keep diagnostic categories only; never include exception text or bodies."""
        from .jev import JevHTTPError, JevResponseError
        self.errors += 1
        self.last_error_type = type(error).__name__
        self.last_http_status = error.status if isinstance(error, JevHTTPError) else None
        self.last_validation = ({"code":error.validation_code, "question":error.question,
                                "diagnostics":error.diagnostics} if isinstance(error, JevResponseError) else None)

    def record_usage(self, result) -> None:
        """Count provider-reported tokens, not a complete billing total."""
        usage = getattr(result, "usage", None)
        if usage is None:  # Local baseline actions have no provider usage.
            return
        keys = ("input_tokens", "output_tokens")
        if not isinstance(usage, Mapping) or any(
            not isinstance(usage.get(key), int) or isinstance(usage.get(key), bool)
            or usage[key] < 0 for key in keys
        ):
            raise ValueError("Decision has invalid token usage")
        self.responses_with_usage += 1
        self.reported_input_tokens += usage["input_tokens"]
        self.reported_output_tokens += usage["output_tokens"]

    def record_decision(self, result) -> None:
        self.record_usage(result)
        corrections = getattr(result, "choice_corrections", ())
        if corrections:
            self.corrected_responses += 1
            self.choice_corrections += len(corrections)
            self.last_choice_corrections = [dict(item) for item in corrections]

    def summary(self) -> dict:
        d = asdict(self)
        times = sorted(d.pop("latency_ms"))
        d["latency_median_ms"] = round(times[len(times)//2], 1) if times else None
        d["latency_p95_ms"] = round(times[max(0, math.ceil(len(times)*.95)-1)], 1) if times else None
        return d


def combat_snapshot(observation: Observation) -> dict:
    """A small observation summary to distinguish survival from combat progress."""
    data = observation.data
    player = data["player"]
    return {"frame": observation.frame,
            "player": {key: player[key] for key in
                       ("x", "y", "hearts", "soul_hearts", "dead") if key in player},
            "room_clear": data["room"]["clear"],
            "enemy_count": len(data["enemies"]),
            "enemies": [{key: enemy[key] for key in ("id", "hp", "x", "y") if key in enemy}
                        for enemy in data["enemies"]],
            "entities_truncated": bool(data.get("truncated", False))}


def result_is_fresh(source: Observation, current: Observation | None,
                    elapsed: float, limit: float, source_epoch: int, epoch: int,
                    *, allow_clear=False) -> bool:
    return (current is not None and (current.controllable if allow_clear else current.active) and source_epoch == epoch
            and source.identity == current.identity and 0 <= current.frame-source.frame <= MAX_FRAME_AGE
            and 0 <= elapsed <= limit)


class Controller:
    def __init__(self, policy: Callable, *, port: int = 42421, max_hz: float = 5,
                 max_calls: int = 30, duration: float = 30, max_latency: float = .25,
                 hold_frames: int = 6,
                 move_frames: int | None = None, move_distance: float | None = None,
                 goal_mode: bool = False,
                 goal_max_age: float = 2.0,
                 one_room: bool = False,
                 floor_mode: bool = False,
                 startup_guard: bool = False,
                 wait_for_arm: bool = False,
                 excluded_run_id: str | None = None,
                 stay_ready: bool = False,
                 adventure_mode: bool = False,
                 continue_floors: bool = False,
                 jev_player: bool = False,
                 on_navigation_stop: Callable[[dict], None] | None = None,
                 logger: Callable[[str], None] = print):
        if not 0 <= port <= 65535 or not 0 < max_hz <= 10 or not 1 <= max_calls <= 10000:
            raise ValueError("Invalid port, rate, or call cap")
        if not 0 < duration <= 3600 or not 0 < max_latency <= .5:
            raise ValueError("Duration must be 0..3600s and latency limit 0..0.5s")
        if type(hold_frames) is not int or not 1 <= hold_frames <= 15:
            raise ValueError("Action hold must be 1..15 frames")
        if move_frames is not None or move_distance is not None:
            from .protocol import number
            if (type(move_frames) is not int or not 1 <= move_frames <= min(6, hold_frames)
                or not number(move_distance) or not 0 < move_distance <= 24):
                raise ValueError("Invalid movement pulse")
        self.policy, self.port, self.max_hz = policy, port, max_hz
        self.max_calls, self.duration, self.max_latency = max_calls, duration, max_latency
        self.hold_frames = hold_frames
        self.move_frames, self.move_distance = move_frames, move_distance
        self.goal_mode = goal_mode
        if not 0 < goal_max_age <= 2:
            raise ValueError("Goal lifetime must be at most two seconds")
        self.goal_max_age = goal_max_age
        self.one_room = one_room
        if floor_mode and (not goal_mode or one_room):
            raise ValueError("Floor control needs Jev goal mode and cannot use one-room mode")
        self.floor_mode, self.startup_guard = floor_mode, startup_guard
        self.wait_for_arm = wait_for_arm
        if excluded_run_id is not None and (not isinstance(excluded_run_id, str) or not excluded_run_id):
            raise ValueError("Excluded run identity must be a nonempty string")
        self.excluded_run_id = excluded_run_id
        if stay_ready and not floor_mode:
            raise ValueError("Staying ready after a pause requires floor mode")
        self.stay_ready = stay_ready
        if adventure_mode and not floor_mode or continue_floors and not adventure_mode:
            raise ValueError("Adventure needs floor mode; continued floors need adventure mode")
        self.adventure_mode, self.continue_floors = adventure_mode, continue_floors
        if jev_player and not (goal_mode and floor_mode and adventure_mode):
            raise ValueError("Jev player mode needs floor, adventure and goal modes")
        self.jev_player = jev_player
        self.on_navigation_stop = on_navigation_stop
        self.log = logger
        self.stats = Stats()
        self.bound_port = None

    def run(self, ready=None) -> dict:
        from .jev import JevResponseError
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind(("127.0.0.1", self.port))
            sock.settimeout(_RECEIVE_TIMEOUT)
            self.bound_port = sock.getsockname()[1]
            if ready:
                ready.set()
            scope = "floor" if self.floor_mode else "room"
            self.log(f"Ready. F8 in Isaac enables {scope} control; F8 stops it.")
            started = time.monotonic()
            deadline = None if self.wait_for_arm else started + self.duration
            if self.wait_for_arm:
                self.log("Waiting for F8; the trial timer starts when control is enabled.")
            latest = None
            latest_time = 0.0
            peer = None
            epoch = 0
            job = None
            last_dispatch = None
            trial_room = None
            trial_run = None
            stop_reason = "duration reached"
            invalid_streak = 0
            last_brake = None
            current_goal = None
            room_fire = None
            last_local_dispatch = None
            next_call = started
            armed_at = None
            combat_epoch = None
            combat_started = None
            goal_received_in_epoch = False
            floor_run = None
            floor_id = None
            floor_phase = None
            waiting_for_rearm = False
            explorer = None
            pending_ability = None
            ability_blocked = set()
            ability_state = None
            floor_history = []
            navigation_recovery = None
            manual_recovery = None
            pending_observations = deque()
            drain_incomplete = False
            seen_sessions = set()
            stall_watchdog = CombatStallWatchdog()
            switch_navigator = None
            objective_status = None
            last_override = last_player_status = None
            def new_explorer():
                if self.jev_player:
                    from .player_navigation import PlayerNavigator
                    return PlayerNavigator(continue_floors=self.continue_floors)
                from .exploration import FloorNavigator
                return (FloorNavigator(adventure_mode=True, continue_floors=self.continue_floors)
                        if self.adventure_mode else FloorNavigator())
            if self.floor_mode:
                from .exploration import FloorNavigator
                from .switches import SwitchNavigator
                explorer = new_explorer()
                switch_navigator = SwitchNavigator()

            def reset_control_epoch():
                nonlocal room_fire
                nonlocal epoch, current_goal, pending_ability, last_dispatch, last_local_dispatch
                nonlocal combat_epoch, combat_started, goal_received_in_epoch, ability_state
                nonlocal objective_status
                epoch += 1
                current_goal = pending_ability = None
                room_fire = None
                last_dispatch = last_local_dispatch = None
                combat_epoch = combat_started = None
                goal_received_in_epoch = False
                ability_state = None
                ability_blocked.clear()
                stall_watchdog.reset()
                if switch_navigator is not None:
                    switch_navigator.reset()
                if self.jev_player and explorer is not None:
                    explorer.cancel_intent()
                objective_status = None

            def recover_navigation(reason):
                nonlocal navigation_recovery
                recoverable = self.stay_ready and reason in _RECOVERABLE_NAVIGATION_STOPS
                capture_navigation_stop(reason, recoverable=recoverable)
                if not recoverable:
                    return False
                navigation_recovery = _NavigationRecovery(floor_run, floor_id,
                                                          frozenset(seen_sessions), latest.frame)
                reset_control_epoch()
                self.stats.navigation_stops += 1
                self.stats.last_navigation_stop = {"reason": reason, "frame": latest.frame,
                                                   "room_id": latest.identity[1]}
                # Release once. No keepalive may restore floor control while
                # waiting for a new F8 session, even if old enabled packets arrive.
                sock.sendto(encode_action(latest, "none", "none", 1, floor_mode=False,
                                         stop_reason="observation" if reason == "incomplete floor observation"
                                         and latest.data.get("capabilities", {}).get("observation_recovery") == 1
                                         else "navigation"), peer)
                remaining = max(0, deadline-time.monotonic()) if deadline is not None else self.duration
                self.log(f"Navigation paused: {reason}. Control released; listener and key remain ready.")
                self.log(f"Move Isaac manually if needed, then press F8 to rearm. "
                         f"{remaining:.1f}s and {max(0, self.max_calls-self.stats.decisions)} requests remain; "
                         "the timer keeps running. No new requests until a fresh F8 session.")
                return True

            def capture_navigation_stop(reason, *, recoverable):
                self.stats.record_navigation_stop(reason, latest, explorer,
                                                  recoverable=recoverable)
                if self.on_navigation_stop is not None:
                    try:
                        self.on_navigation_stop(copy.deepcopy(self.stats.navigation_stop_snapshots[-1]))
                    except Exception as exc:
                        # A diagnostic file failure must not alter game control
                        # or expose paths, response bodies, or exception text.
                        self.log(f"Navigation checkpoint failed ({type(exc).__name__}).")

            def planner_failed(reason, error):
                """A local planning exception must not throw away the live key.

                Suspend this same attempt until an authenticated new F8 session;
                never loop on a failing planner or renew its time/request limits.
                """
                nonlocal stop_reason
                self.stats.record_error(error)
                self.log(f"{reason.capitalize()} ({type(error).__name__}); releasing control.")
                if self.floor_mode and recover_navigation(reason):
                    return True
                stop_reason = reason
                return False

            def send_local(action, *, startup=False):
                nonlocal last_local_dispatch, pending_ability
                nonlocal last_override, last_player_status
                interaction = getattr(action, "interaction", "none")
                interaction_id = getattr(action, "interaction_id", None)
                if pending_ability is not None and interaction == "none":
                    from .adventure import candidate_valid
                    candidate, identity, offered_at, pulse_id = pending_ability
                    pending_ability = None
                    if (latest.active and latest.identity == identity and time.monotonic() - offered_at <= .5
                            and candidate_valid(latest.data, candidate)):
                        interaction, interaction_id = candidate.interaction, pulse_id
                        ability_blocked.add(candidate.key)
                hold_frames = min(self.hold_frames, getattr(action, "hold_frames", self.hold_frames))
                move_frames = min(self.move_frames, hold_frames) if self.move_frames is not None else None
                sock.sendto(encode_action(latest, action.move, action.shoot,
                                         hold_frames, move_frames=move_frames,
                                         move_distance=self.move_distance,
                                         floor_mode=True if self.floor_mode else None,
                                         interaction=interaction, interaction_id=interaction_id,
                                         transition=getattr(action, "transition", None)), peer)
                self.stats.interaction_pulses += int(interaction != "none")
                self.stats.actions_sent += 1
                override = getattr(action, "override", None)
                if self.jev_player:
                    status = getattr(action, "status", None)
                    if status and status.startswith("Local override:"):
                        override = status.removeprefix("Local override:").strip()
                    if override:
                        self.stats.local_overrides += 1
                        if override != last_override:
                            self.log(f"Local override: {override}.")
                    last_override = override
                    if status != last_player_status and status and status.startswith("Activity "):
                        self.log(status)
                    last_player_status = status
                elapsed_ms = round((time.monotonic()-armed_at)*1000, 1)
                if self.stats.first_local_action_ms is None:
                    self.stats.first_local_action_ms = elapsed_ms
                if action.move != "none" or action.shoot != "none":
                    self.stats.nonneutral_actions += 1
                    if self.stats.first_nonneutral_action_ms is None:
                        self.stats.first_nonneutral_action_ms = elapsed_ms
                if startup:
                    self.stats.startup_local_actions += 1
                control = latest.data.get("control", {})
                self.stats.recent_local_controls.append({
                    "frame": latest.frame, "elapsed_ms": elapsed_ms,
                    "player": {k: latest.data["player"][k] for k in ("x", "y", "vx", "vy")},
                    "move": action.move, "shoot": action.shoot,
                    "status": getattr(action, "status", None),
                    "interaction": interaction,
                    "override": override,
                    "observed_control": dict(control) if isinstance(control, dict) else None,
                })
                del self.stats.recent_local_controls[:-120]
                last_local_dispatch = (peer, epoch, latest.identity, latest.frame)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                try:
                    while deadline is None or time.monotonic() < deadline:
                        try:
                            if not pending_observations:
                                pending_observations, drain_incomplete = _receive_observations(sock, self.stats)
                            if not pending_observations:
                                raise socket.timeout()
                            incoming = pending_observations.popleft()
                            if isinstance(incoming, OSError):
                                raise incoming
                            observation, address, received_at = incoming
                            if self.excluded_run_id is not None:
                                incoming_run = observation.data.get("run_id")
                                # After a death, even delayed packets claiming
                                # the previous player is alive cannot restart it.
                                if (not isinstance(incoming_run, str) or not incoming_run
                                    or incoming_run == self.excluded_run_id):
                                    continue
                            now = time.monotonic()
                            # F8 deliberately replaces the mod's UDP endpoint.
                            # Only a fresh arm of the same previously disarmed
                            # run/room can bypass the ordinary peer pinning delay.
                            rearmed_peer = (latest is not None and latest.data["enabled"] is False
                                            and observation.controllable
                                            and latest.data.get("run_id")
                                            and latest.data.get("run_id") == observation.data.get("run_id")
                                            and latest.identity[1] == observation.identity[1]
                                            and latest.identity[0] != observation.identity[0]
                                            and observation.frame >= latest.frame)
                            if navigation_recovery is not None and navigation_recovery.accepts(observation):
                                rearmed_peer = True
                            if manual_recovery is not None and manual_recovery.accepts(observation):
                                rearmed_peer = True
                            if peer is not None and peer != address and now-latest_time < 1 and not rearmed_peer:
                                continue
                            if latest is not None and peer == address and latest.identity == observation.identity and observation.frame < latest.frame:
                                continue
                            if (latest is not None and peer == address and latest.identity == observation.identity
                                and latest.frame == observation.frame
                                and _coalescing_key(latest, peer) == _coalescing_key(observation, address)):
                                # Preserve same-frame off/pause transitions, but do not refresh replay timestamps.
                                continue
                            if (latest is None or _observation_epoch_key(latest, peer)
                                    != _observation_epoch_key(observation, address)):
                                epoch += 1
                                pending_ability = None
                            latest, peer, latest_time = observation, address, received_at
                            seen_sessions.add(observation.identity[0])
                            if navigation_recovery is not None:
                                navigation_recovery.observe(observation)
                            if manual_recovery is not None:
                                manual_recovery.observe(observation)
                            self.stats.observations += 1
                            control = observation.data.get("control")
                            if isinstance(control, dict):
                                self.stats.observed_control_packets += 1
                                if armed_at is not None and self.stats.first_control_echo_ms is None:
                                    self.stats.first_control_echo_ms = round((now-armed_at)*1000, 1)
                                reason = control.get("move_stop_reason")
                                source_frame = control.get("source_frame")
                                brake = (observation.identity, source_frame)
                                if (isinstance(reason, str) and reason in self.stats.observed_movement_brakes
                                    and type(source_frame) is int and 0 <= source_frame <= observation.frame
                                    and brake != last_brake):
                                    self.stats.observed_movement_brakes[reason] += 1
                                    last_brake = brake
                        except socket.timeout:
                            pass
                        except ValueError:
                            self.stats.invalid_packets += 1
                        except OSError as exc:
                            self.stats.record_error(exc)
                            stop_reason = "local connection closed"
                            self.log("Local game connection closed; stopped.")
                            break
                        now = time.monotonic()
                        if deadline is not None and now >= deadline:
                            break
                        eligible = latest is not None and (latest.controllable if self.floor_mode else latest.active)
                        if latest is not None and self.adventure_mode and explorer is not None:
                            explorer.observe_interaction_pause(latest.data, now)
                        if latest is not None and self.adventure_mode:
                            player = latest.data["player"]
                            observed_ability_state = tuple(player.get(k) for k in
                                ("active_item", "active_charge", "pocket_card", "pocket_pill"))
                            if observed_ability_state != ability_state:
                                ability_blocked.clear()
                                ability_state = observed_ability_state
                        if eligible and armed_at is None:
                            armed_at = now
                            if self.wait_for_arm:
                                deadline = armed_at + self.duration
                            self.stats.first_observation = latest.data
                            self.log("F8 received. Local control ready.")
                            if self.floor_mode:
                                floor_run = latest.data.get("run_id")
                                floor_id = latest.data.get("floor", {}).get("id")
                        if armed_at is not None and latest is not None:
                            self.stats.last_observation = latest.data
                        if self.move_frames is not None and eligible:
                            capabilities = latest.data.get("capabilities", {})
                            if (not isinstance(capabilities, dict)
                                or type(capabilities.get("movement_pulses")) is not int
                                or capabilities["movement_pulses"] != 1):
                                stop_reason = "mod update required"
                                self.log("Restart Isaac with the updated Jev mod before this movement test.")
                                break
                        if self.goal_mode and eligible:
                            capabilities = latest.data.get("capabilities", {})
                            if (not isinstance(capabilities, dict)
                                or type(capabilities.get("local_goal_control")) is not int
                                or capabilities["local_goal_control"] != 1):
                                stop_reason = "mod update required"
                                self.log("Restart Isaac with the updated Jev mod before goal control.")
                                break
                        if self.floor_mode and eligible:
                            support = latest.data.get("capabilities", {}).get("floor_control")
                            if type(support) is not int or support != 1:
                                stop_reason = "mod update required"
                                self.log("Restart Isaac with the floor-control mod before this test.")
                                break
                            if self.adventure_mode and latest.data.get("capabilities", {}).get("interaction_control") != 1:
                                stop_reason = "mod update required"
                                self.log("Restart Isaac with interaction support before this trial.")
                                break
                        if self.floor_mode and armed_at is not None and latest is not None:
                            if latest.data.get("run_id") != floor_run:
                                stop_reason = "run changed"
                            elif latest.data.get("floor", {}).get("id") != floor_id:
                                if (navigation_recovery is None and self.continue_floors and explorer.descent_requested
                                        and latest.data.get("floor_advance_permitted") is True):
                                    floor_history.append({"floor": floor_id, "progress": explorer.stats,
                                                          "interactions": explorer.adventure_stats})
                                    floor_id = latest.data["floor"]["id"]
                                    explorer = new_explorer()
                                    self.stats.floors_advanced += 1
                                    floor_phase = None
                                    stop_reason = "duration reached"
                                    self.log("Next floor reached; continuing with the same remaining limits.")
                                else:
                                    stop_reason = "floor changed"
                            elif latest.data["player"]["dead"]:
                                stop_reason = "player died"
                            elif not latest.data["enabled"] and not self.stay_ready:
                                stop_reason = "control disabled"
                            else:
                                stop_reason = "duration reached"
                            if stop_reason != "duration reached":
                                self.log(f"Floor trial stopped: {stop_reason}.")
                                break
                            if (navigation_recovery is not None and now-latest_time <= .15
                                    and navigation_recovery.accepts(latest)):
                                explorer = explorer.rearmed(latest.data)
                                reset_control_epoch()
                                navigation_recovery = None
                                manual_recovery = None
                                floor_phase, waiting_for_rearm = None, False
                                self.stats.navigation_rearms += 1
                                self.log("Fresh F8 received; navigation resumed with the observed floor map. "
                                         "Continuing with the same remaining time and request limits.")
                            if navigation_recovery is not None:
                                eligible = False
                            elif not latest.data["enabled"]:
                                if not waiting_for_rearm:
                                    self.log("Control stopped; still ready. Resume Isaac and press F8 to continue. No new requests while paused.")
                                    manual_recovery = _NavigationRecovery(floor_run, floor_id,
                                                                          frozenset(seen_sessions), latest.frame)
                                    reset_control_epoch()
                                waiting_for_rearm = True
                            elif waiting_for_rearm:
                                if (manual_recovery is not None and now-latest_time <= .15
                                        and manual_recovery.accepts(latest)):
                                    explorer = explorer.rearmed(latest.data)
                                    reset_control_epoch()
                                    self.log("F8 received; continuing the same attempt with its remaining time and request limits.")
                                    waiting_for_rearm = False
                                    manual_recovery = None
                                else:
                                    eligible = False
                            if navigation_recovery is None and not waiting_for_rearm:
                                try:
                                    explorer.observe(latest.data)
                                except Exception as exc:
                                    if planner_failed("floor navigation failed", exc):
                                        continue
                                    break
                                if explorer.stop_reason:
                                    if recover_navigation(explorer.stop_reason):
                                        continue
                                    stop_reason = explorer.stop_reason
                                    self.log(f"Floor trial stopped: {stop_reason}.")
                                    break
                                phase = (latest.data.get("floor", {}).get("room_index"), latest.data["room"]["clear"])
                                if phase != floor_phase:
                                    floor_phase = phase
                                    self.log(f"Room {phase[0]}: {'cleared; checking pickups and doors' if phase[1] else 'uncleared; checking enemies and room objectives'}.")
                        if latest is not None and latest.active and combat_epoch != epoch:
                            combat_epoch, combat_started = epoch, now
                            goal_received_in_epoch = False
                        if (trial_room is not None and latest is not None
                            and latest.identity[1] == trial_room
                            and (trial_run is None or latest.data.get("run_id") == trial_run)):
                            self.stats.combat_end = combat_snapshot(latest)
                        if self.one_room and trial_room is not None and latest is not None:
                            if trial_run is not None and latest.data.get("run_id") != trial_run:
                                stop_reason = "run changed"
                            elif latest.identity[1] != trial_room:
                                stop_reason = "room changed"
                            elif latest.data["player"]["dead"]:
                                stop_reason = "player died"
                            elif latest.data["room"]["clear"]:
                                stop_reason = "room cleared"
                            else:
                                stop_reason = "duration reached"
                            if stop_reason != "duration reached":
                                self.log(f"Trial stopped: {stop_reason}.")
                                break
                        # Replay each retained transition through the safety and
                        # map bookkeeping above, but plan only after the burst.
                        if pending_observations or drain_incomplete:
                            continue
                        noncombat_room = (self.floor_mode and eligible and latest.active
                                          and no_living_enemies(latest.data))
                        player_activity_room = bool(self.jev_player and eligible and
                            (latest.data["room"]["clear"] or no_living_enemies(latest.data)))
                        if job is not None and job[0].done():
                            future, source, address, dispatched, source_epoch = job
                            job = None
                            elapsed = now-dispatched
                            self.stats.latency_ms.append(elapsed*1000)
                            try:
                                action = future.result()
                                self.stats.record_decision(action)
                                if getattr(action, "choice_corrections", ()):
                                    self.log("Choice inconsistency: using the highest validated probability.")
                                invalid_streak = 0
                                if ((not noncombat_room or self.jev_player and source.data.get("_adventure_options"))
                                    and (not self.jev_player or bool(source.data.get("_adventure_options")) == player_activity_room)
                                    and peer == address and now-latest_time <= self.max_latency
                                    and result_is_fresh(source, latest, elapsed, self.max_latency, source_epoch, epoch,
                                                       allow_clear=bool(source.data.get("_adventure_options")))):
                                    if self.goal_mode:
                                        if self.adventure_mode and action.kind == "adventure":
                                            source_offers = source.data.get("_adventure_options", [])
                                            current_offers = [c.as_dict() for c in explorer.adventure_options]
                                            bound = (source_offers == current_offers if action.target_id is None else
                                                any(c.get("key") == action.target_id and c in current_offers for c in source_offers))
                                            if bound:
                                                accepted = explorer.accept_adventure(action.target_id, latest.data, now)
                                                if self.jev_player and accepted:
                                                    room_fire = (action.fire_direction, dispatched, source_epoch,
                                                                 source.identity, address, source.data["player"].get("weapon_type"))
                                            else:
                                                self.stats.stale_discarded += 1
                                            self.stats.strategic_choices += 1
                                            if self.jev_player:
                                                self.stats.player_decisions.append({"frame": source.frame, "kind": "activity",
                                                    "selected": action.target_id or "wait", "fire": action.fire_direction,
                                                    "fire_judgment": action.fire_judgment,
                                                    "accepted": bool(bound and accepted)})
                                                del self.stats.player_decisions[:-80]
                                                self.log(f"Jev chose activity: {action.target_id or 'wait'} ({'accepted' if bound and accepted else 'state changed; discarded'}).")
                                            else:
                                                self.log(f"frame {source.frame}: interaction choice {action.target_id or 'continue exploring'}")
                                            action = None
                                        if action is None:
                                            pass
                                        elif action.kind not in {"engage", "evade", "hold"}:
                                            raise ValueError("Unknown local control goal")
                                        else:
                                            current_goal = (action, dispatched, source_epoch, source.identity, address)
                                            self.stats.goal_updates += 1
                                            goal_received_in_epoch = True
                                            if self.stats.first_goal_ms is None:
                                                self.stats.first_goal_ms = round((now-armed_at)*1000, 1)
                                            if self.jev_player:
                                                from .combat import build_combat_context, current_firing_view
                                                def aim_audit(observed):
                                                    context = build_combat_context(observed, target_limit=64)
                                                    rows = context.get("targets", [])
                                                    return {"frame": observed["frame"],
                                                        "firing_now": current_firing_view(context),
                                                        "movement_target": next((r for r in rows if r["id"] == action.target_id), None),
                                                        "aligned_with_chosen_fire": [r["id"] for r in rows if any(
                                                            lane["direction"] == action.fire_direction and lane["aligned"] and not lane["blockers"]
                                                            for lane in r["lanes"].values())]}
                                                self.stats.player_decisions.append({"frame": source.frame, "kind": action.kind,
                                                    "target_id": action.target_id, "fire": action.fire_direction,
                                                    "fire_judgment": action.fire_judgment,
                                                    "reply_age_ms": round(elapsed*1000, 1),
                                                    "aim_at_request": aim_audit(source.data), "aim_at_reply": aim_audit(latest.data)})
                                                del self.stats.player_decisions[:-80]
                                                self.log(f"Jev chose: {action.kind} {action.target_id or ''}; fire {action.fire_direction} ({elapsed*1000:.0f} ms).")
                                            else:
                                                self.log(f"frame {source.frame}: goal {action.kind} ({elapsed*1000:.0f} ms)")
                                            ability_key = getattr(action, "ability_key", None)
                                            if self.adventure_mode and ability_key and ability_key not in ability_blocked:
                                                from .adventure import candidates
                                                candidate = next((c for c in candidates(latest.data)
                                                                  if c.key == ability_key and c.kind in ("use_active", "use_pocket")), None)
                                                if candidate is not None and candidate.as_dict() in source.data.get("_ability_options", []):
                                                    pending_ability = (candidate, latest.identity, now, uuid.uuid4().hex)
                                    else:
                                        sock.sendto(encode_action(source, action.move, action.shoot, self.hold_frames,
                                                                 move_frames=self.move_frames,
                                                                 move_distance=self.move_distance), address)
                                        self.stats.actions_sent += 1
                                        self.log(f"frame {source.frame}: {action.move} / shoot {action.shoot} ({elapsed*1000:.0f} ms)")
                                else:
                                    self.stats.stale_discarded += 1
                            except Exception as exc:
                                self.stats.record_error(exc)
                                self.stats.last_failed_observation = source.data
                                if isinstance(exc, JevResponseError):
                                    invalid_streak += 1
                                    self.log(f"Invalid reply discarded ({exc.validation_code}); no action applied.")
                                    if invalid_streak >= 3:
                                        stop_reason = "three consecutive invalid replies"
                                        break
                                else:
                                    stop_reason = "decision failed"
                                    # Provider bodies may contain sensitive content; show only exception type.
                                    self.log(f"Decision failed ({type(exc).__name__}); releasing control.")
                                    break
                        if job is None and self.stats.decisions >= self.max_calls:
                            stop_reason = "request cap reached"
                            self.log("Decision cap reached; stopped.")
                            break
                        if navigation_recovery is not None or waiting_for_rearm:
                            continue
                        if noncombat_room and not self.jev_player:
                            # An uncleared puzzle room is not necessarily combat.
                            # No useful enemy choice exists, so do not repeatedly
                            # pay for hold replies or reuse a previous attack/item.
                            current_goal = pending_ability = None
                            if (now-latest_time <= .15
                                    and last_local_dispatch != (peer, epoch, latest.identity, latest.frame)):
                                try:
                                    local_action = switch_navigator.step(latest.data, now)
                                    diagnostic = stall_watchdog.observe(
                                        latest.data, now, local_objective=switch_navigator.has_objective)
                                    reason = getattr(local_action, "stop_reason", None)
                                    if diagnostic is not None:
                                        reason = diagnostic["reason"]
                                    if reason:
                                        if getattr(local_action, "status", None):
                                            self.log(f"Room objective: {local_action.status}.")
                                        if recover_navigation(reason):
                                            continue
                                        stop_reason = reason
                                        break
                                    if local_action is None:
                                        from .navigation import compute_action
                                        dodge = compute_action(latest.data, "evade")
                                        local_action = Action(dodge.move, "none")
                                    status = getattr(local_action, "status", None) or "waiting for an observed room objective"
                                    if status != objective_status:
                                        self.log(f"Room objective: {status}.")
                                        objective_status = status
                                    if time.monotonic()-latest_time <= .15:
                                        send_local(local_action)
                                except Exception as exc:
                                    if planner_failed("room objective navigation failed", exc):
                                        continue
                                    break
                            continue
                        stall_watchdog.reset()
                        if switch_navigator is not None:
                            switch_navigator.reset()
                        objective_status = None
                        if self.goal_mode and current_goal is not None:
                            goal, goal_started, goal_epoch, goal_identity, goal_peer = current_goal
                            goal_fresh = (not player_activity_room and latest is not None and latest.active and epoch == goal_epoch
                                          and latest.identity == goal_identity and peer == goal_peer
                                          and now-goal_started <= self.goal_max_age and now-latest_time <= .15)
                            if not goal_fresh:
                                current_goal = None
                                self.stats.expired_goals += 1
                                # Floor navigation replaces this goal on a clear room's next frame.
                                # Sending two different commands for one source frame could revoke
                                # the floor lease before the door-steering command arrives.
                                if latest is not None and peer is not None and not self.floor_mode:
                                    sock.sendto(encode_action(latest, "none", "none", 1), peer)
                            elif last_local_dispatch != (peer, epoch, latest.identity, latest.frame):
                                from .navigation import compute_action
                                try:
                                    local_action = (compute_action(latest.data, goal.kind, goal.target_id,
                                                                  fire_direction=goal.fire_direction)
                                                    if self.jev_player else compute_action(latest.data, goal.kind, goal.target_id))
                                except Exception as exc:
                                    if planner_failed("local navigation failed", exc):
                                        continue
                                    break
                                # Navigation is bounded, but never apply a result if local work fell behind.
                                if time.monotonic()-latest_time <= .15:
                                    send_local(local_action)
                        if (self.goal_mode and current_goal is None and latest is not None and latest.active
                            and not player_activity_room
                            and self.startup_guard and not goal_received_in_epoch
                            and now-combat_started <= 2 and now-latest_time <= .15
                            and last_local_dispatch != (peer, epoch, latest.identity, latest.frame)):
                            from .navigation import compute_action
                            # Dodge immediately from the current frame while the first cloud goal is pending.
                            try:
                                local_action = (compute_action(latest.data, "hold", fire_direction="none")
                                                if self.jev_player else compute_action(latest.data, "evade"))
                            except Exception as exc:
                                if planner_failed("local navigation failed", exc):
                                    continue
                                break
                            if time.monotonic()-latest_time <= .15:
                                send_local(local_action, startup=True)
                        if (self.floor_mode and eligible and (latest.data["room"]["clear"] or player_activity_room)
                            and now-latest_time <= .15
                            and last_local_dispatch != (peer, epoch, latest.identity, latest.frame)):
                            try:
                                local_action = explorer.step(latest.data, now)
                            except Exception as exc:
                                if planner_failed("floor navigation failed", exc):
                                    continue
                                break
                            if local_action.stop_reason:
                                if recover_navigation(local_action.stop_reason):
                                    continue
                                stop_reason = local_action.stop_reason
                                self.log(f"Floor trial stopped: {stop_reason}.")
                                break
                            if (self.jev_player and self.on_navigation_stop is not None
                                    and (local_action.status or "").startswith("Activity failed:")
                                    and explorer.activity_failures):
                                # Capture geometry while it still exists. Activity failure
                                # returns control to Jev, so it doesn't pass through the
                                # navigation-stop handler or require a process restart.
                                try:
                                    self.on_navigation_stop(dict(copy.deepcopy(explorer.activity_failures[-1]),
                                        kind="activity_failure", recoverable=True, frame=latest.frame,
                                        room_id=latest.identity[1], recent_local_controls=copy.deepcopy(self.stats.recent_local_controls[-120:])))
                                except Exception as exc:
                                    self.log(f"Activity checkpoint failed ({type(exc).__name__}).")
                            if (getattr(local_action, "interaction", "none") == "bomb"
                                    and deadline is not None and deadline - now < 4):
                                stop_reason = "insufficient trial time for bomb retreat"
                                self.log("Trial ending; no bomb placed without enough retreat time.")
                                break
                            if time.monotonic()-latest_time <= .15:
                                if self.jev_player and room_fire is not None and explorer.allows_independent_fire:
                                    direction, dispatched, fire_epoch, identity, address, weapon = room_fire
                                    if (epoch == fire_epoch and latest.identity == identity and peer == address
                                            and now-dispatched <= self.goal_max_age
                                            and latest.data["player"].get("weapon_type") == weapon):
                                        local_action = replace(local_action, shoot=direction)
                                    else:
                                        room_fire = None
                                send_local(local_action)
                        strategy_options = (getattr(explorer, "adventure_options", ())
                                            if self.adventure_mode and eligible and (latest.data["room"]["clear"] or player_activity_room) else ())
                        if (job is None and latest is not None and ((latest.active and not player_activity_room) or strategy_options) and now >= next_call
                            and now-latest_time <= .15 and last_dispatch != (peer, epoch, latest.identity, latest.frame)):
                            self.stats.decisions += 1
                            if trial_room is None and latest.active:
                                trial_room = latest.identity[1]
                                trial_run = latest.data.get("run_id")
                                self.stats.combat_start = combat_snapshot(latest)
                                self.stats.combat_end = self.stats.combat_start
                            last_dispatch = (peer, epoch, latest.identity, latest.frame)
                            source = latest
                            if strategy_options:
                                source = Observation(dict(latest.data, _adventure_options=[c.as_dict() for c in strategy_options]))
                            elif self.adventure_mode and latest.active:
                                from .adventure import candidates
                                abilities = [c.as_dict() for c in candidates(latest.data)
                                             if c.kind in ("use_active", "use_pocket") and c.key not in ability_blocked]
                                if abilities:
                                    source = Observation(dict(latest.data, _ability_options=abilities))
                            if self.goal_mode:
                                # Expose what the local controller knows/does separately
                                # from the untouched engine observation. Snapshot before
                                # handing it to the worker; no client/key data is included.
                                local_context = {
                                    "source_frame": latest.frame,
                                    "floor_mode": self.floor_mode, "adventure_mode": self.adventure_mode,
                                    "jev_player": self.jev_player,
                                    "remaining_seconds": round(max(0, deadline-now), 2) if deadline is not None else self.duration,
                                    "remaining_requests_after_this": max(0, self.max_calls-self.stats.decisions),
                                    "response_age_limit_ms": self.max_latency*1000,
                                    "goal_lifetime_ms": self.goal_max_age*1000,
                                    "previous_goal": ({"kind": current_goal[0].kind,
                                        "target_id": current_goal[0].target_id,
                                        "age_ms": round((now-current_goal[1])*1000, 1)}
                                        if current_goal is not None else None),
                                    "last_local_command": ({k: self.stats.recent_local_controls[-1][k]
                                        for k in ("frame", "move", "shoot", "interaction", "status", "override")}
                                        if self.stats.recent_local_controls and last_local_dispatch is not None
                                        and last_local_dispatch[:3] == (peer, epoch, latest.identity) else None),
                                    "last_navigation_stop": ({k: self.stats.last_navigation_stop[k]
                                        for k in ("reason", "frame")} if self.stats.last_navigation_stop else None),
                                }
                                if explorer is not None and callable(getattr(explorer, "decision_context", None)):
                                    local_context["exploration"] = explorer.decision_context()
                                source = Observation(dict(source.data, _controller_context=copy.deepcopy(local_context)))
                            job = (pool.submit(self.policy, source.data), source, peer, now, epoch)
                            next_call = now + 1/self.max_hz
                except KeyboardInterrupt:
                    stop_reason = "user stopped"
                    self.log("Stopped by user.")
                finally:
                    # A neutral action is best effort; the mod also expires all commands itself.
                    if latest is not None and peer is not None and navigation_recovery is None:
                        try:
                            sock.sendto(encode_action(latest, "none", "none", 1,
                                                     floor_mode=False if self.floor_mode else None), peer)
                        except OSError:
                            pass
            # Executor shutdown waits for any remaining request. Its result is
            # too late to control the game, but reported API usage still counts.
            if job is not None:
                try:
                    self.stats.record_decision(job[0].result())
                    self.stats.stale_discarded += 1
                except Exception as exc:
                    self.stats.record_error(exc)
                    self.log(f"Decision failed ({type(exc).__name__}); control already released.")
            result = self.stats.summary()
            result["max_latency_ms"] = self.max_latency * 1000
            result["hold_frames"] = self.hold_frames
            result["move_frames"] = self.move_frames
            result["move_distance"] = self.move_distance
            result["control_mode"] = "local-goal" if self.goal_mode else "direct"
            result["jev_player"] = self.jev_player
            result["floor_mode"] = self.floor_mode
            result["wait_for_arm"] = self.wait_for_arm
            result["stay_ready"] = self.stay_ready
            if explorer is not None:
                result["floor_progress"] = explorer.stats
                result["pickup_progress"] = getattr(explorer, "pickup_stats", {})
                result["adventure_progress"] = getattr(explorer, "adventure_stats", {})
                result["floor_history"] = floor_history
            result["stop_reason"] = stop_reason
            self.log(json.dumps({k:v for k,v in result.items()
                                 if k not in {"last_failed_observation", "first_observation", "last_observation",
                                              "recent_local_controls", "navigation_stop_snapshots"}}))
            return result
