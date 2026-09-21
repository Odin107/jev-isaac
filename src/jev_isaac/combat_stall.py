"""Bounded diagnostics for live uncleared rooms with no observed objective.

This does not declare a room clear, invent targets, select movement, or make
model calls. The controller owns receipt freshness, checkpointing, release,
and authenticating a fresh manual rearm. Supported local objectives must have
their own progress watchdog; they bypass this specifically targetless guard.
"""
from collections.abc import Mapping
import math

from .protocol import MAX_HAZARDS, number


NO_COMBAT_OBJECTIVE = "no observed combat objective"


def no_living_enemies(state):
    """True only when complete entity observations establish no living enemy.

    False includes unknown/incomplete data as well as actual living enemies.
    This predicate intentionally ignores enabled/paused/room-clear state: the
    controller must separately enforce control eligibility and packet age.
    A living invulnerable enemy always makes it False. It may be used to avoid
    pointless paid combat choices while the local objective/grace logic runs.
    """
    if not isinstance(state, Mapping) or state.get("truncated") is not False:
        return False
    truncation = state.get("truncated_arrays", {})
    if (not isinstance(truncation, Mapping)
            or any(type(value) is not bool or value for value in truncation.values())):
        return False
    for field, limit in (("enemies", 64), ("hazards", MAX_HAZARDS), ("projectiles", 96)):
        items = state.get(field)
        if (not isinstance(items, list) or len(items) > limit
                or any(not isinstance(item, Mapping)
                       or not all(number(item.get(k)) for k in ("x", "y")) for item in items)):
            return False
    for enemy in state["enemies"]:
        if (not number(enemy.get("hp"))
                or ("dead" in enemy and type(enemy["dead"]) is not bool)
                or (enemy["hp"] > 0 and enemy.get("dead") is not True)):
            return False
    return True


class CombatStallWatchdog:
    def __init__(self, *, grace_seconds=5.0, episode_seconds=12.0, movement_threshold=8.0):
        if (not number(grace_seconds) or not 0 < grace_seconds <= 30
                or not number(episode_seconds) or not grace_seconds <= episode_seconds <= 120
                or not number(movement_threshold) or not 0 < movement_threshold <= 24):
            raise ValueError("Invalid combat stall bounds")
        self.grace_seconds = float(grace_seconds)
        self.episode_seconds = float(episode_seconds)
        self.movement_threshold = float(movement_threshold)
        self.reset()

    def reset(self):
        self._identity = None
        self._started = self._progress_at = self._last_now = None
        self._start_point = self._progress_point = None
        self._first_frame = self._last_frame = None
        self._samples = 0
        self._reported = False

    def observe(self, state, now, *, local_objective=False):
        """Return one diagnostic dict after a fresh, targetless stall, else None.

        Call only with a recent authenticated observation, before another model
        dispatch. New frame numbers alone never reset the grace. Duplicates and
        out-of-order frames cannot advance the watchdog. Any living enemy,
        including an invulnerable one, prevents this no-enemy diagnosis.
        """
        if (not number(now) or now < 0 or (self._last_now is not None and now < self._last_now)
                or not isinstance(state, Mapping) or type(local_objective) is not bool):
            self.reset()
            return None
        player, room, floor = state.get("player"), state.get("room"), state.get("floor", {})
        frame = state.get("frame")
        if (not isinstance(player, Mapping) or not isinstance(room, Mapping)
                or not isinstance(floor, Mapping) or type(frame) is not int or frame < 0
                or not all(isinstance(state.get(k), str) and state[k] for k in ("session", "room_id"))
                or not all(number(player.get(k)) for k in ("x", "y", "vx", "vy"))
                or state.get("enabled") is not True or state.get("paused") is not False
                or player.get("dead") is not False or room.get("clear") is not False
                or not no_living_enemies(state) or local_objective):
            self.reset()
            return None
        identity = (state.get("run_id"), state["session"], state["room_id"],
                    floor.get("id"), floor.get("dimension"))
        point = (float(player["x"]), float(player["y"]))
        if identity != self._identity:
            self.reset()
            self._identity = identity
            self._started = self._progress_at = now
            self._start_point = self._progress_point = point
            self._first_frame = frame
        if self._last_frame is not None and frame <= self._last_frame:
            return None
        self._last_now, self._last_frame = now, frame
        self._samples += 1
        if self._reported:
            return None
        if math.dist(point, self._progress_point) >= self.movement_threshold:
            self._progress_point, self._progress_at = point, now
        elapsed, stationary = now-self._started, now-self._progress_at
        if elapsed < self.episode_seconds and stationary < self.grace_seconds:
            return None
        self._reported = True
        return {
            "reason": NO_COMBAT_OBJECTIVE,
            "timeout_kind": "episode" if elapsed >= self.episode_seconds else "no_progress",
            "elapsed_seconds": round(elapsed, 3),
            "no_progress_seconds": round(stationary, 3),
            "first_frame": self._first_frame, "last_frame": frame,
            "fresh_observations": self._samples,
            "start_player": dict(zip(("x", "y"), self._start_point)),
            "player": {k: player[k] for k in ("x", "y", "vx", "vy")},
            "room_id": state["room_id"], "room_clear": False,
            "enemy_count": len(state["enemies"]), "living_enemy_count": 0,
            "has_supported_local_objective": False,
        }
