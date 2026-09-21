"""Bounded facts about recent observed outcomes, never damage attribution or policy."""
from __future__ import annotations

from collections import Counter, deque
from collections.abc import Mapping
from copy import deepcopy
import math

from .protocol import MOVES, SHOTS, number


def _value(value):
    # Bounds also keep arithmetic finite for malformed direct callers.
    return value if number(value) and abs(value) <= 10**9 else None


def _identity(state):
    if not isinstance(state, Mapping):
        return None
    if any(not isinstance(state.get(k), str) or not 1 <= len(state[k]) <= 128
           for k in ("session", "room_id")):
        return None
    floor = state.get("floor")
    floor = floor if isinstance(floor, Mapping) else {}
    return tuple(state.get(k) for k in ("session", "run_id", "room_id")) + (
        floor.get("id"), floor.get("dimension"), floor.get("room_index"))


def _frame(state):
    value = state.get("frame") if isinstance(state, Mapping) else None
    return value if type(value) is int and 0 <= value < 2**53 else None


def _sample(state):
    player = state.get("player")
    player = player if isinstance(player, Mapping) else {}
    position = tuple(_value(player.get(k)) for k in ("x", "y"))
    control = state.get("control")
    clean_control = {}
    if isinstance(control, Mapping):
        for key, choices in (("requested_move", MOVES), ("applied_move", MOVES), ("shoot", SHOTS)):
            value = control.get(key)
            if isinstance(value, str) and value in choices:
                clean_control[key] = value
        source = control.get("source_frame")
        if type(source) is int and 0 <= source <= state["frame"]:
            clean_control["source_frame"] = source
        reason = control.get("move_stop_reason")
        if isinstance(reason, str) and len(reason) <= 128:
            clean_control["move_stop_reason"] = reason
    rows = state.get("enemies")
    truncation = state.get("truncated_arrays")
    partial = state.get("truncated") is True or (
        isinstance(truncation, Mapping) and bool(truncation.get("enemies")))
    coverage = "missing" if not isinstance(rows, list) else "partial" if partial or len(rows) > 256 else "exported"
    rows = rows[:256] if isinstance(rows, list) else []
    counts = Counter(row.get("id") for row in rows if isinstance(row, Mapping)
                     and isinstance(row.get("id"), str) and 1 <= len(row["id"]) <= 128)
    enemies = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        identity = row.get("id")
        if not isinstance(identity, str) or counts.get(identity) != 1:
            continue
        hp = _value(row.get("hp"))
        enemies[identity] = hp if hp is not None and hp >= 0 else None
    return {"frame": state["frame"], "position": position if None not in position else None,
            "hearts": _value(player.get("hearts")), "soul_hearts": _value(player.get("soul_hearts")),
            "control": clean_control or None, "enemies": enemies, "enemy_coverage": coverage}


def _changes(values):
    deltas = [after - before for before, after in zip(values, values[1:])
              if before is not None and after is not None]
    decreases = sum(-delta for delta in deltas if delta < 0)
    increases = sum(delta for delta in deltas if delta > 0)
    return {"comparable_intervals": len(deltas),
            "decrease_observed": round(decreases, 6) if deltas else None,
            "increase_observed": round(increases, 6) if deltas else None,
            "status": ("not_comparable" if not deltas else "decreased_and_increased" if decreases and increases
                       else "decreased" if decreases else "increased" if increases else "unchanged_at_samples")}


class CombatFeedback:
    """Call observe for each fresh packet; context never updates the rolling window.

    The controller should reset on its own rearm/peer changes as well. Paused,
    disabled, dead, regressed-clock and changed-identity packets reset locally.
    Input observations are samples, not proof a button stayed held between them.
    """

    def __init__(self, window_frames=60, enemy_limit=64):
        if type(window_frames) is not int or not 1 <= window_frames <= 300:
            raise ValueError("Feedback window must be 1..300 simulation frames")
        if type(enemy_limit) is not int or not 1 <= enemy_limit <= 256:
            raise ValueError("Feedback enemy limit must be 1..256")
        self.window_frames = window_frames
        self.enemy_limit = enemy_limit
        self.reset()

    def reset(self):
        self._identity = None
        self._samples = deque(maxlen=self.window_frames + 1)

    def observe(self, state):
        identity, frame = _identity(state), _frame(state)
        player = state.get("player") if isinstance(state, Mapping) else None
        inactive = isinstance(state, Mapping) and (state.get("paused") is True
            or state.get("enabled") is False or isinstance(player, Mapping) and player.get("dead") is True)
        if identity is None or frame is None or inactive:
            self.reset()
            return
        if identity != self._identity or self._samples and frame < self._samples[-1]["frame"]:
            self.reset()
            self._identity = identity
        if self._samples and frame == self._samples[-1]["frame"]:
            return
        self._samples.append(_sample(state))
        while self._samples[0]["frame"] < frame - self.window_frames:
            self._samples.popleft()

    def context(self, state):
        if (not self._samples or _identity(state) != self._identity
                or _frame(state) != self._samples[-1]["frame"]):
            return {"status": "no_matching_observation_history"}
        samples = list(self._samples)
        first, last = samples[0], samples[-1]
        pairs = list(zip(samples, samples[1:]))
        distances = [math.dist(a["position"], b["position"]) for a, b in pairs
                     if a["position"] is not None and b["position"] is not None]
        moving_pairs = [(a, b) for a, b in pairs if a["position"] is not None and b["position"] is not None
                        and all(s["control"] is not None
                                and s["control"].get("applied_move") in MOVES - {"none"} for s in (a, b))]
        displacement = None
        if first["position"] is not None and last["position"] is not None and pairs:
            dx, dy = (last["position"][i] - first["position"][i] for i in (0, 1))
            displacement = {"dx": round(dx, 3), "dy": round(dy, 3), "distance": round(math.hypot(dx, dy), 3)}
        inputs = {"latest": last["control"], "sample_counts": {}, "latest_sampled_streak": {}}
        for key in ("requested_move", "applied_move", "shoot"):
            values = [(s["control"] or {}).get(key) for s in samples]
            inputs["sample_counts"][key] = dict(sorted(Counter(v if v is not None else "unknown" for v in values).items()))
            current = values[-1]
            start = len(values) - 1
            if current is not None:
                while start > 0 and values[start - 1] == current:
                    start -= 1
            inputs["latest_sampled_streak"][key] = {
                "value": current, "span_frames": last["frame"] - samples[start]["frame"] if current is not None else None,
                "samples": len(values) - start if current is not None else 0}
        identities = set().union(*(s["enemies"] for s in samples))
        ordered = sorted(identities, key=lambda identity: (identity not in last["enemies"], identity))
        enemies = []
        for identity in ordered[:self.enemy_limit]:
            values = [s["enemies"].get(identity) for s in samples]
            enemies.append({"id": identity, "latest_hp": values[-1],
                            "present_in_latest_export": identity in last["enemies"], **_changes(values)})
        return deepcopy({
            "status": "observed" if pairs else "insufficient_history",
            "window": {"start_frame": first["frame"], "end_frame": last["frame"],
                       "span_frames": last["frame"] - first["frame"], "limit_frames": self.window_frames,
                       "seconds": round((last["frame"] - first["frame"]) / 30, 3), "sample_count": len(samples),
                       "largest_sample_gap_frames": max((b["frame"] - a["frame"] for a, b in pairs), default=0)},
            "player": {"health_half_hearts": {key: {"latest": last[key], **_changes([s[key] for s in samples])}
                                                for key in ("hearts", "soul_hearts")},
                       "movement": {"net_displacement": displacement,
                                    "sampled_path_distance_lower_bound": round(sum(distances), 3) if distances else None,
                                    "comparable_position_intervals": len(distances),
                                    "moving_input_at_both_endpoints": {
                                        "intervals": len(moving_pairs),
                                        "span_frames_sum": sum(b["frame"] - a["frame"] for a, b in moving_pairs),
                                        "sampled_distance": round(sum(math.dist(a["position"], b["position"])
                                                                      for a, b in moving_pairs), 3) if moving_pairs else None}}},
            "inputs": inputs,
            "enemies": {"latest_coverage": last["enemy_coverage"],
                        "window_coverage": dict(sorted(Counter(s["enemy_coverage"] for s in samples).items())),
                        "reported": enemies, "omitted_count": max(0, len(ordered) - self.enemy_limit)},
            "limits": ("Same room/run observations only; 30 simulation frames per second. Input streaks describe available samples, "
                       "not verified continuous holds. Sampled paths undercount travel between observations. HP changes compare "
                       "adjacent observations of the same enemy id; missing/duplicate ids or missing HP interrupt comparison. "
                       "Decreased HP can come from any source, not necessarily Isaac or the selected shot. Disappearance does not prove "
                       "death. Unchanged sampled HP is not proof that no hits occurred. Player heart changes are resource changes, "
                       "not a hit counter. Firing inputs do not establish shots, hits or weapon charge readiness.")})
