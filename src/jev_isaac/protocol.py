"""Validation at the local UDP boundary. No executable commands cross it."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

MOVES = frozenset(("none", "left", "right", "up", "down", "up_left", "up_right", "down_left", "down_right"))
SHOTS = frozenset(("none", "left", "right", "up", "down"))
MAX_DATAGRAM = 60_000
# Large rooms can contain more than 160 observed walls, pits and rocks.
# The bridge separately bounds dynamic hazards and the total packet size.
MAX_HAZARDS = 512
# Total source-observation age, including inference and any remaining action hold.
# Game():GetFrameCount advances at 30 Hz, so 33 frames caps this at 1100 ms.
MAX_FRAME_AGE = 33


def number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def position(value: Any, velocity: bool = False) -> bool:
    return isinstance(value, dict) and all(number(value.get(k)) for k in (("x", "y", "vx", "vy") if velocity else ("x", "y")))


def valid_visited_rooms(value: Any) -> bool:
    """Validate observed room facts without inferring doors or undiscovered rooms."""
    if not isinstance(value, list) or len(value) > 169:
        return False
    indices, descriptors = set(), set()
    for row in value:
        if not isinstance(row, dict):
            return False
        index = row.get("room_index")
        descriptor = row.get("list_index")
        if (type(index) is not int or not 0 <= index <= 168
                or type(descriptor) is not int or not 0 <= descriptor < 2**31
                or type(row.get("type")) is not int or not 1 <= row["type"] < 2**31
                or type(row.get("clear")) is not bool
                or type(row.get("visited_count")) is not int
                or not 1 <= row["visited_count"] < 2**31):
            return False
        aliases = row.get("room_indices", [index])
        if (not isinstance(aliases, list) or not 1 <= len(aliases) <= 4
                or any(type(alias) is not int or not 0 <= alias <= 168 for alias in aliases)
                or len(set(aliases)) != len(aliases) or index not in aliases
                or indices.intersection(aliases) or descriptor in descriptors):
            return False
        indices.update(aliases)
        descriptors.add(descriptor)
    return True


def valid_switches(value: Any) -> bool:
    """Missing subtype/state stays unknown; only observed integers are valid."""
    if not isinstance(value, list) or len(value) > 32:
        return False
    seen = set()
    for row in value:
        if (not isinstance(row, dict) or not position(row)
                or type(row.get("index")) is not int or not 0 <= row["index"] <= 4095
                or row["index"] in seen
                or type(row.get("type")) is not int or row["type"] != 20
                or type(row.get("collision")) is not int or not 0 <= row["collision"] <= 5):
            return False
        if any(key in row and (type(row[key]) is not int or not 0 <= row[key] <= 100000)
               for key in ("variant", "state")):
            return False
        seen.add(row["index"])
    return True


@dataclass(frozen=True)
class Observation:
    data: dict

    @property
    def identity(self) -> tuple[str, str]:
        return self.data["session"], self.data["room_id"]

    @property
    def frame(self) -> int:
        return self.data["frame"]

    @property
    def controllable(self) -> bool:
        return self.data["enabled"] and not self.data["paused"] and not self.data["player"]["dead"]

    @property
    def active(self) -> bool:
        return self.controllable and not self.data["room"]["clear"]

    @classmethod
    def decode(cls, raw: bytes) -> Observation:
        if len(raw) > MAX_DATAGRAM:
            raise ValueError("Observation exceeds datagram limit")
        try:
            d = json.loads(raw)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise ValueError("Invalid observation JSON") from exc
        if not isinstance(d, dict) or type(d.get("protocol")) is not int or d.get("protocol") != 1 or d.get("type") != "observation":
            raise ValueError("Unknown observation protocol")
        for key in ("session", "room_id"):
            if not isinstance(d.get(key), str) or not 1 <= len(d[key]) <= 128:
                raise ValueError("Invalid observation identity")
        if "run_id" in d and (not isinstance(d["run_id"], str) or not 1 <= len(d["run_id"]) <= 128):
            raise ValueError("Invalid run identity")
        if type(d.get("frame")) is not int or not 0 <= d["frame"] < 2**53:
            raise ValueError("Invalid observation frame")
        if any(type(d.get(k)) is not bool for k in ("enabled", "paused")):
            raise ValueError("Missing observation control state")
        if not position(d.get("player"), True) or type(d["player"].get("dead")) is not bool:
            raise ValueError("Invalid player state")
        room = d.get("room")
        if not isinstance(room, dict) or type(room.get("clear")) is not bool:
            raise ValueError("Invalid room state")
        if "has_trigger_pressure_plates" in room and type(room["has_trigger_pressure_plates"]) is not bool:
            raise ValueError("Invalid room switch state")
        if "switches" in d and not valid_switches(d["switches"]):
            raise ValueError("Invalid room switches")
        if not position(room.get("top_left")) or not position(room.get("bottom_right")):
            raise ValueError("Invalid room bounds")
        if any(room["top_left"][k] >= room["bottom_right"][k] for k in ("x", "y")):
            raise ValueError("Empty room bounds")
        if "floor" in d:
            floor = d["floor"]
            if (not isinstance(floor, dict) or not isinstance(floor.get("id"), str)
                or not 1 <= len(floor["id"]) <= 128 or type(floor.get("room_index")) is not int):
                raise ValueError("Invalid floor identity")
            if "dimension" in floor and (type(floor["dimension"]) is not int
                                         or not 0 <= floor["dimension"] <= 2):
                raise ValueError("Invalid floor dimension")
        if "visited_rooms" in d and ("floor" not in d or not valid_visited_rooms(d["visited_rooms"])):
            raise ValueError("Invalid visited room facts")
        for key in ("enemies", "projectiles", "hazards"):
            if not isinstance(d.get(key), list) or len(d[key]) > 256 or not all(position(e) for e in d[key]):
                raise ValueError("Invalid entity array")
        if "pickups" in d and (not isinstance(d["pickups"], list) or len(d["pickups"]) > 64
                               or not all(position(e, True) for e in d["pickups"])):
            raise ValueError("Invalid pickup array")
        # Also reject nested non-finite numbers, which Python's JSON decoder permits.
        try:
            json.dumps(d, allow_nan=False)
        except (ValueError, RecursionError) as exc:
            raise ValueError("Invalid numeric observation") from exc
        return cls(d)


def encode_action(obs: Observation, move: str, shoot: str, hold_frames: int = 6,
                  *, move_frames: int | None = None, move_distance: float | None = None,
                  floor_mode: bool | None = None, interaction: str = "none",
                  interaction_id: str | None = None, transition: str | None = None,
                  stop_reason: str | None = None) -> bytes:
    if move not in MOVES or shoot not in SHOTS:
        raise ValueError("Unknown action")
    if type(hold_frames) is not int or not 1 <= hold_frames <= 15:
        raise ValueError("Invalid action duration")
    packet = {"protocol": 1, "type": "action", "session": obs.identity[0],
              "room_id": obs.identity[1], "frame": obs.frame, "move": move,
              "shoot": shoot, "hold_frames": hold_frames}
    if floor_mode is not None:
        if type(floor_mode) is not bool:
            raise ValueError("Invalid floor mode")
        packet["floor_mode"] = floor_mode
    if interaction not in ("none", "bomb", "active", "pocket"):
        raise ValueError("Unknown interaction")
    if interaction != "none":
        if (not isinstance(interaction_id, str) or not 0 < len(interaction_id) <= 128
                or any(ord(c) < 32 for c in interaction_id)):
            raise ValueError("Invalid interaction identity")
        packet.update(interaction=interaction, interaction_id=interaction_id)
    elif interaction_id is not None:
        raise ValueError("Unused interaction identity")
    if transition is not None:
        if transition != "floor" or floor_mode is not True:
            raise ValueError("Invalid floor transition")
        packet["transition"] = transition
    if stop_reason is not None:
        if (stop_reason not in ("navigation", "observation") or floor_mode is not False
                or move != "none" or shoot != "none" or hold_frames != 1
                or interaction != "none" or transition is not None
                or move_frames is not None or move_distance is not None):
            raise ValueError("Invalid control release reason")
        packet["stop_reason"] = stop_reason
    if move_frames is not None or move_distance is not None:
        if (type(move_frames) is not int or not 1 <= move_frames <= min(6, hold_frames)
            or not number(move_distance) or not 0 < move_distance <= 24):
            raise ValueError("Invalid movement pulse")
        packet.update(move_frames=move_frames, move_distance=move_distance)
    return json.dumps(packet, separators=(",", ":")).encode()
