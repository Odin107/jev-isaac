"""Observed, free pickups in cleared rooms; no inference or spending."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

from .combat import _point
from .navigation import _radius, _velocity


def _int(value, lower=0, upper=2**31-1):
    return type(value) is int and lower <= value <= upper


def pickup_support(state):
    caps = state.get("capabilities", {})
    return (isinstance(caps, Mapping) and type(caps.get("pickup_collection")) is int
            and caps["pickup_collection"] == 1)


def valid_pickups(state):
    """Old observations remain usable; advertised pickup data must be complete."""
    if not pickup_support(state):
        return "pickups" not in state
    items, player = state.get("pickups"), state.get("player", {})
    if not isinstance(items, list) or len(items) > 64 or not isinstance(player, Mapping):
        return False
    if any(not _int(player.get(key)) for key in ("coins", "bombs", "keys", "active_item")):
        return False
    if any(type(player.get(key)) is not bool for key in
           ("can_pick_red_hearts", "can_pick_soul_hearts", "can_pick_black_hearts", "can_pickup_items")):
        return False
    seen = set()
    for item in items:
        if (not isinstance(item, Mapping) or not isinstance(item.get("id"), str)
                or not 0 < len(item["id"]) <= 128 or item["id"] in seen
                or None in (_point(item), _radius(item, 10), _velocity(item))
                or type(item.get("type")) is not int or item["type"] != 5
                or any(not _int(item.get(key)) for key in
                       ("variant", "subtype", "options_index", "wait", "collectible_kind"))
                or not _int(item.get("price"), -10000, 10000)
                or type(item.get("shop_item")) is not bool):
            return False
        seen.add(item["id"])
    return True


def signature(item):
    # A pedestal can retain its identity when an item is exchanged or rerolled.
    return item["id"], item["variant"], item["subtype"]


def priority(item, player):
    """Lower means earlier. None means this pickup is never a goal."""
    if item["price"] != 0 or item["shop_item"]:
        return None
    variant, subtype = item["variant"], item["subtype"]
    if variant == 10:
        red = subtype in (1, 2, 5, 9) and player["can_pick_red_hearts"]
        soul = subtype in (3, 8) and player["can_pick_soul_hearts"]
        black = subtype == 6 and player["can_pick_black_hearts"]
        blended = subtype == 10 and (player["can_pick_red_hearts"] or player["can_pick_soul_hearts"])
        return 0 if red or soul or black or blended else None
    if variant == 100 and subtype > 0:
        if item["collectible_kind"] in (1, 4):
            return 1
        if item["collectible_kind"] == 3 and player["active_item"] == 0:
            return 1
    supplies = {20: ("coins", (1, 2, 3, 4, 5)),
                30: ("keys", (1, 3, 4)), 40: ("bombs", (1, 2))}
    if variant in supplies:
        field, supported = supplies[variant]
        if subtype in supported and player[field] < 99:
            return 2
    return None


def exclusion_boxes(state, radius, target=None):
    """Prevent incidental spending, replacement, or touching unknown pickups.

    Ordinary free supplies and supported hearts may be crossed. Other pickups,
    including unchosen option-group siblings, are obstacles even on door routes.
    """
    boxes = []
    for item in state.get("pickups", []):
        if signature(item) == target:
            continue
        variant, subtype = item["variant"], item["subtype"]
        if variant == 100 and subtype == 0 and item["price"] == 0 and not item["shop_item"]:
            continue  # Empty pedestal after collection.
        if (variant == 50 and subtype == 0 and item["price"] == 0
                and not item["shop_item"] and item["options_index"] == 0):
            continue  # Open ordinary chest remainder cannot consume another resource.
        ordinary = ((variant == 10 and subtype in (1, 2, 3, 5, 6, 8, 9, 10))
                    or (variant == 20 and subtype in (1, 2, 3, 4, 5))
                    or (variant == 30 and subtype in (1, 3, 4))
                    or (variant == 40 and subtype in (1, 2)))
        if ordinary and item["price"] == 0 and not item["shop_item"] and item["options_index"] == 0:
            continue
        x, y = _point(item)
        pad = radius + _radius(item, 10) + 8
        boxes.append((x-pad, y-pad, x+pad, y+pad))
    return boxes


@dataclass(frozen=True)
class PickupAction:
    move: str = "none"
    status: str = "collecting pickup"
    stop_reason: str | None = None


class PickupCollector:
    """Bounded pickup phase with per-room retry limits and fresh-state checks."""

    def __init__(self):
        self._visit = None
        self._pending = None
        self._skipped = set()
        self._started = self._progress_at = self._progress_distance = None
        self._settle_until = None
        self._animation_since = None
        self._settle_frame = None
        self._paused = False
        self.attempts = self.resolved = self.skipped = 0

    def pause(self):
        self._paused = True

    def reset(self):
        self._visit = self._pending = self._settle_until = None
        self._animation_since = self._settle_frame = None
        self._started = self._progress_at = self._progress_distance = None
        self._skipped.clear()

    def _abandon(self):
        self._skipped.add(self._pending)
        self.skipped += 1
        self._pending = None

    def step(self, state, now, move_to, *, defer_items=False):
        if not pickup_support(state):
            return None
        visit = state["run_id"], state["room_id"]
        if visit != self._visit:
            self.reset()
            self._visit, self._settle_until = visit, now + .6
            self._settle_frame = state["frame"] + 18
        if self._paused:
            self._paused = False
            self._started = self._progress_at = now
            self._settle_until = max(self._settle_until, now + .3)
            self._settle_frame = max(self._settle_frame, state["frame"] + 9)
            self._animation_since = None
        items, player = state["pickups"], state["player"]
        if self._pending is not None:
            present = next((item for item in items if signature(item) == self._pending), None)
            if present is None:
                # It may have been collected, vanished with an option sibling,
                # or transformed. Do not report a disappearance as a collection.
                self.resolved += 1
                self._pending = None
                self._settle_until = now + .4
                self._settle_frame = state["frame"] + 12
            elif priority(present, player) is None:
                self._pending = None
                self._settle_until = now + .3
                self._settle_frame = state["frame"] + 9
        if now < self._settle_until or state["frame"] < self._settle_frame:
            return PickupAction(status="waiting for room rewards")
        if not player["can_pickup_items"]:
            # Bound even an unrecognized animation; never walk through a
            # pedestal while the player's pickup state is unavailable.
            if self._animation_since is None:
                self._animation_since = now
            if now - self._animation_since >= 10:
                return PickupAction(status="pickup animation did not finish",
                                    stop_reason="pickup animation timed out")
            return PickupAction(status="waiting for pickup animation")
        self._animation_since = None
        start = _point(player)
        candidates = [item for item in items if signature(item) not in self._skipped
                      and priority(item, player) is not None
                      and not (defer_items and item["variant"] == 100)]
        candidates.sort(key=lambda item: (signature(item) != self._pending,
                                         priority(item, player), math.dist(start, _point(item)), item["id"]))
        # Bound expensive path searches per frame; unreachable candidates stay
        # skipped for this room visit, while a later visit may reconsider them.
        for item in candidates[:4]:
            key = signature(item)
            distance = math.dist(start, _point(item))
            if key != self._pending:
                self._pending = key
                self._started = self._progress_at = now
                self._progress_distance = distance
                self.attempts += 1
            if self._progress_distance - distance >= 8:
                self._progress_at, self._progress_distance = now, distance
            if now - self._started >= 10 or now - self._progress_at >= 3:
                self._abandon()
                continue
            if item["wait"] > 0:
                return PickupAction(status="waiting for pickup to become available")
            move = move_to(item)
            if move is None:
                self._abandon()
                continue
            return PickupAction(move, f"collecting pickup {item['variant']}:{item['subtype']}")
        if len(candidates) > 4:
            return PickupAction(status="checking pickup routes")
        self._pending = None
        return None
