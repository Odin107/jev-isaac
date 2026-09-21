"""Explicit, health-budgeted visits through observed curse-room doors.

Two half-hearts per crossing is a conservative budget, not a measured damage
claim. Flight, temporary invulnerability and item effects grant no discount.
Only ordinary Isaac's observed red/soul health is used by this first policy.
"""
from .combat import _point

CURSE_ROOM = 10
CROSSING_BUDGET = 2
SURVIVAL_MARGIN = 2


def _health(player):
    if type(player.get("player_type")) is not int or player["player_type"] != 0:
        return None
    parts = [player.get("hearts"), player.get("soul_hearts")]
    if any(type(value) is not int or not 0 <= value <= 48 for value in parts):
        return None
    return sum(parts)


def _door(state, slot, target):
    doors = [door for door in state["doors"]
             if door["slot"] == slot and door["target_index"] == target]
    if len(doors) != 1:
        return None
    door = doors[0]
    if (door.get("curse_room_door") is not True or not door["open"] or door["locked"]
            or type(door.get("current_type")) is not int
            or door.get("current_type") != state["room"]["type"]
            or not 0 <= target <= 168 or target == state["floor"]["room_index"]):
        return None
    entering = state["room"]["type"] in (1, 2, 4) and door["target_type"] == CURSE_ROOM
    leaving = state["room"]["type"] == CURSE_ROOM and door["target_type"] in (1, 2, 4)
    return (door, entering) if entering or leaving else None


def curse_candidates(state, *, leaving=False, return_index=None):
    from .adventure import AdventureCandidate, _context, _parsed
    from .exploration import _Door, _door_move
    parsed = _parsed(state)
    if parsed is None or not state["room"]["clear"]:
        return ()
    health = _health(state["player"])
    budget = CROSSING_BUDGET * (1 if leaving else 2)
    if health is None or health < budget + SURVIVAL_MARGIN:
        return ()
    result = []
    for door in state["doors"]:
        bound = _door(state, door["slot"], door["target_index"])
        if bound is None or bound[1] == leaving:
            continue
        if leaving and return_index is not None and door["target_index"] != return_index:
            continue
        selected = _Door(door["slot"], _point(door), door["target_index"], door["target_type"])
        if _door_move(state, parsed, selected) is None:
            continue
        kind = "leave_curse" if leaving else "enter_curse"
        result.append(AdventureCandidate(f"{kind}:{door['slot']}:{door['target_index']}", kind,
            f"door:{door['slot']}:{door['target_index']}", _point(door), {"health_half_hearts_budget": budget},
            ("Return through the curse-room door; reserve up to one heart for crossing" if leaving else
             "Explore the observed curse room; budget up to two hearts for entry and return. Contents unknown"),
            context=_context(state), details={"slot": door["slot"], "target_index": door["target_index"],
                "target_type": door["target_type"], "origin_index": state["floor"]["room_index"],
                "crossing_budget": CROSSING_BUDGET, "return_budget": 0 if leaving else CROSSING_BUDGET,
                "survival_margin": SURVIVAL_MARGIN, "reward_known": False}))
    return tuple(result)


def curse_valid(state, candidate, *, committed=False):
    from .adventure import _context, _parsed
    from .exploration import _Door, _door_move
    parsed = _parsed(state)
    if (parsed is None or candidate.context != _context(state) or not state["room"]["clear"]
            or candidate.kind not in ("enter_curse", "leave_curse")):
        return False
    info = candidate.details
    bound = _door(state, info["slot"], info["target_index"])
    if bound is None or _point(bound[0]) != candidate.point or bound[0]["target_type"] != info["target_type"]:
        return False
    if (candidate.kind == "enter_curse") != bound[1]:
        return False
    health = _health(state["player"])
    expected = CROSSING_BUDGET * (2 if candidate.kind == "enter_curse" else 1)
    if (health is None or candidate.cost != {"health_half_hearts_budget": expected}
            or health < (1 if committed else expected + SURVIVAL_MARGIN)):
        return False
    return _door_move(state, parsed, _Door(info["slot"], candidate.point,
                                         info["target_index"], info["target_type"])) is not None


def curse_move(state, candidate):
    from .adventure import _parsed
    from .exploration import _Door, _door_move
    info = candidate.details
    return _door_move(state, _parsed(state), _Door(info["slot"], candidate.point,
                                                info["target_index"], info["target_type"]))
