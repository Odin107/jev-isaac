"""General room inputs chosen by Jev, without assuming a weapon's effects."""
from .adventure import AdventureCandidate, _context
from .combat import _inside, _point
from .exploration import _point_waypoint, _room_geometry
from .navigation import _VECTORS, _free, _radius, _steer

def _geometry(state, parsed):
    boxes, _, _, phase = _room_geometry(state, parsed[4])
    # Repositioning does not choose a pickup or enter a door. Those interactions
    # have their own explicit activities, costs and transition permissions.
    for item in state.get("pickups", []) + state["doors"]:
        x, y = _point(item)
        pad = parsed[4] + (_radius(item, 10)+8 if "slot" not in item else 28)
        boxes.append((x-pad, y-pad, x+pad, y+pad))
    return boxes, phase


def room_input_move(state, parsed, point):
    boxes, phase = _geometry(state, parsed)
    start, velocity, bounds = parsed[3], parsed[5], parsed[6]
    if not _inside(start, bounds) or not _free(start, boxes):
        return None
    waypoint = _point_waypoint(start, point, bounds, boxes, phase)
    if waypoint is None:
        return None
    return _steer(start, waypoint, velocity, bounds, boxes, deadband=3, drift_frames=5)


def room_input_candidates(state, parsed):
    """Nearby destinations; the parallel firing question owns all button input."""
    if not state["room"]["clear"]:
        return ()
    result = []
    start = parsed[3]
    for name, vector in _VECTORS.items():
        if name == "none":
            continue
        point = (start[0]+vector[0]*40, start[1]+vector[1]*40)
        if room_input_move(state, parsed, point) is not None:
            result.append(AdventureCandidate(f"move_to:{name}", "move_to", name, point, {},
                "Move to this nearby observed room position; no pickup, door entry or firing is implied",
                context=_context(state), details={"position": {"x": point[0], "y": point[1]}}))
    return tuple(result)


def room_input_valid(state, parsed, candidate):
    if candidate.context != _context(state) or not state["room"]["clear"]:
        return False
    if candidate.kind == "move_to":
        return candidate.point is not None and room_input_move(state, parsed, candidate.point) is not None
    return False
