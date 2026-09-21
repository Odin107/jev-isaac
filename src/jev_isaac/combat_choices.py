"""Source-bound movement/target/fire combinations, within Choice's 255 options."""
from .state_context import CONTEXT_INSTRUCTIONS

FIRE_DIRECTIONS = ("none", "left", "right", "up", "down")
TARGETS_PER_GROUP = 24  # (24 targets * 2 maneuvers + hold + evade) * 5 = 250.

_INSTRUCTIONS = (
    "Choose one coordinated combat action in The Binding of Isaac. Each option binds a "
    "movement goal, its enemy target when needed, and an exact firing button together. "
    "You own all three choices. Engage seeks a cardinal firing position for its bound enemy. "
    "Back off tries a direct checked retreat toward a farther cardinal position for that enemy, "
    "without an inward detour, at most 220 world units away (ordinary-tear range minus 20 "
    "when shorter). This is approximate for unknown or special weapons. It holds if no safe "
    "farther position exists. Hold does not pursue; evade moves away from nearby threats. "
    "Local pathfinding executes the selected movement and immediate collision avoidance may "
    "override movement, but never selects a different enemy or firing button. "
    "`firing_now.directions` describes enemies on each side, currently aligned enemies, and "
    "aligned enemies without an observed solid-grid blocker. Null means unknown. These are "
    "current straight-line geometry facts, not guaranteed hits. `combat_context.firing_positions` "
    "describes hypothetical future positions, not firing directions from here. "
    "Coordinates: x right, y down. Use observed vx/vy: momentum can deflect tears diagonally "
    "and persists after release; exact inheritance is uncalibrated. Equal consecutive firing "
    "directions keep the button held. None releases firing input, which can release a charged "
    "shot; it does not mean a charged weapon cannot fire. Shots can trigger explosives. "
    "`controller_context` reports the previous goal, actual observed inputs, current local "
    "movement/override and request timing. Its `combat_feedback` gives recent observed health "
    "and movement changes. These are feedback, not instructions to repeat or change a choice. "
    "Health changes do not identify what caused damage; disappearance is not proof of a kill. "
    "The game keeps moving while this request is answered. Weapon charge readiness and exact "
    "future hits are unknown. Choose using the available observations and the run objective. "
    + CONTEXT_INSTRUCTIONS
)


def build_combat_choices(targets):
    """Return questions, bound options per question, and factual group metadata.

    A dense room uses speculative branches in the same HTTP request. Jev chooses
    the target group and a complete action within it; code never compares scores
    from different groups or silently drops targets to satisfy the API limit.
    """
    groups = [targets[i:i + TARGETS_PER_GROUP]
              for i in range(0, len(targets), TARGETS_PER_GROUP)] or [[]]
    questions, bindings, descriptions = {}, {}, []
    for index, group in enumerate(groups):
        question = "combat" if len(groups) == 1 else f"combat_{index}"
        group_name = f"group_{index}"
        maneuvers = [("hold", "hold", None), ("evade", "evade", None)]
        maneuvers.extend((t["option"], "engage", t["id"]) for t in group)
        maneuvers.extend((f"back_off_{t['option']}", "back_off", t["id"]) for t in group)
        bound, criteria = {}, {}
        for option, kind, target in maneuvers:
            for fire in FIRE_DIRECTIONS:
                key = f"{option}__{fire}"
                bound[key] = {"kind": kind, "target_id": target, "fire_direction": fire}
                movement = f"{kind} enemy {target}" if target is not None else kind
                firing = "release firing input" if fire == "none" else f"hold firing {fire}"
                criteria[key] = f"{movement}; {firing}."
        prefix = (f"Assuming combat target group {group_name} is selected, choose within this group. "
                  "This answer is used only for that group. " if len(groups) > 1 else "")
        questions[question] = {"type": "choice", "instructions": prefix + _INSTRUCTIONS,
                               "criteria": criteria}
        bindings[question] = bound
        descriptions.append({"option": group_name, "question": question,
                             "target_ids": [t["id"] for t in group]})
    if len(groups) > 1:
        questions["combat_group"] = {
            "type": "choice",
            "instructions": (
                "Choose the group containing the enemy you want your next combat maneuver to "
                "engage or back off from. `combat_groups` gives the observed target IDs in each "
                "group. Each group offers every listed enemy with engage/back-off and all five "
                "firing inputs. Each also offers hold/evade with all firing inputs, so any group "
                "can be selected for a maneuver without a target. These groups are only an API "
                "size partition; their ordering has no strategic meaning. Your selected group's "
                "complete movement/target/fire answer will execute; other answers are unused. "
                "Choose from current observed threats, resources and `controller_context` feedback. "
                + CONTEXT_INSTRUCTIONS),
            "criteria": {g["option"]: f"Use the complete action selected for targets {g['target_ids']}; "
                         "hold and evade also available." for g in descriptions},
        }
    return questions, bindings, descriptions


def selected_combat_question(choices, groups):
    if len(groups) == 1:
        return "combat"
    return next(g["question"] for g in groups if g["option"] == choices["combat_group"])
