"""Jev owns activity selection and explicit firing; code executes bound intents."""
from dataclasses import dataclass
import json

from .goals import GoalClient, GoalDecision, build_goal_request, _candidates
from .jev import MAX_RESPONSE_BYTES, JevResponseError, _parse_choice, _unique_object, _GRID_COORDINATES
from .state_context import CONTEXT_INSTRUCTIONS
from .strategy import StrategyDecision, _options
from .combat import build_combat_context


@dataclass(frozen=True)
class PlayerGoalDecision(GoalDecision):
    fire_direction: str = "none"
    ability_key: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.fire_direction not in ("none", "left", "right", "up", "down"):
            raise ValueError("Invalid player firing direction")


def player_contract(payload, phase):
    payload["state"]["game_context"]["control_contract"] = {
        "decision_kind": phase, "authority": "Jev chooses the activity, destination, target and firing intention.",
        "local_execution": "Pathfinding executes the movement goal; the firing button follows Jev's direction. No independent pickups, room order or puzzle selection.",
        "exceptions": "Fresh immediate collision avoidance may change movement and is reported. Expired or changed-state intent is canceled. A committed explosive retreat finishes before another decision.",
        "firing": "No automatic or substitute shooting. Combat uses only Jev's explicit cardinal firing direction. A selected shoot_prop or demolish_tnt activity authorizes that target's bounded aimed shots and retreat.",
        "limits": "Only offered, implemented actions can execute; missing choices may be a mechanics/geometry limitation. Unknown contents and effects remain unknown.",
    }


class PlayerClient(GoalClient):
    def decide(self, state):
        offered = state.get("_adventure_options")
        abilities = state.get("_ability_options")
        clean = {k: v for k, v in state.items() if k not in ("_adventure_options", "_ability_options")}
        payload = build_goal_request(clean, self.model)
        bindings, ability_bindings = {}, {}
        if offered:
            if (not isinstance(offered, list) or len(offered) > 192
                    or any(not isinstance(v, dict) or not isinstance(v.get("key"), str) for v in offered)
                    or len({v["key"] for v in offered}) != len(offered)):
                raise ValueError("Invalid player activities")
            offered = json.loads(json.dumps(offered, allow_nan=False))
            bindings = {f"action_{i}": c["key"] for i, c in enumerate(offered) if c["kind"] != "wait"}
            payload["state"]["goal_candidates"] = []
            payload["state"]["activity_candidates"] = [dict(c, option=f"action_{i}")
                for i, c in enumerate(offered) if c["kind"] != "wait"]
            payload["questions"] = {"activity": {"type": "choice", "instructions": (
                "Play The Binding of Isaac: choose your next activity from activity_candidates. "
                "You own exploration order, free supplies, purchases, items, prop/TNT destruction, "
                "switches and when to descend. Compare the observed room, resources, inventory, "
                "controller_context.exploration memory and previous outcomes. No local policy "
                "will pick a door or collect a reward if you wait. Choose a useful next step "
                "toward surviving and completing the run; avoid repeating failed activities or "
                "unnecessary revisits. A door's appearance/type does not reveal its contents. "
                "Pressing a switch does not authorize TNT demolition; select demolition explicitly "
                "when needed. Destruction does not automatically select the switch afterward. "
                "Movement and selected prop-shot alignment are executed locally; emergency collision avoidance can intervene. "
                "Wait means pause briefly without firing, then reconsider fresh state." + CONTEXT_INSTRUCTIONS),
                "criteria": {"wait": "Wait briefly without choosing an activity or shooting.",
                             **{option: f"Execute the bound activity {key}." for option, key in bindings.items()}}}}
            player_contract(payload, "activity")
        else:
            targets = _candidates(clean, limit=64)
            payload["state"]["combat_context"] = build_combat_context(clean, target_limit=64)
            bindings = {v["option"]: v["id"] for v in targets}
            payload["state"]["goal_candidates"] = targets
            payload["questions"] = {
                "goal": {"type": "choice", "instructions": (
                    "Choose your movement intention in The Binding of Isaac: engage a listed enemy "
                    "to seek a clear firing position, evade threats, or hold position. You own the "
                    "target; the local executor will not pursue a different one. Emergency collision "
                    "avoidance may change a short movement. This answer does NOT authorize shooting; "
                    "the independent fire answer does. Consider obstacles, health and threat motion. "
                    "Unknown enemy phases and effects stay unknown." + CONTEXT_INSTRUCTIONS),
                    "criteria": {"hold": "Hold position, subject to immediate collision avoidance; no implied shooting.",
                                 "evade": "Move away from nearby threats; no implied shooting.",
                                 **{o: f"Engage observed enemy {ident}; seek alignment with that target."
                                    for o, ident in bindings.items()}}},
                "fire": {"type": "choice", "instructions": (
                    "Choose the firing button direction right now. You own aiming; local code will "
                    "not substitute a target or direction. You may fire while moving, holding or evading. "
                    "Aim to hit a living vulnerable enemy from the CURRENT player position. "
                    "Use combat_context.targets: dx/dy are enemy minus player; positive dx means right, "
                    "negative dx left, positive dy down, negative dy up. Each target has horizontal "
                    "and vertical lanes with direction, perpendicular offset, alignment and blockers. "
                    "A lane direction alone is NOT a hit: compare its offset and blockers. "
                    "combat_context.firing_positions are hypothetical future positions, NOT shots "
                    "available from where you stand now. Do not copy their shoot field as current aim. "
                    "Prefer an unblocked current lane; do not keep a previous button merely because "
                    "it was used before. If no useful shot exists, choose none while repositioning. "
                    "Player momentum affects tear trajectory: moving across the firing axis can make "
                    "shots travel diagonally. Use observation.player.vx/vy and positions to account for "
                    "that drift, lead a target, or settle before a precise shot. Releasing movement input "
                    "does not instantly stop momentum. ShotSpeed is not a measured world-velocity or "
                    "momentum multiplier; exact inheritance is uncalibrated. Emergency dodging can "
                    "change player motion between replies; reassess from fresh observations. These "
                    "questions are answered independently: use observed velocity and applied control, "
                    "not an assumed answer to the movement question. Choose none "
                    "to withhold fire. Coordinates: x right, y down."),
                    "criteria": {"none": "Do not shoot.",
                                 "left": "Shoot toward smaller x. Aim at an enemy LEFT of the player; compare vertical offset and blockers.",
                                 "right": "Shoot toward larger x. Aim at an enemy RIGHT of the player; compare vertical offset and blockers.",
                                 "up": "Shoot toward smaller y. Aim at an enemy ABOVE the player; compare horizontal offset and blockers.",
                                 "down": "Shoot toward larger y. Aim at an enemy BELOW the player; compare horizontal offset and blockers."}},
            }
            if abilities:
                abilities = _options(abilities)
                ability_bindings = {f"ability_{i}": c["key"] for i, c in enumerate(abilities)}
                payload["state"]["ability_candidates"] = [dict(c, option=f"ability_{i}") for i, c in enumerate(abilities)]
                payload["questions"]["ability"] = {"type": "choice", "instructions": (
                    "Independently choose a listed item or consumable to use now, based on health, "
                    "threats, resources and observed effects. Choose none to save it. Code applies "
                    "a single bound pulse after checking the fresh state; unknown effects stay unknown."),
                    "criteria": {"none": "Keep current items for later.",
                                 **{o: f"Use bound item {key} now." for o, key in ability_bindings.items()}}}
            player_contract(payload, "combat")
        if "grid_hazards" in payload["state"]["observation"]:
            for question in payload["questions"].values():
                question["instructions"] += " " + _GRID_COORDINATES
        body, elapsed = self.post(payload)
        try:
            if not isinstance(body, bytes) or len(body) > MAX_RESPONSE_BYTES:
                raise ValueError()
            decoded = json.loads(body.decode("utf-8"), object_pairs_hook=_unique_object)
            if not isinstance(decoded, dict) or not isinstance(decoded.get("answers"), dict):
                raise ValueError()
            if set(decoded["answers"]) != set(payload["questions"]):
                raise ValueError()
            model, usage = decoded.get("model"), decoded.get("usage")
            if not isinstance(model, str) or not model.strip() or not isinstance(usage, dict):
                raise ValueError()
            if any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("input_tokens", "output_tokens")):
                raise ValueError()
            choices, corrections = {}, []
            for key, question in payload["questions"].items():
                answer = decoded["answers"][key]
                if not isinstance(answer, dict) or set(answer) != {"type", "choice", "confidence", "probabilities"}:
                    raise ValueError()
                chosen, _, _, correction = _parse_choice(answer, tuple(question["criteria"]), key,
                                                         self.provider, self.choice_policy)
                choices[key] = chosen
                if correction:
                    corrections.append(correction)
        except (ValueError, TypeError, UnicodeError, RecursionError, KeyError):
            raise JevResponseError("Invalid Jev player decision", latency_ms=elapsed) from None
        usage = {k: usage[k] for k in ("input_tokens", "output_tokens")}
        if offered:
            return StrategyDecision("adventure", bindings.get(choices["activity"]), elapsed,
                                    model, usage, tuple(corrections))
        choice = choices["goal"]
        return PlayerGoalDecision("engage" if choice in bindings else choice, bindings.get(choice),
                                  elapsed, model, usage, tuple(corrections),
                                  choices["fire"], ability_bindings.get(choices.get("ability")))
