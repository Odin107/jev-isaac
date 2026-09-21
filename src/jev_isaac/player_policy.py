"""Jev owns activity selection and explicit firing; code executes bound intents."""
from dataclasses import dataclass
import json

from .goals import GoalClient, GoalDecision, build_goal_request, _candidates
from .jev import MAX_RESPONSE_BYTES, JevResponseError, _parse_choice, _unique_object, _GRID_COORDINATES
from .state_context import CONTEXT_INSTRUCTIONS
from .strategy import StrategyDecision, _options
from .combat import build_combat_context, current_firing_view


@dataclass(frozen=True)
class PlayerGoalDecision(GoalDecision):
    fire_direction: str = "none"
    ability_key: str | None = None
    fire_judgment: dict | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.fire_direction not in ("none", "left", "right", "up", "down"):
            raise ValueError("Invalid player firing direction")


@dataclass(frozen=True)
class PlayerActivityDecision(StrategyDecision):
    fire_direction: str = "none"
    fire_judgment: dict | None = None

    def __post_init__(self):
        if self.fire_direction not in ("none", "left", "right", "up", "down"):
            raise ValueError("Invalid activity firing direction")


def player_contract(payload, phase):
    payload["state"]["game_context"]["control_contract"] = {
        "decision_kind": phase, "authority": "Jev chooses the activity, destination, target and firing intention.",
        "objective": "Play toward completing the run. Exploration, resource use, risk and timing are Jev's decisions within the available controls.",
        "local_execution": "Pathfinding executes the movement goal; the firing button follows Jev's direction. No independent pickups, room order or puzzle selection.",
        "exceptions": "Fresh immediate collision avoidance may change movement and is reported. Expired or changed-state intent is canceled. A committed explosive retreat finishes before another decision.",
        "firing": "Movement/activity and firing are independent choices in the same request. The firing button uses the current weapon. Selected shoot_prop, demolish_tnt and bomb_rock routines own firing/retreat while active; the separate fire answer is ignored for those routines.",
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
                "switches and when to descend. The objective is to complete the run. "
                "The observed room, resources, inventory, controller_context.exploration memory "
                "and previous outcomes are available as state. Revisiting rooms, skipping rewards "
                "and descending with unexplored rooms remaining are your decisions. "
                "controller_context.exploration.item_rooms records whether "
                "one was found or visited and its last observed pickups; game_context.item_rooms explains its upgrade value. "
                "game_context.bombs_and_rocks explains bomb costs, tinted rocks and blast risks. "
                "controller_context.exploration.remembered_pickups lists supplies last seen in other rooms, "
                "so returning for an earlier heart or resource is an available planning choice when routes permit. "
                "No local policy will pick a door or collect a reward if you wait. Waiting here holds the activity; "
                "the independent fire answer may still authorize shooting. "
                "A door's appearance/type does not reveal its contents. "
                "Each door candidate's `details.destination_pickups_last_observed` distinguishes "
                "present pickups, none at the last complete observation, and unknown contents. "
                "Treasure-room type alone is not evidence of an unclaimed item. Remembered contents and "
                "permitted exits are last-seen facts, not guarantees of current contents or hidden routes. "
                "`controller_context.exploration.recent_room_transitions` records actual recent crossings; "
                "door candidates also count recent completed entries from here. These are history, not commands. "
                "General move_to activities reposition inside the room without collecting or entering a door. "
                "Pressing a switch does not authorize TNT demolition; select demolition explicitly "
                "when needed. Destruction does not automatically select the switch afterward. "
                "Movement and selected prop-shot alignment are executed locally; emergency collision avoidance can intervene. "
                "Wait means pause the activity briefly, then reconsider fresh state." + CONTEXT_INSTRUCTIONS),
                "criteria": {"wait": "Hold the activity; no implied movement, item use or shooting.",
                             **{c["option"]: f"{c.get('description', 'Execute this activity')}. "
                                f"Bound activity {c['key']}; details in `activity_candidates[{i}]`."
                                for i, c in enumerate(payload["state"]["activity_candidates"])}}}}
            player_contract(payload, "activity")
            if len(offered) == 1 and offered[0]["kind"] == "continue":
                payload["questions"] = {}
            payload["questions"]["fire"] = {"type": "choice", "instructions": (
                "Choose the firing button now, independently of the activity choice. You can fire while "
                "moving, collecting, using an item, entering a door or holding position. The current weapon "
                "determines the effect; this input does not guarantee a hit or destruction. Inspect observed "
                "objects, their positions/health, inventory and weapon type. Shots can trigger explosives. "
                "Coordinates are x right, y down, relative to the CURRENT player position. Observed velocity "
                "can deflect tears; do not assume the parallel activity answer or a future position. "
                "If shoot_prop, demolish_tnt or bomb_rock is executing, that selected routine owns firing "
                "and this answer is ignored until it finishes. Consecutive fresh equal directions keep the "
                "button held; none releases it, including for charge/release weapons. While a bound activity "
                "continues, fire-only requests update aim without restarting its movement. " + CONTEXT_INSTRUCTIONS),
                "criteria": {"none": "Release firing input.", "left": "Hold firing toward smaller x.",
                             "right": "Hold firing toward larger x.", "up": "Hold firing toward smaller y.",
                             "down": "Hold firing toward larger y."}}
            if not clean["room"]["clear"]:
                # Unfinished room puzzles retain their selected demolition
                # contract, including its committed explosive retreat.
                payload["questions"].pop("fire")
        else:
            targets = _candidates(clean, limit=64)
            payload["state"]["combat_context"] = build_combat_context(clean, target_limit=64)
            payload["state"]["firing_now"] = current_firing_view(payload["state"]["combat_context"])
            bindings = {v["option"]: v["id"] for v in targets}
            payload["state"]["goal_candidates"] = targets
            payload["questions"] = {
                "goal": {"type": "choice", "instructions": (
                    "Choose your movement intention in The Binding of Isaac: engage a listed enemy "
                    "to seek a clear firing position, evade threats, or hold position. You own the "
                    "target; the local executor will not pursue a different one. Emergency collision "
                    "avoidance may change a short movement. This answer does NOT authorize shooting; "
                    "the independent fire answer does. Obstacles, health and threat motion are observed state. "
                    "Unknown enemy phases and effects stay unknown." + CONTEXT_INSTRUCTIONS),
                    "criteria": {"hold": "Hold position, subject to immediate collision avoidance; no implied shooting.",
                                 "evade": "Move away from nearby threats; no implied shooting.",
                                 **{o: f"Engage observed enemy {ident}; seek alignment with that target."
                                    for o, ident in bindings.items()}}},
                "fire": {"type": "choice", "instructions": (
                    "Choose your firing input now to play toward completing the run. You own aiming. "
                    "`firing_now.directions` groups living vulnerable enemies by direction from the CURRENT "
                    "player position: enemies_on_side need not be aligned; aligned_enemies cross that straight "
                    "firing lane; grid_clear_aligned_enemies also have no observed solid-grid blocker. "
                    "Null means unknown. These are geometry facts, not guaranteed hits or required choices. "
                    "`combat_context.targets` gives distances and offsets; `observation` includes weapon, "
                    "objects and threats. Shots can trigger explosives. "
                    "Past `observation.control` and controller memory describe previous inputs, not a request "
                    "to repeat them. `combat_context.firing_positions` are hypothetical future positions; their shoot "
                    "directions do not describe shots from here. "
                    "You can shoot while moving, holding or evading. The movement answer is independent and "
                    "unknown to this question. Use observed player vx/vy: momentum can deflect tears diagonally "
                    "and persists after movement release; exact inheritance is uncalibrated. "
                    "Equal consecutive directions hold the button; none releases it, including for charge/release "
                    "weapons. Local code preserves your fire choice. Coordinates: x right, y down."),
                    "criteria": {"none": "Do not shoot.",
                                 "left": "Press LEFT: firing input toward smaller x, to the LEFT of the player. Momentum can deflect tears.",
                                 "right": "Press RIGHT: firing input toward larger x, to the RIGHT of the player. Momentum can deflect tears.",
                                 "up": "Press UP: firing input toward smaller y, ABOVE the player. Momentum can deflect tears.",
                                 "down": "Press DOWN: firing input toward larger y, BELOW the player. Momentum can deflect tears."}},
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
            choices, corrections, fire_judgment = {}, [], None
            for key, question in payload["questions"].items():
                answer = decoded["answers"][key]
                if not isinstance(answer, dict) or set(answer) != {"type", "choice", "confidence", "probabilities"}:
                    raise ValueError()
                chosen, confidence, probabilities, correction = _parse_choice(answer, tuple(question["criteria"]), key,
                                                         self.provider, self.choice_policy)
                choices[key] = chosen
                if key == "fire":
                    fire_judgment = {"reported_choice": answer["choice"],
                                     "confidence": confidence, "probabilities": probabilities}
                if correction:
                    corrections.append(correction)
        except (ValueError, TypeError, UnicodeError, RecursionError, KeyError):
            raise JevResponseError("Invalid Jev player decision", latency_ms=elapsed) from None
        usage = {k: usage[k] for k in ("input_tokens", "output_tokens")}
        if offered:
            target = "continue" if offered[0]["kind"] == "continue" else bindings.get(choices["activity"])
            return PlayerActivityDecision("adventure", target, elapsed,
                                          model, usage, tuple(corrections), choices.get("fire", "none"),
                                          fire_judgment=fire_judgment)
        choice = choices["goal"]
        return PlayerGoalDecision("engage" if choice in bindings else choice, bindings.get(choice),
                                  elapsed, model, usage, tuple(corrections),
                                  choices["fire"], ability_bindings.get(choices.get("ability")),
                                  fire_judgment=fire_judgment)
