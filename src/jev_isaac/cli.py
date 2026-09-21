from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from .controller import Controller, baseline
from .demo import run_demo, sample
from .setup import find_game, install_mod

PROJECT = Path(__file__).resolve().parents[2]


def save_attempt(report: Path | None, result: dict) -> None:
    """Keep both the latest report and an immutable copy of each floor attempt."""
    if report is None:
        return
    report.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2)+"\n"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = report.with_name(f"{report.stem}-attempt-{stamp}{report.suffix}")
    with archive.open("x", encoding="utf-8") as output:
        output.write(payload)
    report.write_text(payload, encoding="utf-8")


def save_navigation_checkpoint(report: Path | None, snapshot: dict) -> None:
    """Make the latest game-only failure evidence available while waiting for F8."""
    if report is None:
        return
    destination = report.with_name(f"{report.stem}-navigation.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {"recorded_at": datetime.now(timezone.utc).isoformat(),
               "navigation_failure": snapshot}
    temporary.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    temporary.replace(destination)


def read_key(provider_name: str, key_variable: str, use_window: bool) -> str:
    if use_window:
        from .key_prompt import prompt_key_window
        return prompt_key_window(provider_name)
    key = os.environ.get(key_variable)
    if key:
        return key
    return getpass.getpass(f"{provider_name} API key (hidden, used for this process only): ").strip()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Jev + Isaac room and floor experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check Python, game, and key availability without calling an API")
    demo = sub.add_parser("demo", help="Synthetic transport test; no game or API key needed")
    demo.add_argument("--report", type=Path)
    check = sub.add_parser("check-access", help="Make one small synthetic Jev request to verify a key and latency")
    check.add_argument("--provider", choices=("typesafe", "openrouter"), default="typesafe")
    check.add_argument("--model", default=None)
    check.add_argument("--key-window", action="store_true", help="Use a masked window with normal paste support")
    check.add_argument("--report", type=Path)
    timing = sub.add_parser("measure-latency", help="Measure up to four synthetic requests on one HTTPS connection")
    timing.add_argument("--provider", choices=("typesafe", "openrouter"), default="typesafe")
    timing.add_argument("--model", default=None)
    timing.add_argument("--samples", type=int, choices=range(1, 5), default=4)
    timing.add_argument("--state", type=Path, help="Replay a saved game observation without sending game controls")
    timing.add_argument("--key-window", action="store_true", help="Use a masked window with normal paste support")
    timing.add_argument("--report", type=Path)
    installer = sub.add_parser("install-mod", help="Copy only our mod into an installed PC game")
    installer.add_argument("--game-dir", type=Path)
    live = sub.add_parser("run", help="Run a local controller for an already running game")
    live.add_argument("--policy", choices=("baseline", "jev", "jev-goal"), default="baseline")
    live.add_argument("--provider", choices=("typesafe", "openrouter"), default="typesafe")
    live.add_argument("--model", default=None, help="Defaults to the selected provider's Jev model")
    live.add_argument("--key-window", action="store_true", help="Use a masked window with normal paste support")
    live.add_argument("--port", type=int, default=42421)
    live.add_argument("--max-calls", type=int, default=30)
    live.add_argument("--duration", type=float, default=30)
    live.add_argument("--wait-for-arm", action="store_true",
                      help="Wait for F8 before starting the duration limit; later pauses do not reset it")
    live.add_argument("--restart-on-death", action="store_true",
                      help="Floor mode: retain the key in memory after death and wait for a new run plus F8")
    live.add_argument("--stay-ready", action="store_true",
                      help="Floor mode: keep waiting after pause or F8 off; another F8 resumes the same bounded attempt")
    live.add_argument("--one-room", action="store_true", help="Finish after the first combat room clears, changes, or the player dies")
    live.add_argument("--floor", action="store_true", help="Explore this floor using open ordinary doors, then fight the boss; Jev goal mode only")
    live.add_argument("--adventure", action="store_true",
                      help="Floor goal mode: let Jev choose verified pickups, spending and item use")
    live.add_argument("--jev-player", action="store_true",
                      help="Jev chooses rooms, supplies, puzzle targets and combat firing; local code executes")
    live.add_argument("--continue-floors", action="store_true",
                      help="Adventure mode: allow observed boss-room exits within the same time/request budget")
    live.add_argument("--hz", type=float, default=5)
    live.add_argument("--max-latency-ms", type=float, default=None,
                      help="Response cutoff: defaults to 500 ms for Jev or 250 ms for the local baseline")
    live.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            game = find_game()
            print(json.dumps({"python":sys.version.split()[0], "game_installed":game is not None,
                              "game_folder":str(game) if game else None,
                              "api_key_in_environment":bool(os.environ.get("TYPESAFE_API_KEY")),
                              "openrouter_key_in_environment":bool(os.environ.get("OPENROUTER_API_KEY")),
                              "mod_source_ready":(PROJECT/"mod"/"jev_bridge"/"main.lua").is_file()}, indent=2))
            return 0
        if args.command == "install-mod":
            target = install_mod(PROJECT, args.game_dir)
            print(f"Mod installed: {target}")
            print("In Steam, set Isaac Launch Options to --luadebug. Enable Jev Bridge in the game's Mods menu.")
            print("Use a local test run. Launch this controller, then press F8 in an uncleared room.")
            return 0
        if args.command == "check-access":
            from .jev import JevClient, JevError
            provider_name = "OpenRouter" if args.provider == "openrouter" else "TypeSafe"
            key_variable = "OPENROUTER_API_KEY" if args.provider == "openrouter" else "TYPESAFE_API_KEY"
            print(f"One {provider_name} request using a synthetic room. No game controls will be sent.")
            print("The key is hidden and is not saved. Provider billing applies to this one request.")
            key = read_key(provider_name, key_variable, args.key_window)
            if not key:
                raise ValueError("No API key entered. No requests made.")
            client = None
            try:
                client = JevClient(api_key=key, provider=args.provider, model=args.model, timeout=15)
                decision = client.decide(sample())
                result = {"mode":"access-check", "provider":args.provider, "success":True,
                          "requests":1, "synthetic_state":True, "game_controls_sent":False,
                          "model":decision.model, "latency_ms":round(decision.latency_ms, 1),
                          "move":decision.move, "shoot":decision.shoot, "usage":dict(decision.usage)}
            except JevError as exc:
                result = {"mode":"access-check", "provider":args.provider, "success":False,
                          "synthetic_state":True, "game_controls_sent":False, "error":str(exc)}
            finally:
                if client is not None:
                    client.close()
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
            print(json.dumps(result, indent=2))
            return 0 if result["success"] else 1
        if args.command == "measure-latency":
            from .timing import run_timing
            replay_options = {}
            if args.state:
                from .protocol import MAX_DATAGRAM, Observation
                if args.state.stat().st_size > MAX_DATAGRAM:
                    raise ValueError("Saved room observation exceeds the size limit.")
                replay_options["state"] = Observation.decode(args.state.read_bytes()).data
            provider_name = "OpenRouter" if args.provider == "openrouter" else "TypeSafe"
            key_variable = "OPENROUTER_API_KEY" if args.provider == "openrouter" else "TYPESAFE_API_KEY"
            print(f"At most {args.samples} small {provider_name} requests to measure connection and response time.")
            print("Saved or synthetic room data only; no game controls. The key is not saved. Provider billing applies.")
            key = read_key(provider_name, key_variable, args.key_window)
            if not key:
                raise ValueError("No API key entered. No requests made.")
            result = run_timing(key, provider=args.provider, model=args.model, samples=args.samples, **replay_options)
        elif args.command == "demo":
            result = run_demo()
        else:
            policy = baseline
            client = None
            is_jev = args.policy in {"jev", "jev-goal"}
            goal_mode = args.policy == "jev-goal"
            if args.floor and (not goal_mode or args.one_room):
                raise ValueError("Use --floor with --policy jev-goal and without --one-room.")
            if args.adventure and (not args.floor or not goal_mode):
                raise ValueError("Use --adventure with --floor and --policy jev-goal.")
            if args.continue_floors and not args.adventure:
                raise ValueError("Use --continue-floors with --adventure.")
            if args.jev_player and not args.adventure:
                raise ValueError("Use --jev-player with --adventure, --floor and --policy jev-goal.")
            if args.restart_on_death and not args.floor:
                raise ValueError("Use --restart-on-death with --floor.")
            if args.stay_ready and not args.floor:
                raise ValueError("Use --stay-ready with --floor.")
            if is_jev:
                from .jev import JevClient
                request_hz = min(args.hz, 2) if goal_mode else args.hz
                print(f"Jev mode: at most {args.max_calls} HTTP requests over {args.duration:g}s, at most {request_hz:g}/s.")
                if args.jev_player:
                    print("Jev player mode: Jev chooses activities, room order and firing directions.")
                    print("Local code executes selected actions; emergency movement overrides are logged.")
                elif args.adventure:
                    print("Jev chooses combat goals and verified interactions; local movement updates each game frame.")
                    print("Cleared-room interaction choices and combat item-use choices share the same request cap.")
                    if args.continue_floors:
                        print("Selected boss-room exits can continue to the next floor without resetting these limits.")
                else:
                    print("One replaceable goal per request; local movement updates each game frame." if goal_mode
                          else "Each request contains movement and shooting questions.")
                print("Provider billing applies; a request cap is not a dollar cap.")
                if args.restart_on_death:
                    print("Limits apply to each attempt. After death, a new run and F8 are required to start another.")
                provider_name = "OpenRouter" if args.provider == "openrouter" else "TypeSafe"
                key_variable = "OPENROUTER_API_KEY" if args.provider == "openrouter" else "TYPESAFE_API_KEY"
                print(f"Provider: {provider_name}")
                key = read_key(provider_name, key_variable, args.key_window)
                if not key:
                    raise ValueError("No API key entered. No requests made.")
                if args.jev_player:
                    from .player_policy import PlayerClient
                    client_class = PlayerClient
                elif args.adventure:
                    from .strategy import AdventureClient
                    client_class = AdventureClient
                elif goal_mode:
                    from .goals import GoalClient
                    client_class = GoalClient
                else:
                    client_class = JevClient
                client = client_class(api_key=key, provider=args.provider, model=args.model, timeout=2,
                                   choice_policy="argmax")
                policy = client.decide
            else:
                print("Baseline mode: simple local rule, no Jev API calls. For testing controls only.")
            max_latency_ms = args.max_latency_ms if args.max_latency_ms is not None else (500 if is_jev else 250)
            try:
                if client is not None:
                    from .jev import JevError
                    print("Preparing the secure connection; no model request is made yet.")
                    try:
                        client.warm_connect()
                    except JevError:
                        print("Connection preparation unavailable; the first decision will open it.")
                excluded_run_id = None
                while True:
                    result = Controller(policy, port=args.port, max_calls=args.max_calls, duration=args.duration,
                                        max_hz=min(args.hz, 2) if goal_mode else args.hz, max_latency=max_latency_ms/1000,
                                        hold_frames=15 if is_jev else 6,
                                        move_frames=6 if is_jev else None,
                                        move_distance=20 if is_jev else None, goal_mode=goal_mode,
                                        one_room=args.one_room, floor_mode=args.floor, startup_guard=goal_mode,
                                        adventure_mode=args.adventure, continue_floors=args.continue_floors,
                                        jev_player=args.jev_player,
                                        wait_for_arm=args.wait_for_arm or args.restart_on_death or args.stay_ready,
                                        stay_ready=args.stay_ready,
                                        on_navigation_stop=lambda snapshot: save_navigation_checkpoint(args.report, snapshot),
                                        excluded_run_id=excluded_run_id).run()
                    result["mode"] = args.policy
                    if is_jev:
                        result.update(provider=args.provider, model=client.model, choice_policy="argmax")
                    if args.restart_on_death or args.stay_ready:
                        save_attempt(args.report, result)
                    after_death = args.restart_on_death and result.get("stop_reason") == "player died"
                    after_new_run = args.stay_ready and result.get("stop_reason") == "run changed"
                    if not (after_death or after_new_run):
                        break
                    # A manually restarted game may already be in the last
                    # snapshot. Exclude the finished attempt's original run,
                    # never the new run the user wants to arm next.
                    ended = result.get("last_observation" if after_death else "first_observation") or {}
                    excluded_run_id = ended.get("run_id")
                    if not isinstance(excluded_run_id, str) or not excluded_run_id:
                        print("Cannot identify the finished run; restart the controller for another attempt.")
                        break
                    # The completed controller has released input and joined its
                    # API worker. Keep only the key/client in memory; a new
                    # controller has fresh counters, goals, map and time budget.
                    if client is not None:
                        client.close()
                    print("Isaac died. Report saved. Start a new run and press F8 when ready." if after_death
                          else "New run detected. Previous report saved. Press F8 in the new run when ready.")
                    print("The controller is staying open; your key remains in this process only.")
            finally:
                if client is not None:
                    client.close()
            result["mode"] = args.policy
            if is_jev:
                result["provider"] = args.provider
                result["model"] = client.model
                result["choice_policy"] = "argmax"
        if getattr(args, "report", None):
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({k:v for k,v in result.items()
                          if k not in {"last_failed_observation", "first_observation", "last_observation", "recent_local_controls"}}, indent=2))
        if args.command == "measure-latency":
            return 0 if result["success"] else 1
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        # These messages are our own setup errors. Provider errors are handled in Controller.
        print(str(exc), file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("Cancelled.")
        return 130
