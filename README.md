# Jev plays Isaac

An experimental AI player for **The Binding of Isaac: Repentance+**. A small Lua
mod sends structured game state to a local Python controller, which asks
[TypeSafe's Jev](https://typesafe.ai/) what to do.

The aim is to watch Jev play: choose enemies, shooting directions, rooms, items
and puzzle actions. Local code turns those choices into movement and handles
immediate collision avoidance.

**Work in progress.** It can fight and travel between supported rooms, but still
makes poor decisions, misses shots and sometimes gets stuck. On **21 September
2026**, it defeated **the Duke of Flies** and chose to descend to floor 2. That
first boss win is verified in the game and controller logs; consistent floor or
run completion is still unproven. See [the milestone](docs/first-boss.md).

## Who controls what?

| Jev decides | Local code executes |
| --- | --- |
| Engage an enemy, hold position or evade | Pathfinding and movement toward the chosen intent |
| Fire left, right, up, down or not at all | The exact chosen firing button |
| Reposition within a cleared room while choosing fire independently | Movement to the chosen position and the current firing button together |
| Which supported door to enter | Movement through that door |
| Which supplies, items or purchases to take | Eligibility checks and the selected interaction |
| Whether to try an unidentified pill | A bounded use of that pill, without revealing its effect first |
| Which pressure plate or TNT target to use | Movement, short shooting actions and explosive retreat |
| Whether to bomb an observed ordinary or tinted rock | One bomb at a checked placement, retreat and observation of the result |
| Whether to take a supported floor exit | The selected transition |

Emergency dodging can override movement. It does not replace Jev's chosen firing
direction. Waiting does not automatically pick a room, collect supplies or fire.
Unsupported mechanics are still filtered locally; not every skipped action is a
deliberate Jev decision. See [architecture and limits](docs/architecture.md).

Jev receives item-room discovery/visit status and remembers pickups left in
visited rooms, including heart types and shop prices. It can weigh returning
for supplies or searching for an item room. Remembered contents are checked
again on revisiting; unseen rooms and tinted-rock drops remain unknown.
Door choices distinguish unknown contents from a room last observed with no
remaining pickups. They also show remembered exits and recent completed entries;
Jev receives its recent room crossings without a forced exploration order.

With `--stay-ready`, a connection failure releases control while keeping the
controller and key open. Press **F8** to retry from fresh state within the same
remaining time/request limits. Pauses also refresh idle connections before the
next request.

## Requirements

- Windows and a Steam installation of Isaac with Repentance+ and mod support.
  This is the environment used for live development; other versions are unverified.
- Python **3.10 or newer**, including Tkinter for the masked key window.
- A [TypeSafe account and API key](https://console.typesafe.ai/) with access to Jev.
  Live play uses paid API requests.

The Python controller has no third-party runtime dependencies. Game files,
artwork, API credentials and personal run recordings are not included.

## Setup

1. Download or clone this repository and open a terminal in its folder.
2. Check Python with `py -3 --version`. If you use `python` instead, substitute
   it in the commands below. The Windows launchers use `.venv` when present,
   otherwise the Windows Python launcher (`py -3`).
3. Close Isaac, then run `py -3 launch.py install-mod`, or double-click
   **install-mod.cmd**. It copies `mod/jev_bridge` into the game's `mods` folder.
   A different existing bridge is not overwritten; back it up before replacing
   it. For a custom installation, use
   `py -3 launch.py install-mod --game-dir "PATH TO ISAAC"`.
4. In **Steam → Isaac → Properties → General → Launch Options**, add
   **`--luadebug`**. Keep any other launch options you need. This exposes the
   bundled Lua socket module and removes Lua sandbox restrictions, so use trusted
   mods. Fully restart Isaac after changing the option or updating the mod.
5. Enable **Jev Bridge** in Isaac's Mods menu, then start or continue a local
   single-player run. Ordinary Isaac with default tears is the best-tested setup.

## Watch Jev play

1. Double-click **start-floor.cmd**.
2. Enter the TypeSafe key in the masked local window. The launcher does not save
   the key to disk; it remains in that controller process until it exits.
3. Return to Isaac and press **F8** to enable control. **F8** stops it again.

Keep only one controller running. A controller waiting after death can be reused:
start a new run and press F8. It retains the key while that process stays open.

The equivalent command is:

```powershell
py -3 launch.py run --policy jev-goal --floor --adventure --jev-player --continue-floors --wait-for-arm --restart-on-death --stay-ready --duration 900 --max-calls 1200 --key-window --report reports/live-floor.json
```

Each attempt is capped at **900 seconds** and **1,200 HTTP requests**, at most
**two requests per second**. The timer starts at the first activation; pausing or
rearming does not reset it. Multiple questions share a request and can increase
token usage. These are time and request limits, **not a spending limit**.

Logs distinguish `Jev chose...`, `Local override...` and activity outcomes.
Reports are written locally under `reports/`, which Git ignores.

## Try it without the game or a key

```powershell
py -3 launch.py demo --report reports/demo.json
py -3 -m unittest discover -s tests -q
```

The demo uses synthetic state and local rules. The Python tests use offline
provider responses; neither command calls TypeSafe. The test suite includes
recorded room geometry, delayed controls, protocol validation and decision
ownership checks.

With Lua installed, run the separate mock bridge suite from the repository root:

```text
lua tests/test_mod.lua
```

The Lua suite mocks the game and networking. Passing it does not establish live
gameplay performance or reproduce the complete game engine.

Optional **check-typesafe.cmd** makes one paid connection-check request;
**check-speed.cmd** makes up to four. Neither controls the game.
**start-jev.cmd** and **start-baseline.cmd** are older comparison modes with
different local decision responsibilities; use **start-floor.cmd** for Jev
player mode.

## Troubleshooting

**“Jev unavailable: add --luadebug; restart game”** means the Lua bridge could
not load its JSON/socket dependencies. Confirm the Steam launch option is exactly
`--luadebug`, fully exit the game, then launch it through Steam again. F8 alone
cannot change a running game's launch options. Your existing controller can stay
open and retain its key.

**“Route blocked” or repeated backtracking:** the local route calculation can
reject a path because of conservative obstacle padding, while Jev can also choose
to revisit rooms. These are separate known problems. The local failure report
preserves the selected action and geometry for diagnosis; see
[known limitations](docs/architecture.md#known-limitations).

**No control after an update:** both the installed Lua mod and Python controller
must use the new files. A Lua update requires restarting Isaac; a Python update
requires restarting the controller. Pause and stale-state guards can also require
a fresh F8 activation.

**Stopped after using an active item:** update the Lua mod and restart Isaac.
Confirmed item-use animations now release inputs temporarily and resume with
fresh controls, without treating the animation as a manual pause. Actual pause
buttons still stop Jev. The controller can remain open with its existing key.

**“Incomplete room data” after entering a large room:** the bridge now supports
up to 512 hazards, including dense grids of walls, pits and rocks. If an
observation is still incomplete, player mode releases control and keeps the
listener and key open until the attempt's limits expire. Resume from a complete
observed room with F8. Pausing or switching windows does not restart the budget.

This is an independent fan experiment, unaffiliated with the creators of Isaac.
You need your own game installation and TypeSafe access.
