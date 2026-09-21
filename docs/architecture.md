# Architecture and current limits

## State and decisions

```mermaid
flowchart LR
  G[Isaac Lua bridge] -->|Structured state over loopback UDP| C[Python controller]
  C -->|State and typed choices over HTTPS| J[TypeSafe Jev]
  J -->|Chosen activity, intent and firing direction| C
  C -->|Bounded controls over loopback UDP| G
```

Jev receives structured state, not screenshots. Observations include player
position and velocity, health and resources, item metadata where available,
enemies, projectiles, grid collisions, supported ground effects, pickups, doors,
pressure plates and observed floor connections. The controller adds relative
combat geometry, visited-room memory, recent outcomes and remaining budgets.
Missing information is not meant to imply an empty or safe room.

`player_policy.py` builds typed choices for combat and room activities.
`player_navigation.py` executes selected activities through the movement and
interaction helpers. `controller.py` coordinates fresh observations, asynchronous
model replies, pauses, budgets and reports. `mod/jev_bridge/main.lua` exports the
state and applies bounded controls after F8 activation.

Combat intent and firing direction are independent questions in one request.
Neither question can assume it knows the other's answer. Aim hints describe the
current position; potential future firing positions are labeled separately.
Jev is told that movement momentum can bend tears across the firing axis.
Actual tear-velocity inheritance has not been calibrated.

The local loop runs much more frequently than model decisions. Immediate dodges
can override movement and are logged. Local execution preserves the selected
cardinal firing button. A blocked selected activity returns to Jev rather than
automatically choosing a different room or pickup.

Cleared rooms also offer general `move_to` positions and an independent `fire`
question in the same request as activity selection. Jev can move and fire, or hold
the activity and fire, simultaneously. Repositioning does not collect pickups or
cross doors. During an ongoing activity, fire-only requests refresh aim without
restarting the selected route. Equal consecutive directions can keep a button
held; `none` releases it. The same freshness, request-rate and budget limits apply.
Selected prop shooting and TNT/bomb routines own firing while they execute, so
independent firing cannot disrupt their aimed shots or committed retreat.

Firing uses the observed current weapon, including Technology, without promising
a hit or assuming ordinary-tear physics. Weapon changes invalidate a previous
firing choice. Charge/release mechanics and synergies are not fully modeled.
Object-specific actions remain useful shortcuts, but their narrower weapon
support no longer removes all firing choices in cleared rooms.

After combat, a stationary ordinary/red fire can leave Isaac within its extra
navigation buffer. A bounded outward step may leave that buffer while retaining
the player radius plus two units around the observed fire body. Normal routes
retain the larger buffer; contact overlap, inward momentum, unobserved/special
fire behavior and all other obstacle exclusions still block this recovery.

## Known limitations

The design policy is to describe the objective, observed state, game mechanics
and control constraints, and leave strategy to Jev. Prompts do not prescribe
which rooms to visit, collecting rewards before leaving, when to spend resources,
or whether to revisit a room. Model-chosen descent is available from a supported
cleared boss room even while unexplored rooms or rewards remain. Existing local
pathfinding, immediate dodges and implemented-action limits still apply.

- **Aiming and latency:** replies can arrive after enemy geometry changes.
  Recent live responses were commonly around 350 ms. Better state and prompts
  do not guarantee accurate shooting.
- **Pathfinding:** conservative player/obstacle padding can falsely block
  available routes. A shallow overlap near a type-3 rock remains an unresolved
  example. Earlier TNT and door fixes do not solve every layout.
- **Defensive movement:** local emergency dodging can hold near a wall while
  Jev is still choosing to engage an enemy. Its short threat prediction does not
  guarantee an escape from crowding; these movements are logged as overrides.
- **Exploration:** Jev can repeatedly revisit cleared rooms. Known exits and
  recent route outcomes are provided, but no local policy forces exploration.
- **Secret rooms:** observed open secret and supersecret doors are supported.
  Searching for hidden entrances and bombing suspected walls are not implemented.
- **Ground effects:** known enemy creep variants are exported with estimated
  footprints and lifetime fields. Coverage is incomplete; flight, immunities and
  exact damaging areas are not fully modeled. Some surfaces affect movement.
- **Pills:** unidentified pills may be explicitly tried. The hidden effect is
  not queried before identification. Known pills, cards and active items still
  use limited eligibility rules; unusual effects may hit transition guards.
- **Trinkets:** collection is offered when the main trinket slot is empty.
  Deliberate swapping, dropping and additional-slot handling are not implemented.
- **Weapons and enemies:** ordinary tears are the main tested weapon. Lasers,
  enemy phases, synergies and special weapon behaviors are incomplete.
- **Room support:** not every special room or interaction is supported. State
  completeness checks may stop control when observations cannot be trusted.

## Validation and privacy

Python tests use fake model replies and synthetic or recorded game-state
fixtures. Movement simulations approximate game physics. Lua tests mock engine
APIs. These checks establish specific contracts and regressions, not consistent
live performance. The first Duke of Flies win and model-chosen floor transition
were separately verified in live logs on 21 September 2026; see
[the milestone](first-boss.md).

Only the Lua bridge and controller sources, tests and documentation are included.
Full gameplay reports, local setup notes and credentials are excluded. The
default launcher asks for a key in a masked local window. Game state is sent to
TypeSafe when Jev is making decisions; it is not an entirely offline player.

For changes, run `python -m unittest discover -s tests -q` and
`lua tests/test_mod.lua`. Keep the authority boundary explicit: improving local
execution must not silently replace Jev's choices with automatic strategy.
