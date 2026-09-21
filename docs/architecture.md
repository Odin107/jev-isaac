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

Floor memory explicitly tracks item rooms as not observed, found but unvisited,
or visited. Observed locked entrances count as found, without becoming traversable
edges. Large-room aliases are kept together; multiple item rooms remain separate.
The accompanying rules explain their value as sources of run-long upgrades and
their usual key cost, without forcing an exploration order or claiming every
floor contains one. Visiting a room does not prove its item was taken.

Pickups left in directly observed rooms are remembered by room, type/subtype,
count and shop price; heart types have readable labels. These are last-seen
facts, not live offscreen observations. A complete new room observation replaces
its snapshot, removing collected/disappeared supplies and empty pedestals.
Missing pickup data or an imported visited-room summary cannot erase or invent
supplies. Both memories survive authenticated same-floor rearming and reset for
a new floor/run. They do not create routes, collection or spending authority.

`player_policy.py` builds typed choices for combat and room activities.
`player_navigation.py` executes selected activities through the movement and
interaction helpers. `controller.py` coordinates fresh observations, asynchronous
model replies, pauses, budgets and reports. `mod/jev_bridge/main.lua` exports the
state and applies bounded controls after F8 activation.

Player-mode combat uses one source-bound choice for movement, enemy target and
exact firing input together. `combat_choices.py` offers all five button states
for engage/back-off for each target, plus hold/evade. Up to 24 targets fit a single
250-option Choice. Larger observations retain all 64 supported targets through
at most three groups: Jev selects a group and a speculative complete action for
each group in the same HTTP request; only the selected group's action executes.
No cross-group probability multiplication or local target selection is used.
Group ordering has no strategic meaning. Abilities remain a parallel choice;
clear-room activity and fire choices are unchanged. Additional options cost
tokens, and actual latency/survival effects still need live measurement. Request
cadence, freshness limits and budgets are unchanged.

Aim hints describe the current position; potential future firing positions are labeled separately.
Jev is told that movement momentum can bend tears across the firing axis.
Actual tear-velocity inheritance has not been calibrated.

Each player-mode enemy target offers both `engage` and `back_off`. Backing away
seeks a farther cardinal firing position for that same enemy, paired with the
firing button Jev selected in that action. The executor checks a direct retreat
against room bounds and obstacles; it does not take an inward detour or pursue
a replacement target. It uses the existing conservative 220-unit shooting
envelope, tightened by a shorter observed ordinary-tear range with a 20-unit
margin. Special-weapon trajectories remain unmodeled. If a farther position is
blocked or beyond that envelope, intended movement holds; immediate collision
avoidance and obstacle-margin recovery retain their usual overrides. These
geometry checks do not promise a hit. Fresh goals, expiry and changed-target
checks apply just as for engaging; the legacy goal-only policy is unchanged.

`combat_feedback.py` retains at most 61 sanitized observations over 60 simulation
frames (two seconds). It reports adjacent comparable HP changes for stable enemy
IDs, separate red/soul half-heart changes, sampled input streaks, displacement
and a lower bound on path distance. Missing/duplicate IDs or unknown values break
comparisons. Disappearance does not imply a kill, health changes do not identify
their cause, and firing input does not establish a shot or a hit. Sample counts,
gaps, truncation and omitted enemy counts remain explicit. A pause, disarm,
death, new room/run/floor/session, peer change or control reset clears history.
The next request receives this feedback alongside the latest issued command,
observed input echo, previous goal/fire and recent measured reply latency.
Feedback never selects actions or changes inputs.

Combat firing also receives a compact `firing_now` view for each button: enemies
on that side, enemies aligned with the current straight firing lane, and aligned
enemies without observed solid-grid blockers. It summarizes current geometry,
does not rank directions, and does not use previous inputs or future waypoints.
Incomplete grid clearance is unknown, not an empty list of obstacles. These
ordinary-tear estimates do not guarantee hits with momentum, range limits,
moving enemies or special weapons. Previous controls are explicitly history.
Bounded decision reports retain the validated joint-action probability
distribution (and group choice when used), outcome feedback at dispatch, and
current geometry at both request and reply time. Joint probabilities are not
reported as independent firing probabilities. Clear-room fire-only judgments
keep their original distribution. These diagnostics do not override aim.

The local loop runs much more frequently than model decisions. Immediate dodges
can override movement and are logged. Local execution preserves the selected
cardinal firing button. A blocked selected activity returns to Jev rather than
automatically choosing a different room or pickup.

Open-door candidates include the destination's last complete pickup snapshot,
including an explicit `none_observed` status. Imported visits without a pickup
observation remain `unknown`; a remaining or swapped item stays `present`.
Item-room summaries expose the same distinction. Candidate descriptions and
choice criteria include these facts, so treasure-room type does not stand in for
an unclaimed reward. Remembered permitted exits can show a room whose only known
route returns to the current room, without claiming there are no hidden exits.
The model also receives up to twelve confirmed room crossings from its bounded
activity history and per-door completed-entry counts. These are observations,
not revisit bans or a preferred route; every otherwise supported door remains
selectable. Same-floor recovery retains memory; a new floor starts fresh.

Player mode pauses new room choices for 18 simulation frames after combat clears
and 12 frames after observed pickup identities or readiness change. Pickup
animation/readiness can extend the pause for up to 90 simulation frames per
continuous unready period; the limit avoids waiting forever for an unavailable
pickup. A changed reward list invalidates cached choices immediately and versions
the rebuilt candidates, so an earlier door reply cannot become valid again just
because that door still exists. An uncommitted departure is reconsidered if new
rewards appear. Already committed explosive retreats, curse-door crossings and
floor transitions finish their existing contracts. The timing window does not
prove that every future drop has appeared; Jev sees the refreshed observed
choices and remains free to leave rewards behind. Pickup motion and a still
positive countdown do not continually restart the settling pause.

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

The HTTP connection is refreshed before a new request after more than five
seconds idle, including pauses and floor transitions. Failed requests are never
replayed. With `--stay-ready`, connection failures/timeouts release control and
keep the listener and in-memory key open for a fresh same-run/floor F8 session.
The next decision uses fresh state; waiting and recovery retain the original
deadline and request cap. This does not automatically retry HTTP billing/auth
errors or unrelated programming failures.

In player mode, observed ordinary and tinted rocks can also be offered as
standalone `bomb_rock` activities in cleared rooms. A visible reward behind the
rock is no longer required. Jev receives the bomb cost and general purpose;
unexposed drops remain unknown. The executor checks ordinary-bomb inventory,
the exact rock, a placement route and a straight retreat, then places one bomb
and waits outside its range for observed destruction. New drops or openings
require another Jev decision. The existing bomb-to-pickup plan remains available.
Its description explicitly says that collection is a separate choice after the
blast. Once Jev chooses a pickup, collection tracks its fresh observed position:
bomb knockback or bouncing no longer invalidates an unchanged pickup identity.
Current cost, eligibility, option group, room identity and reachability are still
checked; a changed or unavailable pickup never authorizes a replacement.
Walls, pits, special rocks, modified bombs and explosive chains are outside this
bounded planner; absent offers do not imply the game mechanic is impossible.
The bridge converts numeric bomb flags and representable `BitSet128` low/high
halves into JSON integers before export. Passing the userdata directly to the
game's JSON encoder omitted the field and made every ordinary-bomb check fail.
Unknown, unreadable or unrepresentable flags remain unavailable rather than
being assumed zero. A missing field still disables the ordinary-bomb planner.

After combat, a stationary ordinary/red fire can leave Isaac within its extra
navigation buffer. A bounded outward step may leave that buffer while retaining
the player radius plus two units around the observed fire body. Normal routes
retain the larger buffer; contact overlap, inward momentum, unobserved/special
fire behavior and all other obstacle exclusions still block this recovery.

The bridge distinguishes a confirmed item-use animation from an ordinary pause.
Isaac's [`Game:IsPaused()`](https://wofsauge.github.io/IsaacDocs/rep/Game.html#ispaused)
also covers full-screen item animations. A matching item/card/pill use callback
after an emitted Jev input permits one animation in the same room, beginning
within two simulation ticks and 0.55 seconds, and ending within three seconds of
use. Movement and firing are released throughout. Resume requires observations
from after the frozen frame; old commands and item pulses cannot replay. The
controller keeps the same attempt and budgets. Escape, P, controller pause, F8,
death, unexpected room changes and the deadline still revoke control. A use
callback alone cannot authorize a pause without the corresponding Jev pulse.
`last_item_use` and `item_animation` provide observed diagnostics. Teleporting or
unusually long item effects may still require manual rearming.

Geometry supports 512 total hazards, with a separate 160-entity hazard cap and
the existing 60,000-byte datagram limit. Large-room grids no longer share the old
160-cell cutoff. Geometry validation, combat hints, navigation and room-objective
detection use the same total cap. Truncated state remains unusable for decisions.
With `--stay-ready`, incomplete observations release control and preserve the
listener/key until the original deadline; only a fresh F8 session can resume.
Updated bridges display "Incomplete room data" instead of a route failure.

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
