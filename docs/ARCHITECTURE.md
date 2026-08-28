# Architecture & Design Methodology

**Read this before designing ANY change.** It exists because Joe
called the disease by name (2026-08-27): *"This is getting to be a
large project and we're continuing to just add layers on top of
layers until we have a skyscraper of shit."* Every rule below was paid
for with a real incident in this repo; the citations are the case law.

---

## 1. The shape of the system

Data flows ONE WAY through five layers. Each layer reads the one
below through a narrow, named seam — never around it.

```
CAPTURE        camera → ffmpeg → frames + capture timestamps
                 (recording chain is sacred: never disturbed by analysis)
MEASUREMENT    measure/core.py MeasurementCore — THE table truth:
                 detections → tracks → presence → ball events
PERSISTENCE    sidecar (analysis.jsonl), shots.json, recordings
                 (schemas are contracts; consumers never reach past them)
INTERPRETATION shot analysis, outcomes, stats, coaching, close pass
PRESENTATION   Qt UI, voice, schematic, phone app, overlays
```

- **MeasurementCore is the single source of table truth.** Any
  animation, analysis, or interpretation PULLS from it (Joe's
  directive, verbatim, 2026-08-27). Its API is the working draft of
  the C++ engine's contract (`docs/MEASUREMENT_CORE.md` §0.1).
- **Python proves, C++ inherits.** A rule ships to the rewrite only
  after it survives the corpus gates here. Nothing unproven ports.

## 2. The laws (each one bought with a failure)

**L1 — One opinion per fact.** If two components can answer the same
question, one of them is a bug you haven't seen yet. *Case: the ball
tray, the schematic, and the make-announcer each kept a private "what's
on the table" and Joe watched them disagree live (2026-08-27). New
derived state goes INTO MeasurementCore, never beside it — the
standalone TablePresence lasted two hours before it had to be folded
in.*

**L2 — Measure, don't patch.** Fix the measurement; never bandage its
symptom downstream. Anything that only works in post cannot support
during-play features, which is the product. *Case: the blur-compensator
stack (trail_resample, forensic_repass) is post-hoc scaffolding the
dense engine exists to delete; the 3/5 ball confusion gets identifier
training data, not announcement heuristics.*

**L3 — Subtract before you add.** Every new module, flag, or guard
must name what it replaces or why nothing existing can absorb it. A
change that only stacks is debt even when it works. When a fact must
thread through 3+ layers to reach its consumer, fix the seam instead
of threading. *Case: FramePacket accreting per-feature fields
(clock_*, present, feed_*) is the current worst offender — new
consumers should read the core, not grow the packet.*

**L4 — Gates, not hope.** Champions (models, trackers, engines) are
replaced only through the measured scorecard (`MEASUREMENT_CORE.md`
§M2), same-batch, on real footage, with a rehearsed rollback. Every
behavior change lands with the test that pins it. *Case: c6 looked
better and was worse on held-out sessions; the suite-red commits the
pre-push hook blocked.*

**L5 — Evidence before theory.** In any incident, run the cheap
discriminating check before proposing causes — and never blame the
environment while our own code is unmeasured. *Case: three days of
phone-outage theories (cellular, cache, Private Relay) while app.js
died on line 1 in every client; "RF interference" blamed for a mouse
our own 80%-GPU inference loop was starving.*

**L6 — Guards live with the resource they guard.** The component that
owns a resource defends it; callers are never trusted to remember.
*Case: the presence guard that only deferred STARTING heavy jobs — a
running job Joe walked in on kept stealing the GPU until the engine
itself learned to pause.*

**L7 — Contracts outlive implementations.** Sidecar row shape,
shots.json schema, pocket DB keys, the phone protocol: these are
frozen seams. Extend additively, never fork per-consumer variants.

## 3. The checklist (answer before writing code)

1. **Which layer does this belong to?** (If it computes table truth,
   it goes in MeasurementCore. If it renders, it reads and renders.)
2. **What opinion-holder does it create?** If any: fold it into the
   core or don't build it.
3. **What does it delete or replace?** "Nothing, it's purely additive"
   is an answer that requires justification, not a free pass.
4. **Which gate or test measures it?** Before/after, on real footage,
   pinned by a test.
5. **Does it survive during-play?** Post-only crutches serve no
   product goal.
6. **Root cause or symptom?** If symptom, say so explicitly in the
   commit and file the root cause in GOALS.

## 4. Where things go

| You are adding…            | It goes…                                    |
|----------------------------|---------------------------------------------|
| a derived table-state fact | MeasurementCore (`measure/core.py`)          |
| a tracking/identity rule   | `measure/tracker.py`, taught via an M2 gate  |
| a shot/outcome judgment    | Interpretation, reading core output          |
| a UI element               | Presentation, per `docs/DESIGN.md`, reading the core/packet |
| a persistence field        | additive schema change + reader-compat test  |
| a guard                    | inside the layer that owns the resource      |
| a heavy job                | presence/recording-guarded, BelowNormal, one at a time |

## 5. Demolition ledger (subtraction is scheduled work)

The skyscraper gets shorter on purpose, not by accident. The full
audited inventory of every private opinion-holder (file:line, what it
holds, its migration) is
`docs/design/state-opinion-map-2026-08-27.md` — ~40 sites including
three track sources, four independent timebase reconciliations, three
strike authorities, four private copies of shot history, and a phone
API that re-implements correction-merging. Standing demolition
targets, sequenced by the gates that prove them dead:
- Blur-compensator stack (trail_resample, forensic_repass) — dies when
  corpus gates prove dense tracks superseded it.
- Old live tracker in `vision/pipeline.py` — dies at M4 promotion;
  divergence counters in the core are accumulating its case now.
- Per-feature FramePacket fields — collapse into core reads as
  consumers migrate.
- Debt items with measured cost: `MEASUREMENT_CORE.md` §3.
- The full-scale demolition is the post-ladder C++ rebuild
  (`MEASUREMENT_CORE.md` §0) — but no NEW floors go up meanwhile.

---

*Maintenance: when a new rule is bought with a new failure, add it
here WITH its case citation, and delete any rule the architecture has
made impossible to break. This document loses authority the moment it
gets long enough to skim.*
