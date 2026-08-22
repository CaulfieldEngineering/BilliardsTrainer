# Sensor fusion: every method where it is strong

Joe: "Can we consider using multiple of these methods/tools in
conjunction?" Yes — and this is the documented endgame. The literature's
fixed-rig systems never pick one method; they compose them. This is the
standing design for how ours compose, so each session builds toward one
architecture instead of another patch.

## The principle

Each source is good in a REGIME and blind elsewhere. Today they are
chained as binary fallbacks (if the model fails, try subtraction; if the
gap is clean, lift the gate). The target is WEIGHTED FUSION: every source
always contributes observations tagged with its own uncertainty, and one
estimator — the trajectory fit — reconciles them.

| source                  | strong                          | blind                     | today            |
|-------------------------|---------------------------------|---------------------------|------------------|
| neural model            | sharp balls; identity/number    | motion blur (returns 0)   | primary          |
| background subtraction  | presence of ANY non-felt blob,  | identity; two touching    | bolt-on fallback |
|  (blur recovery)        | blurred or not                  | balls merge               | (gated, late)    |
| measured colour         | identity at rest + relative     | washed out at speed;      | veto + consensus |
|                         | contests between candidates     | absolute thresholds fail  |                  |
| straight-roll geometry  | validating a line across holes  | multi-segment paths       | gate-lifter      |
| rail-bounce signature   | naming the first cushion at the | mid-table events          | prototyped, not  |
|                         | slow, well-tracked end          |                           | shipped (phase-  |
|                         |                                 |                           | sensitive alone) |
| audio transient         | exact CONTACT TIME (~ms)        | everything spatial        | unused for this  |
| physics model           | interpolating between sparse    | needs anchor observations | next build       |
|  (trajectory fit)       | observations; uncertainty       |                           |                  |

## The composition

1. DETECTION runs two channels every frame, always:
   - subtraction channel: blobs vs a rolling median background — presence,
     position, crude extent. Cheap, never blind to a moving ball.
   - model channel: identity, class, sharp positions when available.
   Every observation carries (source, quality): model-sharp ~2px,
   subtraction-blob ~r/2, blur-recovered smear centroid ~r (flagged so
   geometry can discount what rendering happily draws).

2. TRACKING consumes both channels. Association order: identity evidence
   (colour contest, number) first, distance second. The accreted gates
   (occlusion budgets, strike gate, growing windows) are hand-rolled
   covariance — they stay until the estimator below replaces them, then
   become deletable one by one, each removal measured.

3. PER-SHOT ANALYSIS is one weighted piecewise fit, not sample heuristics:
   - segments: straight roll + rolling deceleration
   - breakpoints: cushion reflections (known geometry), ball contacts
   - anchored by: audio transient (WHEN contact happened), resting
     positions before/after (the best-measured points in the whole shot),
     rail-bounce signatures (WHERE a segment ended)
   - weights: each observation's per-source quality; smear centroids
     discounted, sharp detections dominate
   Every verdict — departure line, first rail, miss side, cut angle —
   reads off the fitted trajectory WITH the fit's own uncertainty. The
   confidence gates stop being hand-rules and become "the fit's sigma on
   this quantity crosses the labelling threshold".

   The straight-roll validation shipped 2026-08-23 is this fit's first
   segment, degenerate case (one segment, no reflections). It already
   flipped @233 from "can't tell" to a correct, trusted "missed left".

## Ordering (each step measured before the next)

1. Trajectory fit v1 — offline, per-shot, straight segments + reflections,
   validated against the 12-miss label set (scratchpad/labelset.json)
   before it replaces anything. Metric: sides correct on labels, trusted-
   tag count, nonsense-angle count (203 today).
2. Wire audio transients as the fit's time anchor (contact time to ~ms,
   already recorded in the sessions; currently unused for analysis).
3. Promote subtraction to an always-on second detection channel with
   per-source quality tags (kills the "recovery fires too late" class).
4. Only then: replace the tracker's hand gates with filter-style
   uncertainty, deleting rules one at a time, id_hops and duplicate-
   identity as the regression metrics.

## What this is NOT

Not a rewrite. Identity arbitration, the colour system, vacancy pruning,
shot segmentation, the sidecar contract — all keep their jobs. The fusion
replaces exactly two things: single-channel detection, and per-sample
verdict geometry. Everything else feeds it.
