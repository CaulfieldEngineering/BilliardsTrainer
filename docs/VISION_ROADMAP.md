# Vision & Analytics Roadmap

Joe's mandate (2026-08-12, verbatim intent): *"I do not want to be the
technical director of this product, I just want to use it, be amazed by it,
and for it to work well. Take the reins. Self-iterating, self-evolving,
without derailing what works."* The vision/AI/analysis side may be tweaked,
blasted, or overwritten freely. The recording pipeline works — do not derail
it. Reference points he named: DrillRoom (drills, per-attempt stats,
success-rate heatmaps, percentile trends), Railbird (session recording →
per-shot analysis, make/miss, sharing).

## The operating principle

**No vision change ships without a before/after score on real footage.**
The scorer is `billiards_trainer.eval.invariants` — physical laws of pool
checked label-free — run by `tools/score_session.py` (one clip) and
`tools/score_corpus.py` (every session, unattended, resumable). A change
that raises impossible-violation rates or drops coverage does not ship.
This exists because the stack was unmeasurable for years and quality
random-walked; that is over.

Key data facts learned from footage (do not relearn these):
- Joe's felt carries **drill position markers** (donut stickers, chalk dots).
  The model detects them as balls; the size prior (`model_size_lo/hi`, band
  [0.72, 1.55] × geometric radius) is what kills them. Markers ≤0.66×,
  real balls ≥0.89× (measured session-20260812).
- Ball ≈ 30 px diameter at native 936×1640; printed numbers are ~8-10 px —
  too small to read reliably. Identity must come from **colour + solid/stripe**
  (the colour code: 1 yellow, 2 blue, 3 red, 4 purple, 5 orange, 6 green,
  7 maroon, 8 black; 9-15 repeat as stripes). Specular glare from the
  overhead light sits on every ball and fools naive stripe detection.
- `out_of_game_number` (stripes 10-15 claimed in 9-ball) is the identifier
  model misreading — fixing it needs retraining, not tracker heuristics.

## Phases

### Phase 1 — Measurement (DONE, keep green)
- [x] Invariant scorer + degenerate-detector guards (`eval/invariants.py`)
- [x] Session scorer, corpus runner, violation-frame dumper
- [x] Baseline frozen: `_eval/corpus/post_marker_fix/`
- [x] First fixes proven: marker phantoms, cue uniqueness, settled-identity lock

### Phase 2 — Detection & ID rebuilt against the metric (NEXT)
The identifier misreads stripes; the finder misses jaw balls and fires on
glare. The loop that fixes both without hand-labelling:
1. Mine hard examples automatically: every invariant violation names a frame
   and a place — crop it. (`vanished_mid_table` = missed detection;
   `out_of_game_number` = misread; `duplicate_number` = one of the two crops
   is wrong.)
2. Auto-label with the VLM path that exists (`train/autolabel.py`, needs the
   backend configured) + colour-code priors as a cross-check. Discard
   disagreements — never train on a guess.
3. Fine-tune the finder (`pool_yolo11.onnx`) and identifier
   (`pool_ballid_r2.onnx`) on Joe's-table crops; export ONNX.
4. Score corpus before/after; ship only on improvement. Keep the old model
   file — rollback is a file copy.
Add a `marker` class so drill stickers become a *feature* (drill zones)
instead of a rejected nuisance.

### Phase 3 — Event layer (the keystone)
Everything a user loves in Railbird/DrillRoom sits on one primitive: the
**shot**. Segment sessions into shots; detect pots.
- Vision: settled → cue accelerates → balls scatter → settle = one shot.
  A ball vanishing near a pocket during the shot = pot (already have
  pocket geometry + vanish detection in the scorer).
- The BLE cue sensor (JINOU JO-BEC12-2, `cue/`) provides the hardware
  impact timestamp — a ground-truth "shot happened NOW" trigger plus stroke
  mechanics (backswing, pause, acceleration, straightness). Camera-only
  competitors cannot do this; it is the differentiator. SAFETY: never write
  to the device (B3A2 absent by design — a write bricked unit #1).
- Persist shots to the DB layer (`db/`): time span, balls moved, pots,
  cue-ball start/end, stroke metrics when the sensor is on.

### Phase 4 — Analytics on top of shots
- Per-session: shots, pots, success %, session timeline you can scrub.
- Heatmaps: make % by cue-ball position / object-ball position (DrillRoom's
  most-loved view).
- Trends across sessions; streaks; time-of-day.
- Stroke ↔ outcome correlation (sensor): "your misses have 2.3× the lateral
  jerk of your makes" — the thing Joe explicitly asked to see.

### Phase 5 — UI worthy of it
Joe: current UI "screams python". Design system pass: dark, spacious,
typographic hierarchy, animated transitions, live-updating cards. The
schematic view is already close; the chrome around it is not. Do not touch
the recording controls' behaviour (hard-won stability).

### Phase 6 — The self-iterating loop (runs unattended)
Nightly (Task Scheduler or on-demand):
1. `score_corpus.py` over new sessions → append to trend log
2. Mine violations → crops → auto-label → grow the training set
3. Retrain when the set has grown enough; score challenger vs champion on
   the frozen corpus; promote only on strict improvement; keep rollback.
4. Write a one-paragraph plain-English report Joe can read with coffee.
Guard-rails: promotion gate is the corpus score (impossible-rate must not
rise, coverage must not fall); champions are files, rollback is trivial;
the recording path is never touched by this loop.

## Session log
- 2026-08-12: Phase 1 built. Baseline: coverage 10.92 vs 6 real balls,
  stability 32.1%, impossible 4.06/1k. After marker/cue/identity fixes:
  9.49 / 41.0% / 1.14 (session-20260812); 14.54 → 0.69/1k impossible on
  session-20260808. Corpus sweep running: `_eval/corpus/post_marker_fix/`.
- 2026-08-12 (later): Full-corpus champion baseline frozen:
  `_eval/corpus/post_marker_fix/aggregate.json` — 18 sessions, 197,477
  ball-frames, **4.86 impossible/1k**, 0 failed, 0 degenerate. Spread is
  30×: best sessions 0.44-0.78/1k; three outliers 13-22/1k dominated by
  `overlapping_balls` (93-139 each — double-detections, NOT markers). Both
  daytime sessions (2026-07-24) are outliers → sunlight implicated.
  session-20260723 shows 3,248 `out_of_game_number` (stripe misreads) yet
  only 0.44/1k impossible — confirms identifier vs detector are separate
  failure axes. Diagnosis workflow dispatched (outlier frame forensics +
  toolchain audit for Phase 2).
- 2026-08-12 (evening): Outliers diagnosed by parallel frame forensics and
  fixed (573cc68): pocketed balls in the basket (class-gated void filter),
  tile-seam truncated boxes, shadow-fused touching pairs (fixed by
  constraint projection to touching). Re-scored: 22.33→0.43, 15.28→1.59,
  13.17→0.70 impossible/1k. Corpus re-baseline running
  (`_eval/corpus/post_overlap_fix/`). Phase 2 bootstrap: `.trainvenv`
  install running; Claude-as-VLM labelling path VALIDATED end to end on
  session-20260812 — 12 layouts, 59 balls labelled, dataset built
  (`_train/autolabel/dataset`). First identity metric: live heuristic
  agrees with vision labels on 50/59 (84.7%). Learned: montage stride 12
  collapses layouts to single frames (propagation +0) — next labelling
  round needs denser stride for a real training set; finder systematically
  MISSES the purple 4 and green 6 (they appear only as un-boxed slivers) —
  finder recall is a separate axis from identifier accuracy; the felt
  markers are where Joe spots balls for drills, so marker and ball
  routinely share coordinates. NO projector exists — white lines on the
  felt are physical tape/drawn guides.
- 2026-08-12 (night): The loop worked exactly as designed — the corpus
  caught MY regression before it shipped (aggregate improved 4.86 → 3.55
  impossible/1k, but one session went 7.68 → 22.60). Root causes fixed
  (3a5506e): unconditional rigid-body repair laundered duplicate detections
  into legal pairs → repair is now identity-gated (distinct numbers = push
  apart; same/unknown = merge); the settled-lock froze track pairs at
  impossible positions → the published tracks get the same projection; and
  NEITHER diameter ruler is trustworthy alone (boxes run ~25% large on some
  sessions, geometry inherits table-lock error ~25% the other way on
  session-20260802-173553, whose lock includes rail margin — KNOWN ISSUE,
  felt-detection defect) → scorer now uses box-median clamped to ±20% of
  geometric, and score_session judges with the pipeline's own calibration.
  Verified: 33.11 → 0.54/1k on the regressed window, controls unchanged.
  Phase 2 assets: challenger c2 trained (35 imgs/152 balls, all 16 classes,
  smoke-tested); champion/challenger gate tool ready; champion_v3 corpus
  baseline running. Gate c2 when it lands: expect c2 to LOSE (thin data) —
  the point is the loop; c3 gets denser-stride labels + more sessions.
- 2026-08-13/14: **champion_v3 frozen: 1.07 impossible/1k** (was 4.86 at
  first measurement — 4.5× in one day), every session ≤3.44, 0 failed.
  Challenger c2 gated: formal gates said PROMOTABLE but held-out analysis
  (it trained on 2 of 18 corpus sessions) showed a tie — DECLINED, and the
  gate now takes --exclude so train-on-test can't flatter a challenger
  again. Labelling tool now STREAMS video (FrameStore proxies + on-demand
  seek), unlocking the big daylight sessions the identifier is worst on;
  first daylight session labelled (85 balls, heuristic truth 82.4%).
  Challenger c3 training on the 3-session merge (night 9-ball + night
  15-ball + daylight). Gate c3 with:
  `--exclude session-20260812-221333,session-20260723-215012,session-20260724-133950`.
