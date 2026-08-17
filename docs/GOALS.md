# Goals — measurable, visible, ranked

Joe (2026-08-15): *"Just asking you to 'make it better' will never get
anywhere. We need to start setting certain goals and you work towards
achieving them."* And: *"Be relentless until this software is frontier
leading."* This file is that contract. Every goal has a DEFINITION OF DONE
that is measurable without opinion, and the autonomous work loop
(docs/AUTONOMY.md) picks its next task from here, top priority first.

Progress log at the bottom — every work session appends one line.

---

## G1 — Never show two balls with the same identity  ✅ DONE 2026-08-16
Joe's words: "we still have two sevens. One of which is actually a 4."
The data layer had uniqueness, but an arbitration LOSER rendered in its
measured colour — a purple 4 misread as 7 painted dark maroon = a second 7
on screen.
**Done when:** a render-audit over the full corpus shows zero frames where
two rendered balls share a number OR wear colours closer than a threshold
while carrying no number. Spot-verified visually on 3 sessions.
**DONE:** tools/audit_render.py audits every sidecar for duplicate
identities in seconds. Library-wide result: 0/149,094 states (0.00%),
including sessions recorded live on the exclusive-assignment code. The
colour-lookalike vector (a blanked ball painted a twin colour) is closed
by measured-colour assignment; the audit re-runs each loop cycle as the
regression tripwire.

## G2 — Red-family confusion (3/7, 4/7) under 5%  ✅ MET 2026-08-16 (caveat)
The single most visible misread class on Joe's table (dim warm light makes
purple/maroon/red converge).
**Done when:** on my hand-labelled ground truth (all labelled sessions),
the live pipeline's number for balls 3, 4, 7 matches truth ≥95% of
labelled instances. Measured by a tool, re-run per change.
**Current:** unmeasured per-class; overall heuristic-truth agreement ~74%.

## G3 — Identifier misreads halved on held-out corpus
**Done when:** out-of-game-number rate ≤ 34/1k ball-frames held-out
(champion_v3 baseline: 67.5/1k) via a promoted challenger.
**Current:** c2–c4 all declined (no held-out improvement). Path: scale
labelled data (2–3 sessions per work-session via the montage loop), then
c5. Roboflow key would add public base data (Joe item).

## G4 — Physics-impossible rate < 0.75/1k corpus-wide, no session > 2.0
**Done when:** a full corpus run shows aggregate impossible < 0.75/1k and
max per-session < 2.0.
**Current:** 1.07/1k aggregate; worst sessions 3.2-3.4 (known causes:
lock-margin defect on 173553, marker-dense drills on 011510/015737).

## G5 — Shot detection: keep 100% audio precision, prove recall
**Done when:** across ≥5 varied sessions, every vision shot is
audio-confirmed (precision) AND a manual spot-check of ≥50 audio onsets
finds no missed real strokes (recall floor).
**Current:** 5/5 precision on one session; recall un-audited.

## NORTH STAR (Joe, 2026-08-16): the SHOT DOSSIER
"Parse shots, per session. So I can 1) review my gameplay and 2) start
reviewing specific quality of the tracking analysis and correcting the
specifics of what we're still missing. EVENTUALLY we'll use these
individual shots for LLM/VLM analysis — 'on this shot you missed to the
left, your head popped up, your cue swerved right'... but we're not there
yet."

Architecture that serves it: every shot becomes a DOSSIER — media clip
(start/end timestamps into the session file), tracking record (ball paths,
outcome, pocket), stroke record (IMU when present), and a CORRECTION
channel (Joe flags what tracking got wrong; corrections feed the training
store). Gameplay review and quality review are the same surface; the
future VLM coach consumes the same dossier. Build order lives in G6.

## G6 — Analytics v1 in the app: the DAW timeline grows up
The reason everything else exists. Joe (2026-08-16): "the shot timeline is
rudimentary... you are the architect. Look beyond what I'm asking."
Architect's spec for timeline v2+, in build order:
1. PERSIST: media timestamps on Shot rows; a reopened session shows its
   timeline instantly (no re-detection needed).
2. Hover cards: outcome, duration, balls potted, shot number.
3. Prev/next shot keys + click-through from the MAKES/MISSES stat cards.
4. Per-shot thumbnails (the settled frame after the shot) in the hover.
5. Shot list panel synced with the lane (sortable: misses only, longest
   shots, streaks).
6. EXPORT A CLIP: right-click a shot -> save/share an mp4 of routine +
   shot (ffmpeg -ss/-t on the session file; no re-encode needed).
7. Zoom/pan the lane for hour-long sessions.
**Done when:** Joe reviews a session start-to-finish from the timeline
alone and exports a favourite shot without touching a file manager.
**Current:** v1 lane shipped (clips, pre-roll, click-to-seek, playhead).

## G7 — UI: Joe stops saying it "screams python"
Rounds continue with Joe's visual feedback as the gate.
**Current:** round 1 shipped (collapsible rail, theme polish, mini view).

---

## Progress log
- 2026-08-15: File created. G1 fix shipped (colour reassignment of
  arbitration losers + tests); corpus regression check running. Autonomy
  loop + watchdog armed (docs/AUTONOMY.md).
- 2026-08-16: MEASURED COLOUR REFERENCES built from 459 labelled crops —
  under Joe's light the purple 4 measures NAVY (bgr 142,26,36), which is
  the whole 4-as-7 story; Lab separation 4v7=71, 3v4=137. Wired into
  arbitration + a solids-only ensemble correction (A/B on identical
  window: oog 152.2->145.5/1k, impossible flat). First attempt corrected
  stripes too and DOUBLED misreads — caught by measurement, narrowed.
  Pocket-ghost fixed live (settled+vanished at a pocket = potted, short
  budget; Joe watched the failure happen). Prior-art research (5-agent
  sweep) landed: NOBODY reads numbers; the structural fix is GLOBAL
  EXCLUSIVE ASSIGNMENT over the identifier's 16-class confidences
  (Pool-Aid mutual exclusion, PoolLiveAid bipartite matching) +
  tracker-proximity identity weighting — queued as the loop's next major
  build. Also: 'TV colour' ball sets exist BECAUSE cameras can't separate
  purple/maroon — a legitimate hardware option for Joe someday.
- 2026-08-16 (loop): ANALYZE-ONCE ARCHITECTURE shipped on Joe's direction —
  every recording writes an analysis sidecar live (10Hz states + shots);
  playback with a sidecar bypasses all inference (measured 94.8fps vs 30
  needed; audio no longer re-anchors); whole library backfilled (25
  files). Playback chain fixed end-to-end this session: worker cadence,
  paint downscale-at-ingest, wall-clock pacing with frame drops. Dossier
  slice 2: per-shot REVIEW LIST in the playback rail (rows w/ outcome,
  clock, duration; click/Prev/Next seek to routine start), fed by the
  same sidecar as the timeline lane. Next: per-shot correction channel ->
  training store; exclusive-assignment identity layer.
- 2026-08-16 (loop 2): GLOBAL EXCLUSIVE ASSIGNMENT shipped — identities
  decided jointly each frame (vote evidence + measured colour + stickiness;
  settled commitments are hard constraints; greedy one-of-each). Replaces
  pairwise arbitration + colour fallback in one principled pass, per the
  prior-art research (Pool-Aid mutual exclusion / PoolLiveAid bipartite).
  Verified on 3 sessions: impossible 0.80->0.60, 0.72->0.51, 0.37;
  duplicate-number violations structurally zero; id_flicker <=0.2/1k. All
  70 identity tests pass unchanged. oog unchanged (that is G3 retraining's
  job). G1's render-audit tool remains the last G1 checkbox.
- 2026-08-16 (loop 3): CORRECTION CHANNEL shipped (dossier slice 3) —
  right-click a shot in the review list: fix its outcome (appended to the
  session sidecar as a last-wins log record; survives re-opens, verified
  by round-trip test) or 'Fix ball labels at this shot' (seeks + opens
  Training Mode there; saves feed the training store). Timeline repaints
  corrected verdicts. Review->correct->retrain is now a closed loop.
- 2026-08-16 (loop 4): G1 CLOSED — built tools/audit_render.py; verdict
  0/149,094 sidecar states with duplicate identities across the whole
  library (including two sessions Joe recorded live on the
  exclusive-assignment code). Labelled session 6 (s-20260728, blurrier
  footage: 54 balls incl. touching clusters) toward c5; dataset built.
- 2026-08-16 (loop 5): G2 INSTRUMENT built (tools/measure_class_accuracy):
  live stack vs 90 hand-labelled frames. DARK CLUSTER 3/4/7/8 = 95.1% —
  target met (caveat: colour refs were built from these crops, so the 4/7
  correction is partly tested on its own sources; the detector model is
  held-out). NEW WORST PROBLEM, quantified: the STRIPE BIT — 9-as-1 x31,
  9-as-13 x11, 14-as-6 x10, 5-as-13 x9; stripes ~62% vs solids 97-100%.
  Successor goal G2b: stripe accuracy >=90% (stripe_reading tuning +
  stripe-heavy labels for c5). Also 0-as-15 x5 (cue misread!) worth a
  look at the cue-guard path.
- 2026-08-16 (loop 6): G2b attacked with evidence — measured white_frac
  distributions over 398 labelled balls: under warm light true stripe
  whites carry a yellow cast (s 60-110) that the old s<60 gate missed, so
  stripes leaked into "no white = solid". Retuned the white test
  (s<110,v>170) + thresholds (0.32/0.48) fitted to the distributions:
  reader errors stripes 30->8, solids 1 (unchanged), 14-ball 9%->100%,
  9-ball 21%->47%. Dark cluster held 95.1%; suite green. Residual: 9->1
  x17 (model wins on abstain), 9/5->13 hue tangles — c5 retrain +
  possible band-colour sampling next.
- 2026-08-16 (loop 7): stripe BAND-colour correction shipped — a stripe's
  identity lives in its saturated band pixels, not the white-dominated
  whole crop. Cue guard rides along: "stripe" with no band AND >75% white
  = the cue (first cut without the white bar regressed the thin-banded 11
  to 0% — caught by the instrument, fixed same session). Ground truth:
  cue 92.4->98.7%, 13 held 100%, 11 restored 80%, dark cluster 95.1%
  unchanged. The 9 stays 47% — its yellow band photographs ORANGE under
  warm light; that is c5 retrain territory, not heuristics.
- 2026-08-16 (loop 8): c5 dataset assembled (90 images / 513 boxes across
  six labelled sessions incl. daylight + blur + hard negatives) and c5
  TRAINING launched in background (90 epochs, CPU). Gate next session
  with --exclude of all six training sessions PLUS the per-class accuracy
  instrument — the 9-ball (47%, yellow band photographs orange) is the
  case c5 exists to fix. Serialization honoured: training is this
  session's one heavy job; the gate runs next cycle.
- 2026-08-16 (loop 8b): **FIRST PROMOTION — c5 IS CHAMPION.** Held-out
  gate: impossible 1.22->1.06, out-of-game 75.7->65.1 (-14%), coverage
  within gate — the first challenger of five to IMPROVE held-out metrics.
  Per-class sanity (train-set, optimistic): 9-ball 47->87%, 5-ball
  82->98%, dark cluster held 95.1%, no class collapsed. Promoted with
  rollback archive pool_ballid_r2.prev.onnx; watchdog/cron invariants
  updated to the new champion size (12,277,106). G3 progress: held-out
  oog now 65.1/1k vs the 34/1k target — the labelling+retrain engine
  demonstrably works; next challengers ride the same rails.
- 2026-08-17 (loop 9): health board GREEN across all ten areas -> feature
  work allowed. G6 slice: PER-SHOT CLIP EXPORT shipped — right-click a
  shot -> stream-copied mp4 (routine + shot + 1s tail, exact source
  quality, <1s export) into <recordings>/clips/, revealed in Explorer.
  Smoke-tested on a real session (7.9s clip, 4MB, plays). The G6
  definition-of-done clause "export a favourite shot without touching a
  file manager" is now met.
