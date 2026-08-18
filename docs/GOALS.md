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
**Current:** ✅ **DONE** (confirmed 2026-08-17, corpus run
`_eval/corpus/handfilter_c5`): aggregate **0.50/1k** over 22 sessions /
203k ball-frames (target <0.75; was 1.05 at the champion_v4 baseline),
**worst session 1.41** (target <2.0; was 3.90), 0 failed, 0 degenerate.
Delivered by rest-frozen identity + the foreign-hand ingestion filter.
The bar stays enforced: every future challenger gates against this run.

## G5 — Shot detection: keep 100% audio precision, prove recall
**Done when:** across ≥5 varied sessions, every vision shot is
audio-confirmed (precision) AND a manual spot-check of ≥50 audio onsets
finds no missed real strokes (recall floor).
**Current:** FIVE-session audit complete (the letter's count):
283 detected shots, 3,031 onsets audited (every one, not a spot-check).
Precision: 279/283 audio-confirmed (98.6%; the 4 silent shots are
frame-dumped for review). Recall: 25 candidate misses total (~0.8% of
onsets; eyeballing shows many are paired strike+impact clacks and
hand-sweeps v2 could not fully rule out) — the goal's own 50-onset
spot-check procedure would find ~0.4 expected candidates, i.e. passes.
One session (221333) audited PERFECT: 0 missed, 0 unheard. Strict
letter ("every shot confirmed" / zero misses) not claimed: the 4+25
residue is named, dumped, and small. Recall rebuilt ~40% -> ~95%.
Previously: deep-audited on 3 sessions with hand-context sidecars
(v2): 212 detected shots, 21 remaining missed-stroke candidates (~2-10
real strokes; several are paired strike+impact clacks), 1 clean silent
shot, 16 hand-involved silent (gathering suspects). Recall rebuilt
~40% -> ~95% via ball-motion arming, backdated starts, time floors,
vanished-flyer credit. Remaining for the letter of DONE: extend to 5
sessions (2 more v2 backfills queued overnight).

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
- 2026-08-17 (loop 10): board GREEN -> features. Timeline HOVER CARDS
  shipped (pure hover_text + thin event shell). champion_v4 corpus
  baseline measured on c5: 1.05 impossible/1k over 22 sessions / 229k
  ball-frames, 0 failed. G4 verdict: NOT MET yet (target <0.75 aggregate,
  no session >2.0; three sessions above 2.0 — the marker-dense drill
  sessions 015737 at 3.90 and 210621 at 2.07 lead). Honest gap named;
  next lever is the lock-margin defect on 173553 + marker-session
  handling. New gates now compare against champion_v4_c5.
- 2026-08-17 (loop 11): board GREEN -> G4. REST-FROZEN IDENTITY: a ball
  not in demonstrable motion can never change number or class. Motion is
  judged from the published step streak (3 consecutive steps above a bar
  that sits above the scorer's rest threshold, floored at the table ball
  radius), so glare blips, settled-bit resets, post-revival smoothing
  glides and creeping drifts all stay frozen. First commit now waits for
  3 agreeing reads (no more rack-time commit-then-correct); stale votes
  drop at motion start so the deferred re-vote is decided by post-strike
  reads; a contested resting loser that already PUBLISHED its number
  strips to unknown instead of being renamed (never-published losers keep
  the colour-rename "two sevens" fix); never-numbered vanished tracks no
  longer ride the long occlusion budget. Took 3 measure-fix rounds (round
  1 regressed the control — median-based rest missed the glide window;
  honest numbers kept it unshipped). Final: 015737 3.90 -> 1.48, 210621
  2.07 -> 1.09, control 011510 0.32 -> 0.24 impossible/1k; id_flicker 12
  -> 0 across all three; out-of-game misreads 249/592/2562 -> 0. Suite
  green (255). Residuals named: teleports (revival oscillation), ghost
  merge-bar vs scorer-bar sliver (merge uses 1.6*exp_r, scorer uses
  published radii), a few class flickers. NEXT: full-corpus re-baseline
  to re-measure the G4 aggregate, then 173553 lock margin + teleports.
- 2026-08-17 (loop 12): board GREEN. FULL-CORPUS RE-BASELINE on the
  rest-frozen tracker: aggregate 0.59 impossible/1k over 22 sessions /
  217k ball-frames, 0 failed — G4's <0.75 AGGREGATE TARGET IS MET (was
  1.05). Per-session bar still blocked by ONE session: 173553 at 2.58
  (felt-lock rail margin; next worst 1.41, control 011510 scored 0.00).
  While the corpus ran, two G6 slices shipped: timeline ZOOM/PAN (wheel
  zooms around cursor, middle-drag pans, min span 10s, 8 pure-math
  tests) and shot-list OUTCOME DOTS (timeline colours as row icons;
  corrected verdicts ringed + tooltip). Suite 265 green. NEXT: 173553
  felt-lock rail margin (the only G4 blocker), then teleport/revival
  oscillation + ghost merge-bar sliver.
- 2026-08-17 (loop 13): board GREEN -> the 173553 blocker. LOOKED at the
  frames instead of trusting the old theory: the "rail margin" diagnosis
  was WRONG — the locked quad is fine (nose inset applied, verified
  against the video). The violations were Joe's GLOVED BRIDGE HAND
  resting on the cushion for rail shots: knuckles detected as balls, one
  misread #4, two ghosts. Fix: ingestion drops detections whose centre
  lies inside a kept foreign (hand-scale) blob — the foreign mask's size
  floor already excludes lone balls, so an isolated 8-ball is safe, and
  a hand-covered ball correctly coasts occluded. Measured: 173553 2.58
  -> 0.27 (overlaps 25 -> 1); control identical to pre-change (0.24).
  Suite 267 green. Full-corpus confirmation LAUNCHED; aggregate + no-
  session->2.0 verdict lands next session. If clean, G4 is DONE.
- 2026-08-17 (loop 14): board GREEN -> G5. Built the RECALL AUDIT
  (tools/audit_shot_recall.py): every audio onset in 10 sessions
  cross-examined against sidecar track motion — quiet-before +
  moving-after with no claiming shot = missed-stroke candidate, each one
  frame-dumped. HONEST RESULT: 163 detected shots vs 721 candidates, one
  74-min drill session with ZERO detections; eyeballed frames confirmed
  real strokes (a break; a full follow-through) among the misses. Traced
  the mechanism on 011928 @44.5s: one fast ball's frame-diff energy
  hovers at the 0.4 threshold — strike died at run 5/6, then armed 1.2s
  late. FIXED: per-ball banking decoupled from table motion; a struck
  ball (>=3 free frames, >=2.5 ball radii banked travel) arms directly;
  start backdated to the strike; settle now also requires tracked balls
  STILL (Joe's spec verbatim). Verified: all 4 unclaimed onsets in the
  traced window now detect (one a MAKE w/ pot); 3 new tests; suite
  green; app restarted. NEXT: re-backfill sidecars with the new
  detector + re-run the audit for the G5 verdict (watch precision too).
- 2026-08-18 (loop 21b, overnight): c6 trained + GATED, and DECLINED —
  the mechanical gate said PROMOTABLE (impossible 0.60->0.58, coverage
  flat), but the mission metric REGRESSED: held-out out-of-game 26.17 ->
  28.60/1k on the c6 config, 41.44 -> 43.49 on the canonical 6-session
  config. A model whose one job is lowering oog does not ship by raising
  it. Also corrected a would-be false victory: the "26/1k, under the
  <=34 target!" reading was a holdout-slice artifact (excluding the two
  newly-labelled oog-heavy sessions flattered the number) — on the
  canonical config c5 stands at 41.4 vs the <=34 target, so G3 remains
  OPEN. 86 frames was not enough signal. Path: targeted full-rack
  labelling for the starved classes (10: 2, 12: 1, 14: 11 labels) and a
  bigger increment before c7. Champion restored, fingerprint verified.
- 2026-08-18 (loop 21): board GREEN -> G3 (the last open metric goal).
  MONTAGE LABELLING, me as the vision model: read 33 crop sheets across
  two sessions (011510 game + 011928 games), labelled 161 crops
  conservatively (-1 when unsure; markers, rail diamonds, and a finger
  correctly refused). Increments lift the starved classes: 4 -> 17,
  5 -> 33 (first orange-5 labels), 13 -> 21, 15 -> 17, two fresh 8s;
  10/12/14 remain starved (2/1/11) — need a targeted full-rack session.
  Found and dodged a build-tool footgun: increments write to the same
  layout-indexed filenames, so batch 3 silently OVERWROTE batch 1 in the
  shared dataset dir — rebuilt each into its own dir and assembled
  _train/c6/dataset (86 frames) with source-prefixed names. c6 finetune
  LAUNCHED (60 epochs, CPU venv, background). NEXT session: held-out
  gate vs champion c5 (compare against handfilter_c5 corpus baselines);
  promote only on measured improvement, decline counts as progress.
- 2026-08-18 (loops 19-20 + a hard night shift): VANISHED-FLYER CREDIT
  shipped after discard-logging pinned the truth — the "missed" strokes
  were clean pots whose whole flight was motion blur (travel gate saw a
  34px cue nudge; the ball left the table unseen). A free-moving ball
  that vanishes now carries the shot. V7 verification: 011510 13 -> 14
  shots (candidates 6 -> 4), 210621 87 -> 90, 011928 105 -> 108. G5 arc
  to date: 163 shots/721 candidates/39 unheard -> 212/21/1-clean.
  BETWEEN loops, Joe live-reported three regressions and the night went
  to triage: the mouse hog was OUR backfill (killed; presence rules now
  in AUTONOMY); the REC clock clipped a third time (now GROW-ONLY — no
  fixed width exists to clip against); and the empty schematic unwound
  into THREE stacked causes — a dead detect worker nothing restarted
  (now self-heals), a coordinate-space confusion that made me delete a
  CORRECT saved lock (live frames vs letterbox-cropped recordings — now
  documented), and finally the TABLE COVER poisoning restore validation
  (felt-detect found wrinkle-quads; validate_against was the one path
  missing the cloth-saturation guard — fixed, with ground-truth frame
  dumps + a per-link vision heartbeat in the log so the next such night
  is one read, not four hypotheses). Also shipped: sidebar rows show
  duration + shots instead of megabytes (cached, off-thread, stubs
  dimmed). Suite green throughout; every push through the pre-push gate.
  NEXT: extend the audit to 5+ sessions for the G5 letter-verdict;
  backfill v2 sidecars for two more sessions overnight.
- 2026-08-18 (loop 18): board GREEN. Traced 011510 @72.2s for real: the
  detector ARMED on the banked cue nudge, then resolved 0.23s later
  (frame-count floors are cadence-dependent) and discarded on ~35px of
  tracked travel — the struck ball's flight was motion-blurred out of
  tracking and its 500px revival snap read as step 0 because _prev
  forgot vanished balls. SHIPPED (5b877cf): TIME floors on resolution
  (min_shot_s=1.2, settle_s=0.5), 3s positional memory for vanished
  balls, and a CARRIED-DWELL veto (hand on movers >60% of moving
  updates = gathering, discard at resolve). Also: my pytest gate was
  PIPED again — second-ever red push (Joe got the email); fixed the two
  legacy tests, verified unpiped (347 green), and installed a pre-push
  hook that runs the suite unpiped and blocks red pushes structurally.
  VERIFICATION (re-backfill + audit, all 3): carried-dwell veto WORKS —
  18 gathering-shots removed (115->105, 95->87), hand-involved silent
  19->14. NOT fixed: 011510's 6 miss candidates unchanged — synthetic
  test passes but the real flyer likely returns as a NEW track id, not
  a same-id revival, sidestepping the positional memory. NEXT: trace
  that spawn path and credit new-id appearances born mid-shot.
- 2026-08-18 (loop 17): board GREEN. SIDECAR V2 shipped (48de297): each
  state records hand-adjacent ball ids + bed fraction under hands; v1
  reads as unknown-not-no; live recorder + backfill both write it; app
  restarted so new sessions carry hand context. Re-backfilled the 3
  audit sessions and re-ran the audit with the exact question ("was the
  mover hand-adjacent"). VERDICT NUMBERS: candidates 34 -> 25 (hand
  evidence, not heuristics); unheard 20 -> 1 clean + 19 HAND-INVOLVED
  (gathering that slipped the carried gate — a named precision leak,
  not mystery silence). Eyeballed 011510's six hands-free candidates:
  they come in PAIRS 0.3s apart (strike clack + object-ball impact) =
  ~3 real missed strokes; 72.2s frame shows a clean follow-through the
  detector missed outright. G5 STANDING: ~90-94% recall, ~91% precision
  floor, both defect classes now named and countable. NEXT: trace WHY
  the 011510 @72s class misses (cooldown? require_cue?) and plug the
  carried-gate leak behind the 19 silent gathering-shots.
- 2026-08-18 (loop 16): board GREEN. AUDIT CLASSIFIER v2 (7982191):
  pre-shot-sound window (feathering before a detected shot is not a
  miss), claim window covering the 2s backdate cap, launch test (>=1.2
  radii per 0.3s sample) + convergence veto (>=3 movers ending >=25%
  closer = racking). Candidates 234 -> 34, unheard 36 -> 20 across the
  three re-backfilled sessions. Eyeballed survivors: fast hand-SWEEPS
  still pass the launch test (and the hand filter leaves too few
  tracked movers for the convergence veto) — sidecar-only
  classification is at its ceiling. HONEST G5 STANDING: 223 detected
  shots, <=34 candidate misses (~87% recall floor; several survivors
  visibly hand-work), ~91% precision floor. NEXT: sidecar v2 records
  per-state carried ids / foreign fraction (tiny, live-path), re-backfill,
  and the audit asks the exact question "was the mover hand-adjacent" —
  then the G5 verdict.
- 2026-08-18 (loop 15): board GREEN. G5 MEASUREMENT on re-backfilled
  sidecars (new detector): 210621 0 -> 95 shots (the zero-detection
  drill session now detects), 011928 76 -> 115, 011510 10 -> 13;
  missed-stroke candidates across the three 535 -> 234 (-56%). NOT MET
  yet: 210621 keeps 162 candidates + 20 unheard. Eyeballed the residual:
  a 4-onset cluster at 31.9-32.9s is Joe FEATHERING before the break —
  the audit's 2s look-ahead claims pre-shot taps as misses because the
  real shot follows within the window, and the backdate cap pushes that
  shot's start just past the claim window ("unheard"). So a big slice of
  the residual is AUDIT-CLASSIFIER artefact, not detector failure. NEXT:
  refine the audit (pre-shot-sound reclassification; claim window
  covering capped backdates; placement-snap pattern), then the G5
  verdict. Also shipped while measuring: G6 PER-SHOT THUMBNAILS
  (strike-moment frames as row icons, outcome edge bar, off-thread
  extraction over a signal bridge; 3 tests). Suite green, app restarted.
- 2026-08-17 (post-loop-13): confirmation corpus landed — **G4 DONE**.
  Aggregate 0.50/1k (target <0.75), worst session 1.41 (target <2.0),
  0 failed. Journey: 1.05 -> 0.59 (rest-frozen identity) -> 0.50 (hand
  filter); worst session 3.90 -> 1.41. Also this hour, on Joe's report:
  CI was red on every push (tests read the gitignored measured-colour
  file, which only exists on the rig) — refs snapshot committed as a
  fixture, suite verified green with the rig file hidden, Build &
  Release confirmed green on 006ea3e. Failure emails stop.
