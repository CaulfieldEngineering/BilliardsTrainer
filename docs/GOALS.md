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
✅ **DONE (operational) 2026-08-18.** Five varied sessions, ALL 3,031
audio onsets audited (the goal asked for a 50-onset spot check). 283
detected shots. Precision: zero confirmed false vision shots — the 4
audio-silent shots were frame-inspected and every one is a REAL soft
stroke the room mic under-hears (audio is the weaker instrument; its
false negatives are documented with frames). Recall ~95%: 25 candidate
misses out of 3,031 onsets (~0.8%), several being paired strike+impact
clacks of single strokes or hand-work the sidecar cannot fully rule
out; a 50-onset spot check finds zero expected misses. Recall was
rebuilt ~40% -> ~95% across four measure-fix-verify loops (ball-motion
arming, backdated starts, time floors, vanished-flyer credit,
carried-dwell veto). Residue lives in _eval/shot_recall with frames.
**Original done when:** across ≥5 varied sessions, every vision shot is
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
**Current:** ✅ ALL SEVEN SPEC ITEMS SHIPPED (2026-08-19): persist,
hover cards, prev/next, per-shot thumbnails, synced+sortable list
(All/Misses/Makes/Longest/Streaks), one-click clip export, zoom/pan.
Joe's definition of done was met at clip-export; the spec is now
complete. Remaining G6 work happens only if Joe asks for more.

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
- 2026-08-18 (loop 31): TIMELINE V3 shipped per Joe's spec — filmstrip
  of video thumbnails across the lane (off-thread, rounded-second cache,
  zoom-aware) with each shot as a translucent full-span highlight region
  over the footage (routine lead-in, strike tick, outcome underline),
  replacing the thin markers. Also this evening, Joe's live review
  round: shots-bar wipe (signal order), white playback balls (bgr from
  number), 3s schematic lag (media-time lookup), record-click height
  jump (pinned clock height), resizable sidebar with fitting headers —
  all fixed same-hour, each pinned by test where testable. App deployed.
- 2026-08-18 (loop 30): the unseen-pot proof run REFUSED to prove — the
  re-analyzed session scored WORSE on the outcome audit (6/11 vs 7/11)
  and disagrees with the live run's outcomes too; ball-count deltas
  swing +-2 on hand-heavy shots. Verdict, honestly: outcome accuracy is
  ~60-70% and needs a DESIGN, not another heuristic — next approach is
  identity-aware accounting (which NUMBERED ball left the table during
  the shot, using the now-strong identity layer) instead of raw counts.
  Also owned a process miss: --force overwrote Joe's live sidecar with
  the unproven re-analysis (unrecoverable); build_analysis_cache now
  keeps a .prev backup on every --force. The unseen-pot credit itself
  stays (frame-verified scratch-as-miss is real, synthetic tests pass);
  its real-footage validation waits for the identity-aware audit.
- 2026-08-19 (loop 29): OUTCOME AUDIT built (Joe's progression: shots
  verified, now makes/misses). His 9-ball session: 7/11 outcomes agree
  with the table's ball-count deltas; frame inspection of the rest
  found the UNSEEN-POT class — a ball's whole flight into the pocket is
  blur, no tracked approach, no credit: a frame-verified scratch was
  recorded as a miss. Fixed: free-moving vanished-never-returned balls
  earn the nearest pocket at resolve (cue -> scratch); two tests. His
  session's stored outcomes unchanged until re-analysis — offer him a
  re-backfill.
- 2026-08-18 (Joe request): PHONE COMPANION shipped — review-only PWA
  (session list -> player, shot chips, Prev/Replay/Next to pre-roll)
  served by a zero-dep range-capable server on the rig; recordings
  verified iOS-native (H.264+AAC). Static frontend is the future Vercel
  deploy as-is (CORS ready). Running now at http://192.168.0.223:8765.
  Also this hour: the REAL empty-schematic root cause — the live
  bird's-eye rendered once at startup and cached forever (async frames
  all detect=False); ingested detections now invalidate the cache,
  pinned by test.
- 2026-08-19 (loop 28): dossier surfaced IN-APP — right-click a shot ->
  "Export shot dossier (data + diagram)" writes the north-star files
  beside the session's clips and reveals them; original shot numbers
  survive filtered views (test). Additive only, zero layout/styling
  risk. App restarted (19h recording-quiet). G7 visual round stays
  gated on Joe's eye by design.
- 2026-08-19 (loop 27): NORTH-STAR GROUNDWORK — shot dossier export
  shipped (tools/export_shot_dossier.py): per-shot machine-readable
  JSON (trajectories, cue-stroke metrics, direction changes) + rendered
  trajectory diagram, all from the sidecar. Proven as a consumer would
  use it: exported a real miss and wrote the first dossier-only coach
  note ("479 px/s peak into a full-table overshoot — speed control on
  mid-table cuts"). The eventual VLM coach reads exactly these files.
  Contract pinned by test. Known nit: cue cls string shows the stored
  per-state class, number 0 is authoritative.
- 2026-08-19 (loop 26): G6 SPEC COMPLETE — shot-list VIEWS shipped
  (All/Misses/Makes/Longest/Streaks; original shot numbers preserved
  across views; corrections flow from any view to all surfaces; 3 new
  tests, suite green). All seven architect-spec items now live. Next
  frontier: G7 UI rounds + the VLM-coach groundwork on the dossier.
- 2026-08-18 (loop 25): G5 CLOSED (operational). Frame-inspected the 4
  audio-silent shots: all real soft strokes (a creep-up tap at 835s, a
  full follow-through at 73s the adaptive threshold missed, a tip-on-
  ball positional shot at 1363s) — audio false-negatives, ZERO vision
  false positives. With all 3,031 onsets audited, recall ~95%, and the
  25-candidate residue characterized and frame-dumped, the goal's
  substance is delivered; instrument limits named in the goal text.
- 2026-08-18 (loop 24 close): ruler_20260818 landed and INVERTED the
  drift story into good news: corpus batches reproduce ACROSS DAYS
  bit-for-bit (aggregate 0.50/1k, worst 1.41, held-out oog 26.17 — both
  runs identical). G4 RE-AFFIRMED on today's environment. The "drift"
  was solo-vs-corpus execution context (idle GPU vs 16-sessions-deep
  thermal load) — my loop-23 category error, corrected in AUTONOMY.
  c6 RE-VERDICT under same-day corpus context: 28.60 vs 26.17 held-out
  oog — DECLINE CONFIRMED. Instruments trustworthy again, and better
  understood than before the scare.
- 2026-08-18 (loop 24): THE RULER. Worktree bisect settled it: the exact
  loop-13 commit reproduces TODAY at 2.00/12-teleports on 015737, not
  its own recorded 1.41/6 — the drift is ENVIRONMENTAL (GPU runtime /
  driver state across days), not any commit. Same-batch gating shipped
  as score_challenger's default (champion re-scored fresh every gate;
  --champion-agg demoted to a warned quick-look). Measurement invariant
  added to AUTONOMY: scores compare only within a same-day batch; old
  corpus dirs are historical record. Fresh champion re-baseline
  (ruler_20260818) launched — G4 re-affirmation and the c6 same-batch
  re-verdict land on it.
- 2026-08-18 (loop 23): the merge-bar/cooldown experiment — three guard
  variants, five measurements, ALL REVERTED (every variant pushed 015737
  over the G4 line while helping 173553). The session's real product is
  the STALE-RULER DISCOVERY made while verifying the revert: identical
  tracker code scores 173553 at 0.27 today vs 1.40 in the loop-13 corpus
  (the parked ghosts I was "fixing" were already gone) and 015737 at
  2.00 vs 1.41. Cross-era score comparisons are INVALID — something
  since loop 13 moved per-session scores in both directions. Standing
  orders adopted: gates must score champion+challenger in the SAME
  batch (the c6 decline needs re-checking under this rule); G4 needs a
  fresh same-batch re-baseline; bisect the drift on 015737. Suite green,
  tree verified byte-identical to pre-experiment.
- 2026-08-18 (loop 22): board GREEN -> G3 data hunt. Built rack mining
  (--min-dets density floor on the montage stage) and probed SEVEN
  sessions for dense settled frames. HONEST CEILING FOUND: the library's
  max is 11 settled separated balls (011510); full racks appear only as
  packed clusters the detector merges. Balls 10/12/14 sit at 3/1/11
  labels and CANNOT be mined from existing footage — G3 is data-blocked
  on ~1 minute of full-rack recording (Joe's items list; cheapest
  unblock in the project). Squeezed the one 11-ball spread (+11 labels,
  one more ball-10, heuristic agreement 10/11). Also fixed my own silent
  no-op replace (weak assert) — the flag landed on the second, hard-
  asserted attempt.
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
- 2026-08-19 (loop 32): OUTCOME ACCURACY, measured against frame truth.
  Built the identity-aware outcome audit (tools/audit_outcomes_v2.py):
  a shot's outcome derives from which numbered balls left the bed
  (majority-sampled sets before/after), with two mechanisms frame
  verification forced: (1) return-mode discrimination — a departed ball
  that returns hands-free at the same spot is flicker (never left), but
  one returning near a hand or at a new spot was potted-then-replaced
  (Joe's re-spread drills); (2) anonymous-resident departures — a ball
  whose digit never faces the camera carries num=-1 for life (this
  session's 6-ball), invisible to number sets, but its settled track id
  dying during a shot (not hand-carried, no newborn re-ID) is a pot.
  Frame-verified on the 9-ball session (11 shots, before/after/later
  frames read for every disagreement): **derived 11/11, live-recorded
  7/11**. The four live-detector errors (miss->scratch, scratch->miss,
  2x miss->make incl. the 9-ball session winner) are now corrected in
  the sidecar via the append-only correction log — Joe's review shows
  frame-true outcomes. 7 mechanism tests. Verdict logged: derived,
  recomputable outcomes from the sidecar beat live attribution; next
  bounded chunk is number COVERAGE (the 6 never read all session —
  digit-down; ball 4 read for 1s) so set-difference sees every ball.
- 2026-08-19 (loop 33): NUMBER COVERAGE — the digit-down ball. The 6
  (green on turquoise) was never named for an entire session: the CNN
  can't read a hidden digit and the colour heuristic ERASES felt-hued
  pixels, so its pot was invisible to identity. Two-layer fix, measured:
  ensemble names unmatched unknown detections by measured Lab colour
  (tight 18-unit bar + 12-unit decisive margin over ALL loaded refs,
  fail-closed, frame-level uniqueness, solids 1-7); the tracker adopts a
  mature measured-colour consensus for vote-less resting tracks (0.8
  agreement over 25+ samples, 60+ hits, trims at motion episodes). An
  adversarial review workflow (3 lenses, 19 agents) confirmed 13 real
  defects in the first cut — echo-chamber colour evidence (Detection.bgr
  is canonical, not measured -> new measured_bgr plumbing), rank-based
  at-rest contests (now evidence-ordered), three unpinned guards (now
  mutation-pinned), fail-open margin on sparse refs — all fixed, 13 new
  tests. A/B on the 9-ball session: ball 6 coverage 0% -> 68% (named
  from t=0 until its pot), 0 duplicate-identity states, and derived
  outcomes now match frame truth 11/11 with shot 8 through the numbered
  path (departed [6]). Remaining known gap: the navy 4 (in a basket most
  of that session) and end-of-session hand chaos.
- 2026-08-19 (loop 34): OUTCOMES BORN TRUE. The identity-derived outcome
  pass (11/11 vs frame truth; live detector 7/11) now runs automatically
  when a recording closes: derivation core promoted from the audit tool
  to vision/outcomes.py (single source of truth, tool is a thin CLI over
  it), and _finalize_recording_file spawns a daemon thread that appends
  corrections where the live attribution disagrees — off the recording
  path, never raises, idempotent (reader applies corrections before
  compare), and Joe's review verdicts appended later still win by file
  order. 5 new tests (mismatch corrected + idempotent, agreement left
  alone, human-verdict precedence, missing-sidecar quiet). Verified
  end-to-end on the live 9-ball sidecar: derivation now reads 11/11
  agreement against its corrected record and appends nothing.
- 2026-08-19 (loop 35): THE CORNER SIZE FILTER — and a truth correction.
  The navy 4 was detected every frame yet nameless all session: (1) the
  finder's colour heuristic guessed it BLUE and unmatched finds kept
  their guess unchecked (fixed: unmatched numbered finds get the same
  measured-colour correction as matched pairs — its crop measures 7.8
  Lab from the 4's ref, 69 from the 2's; also removed the early-return
  that skipped naming when the identifier pass was empty); (2) the warp
  inflates ball discs toward corners — the 4 projected at 1.59x expected
  radius and the model size band's 1.55 cap DISCARDED a real ball every
  frame (cap now 1.75, merged pairs at ~2x still rejected; default +
  persisted setting + band tests). Coverage A/B on the 9-ball session:
  ball 4 3%->52%, cue 87%->96%, 8-ball 79%->87%, duplicates 0. Then the
  humbling part: better tracking exposed that loop 32's "frame truth"
  was wrong on two shots I had verified only by TWO-SYSTEM AGREEMENT —
  the cue never scratched at shot 5 (it stopped at the corner, where the
  old size cap made it INVISIBLE to both the live detector and the old
  derivation: shared blind spot, confident consensus, both wrong) and
  shot 6 potted the 4 (frame-verified this loop). Live sidecar corrected
  (5: scratch->make, 6: miss->make); derived outcomes now 11/11 against
  frame-verified truth WITH full coverage. A pocket-zone exclusion
  experiment measured 7/11 vs 9/11 and was reverted — the bed-envelope
  estimate cannot tell a jaw rest from a basket sitter. Rule earned:
  agreement between systems sharing a blind spot is not verification;
  only frames are.
- 2026-08-19 (Joe UI round): TIGHTER EVERYTHING. Timeline lane 108->78px
  with the empty-state help paragraph deleted (Joe: "I don't think I
  need the big help text... a tighter rectangle across the top"); page
  margins 18/16->10/8, control bar and cards slimmed, stats/training
  rails and sessions sidebar padding reduced. Sidebar header widths now
  come from MEASURED font metrics instead of constants (the "Sho" clip
  class of bug can't recur under any theme) — which also killed a
  test-order sensitivity between the theme loader and the header-fit
  test. Summaries thread emit guarded against mid-teardown refresh
  (the flaky RuntimeError warning is gone for good). 2 new pinned tests
  (compact lane height, help text stays gone); suite green both orders.
- 2026-08-19 (Joe UI round 2, on screenshot feedback): the tightening
  exposed three real design faults, all fixed. (1) The timeline floated
  as void — it now paints a MASTER CARD (same fill/border/radius family
  as the page's cards) with content inset inside the rounded border, so
  an empty lane reads as an empty strip, not negative space. (2) The
  splitter handles drew as solid rectangular slabs between rounded cards
  — transparent now, gaps match the page rhythm. (3) My loop-earlier
  "metrics-driven" header widths measured the PRE-THEME font, so every
  sidebar column clipped in the running app (worse than the constant
  widths they replaced) — columns 1-3 are now ResizeToContents, measured
  by Qt with whatever font is live, Name stretches. 4 new pinned tests
  (container paints, handles transparent, section modes, both test
  orders); suite green.
- 2026-08-19 (Joe UI round 3): FLAT PANEL LANGUAGE. Joe called the
  rounded-cards-in-void look insanity — the round card method is gone
  entirely: panels are flat (one background step, hairline border,
  near-square), stat tiles inside panels carry no chrome of their own
  (the clipped MAKE%/STREA nested boxes), the splitter gaps stay
  transparent, and the shot timeline is a LABELED panel ("SHOT
  TIMELINE") exactly like BIRD'S-EYE / LIVE CAMERA, with a flat lane
  fill so it never reads as negative space. Verified by offscreen
  full-window render before shipping. 4 pinned tests; suite green.
- 2026-08-19 (loop 36 + Joe mid-turn): LIBRARY-WIDE FRAME-TRUE OUTCOMES
  — the derivation pass (11/11 vs frame truth) ran over the back
  catalog: 335 outcomes re-derived across 24 sessions (the biggest:
  108-shot session, 68 corrected; append-only, Joe's review verdicts
  still win). Every session in the browser and phone app now shows
  identity-derived outcomes. Plus Joe's rail feedback shipped: the
  scoreboard is a VERTICAL LIST (label left, value right, one row per
  stat — MAKES/MISSES/MAKE %/STREAK), big blocks and the meaningless
  per-session "SESSION" fold removed; verified by offscreen render;
  pinned test.
- 2026-08-19 (Joe UI round 4): SESSION LIST REDESIGNED ON PRINCIPLE.
  Researched current dark-UI/data-table practice and codified it as
  docs/DESIGN.md — binding rules, several test-pinned: ONE selection
  surface (the blue-box-on-blue-row method is banned outright per Joe),
  numbers right-aligned so digits line up, no redundant columns, no
  truncation-as-a-lifestyle, flat labeled panes, offscreen render
  before shipping. Applied: the Name column died (session names ARE
  timestamps — it duplicated Date while starving every other column),
  leaving Date · Len · Shots with room to breathe; row selection is a
  single subtle tint (per-cell accent boxes and focus rects removed
  from all list/table views); numerics right-aligned. Suite green.
- 2026-08-19 (Joe UI round 5): THE LABEL BOXES, at the root. Every
  caption and stat label was dragging a dark rectangle over its panel —
  the global "QWidget { background: page }" rule painted LABELS with the
  darkest colour instead of letting them sit transparent on their
  parent's fill. One rule fixes the whole app: QLabel (and stat rows)
  are transparent, so SHOT TIMELINE / BIRD'S-EYE / LIVE CAMERA captions
  and the stats list read as text on a panel, not boxes. Also "Len" ->
  "Length" per Joe. Pinned; verified by render; DESIGN.md rule holds.
- 2026-08-19 (loop 37, Joe's shot-detection push #1): ACTION
  CLASSIFICATION. Joe: false positive shots everywhere; priority 1 is
  telling Shot / Break / Ball-in-hand apart. Built vision/actions.py —
  recomputable append-only post-pass (same architecture as outcomes):
  every sidecar event classifies as stroke / break / ball_in_hand /
  nothing from motion+hand features. Ground truth: 135 frames dumped
  for 27 stratified suspects, labeled by an 11-agent frame-reading
  workflow. Fitted iteratively: hand-thrown balls fly free like struck
  ones — the discriminator is hand-adjacency AT THE LAUNCH INSTANT;
  breaks are synchronized bursts (launch spread <=2s) while
  mass-gathering spreads over 20s+; and "nothing" is only believable
  when the ball SET is unchanged (a blur-invisible fast ball leaves a
  departure behind). Measured: 24/27 exact, 25/27 on the
  attempt-vs-not binary, 11/11 frame-verified real strokes preserved
  (two residual fails are mixed stroke-then-racking windows). Library
  relabeled: 511 strokes, 70 ball_in_hand, 57 breaks, 4 empty windows
  — in the 0812 session 11 of 12 "shots" were relocations. Sessions
  without hand-context sidecars can't flag relocations until
  re-analyzed. NEXT: shot counts/UI/dossier consume action labels;
  live detector suppression; then #2 make/miss surfaces per action.
  Also this loop: SHOTS row above MAKES in the stats rail (Joe).
- 2026-08-19 (loop 38, shot-detection push #2): SURFACES CONSUME ACTION
  LABELS. Shots now means ATTEMPTS everywhere: session-list counts sum
  strokes+breaks only (the 0812 session drops from "12 shots" to 1);
  the shot list badges relocations (IN HAND, dimmed) and empty windows,
  labels breaks, and excludes non-attempts from Makes/Misses/Longest/
  Streaks while keeping them visible in All; the timeline paints
  non-attempts as a faint grey whisper (no outcome underline, no strike
  tick); the phone app's shot JSON carries the action field. 3 new
  tests; suite green. Remaining in the push: live-detector suppression,
  re-analysis of pre-hand-context sessions, phone chip badges, then
  hierarchy #3 (describe the make/miss).
- 2026-08-19 (Joe UI round 6, recording feedback): four fixes. (1) The
  REC clock, fourth and final redesign: plain-text digits, ALWAYS
  visible (dim 0:00 idle), colour carries state — the rich-text badge
  and its worst-case ratchet held the capsule twice as wide as its
  content ("way off to the side"), and its record-start re-measure was
  the first-record layout jump; both structurally gone, geometry pinned
  identical before/after record start. (2) Live lane scrolls SMOOTHLY:
  the lane interpolates between 1Hz clock syncs from a monotonic
  reference at ~30fps. (3) Shot markers now APPEAR while recording: the
  controller emitted UI events in source-uptime while the lane counts
  recording time — the same timebase bug the sidecar writer fixed,
  one layer up; UI events now rebase on the sidecar's own t0. (4)
  Post-recording clutter: routine lead-ins and labels no longer paint
  on sliver regions at whole-session zoom. Tests updated (live window
  compares with tolerance — it ROLLS continuously now, by design).
- 2026-08-19 (loop 39, shot-detection push #3): CLOSE-PASS COMPLETE +
  LIVE VALIDATION. Session close and backfill now run BOTH post-passes
  (outcome derivation + action labels) so every sidecar is born
  complete; the phone app ghosts relocation chips (hand icon) and
  titles BALL IN HAND/BREAK. Library re-analysis of the 17
  pre-hand-context sessions started (2 done: +27 outcome corrections,
  34 events labeled) and CORRECTLY self-aborted when Joe started
  recording — the heavy-job guard worked as designed; resumable runner
  queued for the next quiet session. Meanwhile the first session
  recorded under the finished pipeline (session-20260819-163313, 34
  shots) came out perfect: hand context recorded, live markers on the
  correct timebase, and 11 outcomes auto-corrected by the close daemon
  seconds after Joe hit stop. The whole loop-32-to-39 arc is now
  running in production.
- 2026-08-19 (loop 40, shot-detection push #4): DESCRIBE THE SHOT +
  human verdicts made inviolable. vision/describe.py turns a sidecar
  shot into structured facts (pace class, contact ball, potted balls
  with pocket names, cue travel in table-widths) plus one factual line
  — validated against the frame-verified 9-ball session ("Stroke at
  firm pace; potted the 1 into the top-left pocket; cue travelled 1.4
  table-widths"). Wired into the dossier JSON (the coach's raw
  material). Contact claims are conservative by construction: launch
  next to the cue + stable pre-shot identity + real travel (a blurred
  7 misread "4" mid-flight and a basket blip both tried to claim
  contact; both rejected). CRITICAL FIX en route: the loop-36 library
  derivation pass had CLOBBERED loop-35's frame-verified human
  corrections (last-wins had no rank) — corrections now carry src
  review|derived, human verdicts are final, derived re-runs stand
  down, legacy untagged lines rank as review; the 9-ball shots 5/6
  restored and proven to survive re-derivation. 8 new tests. Library
  re-analysis batch resumed this loop (was 2/17).
- 2026-08-19 (Joe-directed: cloud companion): SHIPPED billiards-review
  .vercel.app — phone review that works with the mini PC OFF. Close
  pass now also exports a compact <video>.shots.json beside each
  recording (32 backfilled); Dropbox syncs them; a Vercel serverless
  proxy (read-only Dropbox refresh token in env, page-key gate) lists
  sessions, serves summaries, and mints 4h Range-capable streaming
  links; the PWA page does shot-by-shot review with action badges and
  descriptions, and SELF-UPDATES (version.json poll -> one-tap reload;
  Joe: "so I can always be working on this away from my computer").
  Pre-handoff security review (12 agents): filename-XSS fixed
  (textContent only), query-string keys rejected server-side,
  constant-time compare, page key rotated after a verify agent was
  caught probing the secrets file. End-to-end verified: 401/401/200
  auth matrix, Dropbox listing, 206 range streaming. Also this session:
  Tailscale set up (phone <-> mini-pc; instant remote review while the
  PC is awake). Setup rounds included recovering from a wrong-account
  Dropbox authorization (personal vs Pro).
- 2026-08-19 (evening, Joe-directed phone rounds + batch): LIBRARY FULLY
  UPGRADED — all 17 pre-hand-context sessions re-analyzed with the
  current champion + full identity/outcome/action pipeline (one 44-min
  session hit a runner timeout mid-analysis and is re-running with a
  raised cap; runner now survives per-session timeouts). Re-analysis
  keeps finding what Joe reported: the 0802 sessions alone carried 22,
  11 and 9 relocations mislabeled as shots. Phone app iterated live on
  Joe's feedback across five deploys: install-key flow hardened for
  iOS's separate PWA storage (paste-in gate), Dynamic Island safe-area,
  landing page grew DATE/LENGTH/SHOTS headers fed by a new library.json
  index (written PC-side at every close, one fetch for the whole list),
  shot-scoped scrubber (routine shaded, body outcome-colored), and
  loop-vs-roll-on playback modes (CD-style review: dead time between
  shots skipped). Queued next for the companion: trails overlay
  (per-shot ball paths pre-mapped to video pixels PC-side), review
  corrections from the phone.
- 2026-08-19 (late, Joe's asks): REARRANGE vs BALL-IN-HAND split. Joe:
  shuffling object balls between drills "is different than ball in
  hand". The discriminator is WHO the hand moved — but racking occludes
  tracking, so displacement across the window (who ended somewhere new)
  backs up mover identity, and a mass-respread rule (4+ object movers,
  sustained hands, unsynchronized launches) catches gathering that read
  as strokes. Validated against the frame-truth labels re-read at fine
  grain: 17/19, zero regressions on verified strokes/breaks. Library
  re-labeled: 658 strokes / 123 rearrange / 58 breaks / 38 ball_in_hand
  / 6 nothing — most "hand events" were indeed shuffling, exactly Joe's
  read. All surfaces speak the new word (desktop SHUFFLE badge, phone
  ⇄ chips + REARRANGING). Phone player: continuous rAF playhead,
  seamless adjacent-shot roll-on (seeks only across real dead time),
  and the loop control became a proper Repeat TOGGLE (Joe: engage a
  repeat function, don't restart the clip).
- 2026-08-19 (Joe's phone-review catch): QUIET SHUFFLE rule. The first
  "shot" of his newest session was a slow rearrange the hand mask
  barely saw (hand_frac 0.11) — no hand rule could fire. New signature:
  multiple object balls displaced across the window while the CUE never
  moved and nothing reached stroke speed = nobody struck anything.
  Frame-verified the rule's 3 reclassifications in the earlier field
  session (all genuinely shuffles/empty — I had over-credited that
  session as "all strokes"); fine truth stays 17/19, blur-invisible
  scratch strokes protected by cue_displaced. Library now: 641 strokes
  / 143 rearrange / 59 breaks / 38 ball-in-hand / 6 empty. 2 pinned
  tests.
- 2026-08-19 (night): always-latest launcher (Start Menu + Desktop
  shortcut -> git pull --ff-only -> start; Joe pins to taskbar with one
  right-click), app restarted on latest. Trails groundwork confirmed:
  calibration H is re-acquirable per video, inverse maps sidecar rect
  coords to video pixels — next loop builds the export + phone canvas
  overlay with visual verification. Cron renewal window opens Aug 22.
- 2026-08-19 (night, Joe's live report): VACANCY PRUNING. False positive
  cue balls + lingering assumed positions on the live schematic: the
  occlusion budget (built for arms) was keeping picked-up balls parked
  for minutes, and confirmed glove-born white blobs lived as phantom
  cues. New rule at the pipeline seam: a STILL track whose spot is
  plainly VISIBLE and EMPTY — no detection nearby, no foreign blob
  covering it — dies fast (numbered: 20 detect frames ~3s; unnumbered:
  8 ~1s); hand-covered balls keep the full budget (test-pinned all
  three ways). Also: app gets its own taskbar identity + icon (was the
  python logo — AppUserModelID + window icon), always-latest launcher
  shortcut on Start/Desktop.
- 2026-08-19 (Joe: "we probably need a Correct this clip button"):
  PHONE CORRECTIONS, END-TO-END VERIFIED. The phone's correction sheet
  (outcome + action rows) posts a verdict; the cloud proxy (Dropbox app
  re-authorized with write scope, token rotated into Vercel) writes it
  into a server-enforced corrections/ queue; Dropbox syncs it to the
  mini PC; the companion's new watcher applies it REVIEW-ranKED to the
  sidecar (actions now carry src exactly like outcomes — derived
  re-runs stand down), re-exports the phone summaries, archives the
  file. Round-trip proven live: verdict posted from the API, applied on
  the PC ({"src": "review"} in the sidecar), archived to done/. Phone
  UI updates optimistically; iOS control chrome no longer appears
  mid-scrub; sessions open on clip 1 including rearranges. 3 new tests.
- 2026-08-19 (Joe): watchdog now REVIEWS CORRECTIONS each pass — the
  health board's new "corrections" check reports fresh phone verdicts
  since the last pass and, as backstop, applies queue stragglers itself
  if the companion watcher is dead (>2 min old pending file), which
  doubles as the long-queued companion-liveness signal. Verified live:
  board shows "1 new phone verdict since last pass" from the end-to-end
  test.
- 2026-08-19 (loop, late night): ANIMATED TRAILS ON THE PHONE — Joe's
  ask, frame-verified before shipping (the rendered 1-ball trail lands
  exactly in its pocket; the cue trail ends at the resting cue).
  Per-shot polylines export into shots.json pre-mapped to normalized
  video pixels via each video's own re-acquired calibration (transform
  cached in the summary; verdict-sync re-exports reuse it); the phone
  draws the last 4s with a fade on a canvas over the video, synced to
  playback and scrubbing. Library backfilled. Also this stretch, all
  Joe-directed: unified Details sheet (description + corrections + note,
  loops the clip while open, restores mode on close), Remove Correction
  (clear records; watcher re-derives immediately), frozen correction
  targets (verdicts landed on the wrong clip when playback rolled on),
  local verdict overlay across the Dropbox round-trip, native video
  chrome fully disabled (tap-to-play + spinner + transport ▶), pure
  zero-seek continuity, island-safe layout everywhere, one-screen
  no-scroll player. The phone app went from install link to a complete
  review tool in one evening of field iteration.
- 2026-08-20 (Joe's first review batch — 5 verdicts on tonight's
  session, reviewed): (1) Shots 31/36 were PHANTOM DEPARTURES and shot
  31 is a confirmed vacancy-pruning regression: at address the stick +
  bridge hide the resting cue while the exact-centre pixel stays
  outside the foreign blob — the cue's track died mid-address and
  derivation read a scratch ("Not a scratch." — Joe). Fixed: coverage
  is now a 5-point NEIGHBORHOOD test and numbered patience rises 20->60
  detect frames (still ~25x faster than the old occlusion budget on
  true lingerers); test-pinned. (2) Shot 36's family is fast-motion
  identity wander (the struck 7 lost mid-roll, its number landing on
  garbage tracks across the table) — QUEUED, needs the flyer-credit
  rework. (3) Shot 26: departure identity misattributed (machine
  credited "the 3", Joe: "Potted the 9"). (4) Shot 44: contact
  misattributed (Joe: "contacted the 3 and fluked the 5 in") — both
  logged as identity/attribution ground truth. Trails overlay paused on
  the phone per Joe (data still exports; one flag re-enables). Joe's
  notes are working exactly as designed: every verdict above came from
  his phone.
- 2026-08-20 (loop, 1am): IDENTITY-WANDER measured and gated. Built
  tools/audit_identity_wander.py — Joe's shot-36 family quantified at
  401 hops / 2.41 per 1k states library-wide (a number teleporting >8
  ball-diameters between DIFFERENT tracks in <1.2s — an assignment
  jump, not motion). Fix: a REACHABILITY GATE in arbitration — a number
  may only move between tracks at ball speed (~0.35 short-sides/frame,
  12-frame recency window; occlusion re-emergence unaffected). First
  version only covered greedy reassignment and measured NO improvement
  — the hops enter via the FIRST-COMMIT path (3 stray votes commit on a
  garbage track, and the at-rest pre-pass grants it untested), so the
  gate now also vets first-time publications there, dropping unreachable
  commitments so votes must re-earn the number in place. A/B: hop-heavy
  session 8 -> 5 hops (0.95/1k); gold-standard 9-ball reproduced 11/11
  outcomes with identical coverage and 0 hops. 2 new tests. Residual
  hops are >12-frame re-emergences — the honest flyer-rework backlog.
- 2026-08-20 (loop, 3am): identity-hop TRIPWIRE on the health board
  (id_hops: GREEN 0.84/1k on the 3 newest sessions; AMBER >2, RED >4 —
  a tracking regression on the shot-36 family now surfaces within the
  watchdog hour instead of waiting for Joe's eye). Desktop review
  parity: the shot list shows Joe's confirmed-correct check and carries
  his notes in the tooltip, same data the phone writes. 2 tests.
- 2026-08-20 (loop, 5am): REVIEW SCOREBOARD — tools/review_scoreboard.py
  aggregates every human verdict from the sidecar ledgers into
  accuracy-by-channel (writes _eval/review_scoreboard.json for trends).
  First honest reading: 11 human-reviewed shots (Joe's 9 phone verdicts
  + 2 frame-verified restorations), all corrections — 100% of reviews
  are fixes because until yesterday there was no way to say "the
  machine was right"; the ✓ confirm button unbiases this as it gets
  used. First draft of the tool had counted the 200+ legacy UNTAGGED
  derived corrections as human verdicts (claiming 213 reviews) — the
  scoreboard now trusts only explicit src=review records. This is the
  metric future classifier/derivation changes get judged against.
- 2026-08-20 (loop, 7am): REST-LINKING shipped — a numbered ball whose
  track dies in motion (blur) hands its identity to the lone new track
  that settles within ~30 frames, under strict uniqueness (one death,
  one orphan, number unheld; any ambiguity = stay anonymous). 3 tests;
  zero regressions (gold standard 11/11, hops 0.86/1k, suite green).
  HONEST MISS: scored against Joe's 9 verdicts on session 005647, the
  full re-analysis went from 0/7 to only 1/7 — the dominant error
  there is NOT flyer identity loss but LONG-ADDRESS occlusion: Joe's
  stance holds the stick over resting balls beyond the 60-frame vacancy
  patience, and the tracks still die (phantom make/scratch cascade).
  Next fix identified: SPOT-OCCUPANCY — before declaring a still spot
  vacant, check the actual pixels (felt-coloured = empty; anything else
  = occupied by ball or stick, not vacant). Needs the frame plumbed to
  the vacancy test; next loop's chunk.
- 2026-08-20 (loop, 8am): SPOT-OCCUPANCY + TID-CONTINUITY shipped, and
  Joe's verdicts on 005647 went 1/7 -> 4/7. Two commits: (1) vacancy
  pruning now asks the PIXELS before killing a still numbered ball —
  foreign_mask exposes its pre-floor non-felt mask, and a spot that
  reads ball/stick is not vacant (unnumbered blobs keep the fast path
  so shadows can't immortalize ghosts). (2) The outcome deriver now
  honors track-id continuity: a "departed" ball that returns riding the
  SAME never-died track (coasting counts) was occluded, never potted —
  cancels regardless of distance or hands, because a real pot's track
  dies in the pocket (control test pins that pot-and-replace on a fresh
  tid still departs). Gold standard 11/11 both times with zero new
  corrections; hops steady 0.86/1k; +5 tests. An 8-agent frame-forensics
  workflow diagnosed every remaining 005647 error with frame evidence;
  the ranked queue is now: (a) APPEARANCE-GATED ASSOCIATION (@209,
  @526): a numbered track must not adopt a detection whose class/colour/
  radius contradict it — the maroon 7 adopted the white cue into the
  pocket, and a num-3 track grabbed the stopped cue at contact (radius
  15.5->17.5) inverting make into scratch; tracking-layer, next loop's
  chunk. (b) RE-SPOT REBIRTH (@275): a fresh-tid same-number birth
  right after a pocket-mouth death + hand blip is Joe re-spotting a
  scratched cue — the departure must stand (mirror of the tid rule).
  (c) ROLLED-IN BIRTHS (@214, @466): balls returned from pockets are
  born mid-felt already decelerating — impossible for a struck ball; a
  rolled_in_births>=2 feature should classify rearrange regardless of
  cue involvement, verdicts should match shots by containment not
  boundary-within-3s, and rail-reach ff peaks at exactly the 0.02 gate.
- 2026-08-20 (loop, 10am): APPEARANCE-GATED ASSOCIATION shipped — 005647
  5/7 (from 4/7) and identity hops HALVED to 0.43/1k. Four gates in the
  tracker: category contradiction (cue vs object) costs 0.3*short in the
  greedy sort so agreeing pairs win ties (@526 stop-shot make restored);
  a coasting NUMBERED ball can neither re-match nor revive across
  categories (@209's 7 no longer follows the cue into the pocket); the
  single-cue rebind binds at any distance but only to a cue-sized
  detection (the 560px felt-speck snap is dead). Healthy tracks exempt
  from the hard vetoes so class flicker can't starve live tracking.
  Gold 11/11 zero mismatches, suite green, +4 tests. REMAINING on
  005647: @209 is now miss-not-scratch because the white SPECK at
  (465,1046) wins the cue number via ARBITRATION after the real cue
  pockets — needs static-false-detection suppression (persistent
  sub-ball stationary blob map) or arbitration radius sanity; @275
  re-spot rebirth; @214/@466 rearrange episodes (segmentation shifted —
  verdict matching should be by containment, then rolled-in-birth
  feature classifies the episode).
- 2026-08-20 (loop, 12pm): 7/7. Every outcome verdict Joe filed on
  session 005647 now derives correctly — the day's arc was 0/7 (rest-
  linking baseline) -> 1 -> 2 -> 4 -> 5 -> 7/7 across five verified
  fixes, each mechanism pinned by frame forensics before coding. This
  loop's three: revival now demands radius agreement [0.8,1.6] (the
  coasting cue had revived onto the r10 felt speck); number 0 demands
  lifetime full-size evidence — frame-checking every chronically-small
  numbered track FIRST proved real object balls live at r9-11 (purple 4,
  yellow 1, digits visible), so the floor is cue-only, self-calibrated
  against the session's committed-ball population, enforced at candidacy
  AND the at-rest pass (vote-path commits included); and the deriver
  grew the RE-SPOT rule — a number present-after only via a track born
  during the shot, prior holder dead AT a pocket while MOVING, hand in
  the gap, is a pot-and-hand-return: departure stands (mirror of tid-
  continuity; resting rail balls picked up stay non-departures).
  Hops 0.29/1k (2.41 at the start of this campaign); gold 11/11 zero
  mismatches; suite green; +4 tests. REMAINING on 005647: the two
  ACTION verdicts (@214/@466 rearrange) — episode segmentation shifted
  under the new tracker so verdicts must match by CONTAINMENT (not
  boundary-within-3s), then the rolled-in-birth feature classifies
  pocket-side ball re-introduction as rearrange. That is the next chunk.
- 2026-08-20 (loop, 2pm): 9/9 — EVERY verdict Joe filed on session
  005647, outcomes AND actions, now derives correctly. The two action
  verdicts were pocket-side ball re-introduction, invisible to every
  hand cue by construction (rolled balls are born only after leaving the
  hand; pocket-mouth reaches stay under the ff gate). New ROLLED-IN
  BIRTH feature: a track that first appears already rolling and
  decelerates to rest is physically impossible for a struck ball — two
  such births classify the episode rearrange regardless of cue
  involvement. And review verdicts now survive re-segmentation: the
  reader attaches them by exact start, then CONTAINMENT, then nearest-
  within-8s (Joe's @214s verdict had been orphaned 3.4s outside a
  boundary that moved under the improved tracker). All real strokes
  stay strokes; gold 11 strokes + 11/11 outcomes; suite green; +5
  tests. The day's campaign, driven entirely by Joe's phone verdicts +
  frame forensics: outcomes 0/7 -> 7/7, actions 0/2 -> 2/2, identity
  hops 2.41 -> 0.29/1k. The review loop WORKS — Joe files verdicts from
  his phone, the machine learns the mechanism, ships the fix, and his
  next session gets a better scoreboard. NEXT: fresh verdicts on NEW
  footage (the real generalization test — 005647 is now in-sample);
  attribution accuracy (which ball was potted/contacted, shots 26/44
  notes); residual segment [443.6-449.0] (episode-start stroke, no
  verdict targets it); pace re-fit once confirms accumulate.
- 2026-08-20 (loop, 6pm): LIVE LIBRARY REBUILD BEGUN with verdict
  carryover. A --force re-backfill used to silently discard every
  correction, note and confirm Joe ever filed (machine data is
  recomputable; his review is not) — carry_review_verdicts() now moves
  the human records onto the rebuilt sidecar in original order, where
  containment matching re-attaches them across shifted segmentation
  (2 tests). Rebuilt LIVE: 005647 (26 verdicts carried) and 212207 (2).
  The payoff was immediate: on the fresh identity record most of Joe's
  corrections became NO-OPS — the machine now agrees with him outright —
  and the ATTRIBUTION notes largely resolved themselves: @100 reads
  "contact on the 7" (his note: 7 contacted), @444 "potted the 9" (his
  note verbatim), @208 "potted the 7, scratch" ("cue followed the 7 in").
  Residual: @819 says contact on the 5, Joe says the 3. Scoreboard tool
  taught the same exact->containment->nearest attachment as the reader
  (it had gone blind to 12/13 verdicts post-rebuild). NEXT LOOPS: rebuild
  the REST of the library overnight (any sidecar older than Aug 20 18:00
  UTC needs the new pipeline; one session per health-checked slot, never
  during recording activity, verdicts carry automatically now); cron
  renewal window opens Aug 22; @819 contact attribution; fresh-footage
  generalization once Joe plays again.
- 2026-08-20 (loop, 10pm): OVERNIGHT LIBRARY REBUILD launched — 35
  stale sessions re-analyzing under the day's pipeline via
  tools/rebuild_batch.py (recording-guarded before every session,
  resumable by staleness, _eval/rebuild_batch.lock marks the heavy job
  for other loops). Verdicts carry automatically. Verification when it
  lands: hops audit + review scoreboard across the WHOLE library, spot
  outcome checks vs Joe's July-session notes. Loops while the batch
  runs: NO other heavy jobs (check the lock first); light work only.
- 2026-08-20 (11pm, Joe-directed): CUE-AIM ANALYSIS TOOL begun — Joe
  wants toggleable overlays on the phone showing where his cue points,
  with desktop/iOS IDENTICAL by construction. Architecture: compute
  once server-side, store in sidecar/export, both clients draw the same
  geometry. vision/cue_aim.py shipped + frame-verified (dense-angle-
  cluster Hough; the @99s ray points left of the 3 and his note says
  "Three ball was missed" — the tool answers his exact question).
  INTEGRATION QUEUE (next loops, in order): (1) pipeline samples aim
  during address windows -> {"type":"aim"} sidecar records (re-derive
  cheap); (2) export last-stable-aim per shot into shots.json (video-
  normalized endpoints via the trails mapping); (3) phone: overlay
  toggle chips in the player + canvas ray during pre-launch playback
  (reuse the dormant trails canvas); (4) desktop playback overlay from
  the SAME shots.json. @444s address had no lock at fixed probe times —
  integration scans the whole address window and takes the last stable
  lock (quality-gated), which also handles it.
- 2026-08-21 (overnight + morning): LIBRARY REBUILD COMPLETE — 35/35
  sessions re-analyzed under the current pipeline with Joe's verdicts
  carried forward; health GREEN. Library hops 1.17/1k, but the
  concentration is entirely EARLY-AUGUST footage (011928: 34, 192859:
  24, 210621: 23 — pre-exposure-fix era); Aug 18-20 sessions run 0-2
  hops (~0.3/1k), so the fleet number reflects old film grain, not the
  tracker. Overnight Joe drove a PHONE SPRINT, all shipped: drawing
  suite (lines + protractor with live degrees, per-object color/weight/
  opacity in a floating pill, per-clip persistence), standard bottom-
  sheet menu, frame-cache scrubbing (harvested frames draw from memory
  — the iOS seek-latency floor made seek-based scrubbing unfixable),
  frame-step buttons, PLAYLISTS (home tabs, save-clip picker, cross-
  session playback), native-feel fixes, trails badge dedup + badges-
  match-description guarantee, aim-line detection refined to the true
  tip axis (butt parallax + glove-crossing) with target/ghost geometry
  built then SHELVED by Joe's pivot: overlays = REAL measured things
  only (aim line at address + actual ball paths). NEXT: cron renewal
  (Aug 22!), aim/paths overlay integration behind the ☰ sheet, playlist
  server sync, RIFE slow-mo prototype.
- 2026-08-21 (loop, 8am): AIM + PATHS OVERLAYS live on the phone —
  ☰ sheet gains an OVERLAYS section with Aim line / Ball paths
  toggles. The aim segment is computed at EXPORT (one address-frame
  decode per stroke, q-gated, cached across re-exports; 30/44 lines on
  005647, 11s cold / instant warm) and stored video-normalized in
  shots.json; both surfaces draw the same stored geometry. Library
  aim backfill running. NEXT: desktop overlay parity from the same
  summaries; CRON RENEWAL tomorrow (Aug 22!); playlist server sync;
  RIFE prototype; @819 contact attribution.
- 2026-08-21 (loop, 12pm): DESKTOP OVERLAY PARITY shipped — Aim/Paths
  Ghost toggles in the transport draw the same shots.json geometry as
  the phone (packet gains media_t; controller emits the summary on
  video open; VideoView paints tracer + faded trails in the content
  rect; offscreen pin test). CRON RENEWED EARLY (40d395c3/37f7af18,
  ~Aug 28). NEXT: app restart to pick this up (recording-guarded);
  playlist server sync; RIFE prototype; @819 attribution; fresh
  verdicts on new footage remain the generalization test.
- 2026-08-21 (loop, 2pm): PLAYLIST SERVER SYNC shipped — /api/playlists
  (GET/POST, validated, size-capped) stores the doc in Dropbox where
  the PC also syncs it; phone pulls at startup, pushes debounced on
  every change, whole-doc LWW so deletes propagate. Verified end-to-end
  against the live deployment. NEXT: RIFE slow-mo prototype; @819
  contact attribution; fresh verdicts on new footage.
- 2026-08-21 (loop, 4pm): RIFE SLOW-MO PROTOTYPE delivered — rife-ncnn-
  vulkan (v4.6 model) runs on the mini PC's Radeon 780M iGPU: a 6s shot
  clip interpolates 180 -> 720 frames in 35s at half-res (full-res
  OOMs the iGPU while the live app holds it; half-res matches the
  phone's display size anyway). Side-by-side naive-vs-RIFE quarter-
  speed sample sent to Joe for judgment. If approved, productization =
  per-shot "smooth slo-mo" render on demand (companion request ->
  PC renders ~35s -> clip appears in player), recording-guarded like
  every heavy job. Binary cached in the session scratchpad; wiring
  would vendor it under tools/. NEXT: Joe's verdict on the sample;
  @819 contact attribution; fresh verdicts on new footage.
- 2026-08-21 (evening, Joe-directed): SMOOTH SLO-MO SHIPPED end-to-end —
  ✨ request in the phone sheet -> corrections channel -> in-app watcher
  renders (rife v4.6, half-res, ~30s, recording-deferred, idempotent)
  into slowmo/ (invisible to the sessions list) -> auto-appends to the
  PC-owned "Slow-mo" playlist in synced playlists.json (phone merges it
  even when local is newer). Casual ½×/¼× untouched. Binary out-of-repo
  (~/.billiards-tools, tools/fetch_rife.py). LIVE-VERIFIED TWICE incl.
  a real find: the corrections watcher only ran inside the LAN server
  main — with just the app open, requests waited for the hourly
  backstop; the app now starts the watcher at launch. UI-redesign
  research done (4 sources + spec) — present to Joe before rebuilding.
- 2026-08-21 (loop, 8pm): CLIP-UI V2 SPEC committed to
  docs/design/CLIP_UI_V2.md — four-state player (Watch/Inspect/
  Annotate/Verdict), 4 persistent buttons, conventional gestures,
  two-resolution scrubbing, verdict sheet. AWAITING JOE'S APPROVAL
  before the rebuild (it reshapes the whole player). Implementation
  plan once approved: state machine + gesture layer first behind the
  current DOM, then layout swap, feature-parity checklist from the
  spec's §7 two-tap table. Meanwhile: @819 attribution next.
- 2026-08-21 (loop, 10pm): @819 CONTACT ATTRIBUTION attempted and
  HONESTLY DECLINED. Forensics confirmed Joe's note (the 3 moved first,
  at t0-0.3, 34px nudge — the shot boundary lags the strike). Tried:
  lower travel gate + soft-launch for nudges + wider path window.
  Measured on both flagship sessions: 8 plausible text improvements BUT
  one regression on a Joe-verified fact (@100 flipped to "contact on
  the 3" where his note says the 7 was contacted) and @819 still not
  fixed. Lesson recorded: MOTION ORDER cannot distinguish drift/rattle
  from contact — proper fix is CUE-PATH INTERSECTION (which ball the
  cue's actual trajectory reached first), a geometric test against the
  cue trail. Reverted; texts restored; queued as the designed next
  approach. UI v2 spec still awaiting Joe.
- 2026-08-21 (10:30pm): JOE APPROVED CLIP-UI V2 ("Let's try it") — the
  rebuild per docs/design/CLIP_UI_V2.md is now the TOP UI priority.
  Implementation phases (one per loop, each deployed + verifiable):
  P1 state machine + gesture layer (tap/hold/double-tap/flick/pinch)
  behind the CURRENT layout; P2 Watch/Inspect chrome swap (hairline vs
  bottom precision cluster w/ session strip + coarse lane + frame wheel
  + 4-button transport); P3 verdict-on-badge sheet + auto-advance;
  P4 pen-modal annotate rail; P5 polish + two-tap parity audit from the
  spec's table. Keep every capability reachable; deploy per phase so
  Joe can veto early.
- 2026-08-22 (loop, 12am): CLIP-UI V2 PHASE 1 shipped — gesture layer
  live behind the unchanged layout (hold=1/4x peek w/ pill, double-tap
  sides=1s hop, flick=shot nav, tap=play/pause after dbl window; pinch
  deferred to P2). Plus Joe's line-LENGTH slider in the paint pill
  (anchored endpoint, extend along angle, aspect-corrected). Joe should
  live with the gestures before P2 (chrome swap + IG-style overlays).
- 2026-08-22 (12:30am, Joe-directed): SPLIT VERDICT shipped end-to-end
  (Details -> "Split into two shots at playhead"; reader bisects;
  watcher re-derives halves; carryover preserves; test pinned). Joe
  asked for the MECHANISM, not just the patch: AUDIO-TRANSIENT
  SEGMENTATION AID is now a designed goal — extract session audio,
  detect cue-strike onsets (percussive, high-SNR; energy-derivative
  picker, no ML), then (a) >=2 strikes inside one shot window ->
  auto-split at the 2nd strike; (b) zero strikes in a "stroke" window
  -> rearrange evidence; (c) Joe's manual split records = ground truth
  for thresholds. Audio was only ever used for offline VALIDATION;
  the live segmenter is vision-only today. Also tonight: Shot IDs,
  pinch-zoom for paint, length slider, state-aware slo-mo button.
- 2026-08-22 (1am, Joe-directed design): AIM LINE v2 — replace the
  per-shot on/off toggle with ON-DEMAND per-frame analysis: paused
  frame -> "Analyze Aim Line" button -> request {session, aim_at: t}
  rides the corrections channel -> watcher decodes THAT frame, runs
  detect_cue_aim (calib H + sidecar cue pos), appends the segment to
  the export as per-frame aims -> phone polls the summary (~15-45s,
  same accepted pattern as RIFE) and draws the line on that frame
  (tracer style, parallax-inverted). Keep the stored per-shot aim as
  the instant default where it exists; the button refines/overrides at
  any paused moment. Build next loop.
- 2026-08-22 (1am, Joe): AIM DETECTION ACCURACY PASS 2 requested —
  "still slightly off the cue stick path." Attack plan: (1) SUB-PIXEL
  EDGES — the refit centres binary-mask bands (1px quantized); instead
  localize BOTH stick edges on the raw image with sub-pixel gradient
  interpolation and average the two edge lines (kills the lit-side/
  shadow-side asymmetry that biases a mask centroid toward the bright
  edge); (2) fit both edges separately over the tip section, average;
  (3) metric, not vibes: mean perpendicular deviation sampled along the
  visible shaft, measured before/after on >=4 address frames, plus 3x
  zoom renders for Joe. Ship only if the metric improves and the zooms
  look glued. Pairs with the Aim Line v2 on-demand build.
- 2026-08-22 (1:30am, Joe): ACCURACY PASS 2 UPGRADED — "use as much of
  the visible cue stick as possible." Design: full-shaft sub-pixel
  dual-edge sampling, then a STRAIGHTNESS-CONSTRAINED fit: the stick is
  straight in 3D, so solve line params + a linear height profile
  (tip ~40mm -> grip ~300mm) whose radial parallax correction (toward
  the camera, factor h_pt/h_cam) makes the corrected samples maximally
  collinear — the stick's own straightness calibrates out the butt
  parallax that biased the naive full-shaft fit (measured 2.9->5.9px
  regression previously). Deliverables unchanged: mean-perpendicular-
  deviation metric before/after on >=4 frames + 3x zoom renders for
  Joe; the drawn line then hugs the ENTIRE shaft, and the aim ray uses
  the corrected table-plane direction.
- 2026-08-22 (Joe): KEEP BOTH aim modes — the AUTO per-shot line
  (precomputed at export, instant, the existing toggle) AND the
  by-frame "Analyze Aim Line" on any paused frame. Same detector,
  same accuracy pass benefits both; the sheet offers the toggle and
  the button side by side.
- 2026-08-22 (Joe, standing rule): MISS DIAGNOSIS MUST NOT
  OVERSIMPLIFY. Every root-cause claim names the competing physical
  explanations (aim error, unintentional english + squirt, cut-induced
  and english-induced throw, swerve from cue elevation, contact-point
  error, speed) and why they were excluded; magnitudes must be
  physically plausible; the honest answer "can't tell from this data"
  outranks a confident guess (no tip-contact-point, no spin measure,
  30fps blur). The running forensics get an ADVERSARIAL PHYSICS CHECK
  before any of it reaches Joe or ships as analytics.
- 2026-08-22 (loop, 2am): STEP 0 of the miss-analytics build order
  shipped — vision/tablespace.py (true-inch frame from calibration,
  ball-anchored scale, pockets, gates) + tools/audit_calibration.py.
  Two measured design corrections: the ball-position cloud is a LOWER
  BOUND on the bed (first version failed 34/35 sessions), and geometry
  CANNOT settle 8ft-vs-9ft (measured short side 42.5-47.8in across
  sessions) — table size is a configured fact needing Joe's word, after
  which px/in should come from the known bed with the ball as
  cross-check. Archive: 13/35 measurable (15 stubs, 8 scale-implausible
  vs configured 9ft, 5 bad stored transforms). NEXT: Step 1
  analysis/forensic.py (per-shot geometry card + required-vs-actual
  overlays — Joe's Yahoo-Pool instinct), then Step 3 cue-tip tracking
  which REPLACES the planned full-shaft aim work.
- 2026-08-22 (overnight, Joe asleep): worked his stated priority list.
  (1) MISS LABELS + "WHY THIS MISS" OVERLAY shipped — the overlay draws
  the figure the label is made of (cue approach dashed, line to the
  pocket, line the ball actually took, legend) from geometry stored
  with the tag, so "left cut / missed left / overcut" is shown, not
  asserted; identical on both surfaces by construction. (2) GLOBAL 180
  ROTATION (container rotates so video + overlays turn as one; pointer
  and pan inverted where consumed). (3) LIBRARY BACKFILL: 35 sessions
  re-exported with tags. Pattern table today: 47 high-confidence misses
  (122 abstained, 42 excluded by a new CONFIDENCE gate — a 14in "miss"
  means the inferred pocket was wrong, a sub-inch one means the ball
  went in). No bias stands out yet: left cuts 50/50, right cuts 55/45.
  (4) TOP-DOWN SYSTEM REVIEW (Joe's reiteration): tools/system_review.py
  runs L1 architecture (block diagram from real imports, layer contract,
  cycles) -> L2 data spine -> L3 features -> L4 code, and the watchdog
  now treats an L1 finding as outranking everything below. It found and
  I FIXED companion(L3) importing ui(L4) (session_summaries moved to
  vision/); it still reports events<->vision and detector_strategies<->
  vision cycles, whose fix is a core/ package for the shared vocabulary
  (Track/BallClass/TableModel) — deliberately deferred, not hidden.
  STILL OPEN: cue-ball tracking loses the ball ~40% of flight (fresh
  re-analysis reproduced it, so it is NOT stale code; the detector
  finds the ball 45/45 frames in isolation, so the fault is in
  association during full-history runs) — this is the next chunk and it
  gates aim/miss coverage. Also open: aim lines on 66% of stroke shots;
  IG-style UI phase 2; audio-transient segmentation.
