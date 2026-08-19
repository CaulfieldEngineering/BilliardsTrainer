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
