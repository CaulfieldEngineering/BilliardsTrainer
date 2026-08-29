# BACKLOG — the one work queue

**This is THE queue.** Autonomous work sessions, M1 push sessions, and
watchdog recoveries take the TOPMOST unblocked item and do one bounded
chunk. `docs/GOALS.md` is the append-only LOG of what happened; this
file is what's NEXT. Reorder here, never fork a second list (L1).
An agent that "finds nothing to work on" hasn't read Tier 4.

Ground rules for every item: ARCHITECTURE.md checklist first; suite
green before commit; demolition chunks DELETE the superseded code in
the same commit; one GOALS line per chunk landed; heavy GPU jobs are
presence/recording-guarded, one at a time.

---

## TIER 0 — THE ENGINE RELIABILITY CAMPAIGN (Joe, 2026-08-28)

**Everything else waits.** Joe: "just get this engine reliability
going... none of my other features matter if we can't get this right...
keep churning, ideally 24/7... you have autonomy to keep working at
your own direction so long as you perpetually use your LLM Vision to
corroborate the results of the app's analysis and animations."

Loop (repeat forever until the bar is met): pick the top disagreement
→ instrument it → fix → re-measure the bench (~6 min) → **WATCH the
frames and compare** → log the agreement score → next. Heavy GPU work
(marathon, library, retrains) runs overnight. New Joe requests during
the campaign are APPENDED to Tier 3, never built.

BAR: every real shot detected, windowed at its strike, named,
classified (stroke vs setup), and scored correctly — verified by
Claude's vision, not by metrics.

CAPABILITY LADDER (Joe, 2026-08-28: "break it up by clip yes but also
by feature/requirement"). Rungs are ordered so each depends only on
the ones below it; every rung is measured by tools/scorecard.py and
shown in the phone's STATUS view. A rung is DONE on a clip only when
its gate holds AND a vision watch agrees.
  R0 GEOMETRY - table + POCKET DEAD ZONES (Joe's design): pockets are
     explicit zones; anything resting in a zone is never "on the
     table"; a pot = entered a zone and did not come back out.
     Gate: zero furniture tracks; zone overlay matches the video.
     STATUS: partially done via ad-hoc rules - dead zones replace them.
  R1 CUE BALL - found, named, and continuously tracked; trail covers
     the real path from the strike. Gate: cue correctly named >=99% of
     frames, one cue track per shot, no phantom cue.
     NOW: 99.7% on the HONEST metric (round 11 - live sighting
     required). Residual failures are windows where the label rides a
     coast while the real ball is seen-but-unnamed: an R3 symptom.
     PREVIOUSLY: the metric read 100% but it was WEAK - it only checks that
     exactly one track claims "cue", not that the label sits on the
     real ball. Evidence frame 154.8s shows the cue label on EMPTY
     FELT beside an unnamed white ball. R1 is NOT passing until the
     metric checks correctness (label within a ball radius of a live
     detection) - fix the metric first, then the failure it exposes.
  R2 SHOT EVENTS - every stroke found and windowed AT the strike; no
     stroke invented during hand setup. Gate: 10/10 found, 0 fake.
     NOW: 10/10 FOUND (round 10: engine writes hand context into the
     dense stream, so setup can be told from strokes); 3 fake remain.
  R3 ROUND 14 (tried + reverted): "persistence" jaw filter - refuse a
     dim near-pocket read only if a detection sat at the same spot last
     frame. FAILED twice: (a) it read self._last_detections, which ONLY
     the live path writes, so offline it was inert (scores identical to
     deleting the filter); (b) with real history it also killed balls
     RESTING near a pocket - they too are at the same spot every frame -
     and shots found went 10 -> 0. Reverted. LESSON: "has not moved"
     cannot separate leather from ball; both sit still. NEXT: gate on
     ARRIVAL - accept a dim near-pocket read only when a track was
     travelling toward that pocket in the preceding ~0.5s (needs the
     engine's own tracker passed into prepare_detections; the pipeline's
     tracker is empty offline).
  R3 ROUND 13 (tried + reverted): deleting the jaw confidence filter
     recovered ~35% more moving-ball samples (real pots score 0.49-0.74
     at 16-28px from the pocket and were being discarded) BUT the pocket
     leather returned as a phantom "8" in 962 frames (was 18), dropping
     shots 10->8 and calls 8->6. Net negative; reverted. NEXT ATTEMPT:
     make the jaw filter CONDITIONAL - accept a dim read near a pocket
     when a track is moving toward that pocket (a ball arriving), refuse
     it when nothing is approaching (static leather). Also fix the
     furniture-by-time rule so a re-born leather track cannot dodge it.
  R3 ROUND 22 (corpus started): tools/mine_ballid.py mines the
     ENGINE'S OWN failure frames (moving-but-unnamed, or a number not in
     the session inventory), crops every detection, and writes labels
     into the existing TrainingStore YOLO layout - no new format.
     First batch: 87 crops from 12 failure moments, 57 labelled BY
     VISION (RULE 0) -> cue 11, 1:9, 2:9, 3:10, 4:8, 9:10. One frame
     DROPPED whole rather than labelled partially (two balls ambiguous
     even to Claude; a half-labelled frame teaches "real ball =
     background"). GAP: zero examples of the orange 5 - it is absent
     from these frames and is one of the never-read balls.
     NEXT: (a) mine frames where the 5 is on the table + a second
     session for variety; (b) target >=40 examples per ball; (c) train
     c8 on the combined corpus (tools/finetune_ballid.py); (d) gate
     same-batch vs c5 per MEASUREMENT_CORE M2 - promote only if the
     bench naming rises AND held-out sessions do not regress.
  R3 ROUND 21 (CADENCE RULED OUT; ROOT CAUSE IS THE MODEL): ran the
     bench with IDENT_EVERY=1 (6x the identifier passes). Naming was
     BIT-IDENTICAL: 75.7% named, 150 seen-unnamed, 636 estimate-only.
     Direct probe of the 170.6 shot: the finder finds 7 balls, the
     identifier reads 4 numbers, and one is a "7" - not on this table.
     Cropped and LOOKED (RULE 0): blue->2 correct, red->3 correct,
     yellow->9, and the "7" is the PURPLE 4. Balls 1, 4, 5 get no read
     at all. Reverted to IDENT_EVERY=6 (identical output, 20% slower).
     => R3 is a MODEL problem. Confusion pairs seen so far: 4->7 (dark
     colours), cue->9 (round 8), 3<->5 (Joe's original report), and
     non-reads of 1/4/5. NEXT: build the c8 retrain corpus by mining
     these exact failures from the dense sidecars (crop at the frames
     where a ball is seen-but-unnamed or read as a number not in the
     session inventory), label, train, gate same-batch vs c5.
  R3 ROUND 20 (built + reverted; DIAGNOSIS CORRECTED): implemented
     identity stitching (a fresh detection continuing a dead track's
     motion inherits its name; corridor test from the last REAL
     sighting, since coasting drifts x/y and death zeroes v). Bench:
     ZERO change (75.7% named, 8/10, 2/4) - reverted as unproven.
     Looked at what reaches the pocket on the failing pots:
       51.5s  id2 num=2 reaches 5px WITH REAL SIGHTINGS - no fragment,
              so round 19's "anonymous fragment" explanation was WRONG
       170.6s the ball reaching the pocket is num=-1 for the whole shot
       31.7s  attributed to ball 4; the truth is ball 9 (misread)
     => The gap is RECOGNITION, not track continuity: the app follows
     balls but cannot read WHICH ball while they are in play.
     NEXT (measurement, not a guess): the identifier runs every
     IDENT_EVERY=6 frames - only a handful of chances during a fast
     shot. Re-measure the bench with IDENT_EVERY=1 and compare naming;
     if cadence is the limit, buy it back (fp16 identifier / crop
     batching). If not, it is the model and the c8 retrain is the work.
  R4 ROUND 19 (built the spec, reverted - DEPENDENCY FOUND): the
     three-clause rule (in-zone AND vanished AND no hand) got ALL SIX
     misses right - incl. the 0.47R hand-catch and both jaw rattles -
     but credited only 1 of 5 pots (6/10 vs baseline 8). Traced per
     ball: on 4 of 5 pots the NAMED track stops short (the 2-ball's
     freezes at 1.8R, all ESTIMATE rows) and the final approach belongs
     to a NEW anonymous fragment, which _series also drops for being
     short (<250px span). So "did ball N reach the pocket" is false
     even though a ball did. The baseline scores better only by being
     identity-blind (anything vanishing near a pocket = pot), which is
     also why it mis-attributes half its pots.
     ORDER CORRECTED: R4 CANNOT close before R3. Naming is the FLOOR
     under the outcome rules, not polish above them. NEXT: identity
     continuity into the pocket - stitch a dying named track to the
     fragment that continues its motion (same direction/speed, <0.4s
     gap, <2 ball diameters), so the ball keeps its name to the drop.
     The three-clause rule is DONE and waiting; re-apply it after.
  R0/R4 ROUND 18 (built + reverted; produced the SPEC): implemented
     Joe's dead zones (enter a zone, do not come back out = potted;
     replacing the death-point/path-tail/bed-exit/lip-hover heuristics).
     Alone: calls 8->6 (the jaw filter still hides the ball entering).
     With the arrival gate: 5/10, attribution 0/4 - the entering track
     is DIM and unnamed, and its coast freezes ~1.8 radii short of the
     mouth (measured on the 2-ball pot: id7 sits at 44px = 1.8R, all
     ESTIMATE rows). MEASURED SEPARATION (closest approach in pocket
     radii, pots vs misses): 0.15 POT, 0.29 POT, 0.47 MISS (hand catch),
     0.54 POT, 0.82 POT, 1.35 miss, 1.48 POT, 1.63 miss, 1.74 miss,
     1.75 miss. NO distance threshold separates them - which is why
     every attempt traded one shot for another.
     SPEC for the next attempt (all three parts, none optional):
       potted = entered zone (<=1.5R) AND stopped being seen AND NOT
       hand-adjacent at the vanish (carried-ids are already in the
       stream since round 10; the 0.47R false case is Joe's hand catch).
     Keep ZONE_R ~1.5R; the arrival gate stays OFF until this lands.
  R0/R4 ROUND 17 (built + reverted, but it SETTLES the design): the
     arrival-gated jaw filter works as designed - engine tracker piped
     into prepare_detections via MotionTracker.live(), pockets marked
     "ball incoming" for 0.6s - and coverage improved (naming 75.7->76.0,
     estimates 19.6->18.6). But CALLS DROPPED 8->7: fixed 154.2, broke
     101.0 (false pot) and 130.2 (lost pot). CAUSE: seeing the ball as it
     drops keeps its track ALIVE INSIDE the pocket, so the outcome stage
     reads "resting" instead of "vanished into the pocket". Pot credit is
     currently built on LOSING sight of the ball - so better sight makes
     judgement worse. CONCLUSION: implement Joe's POCKET DEAD ZONES (R0)
     FIRST - a ball that enters a zone and does not come back out is
     potted regardless of whether it is still detected in the basket.
     Then the arrival gate is safe to re-enable (the branch is small:
     tracker.live(), hot_pockets param, HOT_S).
  R2 ROUND 16 (kept): fake strokes 5->1, unexplained 10->0, shots hold
     10/10. Two measured rules: (a) a stroke means the CUE MOVED - all
     10 real strokes move it, tossed-in balls never do (and hand-
     adjacency cannot catch a tossed ball, since no hand touches it
     once released); (b) a stroke means the cue WENT somewhere - real
     strokes >=263px, addressing/nudging 8-29px, gate at 150px.
     REMAINING fake: the 13.1s setup where Joe pushes the cue 210px
     with the stick - needs the STICK recognised (it is tracked as an
     unnamed mover); next R2 attempt.
  R3 ROUND 15 (kept): looked at the open-felt losses per RULE 0 - they
     were NOT missed balls. Crops show a ball in the glove, two empty
     pocket corners, and one prediction off the table on the FLOOR. The
     tracker was coasting into nowhere and counting it as ball motion.
     Coasts now stop at the bed edge: naming 72.1->75.7%, guessed
     motion 24.2->19.6%, total moving samples 3598->3241. COST: fake
     strokes 3->5 - with fewer ghost tracks diluting it, CARRIED_SETUP
     (0.5) no longer catches Joe's hand-setup episodes. NEXT: retune
     that threshold against the three known setup windows.
  R3 RE-AIMED AGAIN (2026-08-28, Joe caught a false premise): the
     "motion blur" story was WRONG. Cropped the pixels at last: the
     ball the engine loses is SHARP, sitting in the pocket jaw, and the
     lost samples measure a median 202 px/s (<0.5px smear at 1/500s).
     Joe fixed blur at the camera before these clips were shot. Real
     split of the coasted samples: 55% within 3 pocket radii (leather
     occlusion drops confidence, then the round-1 jaw filter discards
     them - the arrival-gated filter is the fix, now for the RIGHT
     reason) and 45% SLOW balls in open play (cause unknown - measure
     it, do not guess). Blur recovery is OFF the plan.
  R3 REDIRECTED (round 12, tools/naming_audit.py): the loss is NOT
     recognition. Of all moving-ball samples: 72.1% named, only 3.8%
     SEEN-but-unnamed, and 24.0% never seen at all - coasted estimates
     where the detector lost the ball to motion blur. You cannot name
     what you cannot see, so the FIRST R3 job is detection coverage on
     fast balls (targeted blur recovery - currently OFF after a 2026-08-23
     sweep failed its gates; re-attempt it aimed at moving tracks only,
     measured by the naming audit), and the c8 retrain is demoted to the
     3.8% tail. Worst offenders are track ids, not ball numbers - they
     are balls that never got a name at all during their flight.
  R3 NOTE (round 9, FAILED + reverted 2026-08-28): a session-inventory
     prior (reject numbers read far less often than the rest) made every
     line WORSE - it cannot separate a phantom from a real ball the
     detector rarely sees (the orange 5 has ~47 tracked states all
     session). The invented "11" is a RECOGNITION failure, not a
     counting one: it belongs to the c8 retrain, not a filter. Next R3
     attack: measure WHERE naming is lost (never-read vs misread vs
     outvoted) per ball, then mine crops for those exact cases.
  R3 OBJECT BALLS - every moving ball tracked AND correctly numbered;
     no invented numbers. Gate: >=95% of moving time named, zero
     invented. NOW: 74.5% named, "11" invented in 318 frames.
  R4 MAKE/MISS - outcomes correct incl. jaw rattles and hand catches,
     with EVERY pot attributed to the right ball number. Gate: 10/10
     calls AND pot attribution 4/4. NOW: 8/10 calls, 2/4 attributed.
     (Joe, 2026-08-28: "There shouldn't be 'pots by unnamed balls' by
     the satisfaction of object balls right?" - correct: an unnamed
     pot is a symptom of R3 failing, never an R4 requirement. The
     unnamed-pot code path in measure/shots.py is SCAFFOLDING and is
     DELETED when R3 passes - see ARCHITECTURE.md demolition ledger.)
  R5 LIVE PARITY - the same box on the camera; a live session scores
     the same as its own re-process; old tracker DELETED.
Run the ladder clip by clip: bench first (all rungs), then a cold
clip, then the library.

PHASES + GATES (Joe, 2026-08-28: "We can't just say 'see you in two
weeks'... break this into intermediate phases and milestones to prove
we're on the right track"). Gate numbers come from
`python tools/scorecard.py --publish`, scored against
docs/bench_truth.json (established by eye, NEVER edited to match the
app) and published to the phone's pinned STATUS view.
  P1 BENCH PERFECT (target 2026-08-30): 10/10 found, 10/10 outcomes,
     0 fake strokes in setup windows, 0 unexplained episodes.
     Baseline at plan time: 9/10, 8/10, 4 fake, 8 unexplained.
  P2 COLD CLIP (P1 + 1 day): watch an UNTOUCHED session by eye first,
     write its truth file, then run the engine cold: >=9/10 both lines,
     0 fake. Guards against tuning to one clip.
  P3 NAMES (P2 + 2-3 days): every moving ball correctly named >=95% of
     its moving time on both clips; ZERO invented ball numbers.
     (Today ~20% of real motion is unnamed; a nonexistent "11" appears.)
  P4 LIBRARY REBUILD (P3 + 1-2 days): every session regenerated from
     the box (not merged); 3 random sessions score >=9/10 on fresh
     vision watches; shot counts match.
  P5 LIVE (P4 + 3-5 days): a live-played session scores the same as
     its own re-process; the old tracker is DELETED.
REPORTING RULES: scorecard republished after EVERY round (wins and
regressions alike); a journal entry per round; a missed phase date is
posted ON the date with the reason - never silent slippage.

CAMPAIGN STATE (update every round; newest first):
- Bench = **session-20260824-220247** (pinned). Ground truth: 10 real
  strokes. Dossier: docs/design/bench-220247-watch-2026-08-28.md.
- **Round 8 (biggest win so far)**: velocity-aware association gate.
  The overlay proved the tracker manufactured 8 phantom balls per
  frame down a struck ball's path (model detected 6, sidecar carried
  14) and stranded the "cue" label on the first phantom. Fast balls
  now stay ONE track. Bench: tracks 14->7 at t=170.5, cue correctly
  named at its real position, physics gate 0.49 -> 0.14/1k.
- Rounds 1-7: jaw-phantom filter; path-tail + bed-exit pocket credit;
  chain-continuity (rebirth != resting); lip-hover drops;
  unnamed movers seen but pot-credit gated; pocket furniture dies by
  time; engine made hand/stick-aware offline (refresh_foreign).
- OPEN, NEXT (round 9+), all bench-verified by vision each round:
  (a0) DONE round 11: dense rows carry an 8th field - sighting vs
      coasted ESTIMATE (additive; every reader indexing 0..6 is
      unaffected, trails keep their blur coverage). R1's metric now
      demands a live sighting: honest cue tracking is 99.7%, and the
      debug overlay tags estimates 'est' so evidence frames never imply
      the app saw what it inferred.
  (a) episode layer now over-fires (21 episodes) and mints FALSE POTS
      for named balls that lose/regain names mid-flight - e.g. pots
      credited to "4" at 13.1s/29.5s and to a nonexistent "11".
      Outcome agreement is 6/7 on matched strokes but only 7 of 10
      strokes now match cleanly. Fix the naming churn before trusting
      any outcome number again.
  (b) a nonexistent ball number ("11") appears - identifier misread
      class; feeds the c8 retrain corpus.
  (c) Joe's POCKET DEAD ZONES design (his suggestion, 2026-08-28):
      make pockets explicit zones - a ball is potted when it ENTERS a
      zone and does not return quickly, and anything sitting in a
      zone is NEVER "on the table". This should REPLACE the ad-hoc
      lip-hover/bed-exit/furniture heuristics with one clean rule.
  (d) hand-context in engine output -> unlocks unnamed-ball pots,
      kills the glove-carry false pot, labels setup vs stroke.
  (e) window keying at the strike; ball-4 detection dropout at rest;
      the 9 never identified; aim-line artifacts.
  (f) THEN: other clips, marathon + library re-run, regeneration
      (shots.json written wholesale from the box), merge-machinery
      deletion, M3/M4 live promotion.
- Tools for every round: `tools/debug_overlay.py` (app beliefs drawn
  on the real video - the proof tool) and `tools/journal.py` (plain-
  language entry + images to the phone's Dev Journal).

## Tier 0b — Standing automatics (do these the moment they're armed)

- [ ] **Marathon 4 landing** (armed ~2026-08-28 00:30): read
  `scratchpad/m1_marathon4.log` FULL-GATE + REMERGE lines. Green ⇒
  update the measurement log in MEASUREMENT_CORE.md, bump the session
  stamp, write the What's New entry (accuracy chain: notify Joe).
  Red ⇒ diagnose against the slice run (0.25/1k), do NOT roll out.
- [ ] **Library dense rollout** (blocked on gate green): 16 remaining
  sessions through `measure/engine.reprocess` + `trails_merge
  .merge_into_session`, oldest-first, one at a time, guarded. Each
  session: gate slice → merge → stamp → GOALS line. Overnight work.
- [ ] **c8 identifier gate** (blocked on flip-mining, Tier 2): train →
  same-batch scorecard vs c5 → promote only through the gate.

## Tier 1 — Demolition (default work; order = payoff/risk; ledger: ARCHITECTURE.md §5)

1. [ ] **Shot identity** — stamp Joe's `stem@start-second` id into
   every record; kill the four time-proximity joins.
   a. `analysis_cache.py` add_shot/add_stroke/append_correction/
      append_action write `id`; SidecarReader._shot_for joins by id,
      time-tolerance demoted to legacy fallback (`:136`).
   b. controller stroke join (`controller.py:580` 25s wall-clock) and
      cached_shots dual-clock "key" (`:308`) → id.
   c. UI joins: live_page on_stroke_measured ±6s (`:1542`),
      stay_down (`:155`) → id.
   d. DELETE the tolerance constants + dual-clock key contract.
   Accept: a correction lands on the right shot with clocks skewed 10s.
2. [ ] **One timebase** — core-owned recording clock; sidecar meta
   carries t0; DELETE the four reconciliations (SidecarWriter._t0
   `analysis_cache.py:76`, controller rebases `:1766` `:1784`,
   shots_export.session_time_offset `:247`).
   Accept: session_time_offset returns ~0 unused; exports unchanged
   on a reference session (byte-diff shots.json trails windows).
3. [ ] **Clock snapshot** — core.clock {remaining,total,fraction,
   phase,enabled}; packet stops synthesizing (`controller.py:1516`
   reaches into `_run_seconds`); ring (`live_page.py:1434`) and bar
   read the same dict. DELETE per-widget denominators/status logic.
   Accept: break countdown renders identically in ring + bar.
4. [ ] **One reader** — `companion/server.py:47` read_shots calls the
   SidecarReader merge; DELETE its private ±0.2s rule.
   Accept: desktop and phone show identical outcomes on a session
   with corrections (test with a review-ranked correction).
5. [ ] **Geometry in meta** — controller writes table/H/corners at
   sidecar open (`controller.py:643` writes fps only); DELETE both 3s
   re-warmup derivations (`shots_export.py:28`, engine warmup stays
   for legacy files only).
   Accept: export of a NEW session does zero pipeline warmups.
6. [ ] **One shot log** — core publishes the reconciled shot list;
   timeline/list/DB-scoreboard/sidebar-count bind it. DELETE the four
   private copies incl. cross-widget `_shots` mutation
   (`live_page.py:1603`) and `session_summaries._sidecar_shot_count`
   private filter (`:35`).
   Accept: a correction updates every surface from one publish.
7. [ ] **Tracker demolition** (blocked on M4 gate): swap core's
   MotionTracker to authoritative; DELETE BallTracker
   (`vision/tracking.py:144`), vacancy/ghost bookkeeping
   (`pipeline.py:561`), trail-fade triplication (`pipeline.py:114`).
   The divergence counters in the health line are the evidence file.

## Tier 2 — Measurement quality (interleave with Tier 1 when GPU free)

- [ ] **The one Input→Output box** (Joe, 2026-08-28 — supersedes the
  "offline outcome judge" framing; NO more replay-only processing):
  shot detection + outcome judgment become stages INSIDE the
  measurement engine, one code path consumed by BOTH feeds. A clip's
  shots.json is simply the box's output for that video — no merge, no
  graft (that machinery gets DELETED on landing, per §5). Steps:
  (a) port the shot detector's gates onto dense tracks inside
  measure/ (the veto logic already reads the same tracks);
  (b) outcome stage: pocketed-into-which-pocket from 33ms tracks +
  pocket regions (pocket localization feeds this);
  (c) validate against Joe's review verdicts
  (`_eval/review_scoreboard.json` attachment rules) — the 3-ball
  false make is test case #1;
  (d) full-regeneration export: shots.json written wholesale from the
  box's output; delete trails_merge/arbitrate scaffolding;
  (e) M3/M4 then promote the SAME box onto the live feed.
  Prereqs: Tier 1 items 1-2 (shot ids + one timebase) so history
  rewrites can't mis-attach.
  PROGRESS 2026-08-28: stages 1-2 built + validated (measure/shots.py;
  the 3-ball case judges MISS); on-demand UI re-measure shipped
  (sidebar right-click → measure/job.py: engine→gate→merge, progress
  % in the status bar, deterministic, rules_v stamped, own process,
  presence-pause skipped on click). NEXT: hand-context + table/pocket
  geometry in engine output; box-vs-review-verdicts across the 13
  dense sessions.
- [ ] **3/5 flip mining**: scan dense sidecars for 3↔5 number changes
  on resting tracks; harvest crops at those timestamps; label; add to
  c8 training set. CPU-light scan; crop decode when GPU/disk idle.
- [ ] **M3 shadow sidecar**: plumb capture timestamps through
  `_on_detections_ready → ingest` (today borrows last process t);
  write `.analysis.shadow.jsonl`; exit criteria in MEASUREMENT_CORE §M3.
- [ ] **Pocket visual localization** (ladder rung 2): locate real
  pocket mouths vs geometric marks; feeds pot-credit gates.
- [~] **Engine-version stamp**: writing DONE (meta rules_v=2,
  2026-08-28); REMAINING: gate/scorecard refuses rules_v mismatches. Also: Aug-23 sessions (185550/191319/194542) gate red on
  overlaps — investigate EOS AF-rectangle contamination; they stay
  honest-sparse until then.
- [ ] **Shot recall audit**: false-negative hunt on dense output vs
  audio witness; extends the G5 precision work to recall.

## Tier 3 — Product (only when Tiers 0-2 idle, or Joe asks)

- [ ] iOS shot-clock + ball-tray replay overlay polish (data ships).
- [ ] Per-shot clips (design approved; gated on dense rollout).
- [ ] Overlay-baked video exports; data-saver 480p tier; desktop
  Details parity. Full list: FEATURES.md (do not duplicate here).
- SHELVED (do not resume autonomously): clips-player IG redesign.

## Tier 4 — Never idle (when everything above is blocked)

- Run the weekly full state-opinion audit (map style,
  `docs/design/state-opinion-map-2026-08-27.md`); update the §5
  scoreboard. Growth = incident.
- Write the missing pinning test for the NEXT demolition item (each
  merge needs its acceptance test before the merge).
- Tighten one debt item from MEASUREMENT_CORE.md §3 with a measured
  before/after.
- Re-verify guards: presence-pause, idle throttle, recording aborts
  (tools/health_check.py --quick).

---
*Maintenance: completed items move to a GOALS log line and are deleted
from here. New work enters at the tier its risk deserves, never at
the top by recency. If this file and GOALS disagree about status,
GOALS (the log) wins and this file gets corrected.*
