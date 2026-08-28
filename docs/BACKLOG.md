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
     frames, one cue track per shot, no phantom cue.  NOW: 100.0% ✅
  R2 SHOT EVENTS - every stroke found and windowed AT the strike; no
     stroke invented during hand setup. Gate: 10/10 found, 0 fake.
     NOW: 9/10 found, 4 fake.
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
