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

## Tier 0 — Standing automatics (do these the moment they're armed)

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

- [ ] **3/5 flip mining**: scan dense sidecars for 3↔5 number changes
  on resting tracks; harvest crops at those timestamps; label; add to
  c8 training set. CPU-light scan; crop decode when GPU/disk idle.
- [ ] **M3 shadow sidecar**: plumb capture timestamps through
  `_on_detections_ready → ingest` (today borrows last process t);
  write `.analysis.shadow.jsonl`; exit criteria in MEASUREMENT_CORE §M3.
- [ ] **Pocket visual localization** (ladder rung 2): locate real
  pocket mouths vs geometric marks; feeds pot-credit gates.
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
