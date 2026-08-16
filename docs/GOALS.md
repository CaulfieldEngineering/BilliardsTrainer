# Goals — measurable, visible, ranked

Joe (2026-08-15): *"Just asking you to 'make it better' will never get
anywhere. We need to start setting certain goals and you work towards
achieving them."* And: *"Be relentless until this software is frontier
leading."* This file is that contract. Every goal has a DEFINITION OF DONE
that is measurable without opinion, and the autonomous work loop
(docs/AUTONOMY.md) picks its next task from here, top priority first.

Progress log at the bottom — every work session appends one line.

---

## G1 — Never show two balls with the same identity  ⏳ shipped, verifying
Joe's words: "we still have two sevens. One of which is actually a 4."
The data layer had uniqueness, but an arbitration LOSER rendered in its
measured colour — a purple 4 misread as 7 painted dark maroon = a second 7
on screen.
**Done when:** a render-audit over the full corpus shows zero frames where
two rendered balls share a number OR wear colours closer than a threshold
while carrying no number. Spot-verified visually on 3 sessions.
**Current:** colour-reassignment of arbitration losers shipped (the 4 now
GETS purple). Corpus render-audit tool not yet built — build it, run it.

## G2 — Red-family confusion (3/7, 4/7) under 5%
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
