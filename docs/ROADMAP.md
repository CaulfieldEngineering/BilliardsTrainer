# Billiards Trainer — Detection Roadmap

> North star (Joe): *"I can flip Detection: ON on my laptop, point it at my video,
> see all the balls tracked smoothly, and tune any parameter that's off — without
> anyone telling me to download a model or drop weights anywhere."*

Everything is local + free (no paid AI APIs, no required model downloads). The
default detector is `simple_blob` and it works out of the box. Make/miss
attribution is explicitly **deferred until M1–M6 are solid** (Joe's call).

Milestones are ordered **cue-ball-first**: prove we can hold *one* ball rock-solid
before widening to all 16. Each milestone lists a measurable exit criterion and
the eval clip(s) that prove it. Metrics come from `tools/eval_harness.py` against
the clips cut from `testVideo.MP4` (see `tools/sample_clips.py` /
`docs/datasets-catalog.md`).

Progress legend: `[ ]` not started · `[~]` in progress · `[x]` met (with numbers).

---

## [~] M1 — Table lock
**Goal:** detect the four table corners and lock a stable homography on the first
frame; the bird's-eye view must not warp or jump as the clip plays.

**Exit criteria**
- Corners detected and the table polygon locks within the first ~1s of a clip.
- Homography is stable: re-projected corner positions drift < 3 px across the
  whole clip without manual recalibration.
- Holds across **every** eval clip (idle, break, low-light corner) with no manual
  re-calibration.

**Failure modes to watch:** rail glare read as an edge; cloth pattern/logo
fooling the quad fit; perspective so oblique the far rail is foreshortened.

**Proven by:** all clips in the eval set (a stable-lock check in the harness).

---

## [ ] M2 — Persistent cue-ball tracking  ← Joe's #1 detection goal
**Goal:** track the single white cue ball with a stable, persistent ID across an
entire clip — no flicker, no size pumping, survives brief occlusion (cue stick,
hand, motion blur during a shot). One ball, always white = the cleanest signal
on blue felt and the tightest measurable target.

**Exit criteria**
- Cue ball detected in **> 95%** of frames where it is visible.
- Track-ID switches per clip **< 1**.
- Position jitter when stationary: std-dev **< 2 px**.
- Survives occlusion **< 500 ms** without dropping the track.

**Approach (see P3 work):** dedicated `cue_ball_white` strategy (tight HSV
white-on-felt inside the table polygon, largest connected component) + temporal
median preprocessing to kill sensor noise + IoU tracker with a keep-alive budget
+ EMA smoothing on position **and** radius with outlier rejection.

**Failure modes:** specular highlights on object balls read as white; chalk;
bright rail/diamond inlays; the cue ball resting against a cushion.

**Proven by:** `cue_isolated` + `idle_scatter` clips (stillness stability), and
`break_motion` (occlusion/blur survival).

---

## [ ] M3 — Stable all-ball detection
**Goal:** extend M2's persistence to every visible ball.

**Exit criteria**
- Per-clip **frame-stability** (frames-with-expected-N / total) **> 80%** while
  idle, **> 60%** during active play.
- No phantom balls in pockets or on rails.

**Failure modes:** touching balls merged into one blob; shadowed balls dropped;
cluster near a rail.

**Proven by:** `idle_scatter`, `rail_cluster`, `break_motion`.

---

## [ ] M4 — Ball identification (cue / solids / stripes / 8-ball)
**Goal:** now that a track can be held, classify what each track is.

**Exit criteria**
- Classifier **> 80%** accuracy on a hand-labeled subset of frames.

**Failure modes:** white glare turning a solid into "stripe"; low-light hue
collapse; the 8-ball vs dark solids under shadow.

**Proven by:** labeled frames sampled across all clips.

---

## [ ] M5 — Pocket detection
**Goal:** detect when a ball enters a pocket.

**Exit criteria**
- **≥ 90%** recall of real pocket events, **≤ 5%** false-positive rate.

**Failure modes:** a ball passing in front of a pocket; a ball settling on the
lip then staying up.

**Proven by:** clips containing made balls (to be cut/labeled for M5).

---

## [ ] M6 — Shot (cue-strike) detection
**Goal:** detect cue-strike events.

**Exit criteria**
- **≥ 90%** recall on cue strikes.

**Failure modes:** practice strokes without contact; the cue crossing the cue
ball without striking.

**Proven by:** clips containing shots.

---

## [ ] M7 — Make/miss attribution  (deferred until M1–M6 are solid)
**Goal:** combine M5 + M6 to attribute pocket events to shots.

**Exit criteria**
- Make/miss labeled correctly on a hand-verified set of shots.

---

## Beyond M7
Practice modes, drills, stats, training fundamentals. The Practice/Drill tabs are
disabled in the UI until this point (see `live_page` mode strip).

---

### How we hold the line
- Every detector change is re-measured through `tools/eval_harness.py`; numbers
  (before/after) go into `docs/eval/`.
- "Shipped" means verified against the **frozen** PyInstaller build (or a
  frozen-simulating test), not just a dev run.
- Reliability over coverage: no fake detections to inflate a number.
