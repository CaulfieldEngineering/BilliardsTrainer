# Prior-art research — pool/billiards computer vision

Overnight, read-only survey of five public repos + one conference talk, to decide
what to **adopt vs. skip** before building more of BilliardsTrainer. Clones live
in the gitignored `_refs/` (not committed). Nothing here was run on Joe's table —
all claims are from reading source or each project's own self-reported numbers.

**TL;DR.** None of the five repos solves our actual problem (robust real-world
*overhead* ball detection + make/miss + stats) better than we already do
end-to-end. Where we're behind is exactly one thing — a **trained ball
detector** — and the prior art both confirms the fix (train a small YOLO on a
public Roboflow pool dataset → export ONNX → drop into our existing
`OnnxYoloDetector`) and hands us two high-value **bolt-on features** we don't
have: a **ball-type patch classifier** and a **ghost-ball aim-line / best-shot**
engine. Our calibration, tracking, shot state machine, stats, packaging and
auto-update are genuinely differentiated — not duplicated effort.

---

## At-a-glance comparison

| Repo | Camera / calibration | Ball detection | Tracking | Shot / make-miss logic | License | Maturity |
|---|---|---|---|---|---|---|
| **BilliardsComputerVisionEngine** (fearlessit) | **Auto**: flood-fill felt → edge-scan corners → homography → orientation auto-detect; manual-click fallback | Classical: Canny → contour-merge → radius gate → centroid. **No colour/number ID** | Custom NN association + ring-buffer + majority-vote confirm. Positional IDs only | **None** (motion stillness primitive only; README lists make/miss as *future*) | GPL-3.0 | ★21, Groovy/JVM, 2024, prototype, no tests |
| **pool_coach** (fearlessit) | **None** (YOLO on raw frames) | **YOLO11s, single "ball" class** (trained on Roboflow *only_balls v3*, 6085 imgs). No type/number ID | **ByteTrack** (Ultralytics), `track_buffer:120` | "Shot settled" = motion stops; "pot" = ball *count* drops to 1. **No pocket geometry** | Code MIT / **weights AGPL-3.0** | ★2, Python, 2026-active, runnable demo, dead code present |
| **pool-vision** (Radar3699) | **None** (assumes pre-cropped top-down) | Two CNNs: sliding-window "is-ball" + Inception-V3 patch classifier → {cue, solid, stripe, eight, **neither**} | **None** (per-image) | **None** | MIT | ★5, Python/TF1, 2018, abandoned, weights shipped |
| **Interactive-Pool** (Slorrr) | **None** | Hough circles (≈ our demo path). No type ID | **None** (per-frame) | **None** — "interactive" = cue-stick Hough lines + straight aim-ray to first contact | MIT | ★2, Python, 2023, **broken (won't import/run)** |
| **8-Ball-Pool-Analysis** (brandonabela) | **None** for real tables (it analyses the *Miniclip game*; pockets→bbox crop) | Hough on Canny + BGR pixel-count classifier {solid/stripe/8/cue} | **None** (per-frame) | **Best shot geometry**: ghost-ball + line-circle blocking + **Dijkstra** best-pot. No banks/physics | MIT | ★40, Python, 2020, complete demo, real bugs |
| **PyData talk** (Łukasz Kopeć) | **Side-angle phone video**; HSV felt → Canny → open/close → biggest contour → hull → **k-means line clustering → corner intersections** → temporal smooth → homography to 2:1 | `SimpleBlobDetector`; cue ball by colour | OpenCV ROI tracker (needs known start) | Goal is **table-levelness**, not make/miss: **RDP** to segment tracks at collisions + **polyfit** straight-line skew metric | talk (no code) | PyData conference talk, ~18 min |
| **BilliardsTrainer (us)** | **Auto**: felt HSV → 4 corners → homography, **locked one-shot** + deviation watchdog | Classical Hough+HSV (demo-grade) **+ ONNX-YOLO backend (no torch)**, no model yet | **ByteTrack-style** + constant-velocity | **Motion-energy-gated state machine**: cue strike → travel → pocket approach/vanish → make/miss; manual fallback | MIT | shipping desktop app, CI, auto-update, SQLite stats, tests |

---

## Per-repo analysis

### BilliardsComputerVisionEngine (fearlessit) — ★21, GPL-3.0, Groovy/JVM, 2024

The repo Joe flagged as closest to our setup, and the only other one with a real
**automatic table-calibration** stage. Hobby prototype, honestly labelled. ~1.8k
LOC of Groovy + OpenCV Java bindings; single squashed commit; no tests.

- **Calibration (the valuable half).** `BilliardVisionEngine.findCorners()`
  (`:78-100`) runs every 25th frame on a 400×200 downscale: find a felt pixel by
  **spiralling out from a center-biased offset `xFix=0.57,yFix=0.33`** (to dodge
  the head ball/cue — `VisionAlgorithms.findSimilarPointOnCenterSpiral`), then
  `cv.floodFill` the felt to white with a tunable tolerance (~7.5), take the
  bbox, and **scan each edge for the white region's average extent** to recover
  the diamond-oriented apex corners (`findColorOnLine`). `warpTablePerspective()`
  builds the homography to a 400×200 rectangle; `tableNeedSideReverse()`
  compares edge lengths to auto-pick orientation so the warp isn't mirrored.
  Manual middle/left/right-click corner override as fallback.
- **Ball detection (below our bar).** Canny → `RETR_EXTERNAL` contours →
  **merge contours within 5px Manhattan distance** → keep blobs with
  `10 < radius < 40` → centroid (`detectBalls()` `:102-136`). **No colour, no
  number, no class** — a ball is just an `(x,y)`. Touching balls merge and drop;
  far balls shrink below min radius. Their README admits both failures.
- **Tracking.** Greedy nearest-neighbour into a 75-frame `CappedQueue` per ball,
  gated at `tableWidth/10`; unmatched → push `null`; remove when history all
  `null`. **Display filter = "≥25 frames seen AND ≥25% non-null"** — a temporal
  majority vote. Corners smoothed via **geometric median** over a window.
- **Shot logic.** Essentially none. Only primitives: `isStill()` (no pair of
  recent points farther apart than ~27px — motion as *spatial spread*, not
  velocity) and a still→moving edge that drops a fading breadcrumb at the shot
  origin. `isOngoingTurn()`/"all balls still" exists but is **never called**. No
  pockets, no potting, no physics.
- **Steal (reimplement — GPL, don't copy code):** flood-fill felt → edge-scan
  corner finding (table-agnostic, no clicks); orientation auto-detect;
  geometric-median corner smoothing; **temporal majority-vote detection
  confirmation**; null-history track lifecycle (survives players reaching over);
  spread-based stillness as a secondary shot-gate signal.
- **Skip:** the Canny/contour-merge detector (no ID, weaker than ours), the
  Java net/server layer, AWT rendering, every-25-frame re-detection (we lock).

### pool_coach (fearlessit) — ★2, code MIT / weights AGPL-3.0, Python, 2026-active

The most modern stack (YOLO11 + ByteTrack), and the **single most useful find**
for our model gap — but a thinner system than its README implies: no calibration,
no homography, no per-ball ID, no make/miss. "Shot" = motion settled; "pot" =
ball count → 1.

- **Detector + model provenance (the win).** `model.track(..., tracker=
  "./bytetrack.yaml", persist=True, conf=0.5, iou=0.1)` (`inference.py:25-55`).
  Weights = YOLO11s, **single class `{0:'ball'}`**, trained on **Roboflow
  Universe "only_balls v3" — 6085 images, single class** (`model/README.only_
  balls.v3i.yolov11.txt`; CLAUDE.md:118). Self-reported `mAP50 0.995` on their
  val split (not real webcam footage). Weights are gitignored → "obtain
  separately"/retrain from the public dataset.
- **Tracking.** Ultralytics ByteTrack; `bytetrack.yaml` tweaks
  **`track_buffer:120`** (default 30) to survive break occlusion. But the shipped
  `shot.json` has **66 distinct track IDs for ~10 balls** — heavy ID churn, so
  IDs are *not* stable enough to map to specific balls (why nothing downstream
  tries).
- **Shot logic.** `update_shot_settled_state` (`billiard_ui.py:56-79`): was-moving
  → `BALL_MOVEMENT_THRESHOLD` not exceeded for `SHOT_SETTLE_MIN_FRAMES=25`
  consecutive frames → `shot_completed` rising edge. Speed-Pool game = stopwatch
  ending when averaged ball count ≤ 1.05. Velocity = pixels ÷ hardcoded
  `INIT_FPS=60` (not physical).
- **Steal:** the **training recipe** (only_balls v3 → YOLO → **export ONNX** for
  our backend); ByteTrack `track_buffer:120` + thresholds; the motion-settle
  edge-detector as a cheap fallback gate; the compact per-frame `shot.json`
  schema for replay/analysis.
- **Skip:** no calibration; count-based "pot"; non-physical speed units; the dead
  `position.py`/`collisions.py`/`tracking.py`/`show_shot.py` per its own CLAUDE.md.
- **⚠ License trap:** Ultralytics YOLO + any Ultralytics-trained `.pt` are
  **AGPL-3.0** — linking the runtime or redistributing the `.pt` is copyleft and
  incompatible with a closed desktop release. Our **ONNX-without-torch** path is
  the right instinct: it avoids the AGPL *runtime*; weight provenance still needs
  a license check before we ship any model.

### pool-vision (Radar3699) — ★5, MIT, Python/TF1, 2018 (abandoned)

A two-stage **detect-then-classify** PoC. Assumes already-cropped top-down
images (no calibration). TF1-only (won't load on modern TF), but the *idea* maps
straight onto our ball-ID problem.

- **Pipeline.** (A) a 24×24 "priority" CNN scores ball-vs-not as a cheap
  sliding-window detector → heatmap (`priority.py`); (B) hand-rolled greedy NMS
  (`utils.personalspace`, `utils.py:29-44`) → peaks; (C) each 48×48 patch → a
  frozen **Inception-V3** classifier → softmax over **5 classes**:
  `{black/eight, cue, neither, solids, stripes}` (`classifier.py`,
  `trained_labels.txt`). (Note a label-order bug between the script and the
  labels file — trust `trained_labels.txt`.)
- **Key transferable insight:** classify a *cropped patch* with a small CNN, and
  include an explicit **"neither" reject class** to kill false positives (hands,
  chalk, cue, glare). That reject class is the single most reusable design idea.
- **Steal (pattern, not weights):** the 5-class taxonomy
  `{cue, solid, stripe, eight, neither}` as our **Phase-1 classification target**
  (much easier to train reliably than full 1–15 number ID, and enough for
  solids/stripes stats); feed it our *already-clean bird's-eye patches* (skip
  their slow sliding-window stage); train ourselves and **export ONNX**.
- **Skip:** the 87 MB frozen TF1 graph (won't load in our stack; rig-specific,
  2018), TensorFlow/Keras as a dependency, the sliding-window localiser (we
  localise geometrically already).

### Interactive-Pool (Slorrr) — ★2, MIT, Python, 2023 (broken)

Does **not run** — Cyrillic-char filename breaks the import, function name/arity
mismatches, a truncated core iterator. The *idea* is worth more than the code.

- **What's interesting:** classical **cue-stick detection** — `Canny →
  HoughLinesP → cluster segments by similar angle → keep the largest cluster →
  average to one cue line` (`main.py:27-53`) — plus a geometric **aim-ray to
  first contact**: walk the cue line, find the first ball it passes within
  `2*radius` of, draw a straight ray from there (`main.py:85-94`). Felt-colour
  sampling at the line ends orients tip-vs-butt (`needToReverse`).
- **No physics**: no rail reflections, no banks, no spin despite README claims.
  Per-frame, no tracking.
- **Steal (reimplement clean):** cue-stick detection via angle-clustered Hough
  lines (we have *none* today) — but prefer `cv2.fitLine`/PCA on cue-region edge
  pixels over naive angle-bucketing; ray-circle intersection of the cue line vs.
  tracked ball circles in our warped space (exact, since we have positions+radii).
- **Skip:** the codebase as code (faster to rebuild the 2 ideas), Hough ball
  detection (≈ our demo path), the non-existent "trajectory prediction."

### 8-Ball-Pool-Analysis (brandonabela) — ★40, MIT, Python, 2020

Most-starred of the set, but it analyses **Miniclip's 8 Ball Pool video game**
(clean flat top-down render), not a real camera — so its detection/"calibration"
is worthless for us. Its **shot-geometry engine is the gem**, and it's pure,
portable, MIT geometry.

- **Detection/classification (skip):** Hough on Canny for pockets→bbox crop and
  for balls (`param2=15`, `minRadius=7/maxRadius=13`); classify by **counting
  white/black BGR pixels** inside the disc → {solid/stripe/8/cue}. Works only
  because game balls are flat and glare-free; collapses on real felt. Keep at
  most one weak feature ("fraction of bright pixels inside the disc" as a
  solid-vs-stripe hint).
- **Shot prediction (steal — the headline):** `Logic/Path/`. **Ghost-ball
  contact point** = `move_from_two_points(ball, hole, 2*radius)` — one diameter
  behind the object ball along the ball→pocket line (`vectors.py:17-30`).
  **Blocking checks** via point-to-line distance `line_intercept_circle`
  (`vectors.py:72-85`) for both cue→ghost and ball→pocket segments, plus
  cushion-segment intersection. **Pocket-mouth modelling**: fan 5 aim points
  across each pocket opening with `np.linspace` (`ball_path.py:151-177`).
  **Best-shot selection**: build a graph `cue→ghost→ball→pocket` and run
  **Dijkstra**; impossible shots kept with a `+10000` penalty so it degrades to
  "hit the nearest target ball" (`ball_path.py:53-75`). Straight-shots only — no
  banks, no spin, binary make/miss (no probability).
- **Steal:** the ghost-ball + line-circle-blocking + multi-point-pocket +
  Dijkstra best-pot engine, ported into our warped space (we already have
  tracked ball centres + calibrated pocket positions — it drops in almost
  unchanged), then extended with a make-probability/difficulty score they lack.
- **Skip:** all detection/classification/"calibration" (game-specific), no
  tracking.

### PyData talk — "How I learnt computer vision by playing pool" (Łukasz Kopeć, ~18 min)

Different goal (measure how **un-level** a table is, for tournament fairness),
and a *harder* input than ours: **side-angle phone videos from many different
phones, no fixed rig**. No code published, but the pipeline is well described and
several techniques are directly relevant.

- **Table detection:** work in **HLS/HSV, not RGB** — sample a felt patch, take
  the average, threshold within a range; robust across cameras and time-of-day
  (independently validates our HSV felt approach). Then Canny → morphological
  **opening then closing** → **biggest contour** → **convex hull**.
- **Corners (worth stealing):** the hull is many short line segments — **cluster
  the line equations with k-means into 4 clusters**, take centroids as the 4
  trapezoid sides, intersect them (within the image) to get corners. Then
  **temporally smooth corners** to kill per-frame jitter. Homography to a 2:1
  rectangle.
- **Balls:** `SimpleBlobDetector` for initial positions; cue ball by colour
  similarity (same trick as the table). Then OpenCV ROI tracking, which "is
  pretty great if you know where the objects are to begin with" (struggles on
  ball-ball collisions).
- **Collisions / trajectory (novel for us):** apply **Ramer–Douglas–Peucker
  (RDP / `approxPolyDP`)** to the noisy position sequence to find **sharp turns =
  collisions**, and split the path into straight tracks. Quantify skew by fitting
  a **degree-1 polyfit** to (the first half of) a track, rotating it onto the
  x-axis, and taking the **mean-squared deviation** from straight.
- **Honesty/limits:** ~3× real-time on a laptop CPU; no lens-distortion
  correction (acknowledged); linear-motion assumption ignores spin; weak result
  (~10% higher skew for winners — noisy, low participation).
- **Steal:** k-means line-clustering corners + temporal smoothing (calibration
  robustness); **RDP track segmentation + polyfit straightness** as the basis for
  a "shot quality / table-roll" analysis and a sharper collision detector;
  `SimpleBlobDetector` as a cheap detector cross-check.

---

## Recommendations, ranked by leverage

1. **Train a real ball detector from a public Roboflow pool dataset → ONNX
   (highest leverage; unblocks everything).** `pool_coach` proves the recipe:
   *only_balls v3* (6085 images, single "ball" class) trains a YOLO that
   self-reports `mAP50 0.995`. This is the same kind of asset as the Pool V2 we
   parked — single-class detection is the realistic, reliable target. Train
   YOLO11n/s, **export to ONNX**, drop into our existing `OnnxYoloDetector`. Run
   it on the **rectified bird's-eye** crop (our edge over every repo here — none
   feed YOLO a homography-stabilised image, so balls keep constant size/shape →
   should help both detection and ID stability). Still gated on a free Roboflow
   key + one-time training compute, same as `tools/train_pool_model.py` — extend
   that script to accept either dataset.

2. **Add a ball-type patch classifier as stage 2 (solves the "yellow/black",
   ball-ID problem properly).** YOLO gives clean "ball" boxes; a *small* CNN on
   each clean crop classifies `{cue, solid, stripe, eight, neither}` — the
   pool-vision pattern, minus their slow sliding-window and TF1 baggage. The
   **"neither" reject class** is the key robustness trick. Train tiny, export
   ONNX, layer on top of detection. Far more reliable than HSV-on-raw-frames, and
   it makes solids/stripes stats trustworthy. (Number ID 1–15 is a later phase.)

3. **Ghost-ball aim-line + best-shot overlay (new, high-visibility feature;
   MIT-portable today).** Lift the `8-Ball-Pool-Analysis` `Logic/Path/` geometry
   (ghost-ball, line-circle blocking, multi-point pocket mouth, Dijkstra best
   pot) into our warped space — we already have tracked ball centres + calibrated
   pockets, so it drops in with minimal change. Optionally pair with
   `Interactive-Pool`'s cue-stick detection to anchor the aim line to the real
   cue. This is a marketable capability none of our make/miss work touches, and
   it doesn't depend on solving detection first (works off manual/known
   positions too).

4. **Harden calibration with two cheap, proven robustness tricks.** (a) k-means
   line-clustering → corner-intersection (from the talk) and/or flood-fill-felt →
   edge-scan corners (from BilliardsCVEngine) as a **fallback** when our HSV
   4-corner step is poor; (b) compute the locked corners as a **geometric median
   over a short observation window**, not a single frame. Low risk, improves the
   one part of our stack that's already strong.

5. **Tracking + confirmation tweaks.** Adopt ByteTrack `track_buffer≈120` for
   break-cluster occlusion, and add BilliardsCVEngine's **temporal majority-vote
   confirmation** ("seen ≥N frames AND ≥X% hit-rate" before a detection is
   trusted/drawn) — a cheap, detector-agnostic way to kill flicker that
   complements our render-floor.

6. **RDP + polyfit trajectory analysis (differentiating, later).** Segment each
   ball's path at collisions with `cv2.approxPolyDP`, fit straight lines, measure
   deviation. Gives us sharper collision/pot timing *and* an entirely new
   analytic (table-roll heatmap, shot-quality score) no consumer pool app ships.

---

## Where we're differentiated vs. duplicating

**Genuinely ahead (don't second-guess these):**

- **Calibration + locked homography.** Only BilliardsCVEngine and the talk even
  attempt auto-calibration; the three popular Python repos assume top-down game
  renders or pre-cropped images. Our felt-HSV → 4-corner → locked homography +
  deviation watchdog is the most capable calibration in the set.
- **Shot make/miss state machine.** *No repo here actually detects a pot by
  pocket.* pool_coach infers "settled" + count-to-1; the rest do nothing. Our
  motion-energy-gated cue→travel→pocket-approach/vanish machine is the most
  advanced make/miss logic surveyed.
- **Tracking.** ByteTrack-style + constant-velocity beats "none" (3 repos),
  custom NN (BilliardsCVEngine), and matches pool_coach.
- **Product layer.** SQLite stats, PySide6 desktop UI, CI build, checksum'd
  auto-update, manual-first graceful degradation — **none** of these repos have
  any of it. They're scripts/PoCs; we're a shipping app.

**Duplicating (commodity — stop polishing):**

- Classical **Hough + HSV** ball detection. Everyone has it; it's demo-grade
  everywhere (BilliardsCVEngine's contour version is *worse*). This is the known
  dead end — the trained-detector path (rec #1) is the only real fix.

**Behind / missing (the real gaps, all addressable):**

- A **trained detector** (pool_coach has one; we have the ONNX runway, no model).
- **Ball type/number ID** (pool-vision attempts type; we render measured colour).
- **Aim-line / shot prediction** (8-Ball-Pool-Analysis has the best engine).
- **Cue-stick detection** (Interactive-Pool's one good idea).

**Honest conclusion:** we are *not* duplicating wasted effort. The hard,
differentiating parts (calibration, state machine, stats, packaging) are ours and
ahead. The survey's real payoff is a clear, de-risked detector recipe plus two
self-contained features (type classifier, aim-line) we can add without
rearchitecting.

---

## Three concrete proposals for next steps (surfaced, NOT acted on)

> Decide in the morning — nothing below was implemented.

1. **"Detector sprint."** Extend `tools/train_pool_model.py` to also target the
   *only_balls v3* dataset, train YOLO11n, export ONNX, and — crucially — run it
   on our **rectified** fixtures to get *our own* numbers (not the project's
   self-reported mAP). Decision point: is single-class "ball" detection on the
   warped view good enough to flip auto-detection on by default? Needs a free
   Roboflow key + ~1 GPU-hour (or a slow CPU train).

2. **"Aim-line feature" (no model needed).** Port the MIT ghost-ball + Dijkstra
   best-shot engine into our warped coordinate space as an overlay, driven by
   tracked (or manually placed) ball positions + calibrated pockets. Ships value
   even while detection is still manual, and is a clean, self-contained module.

3. **"Calibration hardening."** Add the k-means-line-cluster / flood-fill corner
   fallback + geometric-median corner smoothing behind a feature flag, and A/B it
   against the current HSV-4-corner path on the recorded fixtures. Low risk, and
   it shores up the part of the stack we most rely on.

---

## Appendix — sources & licenses

| Project | URL | License | Last active |
|---|---|---|---|
| BilliardsComputerVisionEngine | github.com/fearlessit/BilliardsComputerVisionEngine | GPL-3.0 | 2024-11 |
| pool_coach | github.com/fearlessit/pool_coach | code MIT, weights AGPL-3.0 | 2026-05 |
| pool-vision | github.com/Radar3699/pool-vision | MIT | 2018-11 |
| Interactive-Pool | github.com/Slorrr/Interactive-Pool | MIT | 2023-12 |
| 8-Ball-Pool-Analysis | github.com/brandonabela/8-Ball-Pool-Analysis | MIT | 2020-10 |
| PyData talk (Kopeć) | youtu.be/8AhsOAz9RSU | — | conference talk |
| Roboflow *only_balls v3* dataset | Roboflow Universe (key-gated, as Pool V2) | per-project | — |

**License caveats for adoption:** BilliardsCVEngine is **GPL-3.0** → reimplement
ideas, never copy its code. Ultralytics YOLO + `.pt` weights are **AGPL-3.0** →
our ONNX-no-torch runtime avoids the AGPL *package*, but any *weights* we ship
need their own provenance/license check. The three MIT repos
(pool-vision, Interactive-Pool, 8-Ball-Pool-Analysis) are safe to reuse code from
with attribution.
