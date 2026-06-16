# Detection baseline — v0.2.16 (current default detector)

**Date:** 2026-06-16 · **Detector:** classical Hough + HSV (the shipped default;
no ONNX model present) · **Eval set:** 14 clips (idle / shot / spread) sampled
from Joe's real 40-min table video (`testVideo.MP4`, 1920×1080@30, blue felt,
low side-angle, window glare, people in frame).

Run it yourself: `python tools/eval_harness.py --timestamp baseline --out docs/eval/baseline`
(browse `docs/eval/baseline/report.html` for the per-clip 3-up visual diffs).

> This is the *measure-before-you-change* baseline. The numbers are bad on
> purpose — they quantify the gap Phase 1 has to close. No detection logic was
> changed this turn.

## Headline numbers (aggregate over labelled frames)

| metric | value | read |
|---|---|---|
| **table calibration rate** | **~100%** | the pipeline almost always locks *a* table region, even on blue felt |
| **ball detection recall** | **~2.6%** | it finds **~1 in 40** of the balls actually on the table |
| **ball detection precision** | **~26%** | ~3 of every 4 things it does detect are not balls |
| **F1** | **~4.7%** | |
| **AP@dist** | **~0.008** | center-distance average precision (not IoU mAP — see README) |
| **ID accuracy** | ~70% (tiny sample) | of the rare correct localizations; not meaningful at this recall |
| **false-positive shots/min (idle)** | **0.0** | static tables produced **zero** phantom shots — the v0.2.11+ gates hold |

**Bottom line:** on Joe's real table the classical detector is effectively
**non-functional for ball detection** — recall ~2.6%. It reliably finds the
table and reliably does *not* hallucinate shots on a still table, but it does not
see the balls. This is the core gap.

## Per-clip (the hand-labelled idle clips carry the detection numbers)

| clip | calib | GT balls | precision | recall | FP shots/min |
|---|---|---|---|---|---|
| `idle_01` (7 balls, cue in shot, side angle) | 100% | 7 | ~53% | ~5.2% | 0 |
| `idle_02` (7 balls, clean static — best GT) | 100% | 7 | **0%** | **0%** | 0 |
| `idle_03` (near-empty, 2 balls + cue stick) | 100% | 2 | ~7% | ~2.3% | 0 |
| 9 × `active_*` + 2 × `shot_*` (unlabelled) | 100% | — | — | — | 2 shots total, all on active clips |

`idle_02` is the highest-confidence label set and the detector found **0 of 7
balls** across all 1200 frames (220 spurious detections instead) — a clean,
damning data point. The 2 shots that fired on active clips are unverified (no
shot ground truth); idle clips, where the truth is unambiguous, produced **zero**
false shots.

### Caveats (so the numbers aren't over-read)

- Ground truth was hand-labelled by the agent from zoomed keyframes
  (~ball-radius precision), with a generous match tolerance (2.5% of width). Good
  enough to size the gap, not a published benchmark.
- Only the 3 idle (static) clips have position GT; active/shot clips contribute
  calibration + shot-count + the visual report, not P/R.
- "calibration rate ~100%" means the table *locked*, not that it locked the
  *correct* region — the blue felt sits at the edge of the default (green-tuned)
  felt HSV range, so the lock may be loose. The near-zero recall is consistent
  with a mediocre warp + a detector tuned for green felt.

## Roboflow "Pool V2" dataset — assessment (downloaded this turn)

Joe provided a Roboflow key to fold the **Pool V2** dataset in as ready-made
ground truth. Downloaded (751 images, CC BY 4.0) and inspected — **it is not what
the project page implies, and is not usable as overhead-table ball GT:**

- **~1.1 boxes per image** (a pool table has 10–16 balls) — it does not annotate
  all balls on a table.
- **70% of boxes cover >30% of the image; only 5% are ball-sized.** Visual
  inspection confirms the images are **close-up photos of individual balls**
  (one ball filling much of the frame), plus a few rack shots — **not overhead /
  wide table footage.**
- The 4 class names in `data.yaml` are **corrupted placeholder text**
  (`'-'`, `'Roboflow is an end-to-end…'`), so the class labels are unusable.

**Implication:** Pool V2 can't be ground truth for our eval — our pipeline needs
a full table in frame to calibrate, which these close-ups don't provide — and it
is a poor training match for Joe's small, foreshortened, wide-view balls (domain
mismatch: zoomed studio-ish ball crops vs. a side-angle room camera). The
v0.2.16 "train a ball detector from Pool V2" plan should be reconsidered in light
of this. The dataset + `tools/fetch_pool_v2.py` are kept locally (gitignored) for
reference; **the real baseline is Joe's testVideo, above.**

> Attribution (CC BY 4.0): "Pool V2" by the Pool Table workspace, Roboflow
> Universe — https://universe.roboflow.com/pool-table/pool-v2

## What Phase 1 must close (and how we'll know)

The detector must go from ~2.6% recall to something usable on Joe's wide
side-view. Per the prior-art survey (`docs/PRIOR_ART.md`), the path is a learned
detector (YOLO → ONNX), but Pool V2 is **not** the right training set (above). We
need either pool footage matching Joe's conditions (his own captures, via the
Capture-for-analysis flow) or a dataset of wide-view tables. Every future change
re-runs this harness; "better" must be a higher number on these clips, not a
vibe. The clean per-clip metrics live in `docs/eval/baseline/metrics.json`.
