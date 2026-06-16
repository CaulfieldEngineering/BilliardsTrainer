# Phase 1 — detector experiments (round 2, all variants)

**Date:** 2026-06-16 · 6 variants, all detecting on the **raw oblique frame**,
scored in raw coords vs hand-labelled GT across the 14 Phase-0 clips (3 idle
clips carry the position GT). Run with `tools/eval_experiments.py` (self-healing
subprocess-per-variant). Local reports: `docs/eval/experiments/round2/comparison.md`
+ `visual.html` (side-by-side). **No default detector changed — Joe picks.**

## Results (ranked by F1)

| variant | precision | recall | F1 | ID acc | FPS | status |
|---|---|---|---|---|---|---|
| **`simple_blob`** | 25.6% | 24.2% | **24.9%** | 44.8% | **69** | ✅ best F1 + fastest |
| **`felt_mask_hough`** | 16.2% | **45.8%** | 23.9% | 41.3% | 18 | ✅ best recall |
| `onnx_cuedetat_ball` | 11.1% | 12.7% | 11.8% | 98.4% | 5.1 | ✅ pretrained, slow |
| `onnx_cuedetat_pocket` | 11.1% | 12.7% | 11.8% | 98.4% | 4.6 | ✅ (identical to ball — see caveat) |
| `classical_rectified` | 19.4% | 6.1% | 9.2% | 82.2% | 31 | ✅ baseline-style reference |
| `yolo_8ball_best` | — | — | — | — | — | ❌ FAILED (self-healed) |

Phase-0 published baseline (classical on the *rectified* view, with the 0.85
render-floor): **P 26% · R 2.6% · F1 4.7%.**

## Headline — the raw-frame pivot works

Moving detection to the **raw frame** (Joe's architectural call) takes ball
recall from the baseline's **2.6% → up to 45.8%**, and F1 from 4.7% → ~25%. Every
single variant beats the rectified baseline — a clean confirmation that
rectify-then-detect was throwing away the signal.

- **`simple_blob`** (SimpleBlobDetector, per snooker-ball-tracker) — **best F1
  (24.9%), balanced P/R, and real-time (69 fps).** The strongest all-round
  candidate to make default in Phase 2.
- **`felt_mask_hough`** (CueDetat-style felt-prior + local Hough) — **highest
  recall (45.8%)**: it finds the most balls, at the cost of precision (16% — lots
  of false positives, e.g. rail/pocket artefacts) and 18 fps. Best base if recall
  matters most and we add a precision filter.

## Honest caveats (don't over-read)

- **The pretrained CueDetat ONNX models *underperform* the simple classical
  strategies on Joe's footage** (F1 11.8% vs ~24%) and run at only 4–5 fps (not
  real-time on CPU). A real domain mismatch — they were trained on different
  tables/angles. ID accuracy reads 98% but that's over very few true positives.
- **The two CueDetat ONNX files gave *identical* metrics** (`best.onnx` vs
  `pocket_detector_final.onnx`) — suspicious; likely the same export or our decode
  treats them alike. Flagged, not yet run down.
- **`yolo_8ball_best` self-healed to FAILED** — we deliberately did NOT install
  torch/ultralytics (it conflicts with our pinned opencv-headless and would
  replace cv2). The probe marked it FAILED with the traceback and the run
  continued — exactly the self-healing contract. To use it, convert the `.pt` to
  ONNX in a separate env and drop it in `_eval/models/`.
- **"All RECOMMENDED" is true but weak signal:** the harness verdict compares to
  the *Phase-0* baseline (F1 4.7%), which everything beats. The real ranking is
  the F1 column above.
- **`classical_rectified` shows F1 9.2% here vs the 4.7% baseline** because this
  variant omits the 0.85 render-floor the shipped pipeline applies (more
  detections → more recall, less precision). Same detector, looser gate.
- **Precision is still low everywhere (16–26%)** — these are recall-forward
  classical detectors; expect false positives. And on `idle_02` (the cleanest but
  hardest clip — small, far, foreshortened balls on blue felt) the best recall was
  **`felt_mask_hough` 27.4%**; **no variant exceeded 50%** there.
- GT is hand-labelled (≈ball-radius precision). Good for ranking, not a benchmark.

## Recommendation for Phase 2 (Joe's call)

1. **Make `simple_blob` the default** (best F1 + real-time) — and/or **combine**
   it with `felt_mask_hough` (union for recall, then a precision filter), since
   they trade off P vs R.
2. **Skip the pretrained models for now** — they're slower and worse on Joe's
   table than the classical raw strategies. Revisit only with a model fine-tuned
   on his angle/felt (the oblique datasets in `docs/datasets-catalog.md`, or his
   own Capture-for-analysis footage).
3. **Push precision next** — all winners over-detect; a felt/shadow/rail
   rejection pass or a light confidence filter is the obvious next experiment
   (one new strategy file).

Every future idea is one strategy file + one harness run away. The numbers above
are the bar to beat.
