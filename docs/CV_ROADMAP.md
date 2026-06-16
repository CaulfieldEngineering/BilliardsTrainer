# Detection CV roadmap — measured

The goal: robust real-time make/miss detection. We measure before/after every
change with `tools/eval_detection.py` so "better" is a number, not a vibe.

## Measurement method

The eval runs two scenarios through the full pipeline:
- **Noisy-idle** — a settled table with heavy sensor noise + a *flickering
  specular highlight* (a real false-positive source: glare on the felt). Metric:
  false shot-starts / min. Lower is better.
- **Demo** — the scripted make-shot. Metric: makes detected (true positives must
  be preserved).

## Tier 1 — multi-modal evidence fusion ✅ (v0.2.12)

Shipped. Shots are gated on a fused **activity** score combining three modalities
instead of one threshold:

| signal | real ball motion | flickering highlight |
|---|---|---|
| pixel-change motion energy | 0.6 | 0.8 (overlaps — weak alone) |
| optical flow (Farneback) | 1.9 | **10.5** (HIGHER — misleading!) |
| **bg-subtraction foreground** | **0.74** | **0.013** (50× separation) |

Key finding: optical flow *reads brightness flicker as spurious motion*, so it's
a poor discriminator (weighted ~0). **Background subtraction (MOG2)** is the
strong signal — it adapts to a flickering region as background, so foreground is
near-zero there but high for a real moving ball. Fusion weights: fg 0.65, motion
0.30, flow 0.05.

**Result (60 s noisy-idle flicker):**

| | false shot-starts / min | demo makes |
|---|---|---|
| baseline (motion-only) | **2.0** | preserved |
| **fused (Tier 1)** | **0.0** | preserved |

Cost: ~31 ms/frame → ~32 fps real-time (optical flow runs every 2nd frame).
Tunable from **Settings → Detection** (fusion toggle, weights, presets:
Conservative / Balanced / Aggressive).

> Note: false *counts* were already 0 in both — the travel + pocket-approach gates
> from v0.2.11 protect counting. Tier 1 specifically removes the false
> activity/start rate (UI flicker + a noise source that could cascade on real
> footage).

## Reality check — classical CV is demo-grade (v0.2.14)

Live test on Joe's real overhead camera: classical detection was unusable —
wrong ball colours/sizes, a tiny phantom cue, jittering IDs, blobs floating in
the pockets. **No amount of HSV/Hough tuning fixes this class of problem on a
real camera.** So v0.2.14 stops pretending:

- **Manual mode is the default.** The app opens straight into a live camera
  *preview* with auto-detection **OFF** — a clean empty overhead + manual
  +Make/−Miss. The Detection toggle is only enabled when a YOLO model is
  present; otherwise a banner says so. Holding the line: *show nothing rather
  than something wrong.*
- **When detection IS on**, three hard filters cull garbage: full pocket-region
  masking (no in-pocket blobs), a strict regulation ball-size band (radius
  ≈ 0.0225·W, ±~50%), and a 0.85 render/track floor.
- **Capture for analysis**: Settings → *Capture 60s for analysis* zips raw
  frames (+ calibration meta) — the training data for the YOLO pivot below.

## Tier 2 — the off-the-shelf model search (v0.2.15) — what's actually out there

Goal: AI detection **with no training data from the user**. We searched + tested.
The honest result:

| Option | Verdict |
|---|---|
| **COCO YOLOv8 "sports ball" (class 32)** | **Does NOT work.** Tested `yolov8n.onnx` on a real rectified table + live frame: **0** sports-ball detections (it's trained on natural-scene sports balls, not top-down billiards). Dead end as a detector — even though it's the only zero-auth, clearly-licensed `.onnx`. |
| **Roboflow Universe pool/snooker models** (some have per-colour classes!) | Weights are **API-key-gated** — no anonymous download; raw weights need a paid plan. Can't bundle/auto-fetch. |
| **Community `.pt` on GitHub** (e.g. white/colored/cue) | Directly downloadable but **no license** (all-rights-reserved) and `.pt` only (needs torch to run/convert). |

**Conclusion:** there is no free, off-the-shelf model that detects top-down pool
out of the box. So v0.2.15 ships the **infrastructure**, not a false headline:

- **`OnnxYoloDetector`** — runs a YOLO `.onnx` via ONNX Runtime, **no torch**
  (~15 MB vs ~2 GB). Decode is numpy + OpenCV NMS. COCO models auto-filter to
  class 32; pool-specific models keep all classes. Drop a `.onnx` in `models/`
  and it's used automatically (`backend = auto`); classical stays the fallback.
- onnxruntime is a `[onnx]` extra, **not bundled** in the `.exe` yet (no working
  model justifies the size + the PyInstaller DLL pitfalls). The lazy import
  degrades to classical when absent.

**Path to a real pool model** (when someone will accept a project's license /
produce data): export a Roboflow snooker model to ONNX once
(`ultralytics … export format=onnx`) and drop it in `models/`, or fine-tune
YOLO-nano on **Capture-for-analysis** zips. No app change needed — it just works.

### Roboflow "Pool V2" (the candidate model, v0.2.16)

Project: <https://universe.roboflow.com/pool-table/pool-v2>. Verified facts
(read off the page; license/metrics are theirs, not measured by us):

- 751 images, object detection, **4 classes**, **mAP@50 66.8%** /
  precision 66.x% / recall 63.4% — a modest starting model, not a guarantee.
- **License: CC BY 4.0** — free to use with attribution. 

**Gating (verified, honest):** Roboflow's public API and the `roboflow` SDK both
**refuse anonymous access** ("A valid API key must be provided"). So neither the
dataset nor the trained weights are anonymously downloadable, and the *hosted*
inference API would stream every frame to Roboflow's servers — which breaks this
app's local/offline rule. The constraint-compliant route uses the CC-BY dataset:

```
pip install -e ".[yolo]" roboflow
python tools/train_pool_model.py --api-key <FREE_ROBOFLOW_KEY>
```

That downloads Pool V2, fine-tunes YOLOv8n locally, exports
`models/pool_balls.onnx`, and the app's `OnnxYoloDetector` runs it **offline**.
Blocked only on a free Roboflow API key (the user's to create) + one-time training
compute. The drop-in path itself is verified: a `pool_balls.onnx` in the models
dir is auto-selected over classical with zero code changes.

## Tier 3 — tracking (planned)

BoT-SORT / OC-SORT with appearance re-ID + a physics-informed Kalman model
(friction deceleration, rail reflections) to kill the ball-ID-churn failure mode
through clusters and occlusion.

## Tier 4 — active learning (planned)

Auto-flag low-confidence moments + a "Review labels" surface; treat manual
overrides / resets as hard negatives that feed the YOLO fine-tune loop.
