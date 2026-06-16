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

## Tier 2 — YOLO (active, v0.2.15)

Classical CV can't *semantically* tell a ball from a ball-shaped artifact — YOLO
can. Plan: re-check Roboflow Universe for permissively-licensed pool-ball weights
(CC-BY-NC is fine for Joe's personal, non-distributed use); if none, use
COCO-pretrained YOLO's "sports ball" class as a sized-correctly stopgap, and
fine-tune YOLO-nano on the **Capture-for-analysis** zips (auto-label with the
classical+bgsub detector via `scripts/label_session.py`, hand-correct, train →
`models/joe_table.pt`). YOLO becomes the **default backend when weights are
present** (`backend = auto`); classical is the fallback. Then run **YOLO +
classical as an ensemble** (agreement = high confidence).

## Tier 3 — tracking (planned)

BoT-SORT / OC-SORT with appearance re-ID + a physics-informed Kalman model
(friction deceleration, rail reflections) to kill the ball-ID-churn failure mode
through clusters and occlusion.

## Tier 4 — active learning (planned)

Auto-flag low-confidence moments + a "Review labels" surface; treat manual
overrides / resets as hard negatives that feed the YOLO fine-tune loop.
