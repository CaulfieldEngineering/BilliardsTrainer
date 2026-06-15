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

## Tier 2 — YOLO (planned, v0.2.13+)

Classical CV can't *semantically* tell a ball from a ball-shaped artifact — YOLO
can. Plan: re-check Roboflow Universe for permissively-licensed pool-ball weights;
if none, bootstrap-label frames from Joe's **Record-session** clips with the
classical+bgsub detector and fine-tune YOLO-nano, iterating. Auto-fetch weights
on first use when Backend = yolo. Then run **YOLO + classical as an ensemble**
(agreement = high confidence). The Recording mode + shot log are the data
pipeline that makes this possible.

## Tier 3 — tracking (planned)

BoT-SORT / OC-SORT with appearance re-ID + a physics-informed Kalman model
(friction deceleration, rail reflections) to kill the ball-ID-churn failure mode
through clusters and occlusion.

## Tier 4 — active learning (planned)

Auto-flag low-confidence moments + a "Review labels" surface; treat manual
overrides / resets as hard negatives that feed the YOLO fine-tune loop.
