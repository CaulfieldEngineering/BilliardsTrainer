# The autonomous development loop

The platform improves itself without a human in the loop. Three subsystems,
each producing artifacts a reviewer (Joe, or a Claude session) can act on.

## 1. Measurement — physics as free ground truth

`tools/eval_tracking.py` runs the REAL pipeline over recorded video and scores
invariants that need no labels:

| metric | invariant it checks |
|---|---|
| `jitter_px` | a settled ball must not move |
| `teleports_per_min` | balls never move faster than physics allows |
| `phantom_rate_per_min` | real balls don't exist for <0.5 s (FP churn) |
| `dup_number_frac` | each ball number exists at most once |
| `overcount_frac` | at most 16 balls exist |
| `id_flips_per_track_min` | a ball's identity doesn't change while tracked |
| `fps` | must hold real-time |

`score()` collapses these to one number (lower = better). History:
2026-07-02 baseline 47.7 → 6.7 after cross-track number arbitration.

Detection accuracy against HUMAN-verified labels lives separately:
`_eval/holdout/eval_holdout.py` (4 labelled frames, precision/recall/ID-acc).
Use both — the physics score can't see a consistently wrong number.

## 2. Tuning — nightly unattended search

`tools/autotune.py` random-searches the tracker/detector knob space, scoring
each candidate with the harness. **Registered as Windows scheduled task
"BilliardsTrainer Autotune", daily 03:00**, 40 trials (~30 min), reports to
`docs/autotune/report-*.md` + `best.json`.

Promotion is deliberate, not automatic: a reviewer (or Claude, next session)
reads the report, changes the defaults in `BallTracker.__init__` /
`config.py`, re-runs `eval_tracking.py` on FULL segments as a regression
check, and commits. This keeps a noisy night from silently changing shipped
behaviour.

## 3. Model improvement — the ball-ID active-learning loop

The store at `%LOCALAPPDATA%/BilliardsTrainer/training/ballid` accumulates
verified frames (in-app Trainer, or scripted labelling as in the 2026-07-02
session). Fine-tune + deploy:

```powershell
_refs\pool_coach\.venv\Scripts\python.exe tools\finetune_ballid.py `
  --data "$env:LOCALAPPDATA\BilliardsTrainer\training\ballid\data.yaml" `
  --out  "$env:LOCALAPPDATA\BilliardsTrainer\models\pool_ballid.onnx"
# dev discovery prefers _eval/models — copy the new onnx there too
```

Measure honestly on the held-out frames (never train on them). Known weak
pairs to target with new labels: 2↔10 (blue solid/stripe), 3↔5 (red/orange).

## Convergence targets ("perfect" operationalized)

- jitter_px < 0.05, teleports = 0, phantoms < 0.2/min
- dup_number_frac = 0, id_flips < 0.5/track-min
- holdout: precision & recall ≥ 0.99, ID-acc ≥ 0.95
- ≥ 30 fps end-to-end at 1080p with the far-rail rescan ON

When a metric stalls across several nightly reports, the knob space is
exhausted — the next lever is code (tracker logic, model capacity) or data
(more labelled frames of the failing case).
