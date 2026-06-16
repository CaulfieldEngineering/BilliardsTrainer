# Detection eval harness (Phase 0)

*Measure before you change.* This is the rig that turns "the detector feels bad"
into honest numbers, so every future detector change can be judged against a
fixed baseline instead of a vibe.

## What's here

| Path | Committed? | What |
|---|---|---|
| `tools/eval_harness.py` | yes | runs the current pipeline over clips, scores vs. ground truth, writes an HTML visual-diff report + `metrics.json` |
| `tools/sample_clips.py` | yes | scans a long video for motion, cuts diverse idle/active clips |
| `tools/eval_ci_check.py` | yes | compares a run's metrics to the committed CI baseline, flags regressions |
| `tests/fixtures/eval/demo_clip.mp4` | yes (~0.5 MB) | tiny synthetic clip the CI eval runs on |
| `docs/eval/baseline.md` | yes | the honest Phase-0 baseline numbers |
| `testVideo.MP4` | **no** (gitignored, ~1.5 GB) | Joe's real 40-min table video — the ground-truth source |
| `_eval/clips/`, `_eval/labels/`, `_eval/inbox/` | **no** (gitignored) | sampled clips, hand labels, flagged-failure bundles |
| `docs/eval/<run>/` | **no** (gitignored) | generated HTML reports |

## Getting the source video

`testVideo.MP4` is too big for git. To reproduce the full eval on a fresh clone,
drop Joe's 40-min table capture in at the repo root as `testVideo.MP4`
(1920×1080, 30 fps). The **old** `testVideo`/`testPhoto` were different tables —
this file is the only ground truth for the current setup. (A dev/test-machine
sync mechanism for moving footage around is tracked in `TODO.md`.)

## Run it

```powershell
# 1. sample diverse clips from the long video -> _eval/clips
python tools/sample_clips.py --video testVideo.MP4

# 2. (optional) hand-label clips -> _eval/labels/<clip-stem>.json  (schema in eval_harness.py)
#    idle/static scenes only need ONE keyframe; it applies to every frame.

# 3. run the eval -> docs/eval/<timestamp>/report.html
python tools/eval_harness.py --timestamp baseline --out docs/eval/baseline
```

Open `report.html` for the per-clip table + 3-up visual diffs
(INPUT · DETECTOR · GROUND TRUTH).

### Metric definitions (so nobody is misled)

- **precision / recall / F1** — over *labelled* frames; a detection matches a GT
  ball if its centre is within `tolerance_px` (default 2.5% of frame width),
  greedy one-to-one.
- **AP@dist** — centre-distance average precision, **not** IoU mAP (our detector
  emits centre+radius; GT has no tight boxes). Labelled as such everywhere.
- **ID accuracy** — of correctly-localised detections, fraction with the right
  class (cue/solid/stripe/eight).
- **false-positive shots/min** — shot events emitted minus `expected_shots`,
  per minute. The cleanest signal comes from idle clips (`expected_shots: 0`).
- **calibration rate** — fraction of frames where the table locked.

Ground truth here was hand-labelled by the agent from zoomed keyframes
(~ball-radius precision) — good enough to size the gap, not a published
benchmark. `idle_02` is the highest-confidence labelled clip.

## CI

`.github/workflows/eval.yml` runs the harness on the committed synthetic clip on
every push/PR that touches vision/detection code, uploads the HTML report as the
**eval-report** artifact, posts a summary comment on PRs, and flags
detector-behaviour regressions vs. `tests/fixtures/eval/ci_baseline.json`. The
full eval (Joe's footage) is local / manual — CI never sees the big video.

To intentionally update the CI baseline after a detector change:
`python tools/eval_ci_check.py --metrics <run>/metrics.json --update` (or run the
workflow manually with `update_baseline: true`).

## Test-machine failure flagging (skeleton)

In the app: **Settings → Debug → "Save clip + flag as failure"** zips the recent
preview buffer + detector state and stages it via
`src/billiards_trainer/debug_upload.py` into the inbox (`_eval/inbox/` in a source
checkout, `%LOCALAPPDATA%\BilliardsTrainer\eval_inbox` in the frozen app). Joe
syncs that folder to the dev machine and the harness picks the clips up.
`stage_bundle()` is the single seam to later retarget at Supabase Storage.
