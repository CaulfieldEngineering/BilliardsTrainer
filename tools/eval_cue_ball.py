"""M2 cue-ball tracking eval — the measurement the roadmap was missing.

Runs the REAL pipeline (calibration -> detector -> tracker) over a clip and
reports the four M2 exit-criteria numbers for the single cue ball:

  - presence %      : frames with a CUE track / frames after table lock   (M2: >95%)
  - id switches     : times the cue track-id changed                       (M2: <1)
  - stationary jitter: std-dev px of cue position while ~still             (M2: <2px)
  - throughput      : mean pipeline ms / fps                              (real-time?)

Presence is a recall PROXY (assumes the cue is visible every frame of the clip);
for true recall, label a clip and compare. It is still the instrument that turns
"is the cue rock solid?" from a vibe into numbers.

Usage:
  .venv/Scripts/python.exe tools/eval_cue_ball.py [--video testVideo.MP4]
        [--frames 300] [--start 1500] [--strategy auto]
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billiards_trainer.config import Settings  # noqa: E402
from billiards_trainer.core.types import BallClass  # noqa: E402
from billiards_trainer.vision.pipeline import Pipeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(ROOT / "testVideo.MP4"))
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--start", type=int, default=1500)
    ap.add_argument("--strategy", default="auto")
    args = ap.parse_args()

    settings = Settings()
    settings.balls.live_strategy = args.strategy
    settings.table.persist_calibration = False  # always calibrate fresh for eval

    pipe = Pipeline(settings, source=args.video)
    pipe.detect_enabled = True

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}")
        return 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    records = []   # (frame_idx, cue_present, x, y, cue_id, n_tracks)
    ms_list = []
    locked_from = None
    for i in range(args.frames):
        ok, frame = cap.read()
        if not ok:
            break
        t = i / 30.0
        t0 = time.perf_counter()
        res = pipe.process(frame, t, annotate=False, detect=True)
        ms_list.append((time.perf_counter() - t0) * 1000.0)
        if res.status in ("tracking", "deviated"):
            if locked_from is None:
                locked_from = i
            cue = next((tr for tr in res.tracks if tr.cls == BallClass.CUE), None)
            if cue is not None:
                records.append((i, True, cue.x, cue.y, cue.id, len(res.tracks)))
            else:
                records.append((i, False, None, None, None, len(res.tracks)))
    cap.release()

    if not records:
        print(f"No table-locked frames in [{args.start}, {args.start+args.frames}). "
              f"status never reached 'tracking'. Try a different --start.")
        print(f"  (mean {np.mean(ms_list):.1f} ms/frame over {len(ms_list)} frames)")
        return 2

    present = [r for r in records if r[1]]
    presence_pct = 100.0 * len(present) / len(records)

    # id switches: count transitions where the cue id changes between consecutive
    # PRESENT frames (a drop-then-reappear with a new id counts as a switch).
    ids = [r[4] for r in present]
    switches = sum(1 for a, b in zip(ids, ids[1:], strict=False) if a != b)
    distinct_ids = len(set(ids))

    # stationary jitter: over sliding windows where the cue moves little, the
    # std-dev of position. Report the median window std (a still cue should be <2px).
    xs = np.array([r[2] for r in present], float)
    ys = np.array([r[3] for r in present], float)
    jit = float("nan")
    if len(xs) >= 15:
        win = 15
        stds = []
        for j in range(0, len(xs) - win):
            wx, wy = xs[j:j + win], ys[j:j + win]
            # "still" if the span is small (< ~6px)
            if (wx.max() - wx.min()) < 6 and (wy.max() - wy.min()) < 6:
                stds.append(float(np.hypot(wx.std(), wy.std())))
        jit = float(np.median(stds)) if stds else float("nan")

    ms = np.array(ms_list)
    print("=" * 64)
    print(f"M2 CUE-BALL EVAL  strategy={args.strategy}  "
          f"frames {args.start}..{args.start+len(records)} (locked@{locked_from})")
    print("=" * 64)
    print(f"  presence %        : {presence_pct:5.1f}%   (M2 target >95%)   "
          f"[{len(present)}/{len(records)} locked frames]")
    print(f"  cue id switches   : {switches:5d}     (M2 target <1)   "
          f"[{distinct_ids} distinct id(s)]")
    js = f"{jit:.2f}px" if jit == jit else "n/a (cue never still)"
    print(f"  stationary jitter : {js:>9}   (M2 target <2px)")
    print(f"  throughput        : {ms.mean():5.1f} ms  ~{1000/ms.mean():4.0f} fps "
          f"(p95 {np.percentile(ms,95):.0f} ms)")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
