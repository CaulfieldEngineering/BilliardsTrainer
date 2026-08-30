"""Score the LIVE path end to end, exactly as the app runs it.

tools/tracker_bakeoff.py compares the two trackers on identical detections.
This is the other half: it drives the real Pipeline.process() - detection,
preparation, tracking, vacancy pruning, everything the app does while Joe is
at the table - and scores the tracks it publishes against the same
pixel-derived naming truth the bench scorecard uses.

That makes the live swap a measured change rather than a hope: run this
before, swap the tracker, run it after.

    python tools/live_path_check.py [--session <name>] [--max-frames N]

Nothing here writes to the app's state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
REC = Path(r"C:/Users/Joe/Dropbox/Billiards/BilliardsTrainer")


class _Reader:
    def __init__(self, hinv):
        self.meta = {"hinv": hinv}


def run(session: str, max_frames: int) -> dict:
    import cv2
    import numpy as np
    from billiards_trainer.config import Settings
    from billiards_trainer.vision.pipeline import Pipeline

    pipe = Pipeline(Settings.load())
    cap = cv2.VideoCapture(str(REC / session))
    times: list = []
    frames: list = []
    fi = 0
    prev = -1.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if fi > 0 and t <= prev:
            t = prev + 1 / 30.0
        prev = t
        res = pipe.process(frame, t, annotate=False, detect=True)
        rows = []
        for tr in (res.tracks or []):
            if not getattr(tr, "active", True):
                continue
            rows.append((int(tr.id), float(tr.x), float(tr.y),
                         float(tr.radius), int(tr.number), 2, 1,
                         bool(getattr(tr, "coasting", False))))
        times.append(t)
        frames.append(rows)
        fi += 1
        if fi % 900 == 0:
            print(f"  {fi} frames ({t:.0f}s), {len(rows)} tracks", flush=True)
        if max_frames and fi >= max_frames:
            break
    cap.release()

    calib = pipe.calib.calib
    if calib is None:
        raise SystemExit("the pipeline never calibrated")
    hinv = np.linalg.inv(np.asarray(calib.H, dtype=float)).tolist()
    from scorecard import _naming_correctness
    return {"frames": fi,
            "naming": _naming_correctness(_Reader(hinv), times, frames)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="session-20260824-220247.mp4")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--label", default="live")
    a = ap.parse_args()
    out = run(a.session, a.max_frames)
    c = out["naming"]
    print(f"\nLIVE PATH ({a.label}) over {out['frames']} frames of {a.session}")
    if not c:
        print("  no score (no truth samples in range)")
        return 0
    print(f"  named correctly : {c['name_right_pct']}%  "
          f"[wrong {c['name_wrong_frames']}, unnamed {c['name_unnamed_frames']}, "
          f"no track {c['name_missing_frames']}]")
    print("  per ball        : " + "  ".join(
        f"{b}:{v['right']}/{v['right'] + v['wrong'] + v['unnamed']}"
        for b, v in c["name_per_ball"].items()))
    if c["name_confusions"]:
        print("  confusions      : " + ", ".join(
            f"{k} x{v}" for k, v in list(c["name_confusions"].items())[:6]))
    p = ROOT / "_train" / "bench_fix" / f"live_{a.label}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
