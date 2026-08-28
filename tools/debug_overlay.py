"""Render what the ENGINE believes onto the real video frames.

Joe, 2026-08-28: "Some screenshots including debug overlays from the
app would be helpful... Can you prove the app has been insistent on a
ghost ball?" This is the proof tool - and the standing evidence tool
for the Dev Journal: every claim about what the app saw is rendered
over the picture it saw it in.

    python tools/debug_overlay.py <video> --at 170.5 171.5 --out dir/

Each track is drawn at its measured position with its id and the
number the app assigned (or "?" when unnamed), so a ghost is visible
as a label floating over felt/leather with no ball under it.
"""

from __future__ import annotations

import argparse
import bisect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

M1 = Path(r"C:/Users/Joe/AppData/Local/BilliardsTrainer/m1")


def render(video: Path, times_wanted, out_dir: Path, crop=None,
           sidecar_dir: Path = M1) -> list[Path]:
    import cv2
    import numpy as np

    from billiards_trainer.vision.analysis_cache import SidecarReader
    r = SidecarReader(sidecar_dir / video.name)
    hinv = np.asarray(r.meta["hinv"], dtype=float)

    def to_vid(x, y):
        v = hinv @ np.array([x, y, 1.0])
        return int(round(v[0] / v[2])), int(round(v[1] / v[2]))

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    made = []
    for t in times_wanted:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, fr = cap.read()
        if not ok:
            continue
        j = min(bisect.bisect_left(r._times, t), len(r._frames) - 1)
        for tr in r._frames[j]:
            tid, x, y, rad, num, _cls, active = tr[:7]
            if not active:
                continue
            px, py = to_vid(x, y)
            col = (60, 220, 60) if num >= 0 else (60, 160, 255)
            cv2.circle(fr, (px, py), 26, col, 2)
            cv2.putText(fr, f"{'?' if num < 0 else num}", (px - 34, py - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            cv2.putText(fr, f"id{tid}", (px + 20, py + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        cv2.putText(fr, f"t={t:.2f}s   green=named  blue=unnamed",
                    (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2)
        if crop:
            cx, cy, half = crop
            fr = fr[max(0, cy - half):cy + half, max(0, cx - half):cx + half]
        p = out_dir / f"dbg_{video.stem}_{t:.2f}.png".replace(".", "_", 1)
        cv2.imwrite(str(p), fr)
        made.append(p)
    cap.release()
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--at", nargs="+", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", nargs=3, type=int, default=None,
                    metavar=("CX", "CY", "HALF"))
    a = ap.parse_args()
    made = render(Path(a.video), a.at, Path(a.out),
                  crop=tuple(a.crop) if a.crop else None)
    for p in made:
        print(p)


if __name__ == "__main__":
    main()
