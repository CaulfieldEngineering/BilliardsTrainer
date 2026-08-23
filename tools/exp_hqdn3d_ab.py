"""A/B: does hqdn3d denoising of the ANALYSIS stream fix grain-era tracking?

Grain from the 2026-08-23 high-ISO settings measurably hurts identity
tracking (hops 4.3x the clean-era rate, ball numbering 97% -> 87%). The
recorder already lightly denoises what it WRITES; the detector eats RAW
frames. This experiment runs the same slice of a grainy session through
the standard offline pipeline twice - raw vs hqdn3d (default strength,
measured: no motion smearing, 98 fps offline) - and scores both with the
identity-wander rules. Gate for wiring it live: hops clearly down,
numbering clearly up, detections not reduced.

    python tools/exp_hqdn3d_ab.py [--video X] [--t0 840] [--dur 360]
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from _lowprio import demote

demote()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

from billiards_trainer.capture.audio import NO_WINDOW, find_ffmpeg  # noqa: E402
from billiards_trainer.config import Settings  # noqa: E402
from billiards_trainer.vision.pipeline import Pipeline  # noqa: E402

WANDER_DIAM = 8.0   # same rules as audit_identity_wander
WANDER_DT = 1.2
MIN_GAP_DT = 0.05


def run_pipeline(video: Path) -> dict:
    cap = cv2.VideoCapture(str(video))
    pipeline = Pipeline(Settings.load())
    states = []          # (t, [(tid, x, y, r, num, active)])
    n = 0
    last_t = -1.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        n += 1
        res = pipeline.process(frame, t, annotate=False, detect=(n % 3 == 0))
        if res.status == "tracking" and t - last_t >= 0.1:
            last_t = t
            states.append((t, [(tr.id, tr.x, tr.y, tr.radius, tr.number,
                                tr.active) for tr in res.tracks]))
    cap.release()
    return score(states)


def score(states: list) -> dict:
    last = {}
    hops = 0
    active_n = numbered_n = 0
    tids = set()
    radii = sorted(r for _, fr in states[:80] for (_, _, _, r, num, act) in fr
                   if act and num >= 0)
    ball_d = 2.0 * radii[len(radii) // 2] if radii else 25.0
    for t, fr in states:
        for tid, x, y, _r, num, act in fr:
            if not act:
                continue
            active_n += 1
            tids.add(tid)
            if num < 0:
                continue
            numbered_n += 1
            if num in last:
                pt, px, py, pid = last[num]
                dt = t - pt
                if (MIN_GAP_DT < dt < WANDER_DT and tid != pid
                        and ((x - px) ** 2 + (y - py) ** 2) ** 0.5
                        > WANDER_DIAM * ball_d):
                    hops += 1
            last[num] = (t, x, y, tid)
    span_min = (states[-1][0] - states[0][0]) / 60.0 if len(states) > 1 else 1.0
    return {
        "states": len(states),
        "hops": hops,
        "hops_per_1k": round(1000.0 * hops / max(1, len(states)), 2),
        "numbered_pct": round(100.0 * numbered_n / max(1, active_n), 1),
        "tracks_per_min": round(len(tids) / span_min, 1),
        "balls_per_state": round(active_n / max(1, len(states)), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="C:/Users/Joe/Dropbox/Billiards/"
                    "BilliardsTrainer/session-20260823-194542.mp4")
    ap.add_argument("--t0", type=float, default=840.0)
    ap.add_argument("--dur", type=float, default=360.0)
    args = ap.parse_args()
    ff = find_ffmpeg()
    tmp = Path(tempfile.gettempdir())
    raw = tmp / "ab_raw.mp4"
    dn = tmp / "ab_dn.mp4"
    print(f"slicing {args.dur:.0f}s @ t={args.t0:.0f}s ...")
    subprocess.run([ff, "-v", "error", "-ss", str(args.t0), "-i", args.video,
                    "-t", str(args.dur), "-c", "copy", "-an", "-y", str(raw)],
                   check=True, creationflags=NO_WINDOW)
    # High-quality re-encode so encode loss can't be mistaken for denoise
    # effect; crf 16 veryfast measured visually transparent on this rig.
    subprocess.run([ff, "-v", "error", "-i", str(raw), "-vf", "hqdn3d",
                    "-c:v", "libx264", "-crf", "16", "-preset", "veryfast",
                    "-an", "-y", str(dn)], check=True, creationflags=NO_WINDOW)
    print("analyzing RAW slice ...")
    a = run_pipeline(raw)
    print("  " + str(a))
    print("analyzing DENOISED slice ...")
    b = run_pipeline(dn)
    print("  " + str(b))
    print("\nVERDICT candidates: hops_per_1k down + numbered_pct up + "
          "balls_per_state not down => wire hqdn3d into the analysis stream")
    for f in (raw, dn):
        f.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
