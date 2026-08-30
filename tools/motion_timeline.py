"""Find shots the engine never reported — by not asking the engine.

Round 53. Every metric in this campaign scores shots that a truth file
already lists, so a stroke the engine MISSED is invisible to all of
them: it is absent from the shot list, absent from the tracked stream,
and therefore absent from the scorecard. On a clip with no truth file
(any cold clip) there is nothing at all standing between a missed shot
and a clean-looking report.

So this measures the one thing the engine cannot suppress: PIXELS
MOVING ON THE CLOTH. The frame is warped to the table and consecutive
frames differenced, which is independent of detection, tracking, naming
and episode logic. Then it asks two questions:

  * where does the cloth move while the engine reports NOTHING?  a
    candidate missed shot, or Joe walking past.
  * where does the cloth move while the TRACKER sees nothing moving?
    the balls are rolling and the engine is blind to them.

It answers neither question by itself - both need a human look, which
is the point. It says WHERE to look, in a clip too long to watch
frame by frame.

    python tools/motion_timeline.py <session.mp4> [--top 12]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: table-warp width; coarse on purpose - this is an energy signal
_W = 192
#: per-pixel 0-255 change that counts as "this moved"
_MOVED = 12


def _calib(video: Path):
    from billiards_trainer.config import Settings
    from billiards_trainer.measure.engine import _acquire_calib
    from billiards_trainer.vision.pipeline import Pipeline
    return _acquire_calib(video, Pipeline(Settings.load()))


def raw_motion(video: Path) -> tuple[list[float], list[float]]:
    """(times, moving-pixel fraction) over the whole clip, from PIXELS."""
    import cv2
    import numpy as np
    calib = _calib(video)
    tbl = calib.table
    w, h = tbl.x1 - tbl.x0, tbl.y1 - tbl.y0
    s = _W / w
    S = np.array([[s, 0.0, -tbl.x0 * s], [0.0, s, -tbl.y0 * s],
                  [0.0, 0.0, 1.0]])
    M = S @ calib.H
    size = (_W, max(1, int(h * s)))
    cap = cv2.VideoCapture(str(video))
    times: list[float] = []
    energy: list[float] = []
    prev = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        small = cv2.warpPerspective(fr, M, size, flags=cv2.INTER_AREA)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            d = cv2.absdiff(g, prev)
            energy.append(float((d > _MOVED).mean()))
            times.append(t)
        prev = g
    cap.release()
    return times, energy


def tracked_speed(sidecar_video: Path) -> dict[float, float]:
    """Fastest tracked ball per frame, from the engine's own stream."""
    from billiards_trainer.vision.analysis_cache import SidecarReader
    r = SidecarReader(sidecar_video)
    out: dict[float, float] = {}
    prev: dict[int, tuple[float, float, float]] = {}
    for t, rows in zip(r._times, r._frames, strict=False):
        fast = 0.0
        cur: dict[int, tuple[float, float, float]] = {}
        for tr in rows:
            if not tr[6]:
                continue
            cur[tr[0]] = (t, tr[1], tr[2])
            p = prev.get(tr[0])
            if p and 0 < t - p[0] < 0.5:
                v = ((tr[1] - p[1]) ** 2 + (tr[2] - p[2]) ** 2) ** 0.5 / (t - p[0])
                fast = max(fast, v)
        prev.update(cur)
        out[round(t, 2)] = fast
    return out


def bursts(times, energy, floor: float, gap_s: float = 1.0) -> list[tuple]:
    """Contiguous runs above `floor`, merged across short gaps."""
    runs: list[list] = []
    for t, e in zip(times, energy, strict=False):
        if e < floor:
            continue
        if runs and t - runs[-1][1] <= gap_s:
            runs[-1][1] = t
            runs[-1][2] = max(runs[-1][2], e)
        else:
            runs.append([t, t, e])
    return [tuple(r) for r in runs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    from billiards_trainer.config import APP_DIR, Settings
    video = Path(a.video)
    if not video.is_absolute():
        d = (Settings.load().recording.directory or "").strip()
        video = Path(d) / a.video
    print(f"reading pixels: {video.name} ...", flush=True)
    times, energy = raw_motion(video)
    med = sorted(energy)[len(energy) // 2] if energy else 0.0
    floor = max(0.004, med * 6.0)
    runs = bursts(times, energy, floor)
    print(f"  {len(times)} frames, median energy {med:.5f}, "
          f"floor {floor:.5f} -> {len(runs)} motion bursts")

    doc = json.loads((video.parent / f"{video.name}.shots.json")
                     .read_text(encoding="utf-8"))
    shots = doc["shots"] if isinstance(doc, dict) and "shots" in doc else doc
    spans = [(float(s["start"]), float(s["end"]), s.get("action", "?"))
             for s in shots]
    spd = tracked_speed(Path(APP_DIR) / "m1" / video.name)

    print("\n  MOTION THE ENGINE REPORTS NOTHING FOR "
          "(candidate missed shots, longest first):")
    orphans = []
    for t0, t1, peak in runs:
        if any(a0 - 1.0 <= t0 <= b0 + 1.0 or a0 - 1.0 <= t1 <= b0 + 1.0
               or (t0 <= a0 and t1 >= b0) for a0, b0, _ in spans):
            continue
        near = [v for k, v in spd.items() if t0 <= k <= t1]
        orphans.append((t1 - t0, t0, t1, peak, max(near) if near else 0.0))
    for dur, t0, t1, peak, fast in sorted(orphans, reverse=True)[:a.top]:
        print(f"     {t0:7.2f} -> {t1:7.2f}  ({dur:4.1f}s)  "
              f"peak energy {peak:.4f}   fastest TRACKED ball "
              f"{fast:6.0f} px/s")
    if not orphans:
        print("     none - every burst of cloth motion is inside an entry")
    print(f"\n  ({len(orphans)} unexplained bursts, "
          f"{len(runs) - len(orphans)} inside reported entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
