"""Where does object-ball naming actually get lost?

Round 9's failed filter taught the lesson: the phantom number is a
RECOGNITION problem, not a counting one. This splits the unnamed
moving time into causes so the fix is aimed, not guessed:

  seen-unnamed   the app SAW the ball this frame and still had no
                 number for it  -> recognition / arbitration
  estimate-only  the position was a coasted guess, so there was no
                 picture to read at all -> tracking coverage

    python tools/naming_audit.py [session.mp4]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
M1 = Path(r"C:/Users/Joe/AppData/Local/BilliardsTrainer/m1")
MOVE = 60.0


def main() -> None:
    from billiards_trainer.vision.analysis_cache import SidecarReader
    name = sys.argv[1] if len(sys.argv) > 1 else "session-20260824-220247.mp4"
    r = SidecarReader(M1 / name)
    times, frames = r._times, r._frames
    prev: dict = {}
    named = seen_unnamed = est_unnamed = 0
    per_track: dict = {}
    for j, rows in enumerate(frames):
        for tr in rows:
            if not tr[6]:
                continue
            tid, x, y, n = tr[0], tr[1], tr[2], tr[4]
            est = len(tr) > 7 and bool(tr[7])
            p0 = prev.get(tid)
            prev[tid] = (times[j], x, y)
            if p0 is None:
                continue
            dt = times[j] - p0[0]
            if not (0 < dt < 0.2):
                continue
            if ((x - p0[1]) ** 2 + (y - p0[2]) ** 2) ** 0.5 / dt <= MOVE:
                continue
            if n >= 0:
                named += 1
            elif est:
                est_unnamed += 1
            else:
                seen_unnamed += 1
                d = per_track.setdefault(tid, 0)
                per_track[tid] = d + 1
    tot = named + seen_unnamed + est_unnamed
    print(f"MOVING SAMPLES: {tot}")
    print(f"  named                : {named} ({100*named/max(1,tot):.1f}%)")
    print(f"  SEEN but unnamed     : {seen_unnamed} "
          f"({100*seen_unnamed/max(1,tot):.1f}%)  <- recognition")
    print(f"  estimate only        : {est_unnamed} "
          f"({100*est_unnamed/max(1,tot):.1f}%)  <- tracking coverage")
    worst = sorted(per_track.items(), key=lambda kv: -kv[1])[:6]
    print("worst seen-but-unnamed tracks (id, moving samples):", worst)


if __name__ == "__main__":
    main()
