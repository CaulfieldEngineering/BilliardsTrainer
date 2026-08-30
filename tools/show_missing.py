"""The 88 blind moments: where truth says a ball is, and the app has none.

Round 49. The naming headline divides by right+wrong+unnamed, so every
check where the app had NO live sighting was dropped from it - 88 of
1096 on the bench, 8% of the evidence, invisible. `--all` in
tools/scorecard.py now counts them, and this renders them so the cause
can be SEEN rather than guessed at.

A blind moment is not one failure but at least three, and only the
pixels tell them apart:
  * OCCLUDED   - a hand, the cue stick or Joe's body is over the ball.
                 Nothing to fix in tracking; the ball is genuinely
                 invisible.
  * COASTING   - the app HAS the track but it is a prediction this
                 frame, not a sighting. The metric is right to refuse
                 it, and the fix is detection recall.
  * PLAIN SIGHT - the ball is clearly visible, unoccluded, and the app
                 has nothing within 30px. That is the real defect.

    python tools/show_missing.py            # classify + contact sheet
    python tools/show_missing.py --limit 12
"""

from __future__ import annotations

import argparse
import bisect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

BENCH = "session-20260824-220247.mp4"
NAME_TOL_PX = 30.0
OUT = ROOT / "_train" / "bench_fix" / "asjoesees"


def _paths():
    from billiards_trainer.config import APP_DIR, Settings
    sidecar = Path(APP_DIR) / "m1" / f"{BENCH}.analysis.jsonl"
    d = (Settings.load().recording.directory or "").strip()
    return sidecar, Path(d) / BENCH


def find_missing() -> list[dict]:
    """Every truth check with no live sighting within NAME_TOL_PX."""
    import numpy as np
    from billiards_trainer.vision.analysis_cache import SidecarReader
    sidecar, _ = _paths()
    r = SidecarReader(sidecar.with_suffix("").with_suffix(""))
    hinv = np.asarray(r.meta["hinv"], dtype=float)

    def to_video(x, y):
        v = hinv @ np.array([x, y, 1.0])
        return v[0] / v[2], v[1] / v[2]

    truth = json.loads((ROOT / "docs" / "bench_naming_truth.json")
                       .read_text(encoding="utf-8"))
    times, frames = r._times, r._frames
    out = []
    for s in truth.get("samples", []):
        t = s["t"]
        j = bisect.bisect_left(times, t)
        if j and (j >= len(times) or abs(times[j - 1] - t) < abs(times[j] - t)):
            j -= 1
        if j >= len(times) or abs(times[j] - t) > 0.15:
            continue
        live = [x for x in frames[j] if x[6]]
        allt = frames[j]
        for n, tx, ty in s["balls"]:
            def near(rows):
                best, bd = None, 1e9
                for row in rows:
                    vx, vy = to_video(row[1], row[2])
                    d = ((vx - tx) ** 2 + (vy - ty) ** 2) ** 0.5
                    if d < bd:
                        bd, best = d, row
                return best, bd
            _, bd_live = near(live)
            if bd_live <= NAME_TOL_PX:
                continue                      # the app has it; not blind
            coast, bd_any = near(allt)
            out.append({"t": t, "n": int(n), "x": tx, "y": ty,
                        "nearest_live_px": round(bd_live, 1),
                        "coasting_track": (bd_any <= NAME_TOL_PX),
                        "coast_n": int(coast[4]) if coast is not None else -1})
    return out


def render(rows: list[dict], limit: int) -> Path | None:
    import cv2
    import numpy as np
    _, video = _paths()
    OUT.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    tiles = []
    step = max(1, len(rows) // limit)
    for row in rows[::step][:limit]:
        cap.set(cv2.CAP_PROP_POS_MSEC, row["t"] * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        x, y = int(row["x"]), int(row["y"])
        x0, y0 = max(0, x - 130), max(0, y - 130)
        crop = fr[y0:y0 + 260, x0:x0 + 260].copy()
        if crop.shape[0] < 40 or crop.shape[1] < 40:
            continue
        cv2.circle(crop, (x - x0, y - y0), 20, (0, 0, 255), 2)
        tag = f"t={row['t']:.0f} ball{row['n']}"
        cv2.putText(crop, tag, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)
        cv2.putText(crop, f"{'COAST' if row['coasting_track'] else 'NONE'}"
                    f" {row['nearest_live_px']:.0f}px", (6, 246),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        tiles.append(cv2.resize(crop, (260, 260)))
    cap.release()
    if not tiles:
        return None
    per = 6
    rowsimg = [np.hstack(tiles[i:i + per])
               for i in range(0, len(tiles) - per + 1, per)]
    if not rowsimg:
        rowsimg = [np.hstack(tiles)]
    w = min(r.shape[1] for r in rowsimg)
    sheet = np.vstack([r[:, :w] for r in rowsimg])
    p = OUT / "missing_moments.png"
    cv2.imwrite(str(p), sheet)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=18)
    a = ap.parse_args()
    rows = find_missing()
    print(f"blind checks: {len(rows)}")
    coasting = sum(1 for r in rows if r["coasting_track"])
    print(f"  app HAS a track there but it is COASTING : {coasting}")
    print(f"  app has NOTHING within {NAME_TOL_PX:.0f}px        : "
          f"{len(rows) - coasting}")
    by_ball: dict = {}
    for r in rows:
        by_ball[r["n"]] = by_ball.get(r["n"], 0) + 1
    print("  by ball: " + "  ".join(f"{k}:{v}" for k, v in sorted(by_ball.items())))
    spans: list = []
    for r in sorted(rows, key=lambda r: (r["n"], r["t"])):
        if spans and spans[-1][0] == r["n"] and r["t"] - spans[-1][2] <= 2.0:
            spans[-1][2] = r["t"]
            spans[-1][3] += 1
        else:
            spans.append([r["n"], r["t"], r["t"], 1])
    print("  longest blind spans:")
    for n, t0, t1, k in sorted(spans, key=lambda s: -s[3])[:8]:
        print(f"     ball {n}: {t0:6.1f}s -> {t1:6.1f}s  ({k} checks)")
    p = render(rows, a.limit)
    print(f"  contact sheet: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
