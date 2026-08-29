"""Build a per-frame NAMING TRUTH stream for the bench clip.

Round 27 landed a change that lifted every scorecard naming number while
silently renaming the red 3 to "1" in 1843 frames. Nothing caught it: the
scorecard scores whether a ball HAS a name and whether that name is on the
table's inventory, so a wrong-but-valid name reads as success. This builds the
missing yardstick.

Truth here is derived from PIXELS, never from the app's naming:
  * positions come from the finder (position has been the reliable part -
    R1 cue tracking passes at 100%);
  * identity comes from this table's measured colours plus the stripe
    white-fraction window verified in round 27 (solids <=0.211, the striped
    9 >=0.418, cue >=0.903 - clear daylight in between);
  * the yellow family is split by that stripe reading, which is the one
    confusion colour alone cannot resolve (measured 1-vs-9 separation is
    only 7.3 Lab).

Low-confidence detections are dropped: the bench's three fixed phantoms score
0.30-0.49 at the finder while every real ball, resting or rolling, scores
0.84-0.88 (round 26).

    python tools/build_naming_truth.py             # build + verification sheet
    python tools/build_naming_truth.py --step 2.0

The output feeds tools/scorecard.py. It is checked by EYE before use - the
sheet it writes draws each verdict on the frame it came from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
REC = Path(r"C:/Users/Joe/Dropbox/Billiards/BilliardsTrainer")
OUT = ROOT / "docs" / "bench_naming_truth.json"
SHEET = ROOT / "_train" / "bench_fix" / "naming_truth_check.png"

MIN_SCORE = 0.70      # phantoms 0.30-0.49, real balls 0.84-0.88 (round 26)
STRIPE_AT = 0.314     # midpoint of the measured gap 0.211 -> 0.418


def _white_frac(crop) -> float:
    """Fraction of the ball's disc that reads as white.

    Deliberately self-contained: a yardstick that imports the app's own
    appearance code would move whenever the app does, which is exactly how
    round 27's regression stayed invisible. The 95% window matters - a
    stripe's white sits at the POLES, and the app's inner-62% window
    discards it (all 20 labelled 9s read solid under it)."""
    import cv2
    import numpy as np
    h, w = crop.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rad = 0.95 * min(h, w) / 2.0
    sel = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad * rad
    if not sel.any():
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1][sel].astype(np.float32)
    v = hsv[:, :, 2][sel].astype(np.float32)
    return float(np.mean((s < 110) & (v > 150)))


def identify(frame, x: float, y: float, r: float) -> int | None:
    """This table's six balls from pixels alone. None = not a ball we know."""
    import cv2
    import numpy as np
    rr = max(2, int(round(r * 0.7)))
    y0, x0 = max(0, int(y) - rr), max(0, int(x) - rr)
    crop = frame[y0:int(y) + rr + 1, x0:int(x) + rr + 1]
    if crop.size < 30:
        return None
    px = crop.reshape(-1, 3).astype(np.float32)
    keep = px[px.mean(1) <= np.percentile(px.mean(1), 75)]   # trim glare
    if len(keep) < 10:
        return None
    b, g, red = (float(v) for v in np.median(keep, axis=0))
    if b > 200 and g > 210 and red > 200:
        return 0                                   # cue
    if b < 115 and g < 95 and red > 150:
        return 3                                   # red
    if b > 190 and g < 170 and red < 95:
        return 2                                   # blue
    if 120 < b < 200 and g < 75 and red < 120:
        return 4                                   # purple
    if b < 95 and g > 180 and red > 200:
        # the one pair colour cannot split: 1 and 9 share a pigment
        rr2 = max(2, int(round(r)))
        y2, x2 = max(0, int(y) - rr2), max(0, int(x) - rr2)
        full = frame[y2:int(y) + rr2 + 1, x2:int(x) + rr2 + 1]
        if full.size < 30:
            return None
        return 9 if _white_frac(full) >= STRIPE_AT else 1
    return None


def build(session: str, step: float) -> list:
    import cv2
    from billiards_trainer.config import Settings
    from billiards_trainer.detector_strategies import discover
    from billiards_trainer.measure.engine import _acquire_calib
    from billiards_trainer.vision.pipeline import Pipeline

    pipe = Pipeline(Settings.load())
    calib = _acquire_calib(REC / session, pipe)
    strat = discover()["ensemble_findid"]
    strat.inference_provider = "dml"
    cap = cv2.VideoCapture(str(REC / session))
    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    out = []
    t = 16.0
    while t < dur - 1.0:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            break
        balls = []
        for d in (strat._finder.detect(fr, calib) or []):
            if float(getattr(d, "score", 1.0)) < MIN_SCORE:
                continue
            n = identify(fr, d.x, d.y, d.radius)
            if n is not None:
                balls.append([n, round(float(d.x), 1), round(float(d.y), 1)])
        # a number may appear at most once per frame; drop the whole
        # reading rather than guess which copy is real
        seen = [b[0] for b in balls]
        balls = [b for b in balls if seen.count(b[0]) == 1]
        out.append({"t": round(t, 2), "balls": balls})
        t += step
    cap.release()
    return out


def sheet(session: str, truth: list, at=(30.0, 120.0, 160.0, 200.0)) -> None:
    """Draw the verdicts on their own frames - this gets checked by eye."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(REC / session))
    tiles = []
    for want in at:
        row = min(truth, key=lambda e: abs(e["t"] - want))
        cap.set(cv2.CAP_PROP_POS_MSEC, row["t"] * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        for n, x, y in row["balls"]:
            cv2.circle(fr, (int(x), int(y)), 26, (0, 255, 0), 2)
            cv2.putText(fr, str(n), (int(x) - 10, int(y) - 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(fr, f"t={row['t']:.0f}s  truth={sorted(b[0] for b in row['balls'])}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        h = 620
        tiles.append(cv2.resize(fr, (int(fr.shape[1] * h / fr.shape[0]), h)))
    cap.release()
    if tiles:
        SHEET.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(SHEET), np.hstack(tiles))
        print(f"verification sheet -> {SHEET}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="session-20260824-220247.mp4")
    ap.add_argument("--step", type=float, default=1.0)
    a = ap.parse_args()
    truth = build(a.session, a.step)
    OUT.write_text(json.dumps({
        "session": a.session,
        "note": ("Per-frame naming truth from PIXELS - colour family plus the "
                 "measured stripe window - never from the app's naming. Checked "
                 "by eye via naming_truth_check.png before use."),
        "min_score": MIN_SCORE, "stripe_at": STRIPE_AT,
        "samples": truth}, indent=1), encoding="utf-8")
    from collections import Counter
    c = Counter(b[0] for e in truth for b in e["balls"])
    print(f"{len(truth)} samples -> {OUT}")
    print("ball sightings:", dict(sorted(c.items())))
    sheet(a.session, truth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
