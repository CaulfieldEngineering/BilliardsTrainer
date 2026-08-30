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
MARGIN_LAB = 12.0     # nearest palette entry must beat the runner-up
                      # by this much, or the sample is left UNNAMED.
                      # A yardstick abstains; it never guesses.


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


def identify_by_palette(frame, x: float, y: float, r: float, pal: dict):
    """Nearest entry in a PER-TABLE palette, or None when it is not decisive.

    The bench windows below are hand-fitted to one table's six balls and
    cannot describe another rack. A palette is that same idea measured
    per table: every ball on the cloth, labelled BY EYE from a zoomed
    grid, with its crop median and its white fraction.

    Two things make this a yardstick rather than a guess:

      * IT ABSTAINS. A sample is only named when the nearest palette
        entry beats the runner-up by MARGIN_LAB. On the cold clip the
        closest pairs are 19.0 Lab (the gold 1 against the orange 13),
        22.8 (burgundy 7 against black 8) and 23.3 (crimson 3 against
        orange 5), so colour alone genuinely cannot decide many samples -
        and a truth file must say "I don't know" rather than invent one.

      * THE STRIPE BAR IS PER TABLE. 1-vs-13 is 19 Lab apart in colour
        and 0.03-vs-0.21 apart in white fraction, so white decides it -
        but only against THIS table's own solids. The engine's absolute
        0.32 sits above both and is exactly why it calls that stripe a
        solid (round 59).
    """
    import cv2
    import numpy as np
    rr = max(2, int(round(r * 0.7)))
    y0, x0 = max(0, int(y) - rr), max(0, int(x) - rr)
    crop = frame[y0:int(y) + rr + 1, x0:int(x) + rr + 1]
    if crop.size < 30:
        return None
    px = crop.reshape(-1, 3).astype(np.float32)
    keep = px[px.mean(1) <= np.percentile(px.mean(1), 75)]
    if len(keep) < 10:
        return None
    med = np.median(keep, axis=0)
    lab = cv2.cvtColor(np.uint8([[med]]), cv2.COLOR_BGR2LAB)[0][0].astype(float)
    rr2 = max(2, int(round(r)))
    y2, x2 = max(0, int(y) - rr2), max(0, int(x) - rr2)
    wf = _white_frac(frame[y2:int(y) + rr2 + 1, x2:int(x) + rr2 + 1])
    skip = {str(n) for n in pal.get("not_scored", ())}
    balls = {k: v for k, v in pal["balls"].items() if k not in skip}
    bar = pal["stripe_bar"]
    ranked = sorted(
        ((float(np.linalg.norm(lab - np.array(v["lab"]))), int(k))
         for k, v in balls.items()), key=lambda z: z[0])
    if not ranked:
        return None
    (d0, n0), rest = ranked[0], ranked[1:]
    if rest and (rest[0][0] - d0) < MARGIN_LAB:
        # not decisive on colour - the stripe bit may still decide it,
        # but only when the two candidates differ in class
        cand = [n for _, n in ranked[:2]]
        cls = [bool(balls[str(n)]["stripe"]) for n in cand]
        if cls[0] == cls[1]:
            return None
        want = wf >= bar
        return cand[0] if cls[0] == want else cand[1]
    # The CUE is white all over, so a stripe/solid test says nothing
    # about it - applying one dropped every cue sample in the first
    # build. The 8 is exempt for the same reason in reverse: it is the
    # darkest thing on the cloth and no white test can inform it.
    if n0 not in (0, 8) and bool(balls[str(n0)]["stripe"]) != (wf >= bar):
        return None            # colour and appearance disagree: abstain
    return n0


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


def build(session: str, step: float, pal: dict | None = None) -> list:
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
            n = (identify_by_palette(fr, d.x, d.y, d.radius, pal)
                 if pal else identify(fr, d.x, d.y, d.radius))
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
    ap.add_argument("--palette", default=None,
                    help="per-table palette JSON (docs/<clip>_palette.json). "
                         "Without it the bench's hand-fitted colour windows "
                         "are used, which describe ONE table's six balls.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pal = None
    if a.palette:
        pal = json.loads(Path(a.palette).read_text(encoding="utf-8"))
        wfs = sorted(v["white_frac"] for v in pal["balls"].values()
                     if not v["stripe"] and v["white_frac"] < 0.5)
        stripes = sorted(v["white_frac"] for v in pal["balls"].values()
                         if v["stripe"])
        # the bar sits in THIS table's own gap, not in another table's
        pal["stripe_bar"] = ((max(wfs) + min(stripes)) / 2.0
                             if wfs and stripes else STRIPE_AT)
        print(f"palette: {len(pal['balls'])} balls, stripe bar "
              f"{pal['stripe_bar']:.3f} (solids <= {max(wfs):.2f}, "
              f"stripes >= {min(stripes):.2f})")
    global OUT, SHEET
    if a.out:
        OUT = Path(a.out)
        SHEET = ROOT / "_train" / "bench_fix" / (OUT.stem + "_check.png")
    truth = build(a.session, a.step, pal)
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
