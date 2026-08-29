"""Mine ball-ID training crops from the engine's OWN failures.

Round 21 measured why naming is stuck at 75.7%: the recogniser reads
only 4 of 7 balls and calls the purple 4 a "7". Cadence, coverage,
stitching and dead zones all failed to move it - it is the model. This
mines exactly the frames where it fails, so the retrain sees the cases
that matter instead of more of what already works.

    python tools/mine_ballid.py --sheet     # build a labelling sheet
    python tools/mine_ballid.py --labels "0:0,1:2,..."   # write labels

A failure frame is one where a detected ball is either unnamed while
moving, or named something not in this table's inventory. Every ball in
the frame is boxed; Claude labels them by LOOKING (RULE 0) and the
labels are written in the existing TrainingStore YOLO layout - no new
format, no new pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
M1 = Path(r"C:/Users/Joe/AppData/Local/BilliardsTrainer/m1")
REC = Path(r"C:/Users/Joe/Dropbox/Billiards/BilliardsTrainer")
OUT = ROOT / "_train" / "bench_fix"
SHEET = OUT / "sheet"
TAG = "bench"


def failure_times(name: str, inventory: set[int], limit: int = 12,
                  min_gap: float = 1.0) -> list:
    """Times where the engine's own output shows a naming failure."""
    from billiards_trainer.vision.analysis_cache import SidecarReader
    r = SidecarReader(M1 / name)
    times, frames = r._times, r._frames
    prev: dict = {}
    hits: list = []
    for j, rows in enumerate(frames):
        bad = False
        for tr in rows:
            if not tr[6]:
                continue
            n, est = tr[4], (len(tr) > 7 and bool(tr[7]))
            p0 = prev.get(tr[0])
            prev[tr[0]] = (times[j], tr[1], tr[2])
            if n >= 0 and n not in inventory:
                bad = True                       # invented number
            if n < 0 and not est and p0:
                dt = times[j] - p0[0]
                if 0 < dt < 0.2:
                    v = (((tr[1] - p0[1]) ** 2
                          + (tr[2] - p0[2]) ** 2) ** 0.5) / dt
                    if v > 60.0:
                        bad = True               # moving, seen, unnamed
        if bad:
            hits.append(times[j])
    # spread them out: one per second at most, capped
    spread, last = [], -9.0
    for t in hits:
        if t - last >= min_gap:
            spread.append(t)
            last = t
        if len(spread) >= limit:
            break
    return spread


def build_sheet(name: str, stamps: list) -> None:
    import cv2
    import numpy as np

    from billiards_trainer.config import Settings
    from billiards_trainer.detector_strategies import discover
    from billiards_trainer.measure.engine import _acquire_calib
    from billiards_trainer.vision.pipeline import Pipeline
    SHEET.mkdir(parents=True, exist_ok=True)
    pipe = Pipeline(Settings.load())
    calib = _acquire_calib(REC / name, pipe)
    strat = discover()["ensemble_findid"]
    strat.inference_provider = "dml"
    cap = cv2.VideoCapture(str(REC / name))
    index, tiles = [], []
    for fi, t in enumerate(stamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        dets = strat._finder.detect(fr, calib) or []
        cv2.imwrite(str(SHEET / f"frame_{fi:02d}.jpg"), fr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        boxes = []
        for di, d in enumerate(dets):
            x, y, r = int(d.x), int(d.y), max(int(d.radius), 10)
            crop = fr[max(0, y - 3 * r):y + 3 * r, max(0, x - 3 * r):x + 3 * r]
            if crop.size == 0:
                continue
            tile = cv2.resize(crop, (150, 150), interpolation=cv2.INTER_NEAREST)
            cv2.putText(tile, f"{fi}-{di}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            # position matters for labelling: two similar yellows are
            # separable by WHERE they are when colour alone is ambiguous
            cv2.putText(tile, f"{x},{y}", (5, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            tiles.append(tile)
            boxes.append({"i": di, "x": float(d.x), "y": float(d.y),
                          "r": float(d.radius)})
        index.append({"f": fi, "t": t, "h": fr.shape[0], "w": fr.shape[1],
                      "boxes": boxes})
    cap.release()
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    per_row = 8
    rows = []
    for k in range(0, len(tiles), per_row):
        row = tiles[k:k + per_row]
        while len(row) < per_row:
            row.append(np.zeros((150, 150, 3), np.uint8))
        rows.append(np.hstack(row))
    if rows:
        cv2.imwrite(str(OUT / "labelsheet.png"), np.vstack(rows))
    print(f"{len(tiles)} crops across {len(index)} frames -> "
          f"{OUT / 'labelsheet.png'}")


def write_labels(spec: str) -> None:
    """spec: "f-d:number,f-d:number,..." - everything unlisted is skipped."""
    import cv2

    from billiards_trainer.train.store import LabeledBall, TrainingStore
    index = json.loads((OUT / "index.json").read_text())
    want: dict = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        key, num = part.split(":")
        f, d = key.split("-")
        want[(int(f), int(d))] = int(num)
    store = TrainingStore(OUT / "store")
    total = 0
    for entry in index:
        fi, h, w = entry["f"], entry["h"], entry["w"]
        balls = []
        for b in entry["boxes"]:
            n = want.get((fi, b["i"]))
            if n is None:
                continue
            r = b["r"]
            balls.append(LabeledBall(number=n, cx=b["x"] / w, cy=b["y"] / h,
                                     w=2 * r / w, h=2 * r / h))
        if not balls:
            continue
        img = cv2.imread(str(SHEET / f"frame_{fi:02d}.jpg"))
        if img is None:
            continue
        total += store.add_frame(img, balls, stamp=f"{TAG}_{fi:02d}")
    print(f"wrote {total} labelled boxes -> {store.root}")
    print("class counts:", store.class_counts())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="session-20260824-220247.mp4")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--min-gap", type=float, default=1.0,
                    help="seconds between mined frames - spread them across "
                         "the whole session so late-session balls (the "
                         "orange 5) are represented, not just the opening")
    ap.add_argument("--tag", default="bench")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--at", nargs="+", type=float, default=None,
                    help="mine these exact times instead of hunting "
                         "failures - for balls a failure frame never "
                         "happens to contain (the orange 5)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    global TAG
    TAG = a.tag
    if a.labels:
        write_labels(a.labels)
        return
    truth = json.loads((ROOT / "docs" / "bench_truth.json").read_text())
    stamps = a.at or failure_times(
        a.session, set(truth.get("balls_on_table", [])),
        limit=a.limit, min_gap=a.min_gap)
    print("failure frames:", [round(t, 2) for t in stamps])
    if a.sheet:
        build_sheet(a.session, stamps)


if __name__ == "__main__":
    main()
