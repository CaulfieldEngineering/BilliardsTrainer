"""Rebuild colour_refs.json from a LABELLED store — the missing owner.

``core/balls.py`` arbitrates ball identity with per-number measured colours
loaded from ``APP_DIR/colour_refs.json``. Round 25 found that file had NO
writer anywhere in the tree: it was an orphan dated 2026-08-15 whose ball-4
entry recorded the purple 4 as navy (BGR 142,26,36), sitting next to the real
blue 2 in Lab space — exactly the "purple 4 guesses BLUE" misread the ensemble
works around downstream. A fact with no owner goes stale silently; this gives
it one.

Measurement uses the SAME crop convention as the live naming path
(``FindIdEnsemble._fix_colour``): a tight r*0.7 box, brightest quartile
trimmed as glare, per-channel median. Measuring any other way would produce
references the live path cannot reproduce.

Numbers absent from the store are LEFT ALONE — this table carries six balls,
and dropping the others would blind the app on every clip that has them.

    python tools/build_colour_refs.py --store _train/bench_fix/store
    python tools/build_colour_refs.py --store <store> --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def measure(store: Path) -> dict[int, dict]:
    """Per-number glare-trimmed median colour over every labelled box."""
    import cv2
    import numpy as np

    samples: dict[int, list] = {}
    for lab_f in sorted((store / "labels").glob("*.txt")):
        img_f = store / "images" / (lab_f.stem + ".jpg")
        if not img_f.exists():
            continue
        frame = cv2.imread(str(img_f))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        for line in lab_f.read_text(encoding="utf-8").split("\n"):
            parts = line.split()
            if len(parts) != 5:
                continue
            num = int(parts[0])
            cx, cy, bw, _bh = (float(v) for v in parts[1:])
            x, y, r = cx * w, cy * h, (bw * w) / 2.0
            # identical to FindIdEnsemble._fix_colour
            rr = max(2, int(round(r * 0.7)))
            y0, x0 = max(0, int(y) - rr), max(0, int(x) - rr)
            crop = frame[y0:int(y) + rr + 1, x0:int(x) + rr + 1]
            if crop.size < 30:
                continue
            px = crop.reshape(-1, 3).astype(np.float32)
            keep = px[px.mean(1) <= np.percentile(px.mean(1), 75)]
            if len(keep) < 10:
                continue
            samples.setdefault(num, []).append(np.median(keep, axis=0))

    out: dict[int, dict] = {}
    for num, arr in samples.items():
        bgr = np.median(np.array(arr, np.float32), axis=0)
        lab = cv2.cvtColor(np.array([[bgr]], np.uint8),
                           cv2.COLOR_BGR2LAB)[0, 0].astype(float)
        out[num] = {"bgr": [round(float(v), 1) for v in bgr],
                    "lab": [round(float(v), 1) for v in lab],
                    "n": len(arr)}
    return out


def main() -> int:
    import numpy as np

    from billiards_trainer.config import APP_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", help="labelled TrainingStore to measure from")
    ap.add_argument("--write", action="store_true",
                    help="without this the tool only reports the diff")
    ap.add_argument("--install", action="store_true",
                    help="restore APP_DIR's refs from the version of record "
                         "in docs/colour_refs.json (a fresh clone has no "
                         "labelled corpus to measure, but the engine's naming "
                         "depends on these - see round 33)")
    a = ap.parse_args()

    if a.install:
        canon = json.loads((ROOT / "docs" / "colour_refs.json")
                           .read_text(encoding="utf-8"))["refs"]
        live = {k: {kk: vv for kk, vv in v.items() if kk != "source"}
                for k, v in canon.items()}
        path = APP_DIR / "colour_refs.json"
        path.write_text(json.dumps(live, indent=1), encoding="utf-8")
        print(f"installed {len(live)} references -> {path}")
        return 0
    if not a.store:
        print("--store is required unless --install is given", file=sys.stderr)
        return 2

    fresh = measure(Path(a.store))
    if not fresh:
        print("no labelled boxes found", file=sys.stderr)
        return 2
    path = APP_DIR / "colour_refs.json"
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = {}

    print(f"{'ball':>5} {'n':>4}  {'measured BGR':<22}{'previous BGR':<22}drift(Lab)")
    for num in sorted(fresh):
        f = fresh[num]
        o = old.get(str(num))
        if o:
            drift = float(np.linalg.norm(np.array(f["lab"]) - np.array(o["lab"])))
            print(f"{num:>5} {f['n']:>4}  {str(f['bgr']):<22}{str(o['bgr']):<22}{drift:6.1f}")
        else:
            print(f"{num:>5} {f['n']:>4}  {str(f['bgr']):<22}{'-- none --':<22}")
    kept = sorted(k for k in old if int(k) not in fresh)
    print(f"\nleft untouched (not on this table): {kept}")

    if not a.write:
        print("\ndry run - pass --write to update", path)
        return 0
    merged = dict(old)
    for num, f in fresh.items():
        merged[str(num)] = f
    path.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
