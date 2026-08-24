"""A/B: does a colour/glare pre-filter improve ball identification?

Joe's ask: "implement a color filter to reduce glare and emphasize
colors". This measures CANDIDATE filters against the hand-labelled truth
(same harness as measure_class_accuracy) before anything ships: the live
ensemble runs on filtered frames and per-class accuracy is compared to
the unfiltered baseline. Gate: overall AND stripe accuracy up, dark
cluster (3/4/7/8) not down.

    python tools/exp_colorfilter_ab.py
"""
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from _lowprio import demote

demote()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billiards_trainer.detector_strategies import discover  # noqa: E402

STRIPES = list(range(9, 16))
DARK = [3, 4, 7, 8]


def f_none(img):
    return img


def f_sat(img):
    """Saturation boost 1.35x: 'emphasize colors'."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.35, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def f_glare(img):
    """Specular-glare cap: compress the top of V so highlights stop
    whiting-out ball colour (the washed-out-9-ball complaint)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[:, :, 2]
    hsv[:, :, 2] = np.where(v > 210, 210 + (v - 210) * 0.35, v)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def f_clahe(img):
    """CLAHE on L: local contrast without hue shift."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def f_glare_sat(img):
    return f_sat(f_glare(img))


FILTERS = [("baseline", f_none), ("sat1.35", f_sat), ("glare_cap", f_glare),
           ("clahe", f_clahe), ("glare+sat", f_glare_sat)]


def load_labels(lbl: Path, w: int, h: int):
    out = []
    for line in lbl.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        n = int(parts[0])
        out.append((n, float(parts[1]) * w, float(parts[2]) * h,
                    max(float(parts[3]) * w, float(parts[4]) * h) / 2.0))
    return out


def _make_acc(right, wrong):
    """Bind the per-filter counters explicitly (B023: no loop-var closure)."""
    def acc(nums):
        t = sum(right[n] + wrong[n] for n in nums)
        return 100.0 * sum(right[n] for n in nums) / t if t else 0.0
    return acc


def main() -> int:
    strat = discover().get("ensemble_findid")
    if strat is None:
        print("ensemble not available", file=sys.stderr)
        return 2
    frames = []
    for ds in sorted(ROOT.glob("_train/autolabel*/dataset")):
        for img_path in sorted((ds / "images").glob("*")):
            lbl = ds / "labels" / (img_path.stem + ".txt")
            if lbl.exists():
                frames.append((img_path, lbl))
    print(f"{len(frames)} labelled frames, {len(FILTERS)} filters\n")
    print(f"{'filter':>10s} {'overall%':>9s} {'stripes%':>9s} {'dark%':>7s} "
          f"{'missed':>7s}")
    for name, fn in FILTERS:
        right, wrong, missed = Counter(), Counter(), Counter()
        for img_path, lbl in frames:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            truth = load_labels(lbl, w, h)
            dets = strat.detect(fn(frame), None, rescan=True)
            used = set()
            for (n, cx, cy, r) in truth:
                best, bd = None, (1.2 * r) ** 2
                for i, d in enumerate(dets):
                    if i in used:
                        continue
                    d2 = (d.x - cx) ** 2 + (d.y - cy) ** 2
                    if d2 < bd:
                        bd, best = d2, i
                if best is None:
                    missed[n] += 1
                    continue
                used.add(best)
                (right if dets[best].number == n else wrong)[n] += 1
        allc = list(range(16))
        acc = _make_acc(right, wrong)
        print(f"{name:>10s} {acc(allc):>8.1f}% {acc(STRIPES):>8.1f}% "
              f"{acc(DARK):>6.1f}% {sum(missed.values()):>7d}")
    print("\nGate to ship: overall AND stripes above baseline, dark not "
          "down, missed not up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
