"""G2 instrument: per-class identity accuracy against hand-labelled truth.

Runs the LIVE detection stack (the same ensemble + colour corrections the
app uses) over every labelled dataset frame and scores each class:
right/wrong/missed, plus a confusion table for the pairs that matter
(3/4/7/8 — the dark cluster). This is the tool G2's definition of done
names: red-family >=95% on truth.

    python tools/measure_class_accuracy.py
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billiards_trainer.detector_strategies import discover  # noqa: E402


def load_labels(lbl: Path, w: int, h: int):
    out = []
    for line in lbl.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        n = int(parts[0])
        cx, cy = float(parts[1]) * w, float(parts[2]) * h
        r = max(float(parts[3]) * w, float(parts[4]) * h) / 2.0
        out.append((n, cx, cy, r))
    return out


def main() -> int:
    strategies = discover()
    strat = strategies.get("ensemble_findid")
    if strat is None:
        print("ensemble not available", file=sys.stderr)
        return 2

    right: Counter = Counter()
    wrong: Counter = Counter()
    missed: Counter = Counter()
    confusion: dict[int, Counter] = defaultdict(Counter)

    datasets = sorted(ROOT.glob("_train/autolabel*/dataset"))
    n_imgs = 0
    for ds in datasets:
        for img_path in sorted((ds / "images").glob("*")):
            lbl = ds / "labels" / (img_path.stem + ".txt")
            if not lbl.exists():
                continue
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            n_imgs += 1
            h, w = frame.shape[:2]
            truth = load_labels(lbl, w, h)
            dets = strat.detect(frame, None, rescan=True)
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
                got = dets[best].number
                if got == n:
                    right[n] += 1
                else:
                    wrong[n] += 1
                    confusion[n][got] += 1

    print(f"{n_imgs} labelled frames\n")
    print(f"{'class':>5s} {'right':>6s} {'wrong':>6s} {'miss':>5s} {'acc%':>6s}")
    for n in sorted(set(right) | set(wrong) | set(missed)):
        tot = right[n] + wrong[n]
        acc = 100.0 * right[n] / tot if tot else 0.0
        print(f"{n:>5d} {right[n]:>6d} {wrong[n]:>6d} {missed[n]:>5d} {acc:>5.1f}%")
    dark = [3, 4, 7, 8]
    dtot = sum(right[n] + wrong[n] for n in dark)
    dright = sum(right[n] for n in dark)
    if dtot:
        print(f"\nDARK CLUSTER (3/4/7/8): {dright}/{dtot} = {100.0*dright/dtot:.1f}%"
              f"   (G2 target: >=95%)")
    print("\nconfusions (truth -> got):")
    for n in sorted(confusion):
        for got, c in confusion[n].most_common(3):
            print(f"   {n} -> {got}: {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
