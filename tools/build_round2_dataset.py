"""Combine per-session auto-label datasets into ONE round-2 training set,
split every frame into the same two tiles the live detector uses (top 60% +
bottom 60%), and make a grouped train/val split.

Why tiles: inference letterboxes 60%-height tiles into the model's 640px
input. Training on identically-shaped crops means the model sees balls at the
same ~15px scale at train and test time — the round-1 failure trained on
full frames where balls letterboxed down to ~9px.

Why grouped split: propagation multiplies each hand-labeled layout into ~N
near-identical frames; a random split would leak them across train/val and
fake the validation score. All samples from one layout stay on one side.
"""

import random
import re
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SESSIONS = ["s1", "s2", "s3"]
SRC = ROOT / "_train" / "autolabel"
OUT = ROOT / "_train" / "round2"
TILE_FRAC = 0.60
VAL_FRAC = 0.15


def tile_labels(lines, y0_frac, y1_frac):
    """Remap normalized YOLO boxes into the [y0_frac, y1_frac) horizontal band."""
    span = y1_frac - y0_frac
    out = []
    for ln in lines:
        parts = ln.split()
        if len(parts) != 5:
            continue
        cls, cx, cy, w, h = parts[0], *map(float, parts[1:])
        if not (y0_frac <= cy < y1_frac):
            continue                      # box centre outside this tile
        ncy = (cy - y0_frac) / span
        nh = min(h / span, 1.0)
        out.append(f"{cls} {cx:.6f} {ncy:.6f} {w:.6f} {nh:.6f}")
    return out


def main() -> int:
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    # group key: session + layout number (L07 and its propagated variants stay
    # together on one side of the split)
    groups: dict[str, list[tuple[Path, Path]]] = {}
    for s in SESSIONS:
        ds = SRC / s / "dataset"
        for img in sorted((ds / "images").glob("*")):
            lbl = ds / "labels" / (img.stem + ".txt")
            if not lbl.is_file():
                continue
            m = re.match(r"(L\d+)", img.stem)
            key = f"{s}_{m.group(1) if m else img.stem}"
            groups.setdefault(key, []).append((img, lbl))

    keys = sorted(groups)
    random.Random(42).shuffle(keys)
    n_val = max(1, int(len(keys) * VAL_FRAC))
    val_keys = set(keys[:n_val])
    counts = {"train": 0, "val": 0, "boxes": 0}

    for key, pairs in groups.items():
        split = "val" if key in val_keys else "train"
        s = key.split("_")[0]
        for img_path, lbl_path in pairs:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h = img.shape[0]
            th = int(h * TILE_FRAC)
            lines = [ln for ln in lbl_path.read_text().splitlines() if ln.strip()]
            for tag, (a, b), (fa, fb) in (
                ("top", (0, th), (0.0, TILE_FRAC)),
                ("bot", (h - th, h), (1.0 - TILE_FRAC, 1.0)),
            ):
                tl = tile_labels(lines, fa, fb)
                if not tl:
                    continue              # empty tiles teach nothing here
                name = f"{s}_{img_path.stem}_{tag}"
                cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"),
                            img[a:b], [cv2.IMWRITE_JPEG_QUALITY, 92])
                (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(tl) + "\n")
                counts[split] += 1
                counts["boxes"] += len(tl)

    names = "\n".join(f"  {i}: '{i}'" for i in range(16))
    (OUT / "data.yaml").write_text(
        f"path: {OUT.resolve()}\ntrain: images/train\nval: images/val\n"
        f"nc: 16\nnames:\n{names}\n")
    print(f"round2 dataset: {counts['train']} train + {counts['val']} val tiles, "
          f"{counts['boxes']} boxes, {len(keys)} layout groups -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
