"""Combine the vetted Roboflow pool datasets into ONE cue+1..15 dataset.

Unified class scheme (class index == ball number): 0 = cue, 1..15 = the numbered
balls. Each source dataset's classes are remapped to that scheme; boxes that can't
be assigned a number (generic 'Object_Ball', 'rack', 'pool table', 'Break') are
dropped. Images are copied; label files rewritten with the remapped class ids.

Output: _train/ballid/{train,valid,test}/{images,labels} + data.yaml — ready for
`yolo detect train`. Run:  _refs/pool_coach/.venv/Scripts/python.exe tools/build_ballid_dataset.py
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "_eval" / "datasets"
OUT = ROOT / "_train" / "ballid"

# per-dataset: source-class-index -> unified ball number (or None to drop).
# Built from each data.yaml's names list (order matters).
POOL_BALL = ["0", "1", "10", "11", "12", "13", "14", "15", "2", "3", "4", "5", "6", "7", "8", "9"]
NWMSH = ["Break", "Cue_Ball", "Eight", "Five", "Four", "Nine", "Object_Ball",
         "One", "Seven", "Six", "Three", "Two"]
WPB3Z = ["ball 1", "ball 2", "ball 3", "ball 4", "ball 5", "ball 6", "ball 7",
         "ball 8", "ball 9", "pool table", "rack", "white ball"]
_WORD = {"Cue_Ball": 0, "One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5,
         "Six": 6, "Seven": 7, "Eight": 8, "Nine": 9}


def _map_pool_ball(i):
    return int(POOL_BALL[i])                       # name is the number; 0=cue


def _map_nwmsh(i):
    return _WORD.get(NWMSH[i])                      # Break/Object_Ball -> None


def _map_wpb3z(i):
    n = WPB3Z[i]
    if n == "white ball":
        return 0
    if n.startswith("ball "):
        return int(n.split()[1])
    return None                                     # table/rack -> None


SOURCES = [
    ("pool-ball-detection", _map_pool_ball),
    ("pool-billiard-nwmsh", _map_nwmsh),
    ("billiard-pool-wpb3z", _map_wpb3z),
]


def remap_split(ds: str, mapper, split: str, counts: dict) -> None:
    base = SRC / ds / split
    imgs = base / "images"
    labels = base / "labels"
    if not imgs.is_dir():
        return
    out_img = OUT / split / "images"
    out_lbl = OUT / split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    for img in imgs.iterdir():
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl = labels / (img.stem + ".txt")
        new_lines = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                num = mapper(int(parts[0]))
                if num is None:
                    continue
                new_lines.append(f"{num} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
        # unique name (datasets share stems); skip images with no usable boxes
        if not new_lines:
            continue
        stem = f"{ds[:6]}_{img.stem}"
        shutil.copy2(img, out_img / (stem + img.suffix))
        (out_lbl / (stem + ".txt")).write_text("\n".join(new_lines))
        counts[split] = counts.get(split, 0) + 1
        for _ln in new_lines:
            counts["boxes"] = counts.get("boxes", 0) + 1


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    counts: dict = {}
    for ds, mapper in SOURCES:
        for split in ("train", "valid", "test"):
            remap_split(ds, mapper, split, counts)
    names = ["cue"] + [str(i) for i in range(1, 16)]
    yaml = ["path: " + str(OUT).replace("\\", "/"),
            "train: train/images", "val: valid/images", "test: test/images",
            "nc: 16", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(names)]
    (OUT / "data.yaml").write_text("\n".join(yaml) + "\n")
    print("Combined dataset ->", OUT)
    print("  images:", {k: v for k, v in counts.items() if k != "boxes"},
          " total boxes:", counts.get("boxes", 0))
    print("  data.yaml: 16 classes (0=cue, 1..15)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
