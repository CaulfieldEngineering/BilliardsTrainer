"""Persistent labeled-data store for ball-ID fine-tuning.

Lives in the app data dir so corrections accumulate across sessions and survive
updates — the growing "foundation" the user builds for their table. Stored as a
plain YOLO detection dataset (image + ``.txt`` with ``class cx cy w h`` in
normalised coords), where the class index IS the ball number (0 = cue, 1..15).
That's the exact format ``finetune_ballid`` trains on and the numbered-model path
consumes, so the loop closes with no conversion.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class LabeledBall:
    number: int           # 0 = cue, 1..15; negative => not a ball (skip)
    cx: float             # normalised centre / size (0..1) in the full frame
    cy: float
    w: float
    h: float


class TrainingStore:
    """Append-only labeled-frame store under ``<app data>/training/ballid``."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.images = self.root / "images"
        self.labels = self.root / "labels"

    def ensure(self) -> None:
        self.images.mkdir(parents=True, exist_ok=True)
        self.labels.mkdir(parents=True, exist_ok=True)

    def count(self) -> int:
        """Number of saved frames."""
        if not self.images.is_dir():
            return 0
        return sum(1 for _ in self.images.glob("*.jpg"))

    def class_counts(self) -> dict[int, int]:
        """How many labelled boxes exist per ball number (coverage at a glance)."""
        out: dict[int, int] = {}
        if not self.labels.is_dir():
            return out
        for lbl in self.labels.glob("*.txt"):
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if parts:
                    n = int(parts[0])
                    out[n] = out.get(n, 0) + 1
        return out

    def add_frame(self, frame_bgr: np.ndarray, balls: list[LabeledBall],
                  stamp: str | None = None) -> int:
        """Save one labelled frame (image + YOLO label). Balls with number < 0 are
        dropped (the user marked them 'not a ball'). Returns boxes written."""
        self.ensure()
        usable = [b for b in balls if 0 <= b.number <= 15]
        if not usable:
            return 0
        name = stamp or f"{int(time.time() * 1000)}"
        cv2.imwrite(str(self.images / f"{name}.jpg"), frame_bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        lines = [f"{b.number} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}" for b in usable]
        (self.labels / f"{name}.txt").write_text("\n".join(lines) + "\n")
        return len(usable)

    def write_data_yaml(self, path: Path | None = None) -> Path:
        """Write a YOLO data.yaml for this store (single split — small fine-tune
        sets train+val on the same data, which is fine for a per-table tune)."""
        self.ensure()
        path = path or (self.root / "data.yaml")
        names = ["cue"] + [str(i) for i in range(1, 16)]
        lines = ["path: " + str(self.root).replace("\\", "/"),
                 "train: images", "val: images", "nc: 16", "names:"]
        lines += [f"  {i}: {n}" for i, n in enumerate(names)]
        path.write_text("\n".join(lines) + "\n")
        return path
