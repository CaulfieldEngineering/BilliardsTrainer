"""Ultralytics YOLO (.pt) strategies — RAW frame, requires torch+ultralytics.

Auto-registers one strategy per ``.pt`` in the eval models dir (e.g. the
lakpahana/8ball weights). Ultralytics is imported lazily; if it (or torch) isn't
installed, the strategy raises on first use and the harness marks the variant
FAILED — a deliberate, honest self-healing demonstration, since we do NOT install
torch into this venv (it conflicts with the pinned opencv-headless). To actually
run these, convert the .pt to .onnx in a separate env and drop the .onnx in
``_eval/models/`` — it then runs via the no-torch ONNX strategy.
"""

from pathlib import Path

import numpy as np

from ..vision.types import Detection
from . import DetectorStrategy, ball_radius_raw, classify_crop, table_polygon_mask

ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "_eval" / "models"


class UltralyticsPtStrategy(DetectorStrategy):
    def __init__(self, model_path: Path, conf: float = 0.25):
        self._path = Path(model_path)
        self.name = f"yolo_{self._path.stem}"
        self.description = f"Ultralytics YOLO (.pt) on raw frame: {self._path.name}"
        self._conf = conf
        self._model = None

    def detect(self, frame_bgr, calib):
        if self._model is None:
            from ultralytics import YOLO  # raises if torch/ultralytics absent -> self-heal
            self._model = YOLO(str(self._path))
        res = self._model(frame_bgr, conf=self._conf, verbose=False)[0]
        table = table_polygon_mask(frame_bgr.shape, calib)
        rmax = ball_radius_raw(calib, frame_bgr.shape) * 6.0
        h, w = frame_bgr.shape[:2]
        out = []
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy, r = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1 + y2 - y1) / 4
            if r > rmax or r < 2:
                continue
            ix, iy = int(np.clip(cx, 0, w - 1)), int(np.clip(cy, 0, h - 1))
            if table[iy, ix] == 0:
                continue
            cls, bgr = classify_crop(frame_bgr, cx, cy, r)
            out.append(Detection(cx, cy, r, bgr, cls, float(box.conf[0])))
        return out


def _build():
    if not MODELS_DIR.exists():
        return []
    return [UltralyticsPtStrategy(p) for p in sorted(MODELS_DIR.glob("*.pt"))]


STRATEGIES = _build()
