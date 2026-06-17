"""Export pool_coach's YOLO11 ball detector to ONNX for the in-app A/B strategy.

We ship the YOLO11 detector via ONNX + onnxruntime (NO torch in the app — torch
collides with our opencv-headless pin and is ~2 GB). This script does the one-time
export in an ISOLATED env that has ultralytics, then you drop the resulting .onnx
into your models folder so the app's `onnx_<name>` strategy picks it up.

Setup + run (one time)::

    git clone --depth 1 https://github.com/fearlessit/pool_coach _external/pool_coach
    python -m venv _external/pool_coach/.venv
    _external/pool_coach/.venv/Scripts/python -m pip install ultralytics
    _external/pool_coach/.venv/Scripts/python tools/export_pool_coach_onnx.py

Then copy the printed .onnx into your models dir (printed by the app as
MODELS_DIR, e.g. %LOCALAPPDATA%\\BilliardsTrainer\\models\\pool_yolo11.onnx) and
pick "onnx_pool_yolo11" in Settings -> Live detector.

LICENSE: Ultralytics YOLO11 weights are AGPL-3.0. Exporting to ONNX does not
change that — redistributing the model (e.g. bundling it in a release) needs
AGPL compliance. Running it locally for your own A/B is fine.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WEIGHTS = REPO / "_external" / "pool_coach" / "model" / "best_weights.pt"


def main(imgsz: int = 640):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Run this with the ISOLATED env python that has ultralytics "
                         "(see module docstring) — not the project venv.") from exc
    if not WEIGHTS.exists():
        raise SystemExit(f"weights not found: {WEIGHTS}\nClone pool_coach first (see docstring).")
    out = YOLO(str(WEIGHTS), task="detect").export(format="onnx", imgsz=imgsz, opset=12)
    print(f"\nexported: {out}")
    print("Copy it to your models dir as pool_yolo11.onnx, then select "
          "'onnx_pool_yolo11' in Settings -> Live detector.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 640)
