"""Train the cue+1..15 ball-ID detector on the combined dataset, export ONNX.

Run with the torch+ultralytics env (pool_coach venv):
  _refs/pool_coach/.venv/Scripts/python.exe tools/train_ballid.py
Output: _eval/models/pool_ballid.onnx  (class index == ball number; 0 = cue).
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_train" / "ballid" / "data.yaml"


def main() -> int:
    from ultralytics import YOLO
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    model = YOLO("yolov8n.pt")
    model.train(data=str(DATA), epochs=epochs, imgsz=640, device=0, batch=16,
                patience=20, project=str(ROOT / "_train"), name="ballid", verbose=False,
                plots=False)
    onnx = Path(model.export(format="onnx", imgsz=640, opset=12))
    dest = ROOT / "_eval" / "models" / "pool_ballid.onnx"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx, dest)
    print(f"EXPORTED {dest} ({dest.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
