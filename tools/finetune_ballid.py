"""Fine-tune the cue+1..15 ball-ID model on the user's labelled store.

Runs in a torch env (NOT the shipped, torch-free app) — the in-app trainer shells
out to it, or run it by hand. Starts from the pretrained ball-ID weights so it
ADAPTS to this table while keeping the base knowledge, then exports ONNX into the
models dir where the app's numbered-model path uses it.

  <torch-python> tools/finetune_ballid.py --data <store>/data.yaml --out <models>/pool_ballid.onnx
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="store data.yaml")
    ap.add_argument("--out", required=True, help="output .onnx path")
    ap.add_argument("--base", default="", help="base .pt weights (default: prior ball-ID run, else yolov8n)")
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics/torch not available in this python. Run with a "
              "torch env (e.g. the pool_coach venv).", file=sys.stderr)
        return 2

    base = args.base
    if not base:
        prior = ROOT / "_train" / "ballid" / "weights" / "best.pt"
        prior2 = ROOT / "_train" / "ballid-2" / "weights" / "best.pt"
        base = str(prior if prior.exists() else prior2 if prior2.exists() else "yolov8n.pt")
    print(f"Fine-tuning from {base} on {args.data} for {args.epochs} epochs …")
    model = YOLO(base)
    model.train(data=args.data, epochs=args.epochs, imgsz=640, device=0, batch=16,
                patience=15, project=str(ROOT / "_train"), name="ballid_finetune",
                plots=False, verbose=False)
    onnx = Path(model.export(format="onnx", imgsz=640, opset=12))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx, out)
    print(f"FINETUNE_OK {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
