"""Turn the Roboflow **Pool V2** dataset into a local, offline ONNX detector.

Why this script exists
----------------------
Roboflow Universe gates *everything* behind a free API key — the public API and
the `roboflow` SDK both refuse anonymous access (verified). And their *hosted*
inference API would send every camera frame to Roboflow's servers, which breaks
this app's hard rule of staying local/offline. The dataset, however, is
**CC BY 4.0** (free, attribution), so the constraint-compliant path is:

    download the dataset  ->  train YOLOv8n locally  ->  export ONNX  ->  drop in

The exported ``pool_balls.onnx`` lands in the app's models dir, where the
existing :class:`OnnxYoloDetector` (added in v0.2.15) picks it up automatically —
running fully offline with no torch at inference time.

Usage
-----
1. Make a free Roboflow account, copy your API key from
   https://app.roboflow.com/settings/api
2. Install the training extra (this is a one-time, offline step; it pulls in
   torch, ~2 GB — only needed to TRAIN, never to run the app):

       pip install -e ".[yolo]" roboflow

3. Train + export (≈ tens of minutes on CPU, minutes on a GPU):

       python tools/train_pool_model.py --api-key YOUR_KEY            # full run
       python tools/train_pool_model.py --api-key YOUR_KEY --epochs 50
       python tools/train_pool_model.py --api-key YOUR_KEY --download-only

   The key may also be supplied via the ROBOFLOW_API_KEY environment variable.

4. Launch the app — Settings → Detection toggle is now available and runs the
   pool model locally.

Attribution: "Pool V2" by the *Pool Table* workspace on Roboflow Universe,
CC BY 4.0 — https://universe.roboflow.com/pool-table/pool-v2
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

WORKSPACE = "pool-table"
PROJECT = "pool-v2"
DATASET_URL = "https://universe.roboflow.com/pool-table/pool-v2"


def _die(msg: str) -> "int":
    print(f"\nERROR: {msg}", file=sys.stderr)
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a local ONNX pool-ball detector "
                                             "from the Roboflow Pool V2 dataset.")
    ap.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""),
                    help="Roboflow API key (or set ROBOFLOW_API_KEY).")
    ap.add_argument("--version", type=int, default=0,
                    help="Dataset version number (default: latest available).")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--model", default="yolov8n.pt", help="base weights to fine-tune")
    ap.add_argument("--download-only", action="store_true",
                    help="Just download the dataset; skip training.")
    ap.add_argument("--out", default="", help="Where to write pool_balls.onnx "
                    "(default: the app's models dir).")
    args = ap.parse_args()

    if not args.api_key:
        return _die("No Roboflow API key. Get a free one at "
                    "https://app.roboflow.com/settings/api and pass --api-key "
                    "or set ROBOFLOW_API_KEY. The dataset is CC BY 4.0 but Roboflow "
                    "requires a (free) key to download it — anonymous access is "
                    "refused.")

    try:
        from roboflow import Roboflow
    except ImportError:
        return _die("The 'roboflow' package isn't installed. Run: pip install roboflow")

    print(f"Connecting to Roboflow project {WORKSPACE}/{PROJECT} …")
    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)

    version_num = args.version
    if not version_num:
        try:
            versions = project.versions()
            version_num = max(int(v.version.split("/")[-1]) for v in versions)
        except Exception:  # noqa: BLE001 - SDK shape varies; fall back to 1
            version_num = 1
    print(f"Using dataset version {version_num}")

    version = project.version(version_num)
    dataset = version.download("yolov8")
    data_yaml = Path(dataset.location) / "data.yaml"
    print(f"Dataset downloaded to {dataset.location}")
    print(f"  classes/data.yaml: {data_yaml}")

    if args.download_only:
        print("\n--download-only: stopping before training.")
        print(f"Train later with: yolo detect train data={data_yaml} "
              f"model={args.model} epochs={args.epochs} imgsz={args.imgsz}")
        return 0

    try:
        from ultralytics import YOLO
    except ImportError:
        return _die("ultralytics isn't installed. Run: pip install -e \".[yolo]\"")

    print(f"\nTraining {args.model} for {args.epochs} epochs (imgsz={args.imgsz}) …")
    model = YOLO(args.model)
    results = model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz,
                          project=str(ROOT / "_train"), name="pool_v2")
    print("Exporting to ONNX …")
    onnx_path = Path(model.export(format="onnx", imgsz=args.imgsz))

    if args.out:
        dest = Path(args.out)
    else:
        from billiards_trainer.config import MODELS_DIR
        dest = MODELS_DIR / "pool_balls.onnx"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, dest)

    print("\n" + "=" * 64)
    print(f"DONE. Pool model written to:\n  {dest}")
    print("Launch the app — Settings → turn Detection ON. It runs locally, offline.")
    print(f"Best metrics dir: {getattr(results, 'save_dir', ROOT / '_train')}")
    print("\nAttribution (CC BY 4.0): 'Pool V2' by the Pool Table workspace,")
    print(f"  {DATASET_URL}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
