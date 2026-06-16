"""Download the Roboflow **Pool V2** dataset locally for Phase-0 eval ground truth.

The dataset (workspace ``pool-table``, project ``pool-v2``) is CC BY 4.0 but
Roboflow gates downloads behind a free API key. This script fetches the dataset
(images + YOLO bbox annotations) into ``_eval/datasets/pool_v2`` (gitignored) so
the eval harness can score the detector against real labelled pool stills.

SECURITY: the API key is read from ``--api-key`` or the ``ROBOFLOW_API_KEY`` env
var and is NEVER printed, logged, or written to any file. After download you
don't need it again — everything downstream is local.

    set ROBOFLOW_API_KEY=...        (or pass --api-key ...)
    python tools/fetch_pool_v2.py

Attribution (CC BY 4.0): "Pool V2" by the Pool Table workspace, Roboflow Universe
— https://universe.roboflow.com/pool-table/pool-v2
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE, PROJECT = "pool-table", "pool-v2"
DEST = ROOT / "_eval" / "datasets" / "pool_v2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""))
    ap.add_argument("--version", type=int, default=0, help="0 = latest")
    ap.add_argument("--format", default="yolov8")
    args = ap.parse_args()
    if not args.api_key:
        print("ERROR: no API key. Pass --api-key or set ROBOFLOW_API_KEY "
              "(the key is never stored or logged).", file=sys.stderr)
        return 2

    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: pip install roboflow", file=sys.stderr)
        return 2

    DEST.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=args.api_key)            # key used here only, never echoed
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version_num = args.version
    if not version_num:
        try:
            version_num = max(int(str(v.version).split("/")[-1]) for v in project.versions())
        except Exception:  # noqa: BLE001
            version_num = 1
    print(f"Downloading {WORKSPACE}/{PROJECT} v{version_num} ({args.format}) -> {DEST}")
    # overwrite=True so a pre-existing (e.g. empty) dest dir doesn't skip the fetch
    ds = project.version(version_num).download(args.format, location=str(DEST), overwrite=True)

    loc = Path(ds.location)
    yaml = loc / "data.yaml"
    classes = "?"
    if yaml.exists():
        import re
        txt = yaml.read_text(encoding="utf-8")
        m = re.search(r"names:\s*(\[.*\]|\n(?:\s*-\s*.*\n?)+)", txt)
        classes = (m.group(1).strip() if m else "?")
    n_imgs = sum(1 for _ in loc.rglob("*.jpg")) + sum(1 for _ in loc.rglob("*.png"))
    print(f"\nDone. {n_imgs} images at {loc}")
    print(f"classes: {classes}")
    print("License: CC BY 4.0 — 'Pool V2' by the Pool Table workspace, Roboflow Universe.")
    print("The API key is no longer needed; all downstream use is local.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
