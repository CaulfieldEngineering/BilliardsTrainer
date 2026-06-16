"""Batch-download Roboflow Universe datasets for the eval/dataset catalog.

Takes one or more ``workspace/project`` slugs, downloads each (latest version,
YOLOv8 format) into ``_eval/datasets/<project>`` (gitignored), and prints a
one-line summary per dataset (image count, classes, license) — WITHOUT ever
printing the API key.

SECURITY: key from ``--api-key`` or ``ROBOFLOW_API_KEY`` only; never logged.

    python tools/fetch_datasets.py ben-gann-lscqy/pool-ball-detection ...
"""

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST_ROOT = ROOT / "_eval" / "datasets"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass


def summarize(loc: Path) -> dict:
    yaml = loc / "data.yaml"
    info = {"images": 0, "classes": "?", "license": "?", "nc": "?"}
    info["images"] = sum(1 for _ in loc.rglob("*.jpg")) + sum(1 for _ in loc.rglob("*.png"))
    if yaml.exists():
        txt = yaml.read_text(encoding="utf-8")
        ncm = re.search(r"nc:\s*(\d+)", txt)
        info["nc"] = ncm.group(1) if ncm else "?"
        lic = re.search(r"license:\s*(.+)", txt)
        info["license"] = lic.group(1).strip() if lic else "?"
        nm = re.search(r"names:\s*(\[.*?\]|(?:\n\s*-\s*.*)+)", txt)
        info["classes"] = re.sub(r"\s+", " ", nm.group(1).strip())[:200] if nm else "?"
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("slugs", nargs="+", help="workspace/project slugs")
    ap.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""))
    ap.add_argument("--fmt", default="yolov8")
    args = ap.parse_args()
    if not args.api_key:
        print("ERROR: no API key (--api-key or ROBOFLOW_API_KEY).", file=sys.stderr)
        return 2
    from roboflow import Roboflow
    rf = Roboflow(api_key=args.api_key)   # key used only here; never echoed

    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for slug in args.slugs:
        ws, _, proj = slug.partition("/")
        dest = DEST_ROOT / proj
        try:
            project = rf.workspace(ws).project(proj)
            try:
                ver = max(int(str(v.version).split("/")[-1]) for v in project.versions())
            except Exception:  # noqa: BLE001
                ver = 1
            print(f"[{slug}] downloading v{ver} -> {dest}")
            project.version(ver).download(args.fmt, location=str(dest), overwrite=True)
            info = summarize(dest)
            print(f"  OK images={info['images']} nc={info['nc']} license={info['license']}")
            results.append((slug, dest, info, None))
        except Exception as exc:  # noqa: BLE001 - one bad slug shouldn't abort the batch
            msg = re.sub(r"api_key=[^&\s]+", "api_key=***", str(exc))   # never leak key
            print(f"  FAILED: {msg[:200]}")
            results.append((slug, dest, None, msg[:200]))
    print(f"\nDownloaded {sum(1 for r in results if r[2])}/{len(args.slugs)} datasets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
