"""Fetch optional detector models into the user models dir (no torch).

Lets the shipped app pull a pre-exported .onnx detector on demand so it appears in
Settings -> Live detector as an A/B option. We host the model rather than bundle it
(keeps the base download small; and lets the AGPL model live as a release asset with
attribution rather than inside the app binary).
"""

import hashlib
import logging

import requests

from ..config import MODELS_DIR

log = logging.getLogger("model_fetch")

# key -> spec. `strategy` is the name it shows up as once present (onnx_<stem>).
FETCHABLE = {
    "pool_yolo11": {
        "filename": "pool_yolo11.onnx",
        "url": ("https://github.com/CaulfieldEngineering/BilliardsTrainer/"
                "releases/download/model-assets-v1/pool_yolo11.onnx"),
        "sha256": "7c84d297aaf271c5e7aefdcefe4430f476a7619849aa701f018f7e5aefad3f8b",
        "label": "YOLO11 ball detector (~38 MB)",
        "strategy": "onnx_pool_yolo11",
        "license": "AGPL-3.0 (derived from Ultralytics YOLO11 / pool_coach)",
    },
}


def model_path(key: str):
    return MODELS_DIR / FETCHABLE[key]["filename"]


def is_present(key: str) -> bool:
    return model_path(key).exists()


def fetch_model(key: str, progress=None):
    """Download the model to MODELS_DIR (atomic + sha-verified). progress(frac 0..1)
    is called as bytes arrive. Returns the destination Path; raises on failure."""
    spec = FETCHABLE[key]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = model_path(key)
    tmp = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    log.info("fetching model %s from %s", key, spec["url"])
    with requests.get(spec["url"], stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                if not chunk:
                    continue
                f.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if progress and total:
                    progress(min(1.0, done / total))
    expected = spec.get("sha256", "")
    if expected and h.hexdigest() != expected:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"checksum mismatch for {key} (got {h.hexdigest()[:12]})")
    tmp.replace(dest)
    log.info("model %s saved to %s", key, dest)
    return dest
