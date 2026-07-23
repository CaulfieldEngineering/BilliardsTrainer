"""Frame-by-frame replay of Joe's 17:55 session through the LIVE pipeline.

Logs per-frame state compactly (JSONL) + an independent raw-motion signal, and
dumps side-by-side (annotated camera | schematic) JPEGs around every motion
burst so the interesting moments can be watched by eye.
"""
import json
import sys
import time

import pathlib as _pl; sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))
import sys as _sys

import cv2
import numpy as np

from billiards_trainer.config import Settings
from billiards_trainer.vision.pipeline import Pipeline

VIDEO = _sys.argv[1]
OUT = "/private/tmp/claude-501/-Users-joe-Documents-GitHub--CaulfieldEngineering-BilliardsTrainer/d52d3d24-6c8e-4160-8774-20b9bffe9ca7/scratchpad/review"
import os

os.makedirs(f"{OUT}/frames", exist_ok=True)

settings = Settings.load()
pipe = Pipeline(settings, source=VIDEO)
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 18.0
log = open(f"{OUT}/replay.jsonl", "w")

prev_small = None
t = 0.0
i = -1
t_start = time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    i += 1
    t += 1.0 / fps
    # independent motion signal: mean abs diff on a small gray image
    small = cv2.cvtColor(cv2.resize(frame, (240, 426)), cv2.COLOR_BGR2GRAY)
    motion = float(np.mean(cv2.absdiff(small, prev_small))) if prev_small is not None else 0.0
    prev_small = small

    res = pipe.process(frame, t)

    rec = {
        "i": i, "motion": round(motion, 2), "status": res.status,
        "ndet": len(res.detections),
        "tracks": [[tr.id, round(tr.x), round(tr.y), tr.number,
                    round((tr.vx**2 + tr.vy**2) ** 0.5, 1), tr.misses]
                   for tr in res.tracks],
    }
    ev = getattr(res, "shot_event", None)
    if ev is not None:
        rec["shot"] = str(getattr(ev, "outcome", ev))
    log.write(json.dumps(rec) + "\n")

    if i % 500 == 0:
        el = time.time() - t_start
        print(f"frame {i} ({el:.0f}s elapsed) motion={motion:.1f} ndet={rec['ndet']}", flush=True)

log.close()
cap.release()
print(f"DONE {i+1} frames in {time.time()-t_start:.0f}s -> {OUT}/replay.jsonl")
