"""Second replay pass: render (annotated camera | schematic) side-by-sides
for flagged windows so the failure moments can be reviewed by eye.

Renders every 3rd frame in a window of [-10, +25] around each flagged index,
stacked into per-window filmstrips (2 columns x N rows of paired panels).
"""
import json
import os
import sys

import pathlib as _pl; sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "src"))
import sys as _sys

import cv2
import numpy as np

from billiards_trainer.config import Settings
from billiards_trainer.vision.pipeline import Pipeline

VIDEO = _sys.argv[1]
OUT = "/private/tmp/claude-501/-Users-joe-Documents-GitHub--CaulfieldEngineering-BilliardsTrainer/d52d3d24-6c8e-4160-8774-20b9bffe9ca7/scratchpad/review"

flags = json.load(open(f"{OUT}/flags.json"))
windows = []
for kind, i in flags:
    windows.append((max(0, i - 10), i + 25, kind, i))
windows.sort()
# merge overlapping windows
merged = []
for w in windows:
    if merged and w[0] <= merged[-1][1]:
        merged[-1] = (merged[-1][0], max(merged[-1][1], w[1]),
                      merged[-1][2] + "+" + w[2], merged[-1][3])
    else:
        merged.append(list(w))
        merged[-1] = tuple(merged[-1])
print("render windows:", [(w[0], w[1], w[2]) for w in merged])

settings = Settings.load()
pipe = Pipeline(settings, source=VIDEO)
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 18.0

os.makedirs(f"{OUT}/strips", exist_ok=True)
t = 0.0
i = -1
widx = 0
panels = []


def flush(kind, anchor):
    global panels
    if not panels:
        return
    rows = [np.hstack(panels[k:k + 2]) if k + 1 < len(panels)
            else np.hstack([panels[k], np.zeros_like(panels[k])])
            for k in range(0, len(panels), 2)]
    strip = np.vstack(rows)
    path = f"{OUT}/strips/{anchor:05d}_{kind.replace('+', '_')[:40]}.jpg"
    cv2.imwrite(path, strip, [cv2.IMWRITE_JPEG_QUALITY, 82])
    print("wrote", path, strip.shape)
    panels = []


while True:
    ok, frame = cap.read()
    if not ok:
        break
    i += 1
    t += 1.0 / fps
    res = pipe.process(frame, t)
    if widx >= len(merged):
        break
    w0, w1, kind, anchor = merged[widx]
    if i < w0:
        continue
    if i > w1:
        flush(kind, anchor)
        widx += 1
        continue
    if (i - w0) % 3 != 0:
        continue
    cam = res.frame_bgr if res.frame_bgr is not None else frame
    cam = cv2.resize(cam, (297, 528))
    schem = res.rect_bgr
    if schem is None:
        schem = np.zeros((528, 297, 3), np.uint8)
    else:
        schem = cv2.resize(schem, (int(schem.shape[1] * 528 / schem.shape[0]), 528))
    pair = np.hstack([cam, schem])
    cv2.putText(pair, f"f{i}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 2)
    panels.append(pair)
flush(merged[widx][2] if widx < len(merged) else "end",
      merged[widx][3] if widx < len(merged) else i)
print("DONE")
