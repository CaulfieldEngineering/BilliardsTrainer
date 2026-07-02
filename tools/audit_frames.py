"""Audit the app's LIVE tracked output against the raw video, Joe's way:
run the pipeline continuously (warm tracker state), and at sample points save
a side-by-side of the camera view and the schematic (with numbers) for human
(Claude) review."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"c:\Users\jpcfo\Documents\_GitHub\_CaulfieldEngineering\BilliardsTrainer")
sys.path.insert(0, str(ROOT / "src"))
OUT = Path(__file__).parent / "audit"
OUT.mkdir(exist_ok=True)

from billiards_trainer.config import Settings  # noqa: E402
from billiards_trainer.vision.pipeline import Pipeline  # noqa: E402

# (warmup_start_s, [sample offsets within the window]) — pipeline runs the whole
# window so tracker state is warm and realistic at each sample.
WINDOWS = [
    (100, [8, 20]),      # settled mid-game
    (478, [10, 16, 24]), # active play: strike + rolling + after
    (1140, [10]),        # late-game settled
    (2280, [12]),        # end-game
]

cap = cv2.VideoCapture(str(ROOT / "testVideo.MP4"))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
settings = Settings()

for start, offsets in WINDOWS:
    pipe = Pipeline(settings, source=str(ROOT / "testVideo.MP4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    marks = {int(o * fps) for o in offsets}
    dur = int((max(offsets) + 1) * fps)
    for i in range(dur):
        ok, frame = cap.read()
        if not ok:
            break
        res = pipe.process(frame, i / fps)
        if i in marks and res.rect_bgr is not None:
            # camera crop of the table, schematic beside it, same height
            cam = frame[100:860, 380:1530]
            h = 700
            cam_r = cv2.resize(cam, (int(cam.shape[1] * h / cam.shape[0]), h))
            sch = res.rect_bgr
            sch_r = cv2.resize(sch, (int(sch.shape[1] * h / sch.shape[0]), h))
            canvas = np.full((h, cam_r.shape[1] + sch_r.shape[1] + 12, 3), 30, np.uint8)
            canvas[:, :cam_r.shape[1]] = cam_r
            canvas[:, cam_r.shape[1] + 12:] = sch_r
            t = start + i / fps
            nums = sorted(tr.number for tr in res.tracks)
            cv2.putText(canvas, f"t={t:.1f}s tracks={len(res.tracks)} nums={nums}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imwrite(str(OUT / f"audit_t{int(t):04d}.png"), canvas)
            print(f"audit_t{int(t):04d}: {len(res.tracks)} tracks nums={nums}")
cap.release()
print("done ->", OUT)
