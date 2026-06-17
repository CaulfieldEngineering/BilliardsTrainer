"""Reproduce the pool_coach YOLO11+ByteTrack evaluation (see docs/PRIOR_ART.md).

This is deliberately run from an ISOLATED env so torch never collides with our
opencv-headless pin. Setup (one time)::

    git clone --depth 1 https://github.com/fearlessit/pool_coach _external/pool_coach
    python -m venv _external/pool_coach/.venv
    _external/pool_coach/.venv/Scripts/python -m pip install ultralytics

Then run with THAT interpreter (not the project venv)::

    _external/pool_coach/.venv/Scripts/python tools/eval_pool_coach.py <video.mp4>

It prints honest metrics (dets/frame, CPU fps, unique track IDs + persistence)
and writes a few annotated frames next to the video for visual judgement. There
is intentionally NO accuracy-vs-ground-truth number — we have no trustworthy GT,
so this is visual + stability evidence only (see PRIOR_ART caveats).
"""
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL = REPO / "_external" / "pool_coach" / "model" / "best_weights.pt"
TRACKER = REPO / "_external" / "pool_coach" / "bytetrack.yaml"


def evaluate(video: str, start_frame: int = 0, n: int = 150, conf: float = 0.25):
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(MODEL), task="detect")
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    out_dir = Path(video).with_suffix("")
    out_dir.mkdir(exist_ok=True)
    counts, lifetimes, saved = [], defaultdict(int), 0
    t0 = time.perf_counter()
    done = 0
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        r = model.track(frame, persist=True, tracker=str(TRACKER), conf=conf, verbose=False)[0]
        done += 1
        counts.append(0 if r.boxes is None else len(r.boxes))
        if r.boxes is not None and r.boxes.id is not None:
            for tid in r.boxes.id.int().tolist():
                lifetimes[tid] += 1
        if i % 30 == 0 and saved < 6:
            cv2.imwrite(str(out_dir / f"frame_{i:04d}.png"), r.plot())
            saved += 1
    cap.release()
    dt = time.perf_counter() - t0
    lt = sorted(lifetimes.values(), reverse=True)
    print(f"frames={done} cpu_fps={done / dt:.1f} "
          f"dets/frame min={min(counts)} med={int(st.median(counts))} max={max(counts)} "
          f"unique_ids={len(lifetimes)} top_lifetimes={lt[:5]} annotated->{out_dir}")


if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "testVideo.MP4")
    sf = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    evaluate(vid, start_frame=sf)
