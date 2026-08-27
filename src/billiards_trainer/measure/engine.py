"""M1 offline engine: re-process a recorded session into a DENSE sidecar.

Every frame decoded, the finder runs EVERY frame (positions), the
identifier every IDENT_EVERY frames (numbers persist between votes),
and the motion tracker carries identity through blur. Output goes to a
scratch path as a normal analysis sidecar (SidecarReader-compatible),
never anywhere near the session's real files until promotion.

    python -m billiards_trainer.measure.engine <video> [out_dir]

Guards: BelowNormal priority; aborts between frames if a recording
starts (the GPU belongs to the table).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

log = logging.getLogger("measure.engine")

IDENT_EVERY = 6          # identifier cadence (votes persist on tracks)
PROGRESS_EVERY_S = 30.0


def reprocess(video: str, out_dir: str | None = None,
              max_frames: int = 0) -> dict:
    import cv2

    from ..config import EXPORTS_DIR
    from ..detector_strategies import discover
    from ..vision.analysis_cache import SidecarWriter
    from .tracker import MotionTracker

    video = Path(video)
    out_root = Path(out_dir) if out_dir else (video.parent / "m1")
    out_root.mkdir(parents=True, exist_ok=True)
    out_video_alias = out_root / video.name    # sidecar naming anchor

    strat = discover()["ensemble_findid"]
    strat.inference_provider = "dml"
    finder, ident = strat._finder, strat._identifier

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = SidecarWriter(out_video_alias, {"fps": fps, "engine": "m1",
                                             "dense": True,
                                             "source": video.name})
    tracker = MotionTracker()
    ident_by_pos: list = []      # latest identifier detections (x, y, n)

    def recording_live() -> bool:
        return bool(list(EXPORTS_DIR.glob(".session-*.part.mp4")))

    t0_wall = time.time()
    last_prog = t0_wall
    fi = 0
    written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = fi / fps
            if fi % 150 == 0 and recording_live():
                log.warning("recording started - engine run aborted at %.1fs", t)
                return {"aborted": True, "frames": fi}
            found = finder.detect(frame, None) or []
            if fi % IDENT_EVERY == 0:
                ids = ident.detect(frame, None) or []
                ident_by_pos = [(d.x, d.y, d.number) for d in ids
                                if getattr(d, "number", -1) >= 0]
            dets = []
            for d in found:
                num = -1
                best = 2.5 * max(d.radius, 8.0)
                for (ix, iy, n) in ident_by_pos:
                    dd = ((d.x - ix) ** 2 + (d.y - iy) ** 2) ** 0.5
                    if dd < best:
                        best, num = dd, n
                if getattr(d, "number", -1) >= 0:
                    num = d.number       # the finder's own read wins
                dets.append((float(d.x), float(d.y),
                             float(d.radius), int(num)))
            rows = tracker.update(dets, t)
            writer.add_frame(t, rows)
            written += 1
            fi += 1
            if max_frames and fi >= max_frames:
                break
            now = time.time()
            if now - last_prog > PROGRESS_EVERY_S:
                rate = fi / (now - t0_wall)
                eta = (n_total - fi) / max(0.1, rate) / 60
                print(f"[m1] {fi}/{n_total} frames, {rate:.1f} fps, "
                      f"eta {eta:.0f} min", flush=True)
                last_prog = now
    finally:
        writer.close()
        cap.release()
    wall = time.time() - t0_wall
    out = {"frames": fi, "written": written, "wall_s": round(wall, 1),
           "proc_fps": round(fi / max(0.1, wall), 1),
           "sidecar": str(out_video_alias) + ".analysis.jsonl"}
    log.info("m1 reprocess done: %s", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    v = sys.argv[1]
    res = reprocess(v, sys.argv[2] if len(sys.argv) > 2 else None)
    print(res)
