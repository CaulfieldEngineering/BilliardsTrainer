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


def _acquire_calib(video):
    """The session's own rectification, re-acquired from its frames
    (same warmup shots_export._video_transform uses for hinv)."""
    import cv2

    from ..config import Settings
    from ..vision.pipeline import Pipeline
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    try:
        pipe = Pipeline(Settings.load())
        for n in range(1, 91):
            ok, fr = cap.read()
            if not ok:
                break
            pipe.process(fr, n / 30.0, annotate=False, detect=False)
            if pipe.calib.calib is not None and n >= 20:
                break
        return pipe.calib.calib
    finally:
        cap.release()


def reprocess(video: str, out_dir: str | None = None,
              max_frames: int = 0, calib=None) -> dict:
    import cv2

    from ..config import EXPORTS_DIR
    from ..detector_strategies import discover
    from ..vision.analysis_cache import SidecarWriter
    from .tracker import MotionTracker

    video = Path(video)
    out_root = Path(out_dir) if out_dir else (video.parent / "m1")
    out_root.mkdir(parents=True, exist_ok=True)
    out_video_alias = out_root / video.name    # sidecar naming anchor

    if calib is None:
        # Acquire calibration FROM THE VIDEO (the export's own recipe) so
        # engine coords live in the session's rectified space by
        # construction. calib=None auto-find drifted per segment on the
        # first marathon run and no single transform could repair it
        # (229px median even borrowing a same-dims neighbour lock).
        calib = _acquire_calib(video)
        if calib is None:
            log.warning("calibration warmup failed for %s - auto-find "
                        "fallback (coords may not match session space)",
                        video.name)

    strat = discover()["ensemble_findid"]
    strat.inference_provider = "dml"
    finder, ident = strat._finder, strat._identifier

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = SidecarWriter(out_video_alias, {"fps": fps, "engine": "m1",
                                             "dense": True,
                                             "calibrated": calib is not None,
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
            found = finder.detect(frame, calib) or []
            if fi % IDENT_EVERY == 0:
                ids = ident.detect(frame, calib) or []
                ident_by_pos = [(d.x, d.y, d.number) for d in ids
                                if getattr(d, "number", -1) >= 0]
            # EXCLUSIVE finder<->identifier pairing (one identifier read
            # feeds ONE finder detection - the first marathon run let
            # neighbours share a read and spread duplicate numbers)
            cand = []
            for fi_d, d in enumerate(found):
                lim = 2.5 * max(d.radius, 8.0)
                for ii, (ix, iy, n) in enumerate(ident_by_pos):
                    dd = ((d.x - ix) ** 2 + (d.y - iy) ** 2) ** 0.5
                    if dd < lim:
                        cand.append((dd, fi_d, ii))
            cand.sort()
            num_for: dict = {}
            used_i: set = set()
            for _dd, fi_d, ii in cand:
                if fi_d in num_for or ii in used_i:
                    continue
                num_for[fi_d] = ident_by_pos[ii][2]
                used_i.add(ii)
            dets = []
            for fi_d, d in enumerate(found):
                num = num_for.get(fi_d, -1)
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
