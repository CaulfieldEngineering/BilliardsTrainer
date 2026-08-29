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

import json
import logging
import sys
import time
from pathlib import Path

log = logging.getLogger("measure.engine")

IDENT_EVERY = 6          # identifier cadence (votes persist on tracks)
PROGRESS_EVERY_S = 30.0  # log-line cadence
PROGRESS_FILE_S = 2.0    # UI progress-file cadence (Joe: see percent)

#: bump when tracker/filter RULES change - the gate refuses sidecars
#: from older rules (a stale pre-hardened sidecar once gated at 184/1k
#: and nearly condemned a good session)
ENGINE_RULES_V = 12  # v12: coasts stop at the bed edge


def _joe_present(idle_min: float = 10.0) -> bool:
    """tools/_presence with a fail-toward-present default."""
    try:
        import sys as _sys
        tools = str(Path(__file__).resolve().parents[3] / "tools")
        if tools not in _sys.path:
            _sys.path.insert(0, tools)
        from _presence import joe_present
        return joe_present(idle_min)
    except Exception:  # noqa: BLE001 - no presence signal = assume present
        return True


def _acquire_calib(video, pipe):
    """The session's own rectification, re-acquired from its frames
    (same warmup shots_export._video_transform uses for hinv)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    try:
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


def _pair_identities(found, ident_by_pos) -> None:
    """EXCLUSIVE finder<->identifier pairing, in place: one identifier
    read feeds ONE finder detection (the first marathon run let
    neighbours share a read and spread duplicate numbers)."""
    cand = []
    for fi_d, d in enumerate(found):
        lim = 2.5 * max(d.radius, 8.0)
        for ii, (ix, iy, _n) in enumerate(ident_by_pos):
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
    for fi_d, d in enumerate(found):
        num = num_for.get(fi_d, -1)
        if num >= 0 and getattr(d, "number", -1) < 0:
            d.number = num               # carry the identifier's read


def reprocess(video: str, out_dir: str | None = None,
              max_frames: int = 0, calib=None, start_s: float = 0.0,
              presence_pause: bool = True) -> dict:
    import cv2

    from ..config import EXPORTS_DIR, Settings
    from ..detector_strategies import discover
    from ..vision.analysis_cache import SidecarWriter
    from ..vision.pipeline import Pipeline
    from .tracker import MotionTracker

    video = Path(video)
    out_root = Path(out_dir) if out_dir else (video.parent / "m1")
    out_root.mkdir(parents=True, exist_ok=True)
    out_video_alias = out_root / video.name    # sidecar naming anchor

    # A Pipeline instance carries BOTH the session-warmup calibration and
    # prepare_detections - the full bought filter stack (size prior,
    # foreign veto, rigid-body repair, geometry sanity...). The engine
    # once bypassed it and got a 229px coordinate offset plus every
    # phantom class the filters exist for.
    pipe = Pipeline(Settings.load())
    if calib is None:
        calib = _acquire_calib(video, pipe)
        if calib is None:
            log.warning("calibration warmup failed for %s", video.name)
            return {"aborted": True, "reason": "no calibration"}

    strat = discover()["ensemble_findid"]
    strat.inference_provider = "dml"
    finder, ident = strat._finder, strat._identifier
    # ENGINE-ONLY fp16 finder (1.58x; verified 0/60 count mismatches,
    # p99 position diff 0.47px on real frames). The LIVE app keeps the
    # fp32 champion until the corpus gates pass judgement.
    fp16 = Path(r"C:\Users\Joe\AppData\Local\BilliardsTrainer\models"
                r"\m1\pool_yolo11.fp16.onnx")
    if fp16.is_file():
        finder._path = fp16
        finder._sess = None
        log.info("engine finder: fp16 (%s)", fp16.name)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if start_s > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
    # the ENGINE's own rect->video transform rides in the meta, so the
    # trail exporter normalizes through a SELF-CONSISTENT chain (dense
    # coords -> this hinv -> video pixels) instead of a second warmup's
    # slightly different lock
    import numpy as np
    hinv = np.linalg.inv(np.asarray(calib.H, dtype=float))
    writer = SidecarWriter(out_video_alias, {"fps": fps, "engine": "m1",
                                             "rules_v": ENGINE_RULES_V,
                                             "clock": "pts",
                                             "dense": True,
                                             "calibrated": True,
                                             "hinv": [[round(float(v), 8)
                                                       for v in row]
                                                      for row in hinv],
                                             "w": w_px, "h": h_px,
                                             "source": video.name})
    tracker = MotionTracker(
        pockets=[(pk.x, pk.y) for pk in calib.table.pockets],
        pocket_r=float(calib.table.pocket_radius))
    ident_by_pos: list = []      # latest identifier detections (x, y, n)

    def recording_live() -> bool:
        return bool(list(EXPORTS_DIR.glob(".session-*.part.mp4")))

    # UI progress (Joe: "see in the UI what percent of progress"):
    # a tiny JSON the app's status bar polls; deleted on exit. One
    # engine runs at a time (RUNNING-marker discipline), one file.
    from ..config import APP_DIR
    prog_path = APP_DIR / "m1_progress.json"

    def _write_progress(done: int, rate: float) -> None:
        try:
            eta_s = (n_total - done) / rate if rate > 0 else None
            prog_path.write_text(json.dumps(
                {"video": video.name, "done": done, "total": n_total,
                 "proc_fps": round(rate, 1),
                 "eta_min": round(eta_s / 60, 1) if eta_s else None}))
        except OSError:
            pass

    t0_wall = time.time()
    last_prog = t0_wall
    last_file = 0.0
    fi = 0
    written = 0
    prev_pts = -1.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # REAL frame pts, not frame_index/fps (forensics 2026-08-28:
            # recordings carry a first-frame HOLD of 120-155ms, so the
            # grid assumption made every trail LEAD the video by ~4
            # frames on the phone, decaying with an avg-fps drift).
            # After read(), CAP_PROP_POS_MSEC is the pts of the frame
            # just returned (verified against ffprobe packet times).
            pts_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if fi > 0 and pts_s <= prev_pts:      # backend hiccup
                pts_s = prev_pts + 1.0 / fps
            prev_pts = pts_s
            t = start_s + pts_s
            if fi % 150 == 0:
                if recording_live():
                    log.warning("recording started - engine aborted at %.1fs", t)
                    return {"aborted": True, "frames": fi, "reason": "recording"}
                # Joe-presence pauses the GPU (2026-08-28: his mouse bogged
                # down DAILY - the presence guard deferred STARTING runs
                # but never paused one he walked in on; 57% GPU from this
                # engine + 30% from live inference starved the compositor)
                # on-demand runs (Joe clicked Re-measure) skip the pause:
                # his request IS the consent to use the GPU now
                if presence_pause and _joe_present():
                    log.info("Joe present - engine pausing at %.1fs", t)
                    while _joe_present():
                        if recording_live():
                            return {"aborted": True, "frames": fi,
                                    "reason": "recording"}
                        time.sleep(30)
                    log.info("Joe idle again - engine resuming")
            found = finder.detect(frame, calib) or []
            if fi % IDENT_EVERY == 0:
                ids = ident.detect(frame, calib) or []
                ident_by_pos = [(d.x, d.y, d.number) for d in ids
                                if getattr(d, "number", -1) >= 0]
            _pair_identities(found, ident_by_pos)
            # THE shared stage: raw-frame -> rect space + every filter
            prepared = pipe.prepare_detections(found, calib, frame.shape,
                                               frame=frame,
                                               refresh_foreign=True)
            dets = [(float(d.x), float(d.y), float(d.radius),
                     int(getattr(d, "number", -1))) for d in prepared]
            rows = tracker.update(dets, t)
            # HAND CONTEXT (bench R2: four strokes invented while Joe was
            # placing balls by hand). The live path has always recorded
            # which balls are hand-adjacent; the engine computed the mask
            # (round 7) but threw the answer away. Now it ships, so the
            # shot stage can tell a stroke from ball-gathering.
            try:
                foreign = pipe._foreign_state(frame, calib)
                carried = pipe._carried_ids(rows, foreign)
                ffrac = float(foreign[0]) if foreign else 0.0
            except Exception:  # noqa: BLE001 - context is a bonus, never fatal
                carried, ffrac = set(), 0.0
            writer.add_frame(t, rows, carried_ids=carried,
                             foreign_frac=ffrac)
            written += 1
            fi += 1
            if max_frames and fi >= max_frames:
                break
            now = time.time()
            if now - last_file > PROGRESS_FILE_S:
                _write_progress(fi, fi / max(0.1, now - t0_wall))
                last_file = now
            if now - last_prog > PROGRESS_EVERY_S:
                rate = fi / (now - t0_wall)
                eta = (n_total - fi) / max(0.1, rate) / 60
                print(f"[m1] {fi}/{n_total} frames, {rate:.1f} fps, "
                      f"eta {eta:.0f} min", flush=True)
                last_prog = now
    finally:
        writer.close()
        cap.release()
        prog_path.unlink(missing_ok=True)
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
