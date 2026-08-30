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
ENGINE_RULES_V = 20  # v20: a stroke requires the CUE to move FIRST -
                     # nothing on the table moves until the cue reaches
                     # it - which separates hand setup from shooting
                     # without needing to see the hand. Bench: all 10
                     # real strokes lead by 0.07-0.43s, the hand setup
                     # trails by 2.90s. Plus ONE copy of that judgement,
                     # shared by the engine and the scorecard.
                     # v19: the rect radius was measured between two
                     # different coordinate frames (parallax applied to
                     # the centre, not to the offset point it is measured
                     # against), so a real ball in plain sight could fall
                     # 0.22px under the size floor and be discarded for
                     # 82 seconds. Ball radii across a frame tightened
                     # from +-24% to +-4%; blind checks 88 -> 7


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


def _pair_identities(found, ident_by_pos, frame_bgr=None) -> None:
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
        # THE IDENTIFIER WINS. It is the trained 16-class naming model;
        # any number already on a find is the finder's crude colour
        # heuristic, which exists only for balls the identifier cannot
        # see. The old guard (`and d.number < 0`) had that precedence
        # backwards, so a ball the heuristic had already guessed could
        # never be corrected - measured on the bench: the heuristic calls
        # the yellow-STRIPED 9 a "1" in 43 of 43 samples (both are
        # yellow) while the identifier reads it 9 in 71 of 72, and the
        # correct read was discarded every frame. The 9 scored 0/221 and
        # collected the name "1" whenever the real 1 left the table. The
        # heuristic is also where the invented numbers come from (it
        # emits 8s and an 11 across this clip).
        # TRIED AND REVERTED (round 58): refuse a number whose class
        # contradicts the finder's solid/stripe judgement (1-7 solid,
        # 9-15 stripe). It looked well-founded - measured over 900 cold
        # frames the two agree 6357 times and contradict 1727, almost
        # all of it a gold SOLID called "9" - and the scorecard threw it
        # out at once: the BENCH's 9 is a yellow STRIPE whose body reads
        # SOLID to the finder, so the rule vetoed the correct name and
        # took naming 99.6% -> 76.8%, outcomes 10/10 -> 8/10, pots
        # 4/4 -> 2/4. The finder's class is not evidence about
        # stripes; it is a guess about them, and round 33 already bought
        # the 9's name the hard way.
        if num >= 0:
            d.number = num
            if frame_bgr is not None:
                # THE MODEL READS THE PURPLE 4 AS THE 7. That is a real
                # misread of the dark 4/7/8 cluster under Joe's warm
                # light, and until now the tracker's rest-freeze hid it:
                # the 4 settled on the heuristic's correct guess before
                # the "7" could accumulate. With a resting ball made
                # correctable (round 33), the misread wins instead - 129
                # sightings - so the correction the ensemble already owns
                # has to run here too. It compares the crop against THIS
                # table's measured colours and only overrules the model
                # when the claim is far and another number is close.
                from ..detector_strategies.ensemble import FindIdEnsemble
                FindIdEnsemble._fix_colour(frame_bgr, d)


def _write_shots(writer, times, frames, carried, calib) -> int:
    """Derive the session's SHOT LIST from the measured stream and write
    it into the sidecar beside the frames.

    One engine, one answer: the same episode stage the scorecard is
    judged on decides what a shot is here, so what Joe sees on his phone
    and what the bench measures can no longer drift apart. A shot is a
    STROKE only if the cue ball was actually struck - hand-driven
    gathering fails either the travel bar or the speed bar (round 35:
    hand-placed 213 px/s vs 691+ for every real stroke)."""
    from types import SimpleNamespace

    from .shots import analyze, is_stroke
    if not times:
        return 0
    eps = analyze(times, frames,
                  pockets=[(p.x, p.y) for p in calib.table.pockets],
                  pocket_r=float(calib.table.pocket_radius),
                  carried=carried)
    _struck = is_stroke      # ONE copy of the rule; see shots.is_stroke

    # TABLE CHANGE (Joe, 2026-08-31: "I don't think we need to separate
    # out rearranging as discrete events. Perhaps we can add a post
    # processor to merge consecutive rearranging - I call it Table
    # Change - into one event"). Gathering balls is not a sequence of
    # events, it is one interruption between shots: the bench produced
    # NINE separate "rearranging" entries in a 19-item list, one of
    # which was only Joe walking to his next shot. Consecutive
    # non-strokes collapse into a single span.
    merged: list = []
    for e in eps:
        if not _struck(e) and merged and not _struck(merged[-1]):
            merged[-1] = merged[-1].__class__(
                **{**merged[-1].__dict__, "t_settle": e.t_settle})
            continue
        merged.append(e)
    eps = merged
    n = 0
    for e in eps:
        struck = _struck(e)
        # NOTHING IS POTTED WITHOUT A STROKE. A ball that disappears while
        # Joe is gathering by hand has been PICKED UP, not made - and the
        # episode stage cannot tell those apart from the track alone. The
        # first derived list credited three "makes" to the purple 4 during
        # the 24-28s setup window, which is exactly the kind of phantom
        # make he has reported before.
        balls = [int(b) for b, _x, _y in e.pocketed if b >= 0] if struck else []
        # WHICH POCKET, decided here and carried. The engine knows the
        # pocket it credited; the sentence layer used to re-derive one
        # from the last point of the ball's TRAIL, which keeps going
        # after the ball is gone (its track gets re-attached to whatever
        # is nearby). On the bench that put the 31.7 pot "into the
        # right-middle pocket" when it fell bottom-right, because the
        # trail ended mid-table at (489,792). One measurement, carried,
        # instead of two derivations that can disagree.
        at, at_xy = [], []
        if struck:
            for b, px, py in e.pocketed:
                if b < 0:
                    continue
                p = min(calib.table.pockets,
                        key=lambda q: (q.x - px) ** 2 + (q.y - py) ** 2)
                at.append(p.name)
                at_xy.append([round(float(p.x), 1), round(float(p.y), 1)])
        writer.add_shot(SimpleNamespace(
            start_t=e.t_strike, end_t=e.t_settle,
            outcome="make" if (struck and e.pocketed) else "miss",
            num_pocketed=len(e.pocketed) if struck else 0,
            action="stroke" if struck else "rearrange",
            pocketed_balls=balls, pocketed_at=at,
            pocketed_xy=at_xy))
        n += 1
    # ...and count them with the SAME function that classified them.
    # This line still spelled the rule out by hand after round 51 moved
    # it into shots.is_stroke, and it named two constants the import no
    # longer brought in - so _write_shots raised NameError on EVERY run
    # from round 51 to round 62, eleven rounds, swallowed by the
    # "shot derivation failed" except in reprocess(). The shot list
    # survived because every add_shot happens above this line, which is
    # exactly why nothing looked wrong. Found only because a probe ran
    # with the traceback visible.
    log.info("m1 shots derived: %d (%d strokes)", n,
             sum(1 for e in eps if _struck(e)))
    return n


def reprocess(video: str, out_dir: str | None = None,
              max_frames: int = 0, calib=None, start_s: float = 0.0,
              presence_pause: bool = True) -> dict:
    import cv2

    from ..config import EXPORTS_DIR, Settings
    from ..detector_strategies import discover
    from ..vision.analysis_cache import SidecarWriter
    from ..core.balls import use_session_refs
    from ..detector_strategies.ensemble import FindIdEnsemble as _ENSEMBLE
    from ..vision.pipeline import Pipeline

    video = Path(video)
    # COLOUR NAMING IS A PER-TABLE FACT (round 61). The measured colour
    # references correct the model's misreads - the purple 4 read as a 7
    # is the case they were built for - and there was ONE global set,
    # describing the bench's rack. Measured on the first cold clip:
    # measured_identity() returns -1 for every ball on that table, so the
    # correction could never fire, and the 4 was called the 7 in 25
    # sightings. Installing that table's own references took its naming
    # from 85.7% to 93.3% (the 4 66/111 -> 110/111, the 7 126/151 ->
    # 151/151, unnamed 46 -> 1). A clip now uses its own set when the
    # repo has one, and the global set otherwise.
    use_session_refs(video.name)
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
    # ONE TRACKER INSTANCE, not merely one tracker CLASS (round 47).
    # Round 39 put live and offline on the same MotionTracker code, but
    # this line still built a SECOND one while the Pipeline kept its own,
    # and prepare_detections' blur-recovery pass reads pipe.tracker. That
    # tracker is never updated offline, so `lost` was empty on every
    # frame and recovery could not fire once in a whole clip - which is
    # why Joe's @85 report (the 3's opening tail replaced by a straight
    # line across a 360px jump) survived a feature built to prevent
    # exactly it. Sharing the instance costs nothing: the offline loop is
    # the only thing driving it.
    tracker = pipe.tracker
    tracker.reset()
    tracker.set_geometry([(pk.x, pk.y) for pk in calib.table.pockets],
                         float(calib.table.pocket_radius))
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

    # the measured stream, kept so the SHOT stage can run over it at the
    # end of the pass instead of re-reading the sidecar
    ep_times: list = []
    ep_frames: list = []
    ep_carried: list = []
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
            # NOT start_s + pts_s (round 47). start_s is applied by
            # seeking the capture, so pts_s is ALREADY absolute in the
            # source video; adding it again doubled the offset. Harmless
            # for full runs (start_s == 0) and poison for every partial
            # one: a probe of the @85 bank reported its frames at
            # t=160-171, so a window filter on real time matched nothing
            # and the investigation looked like "recovery never runs".
            # An investigation tool that lies about its own clock is
            # worse than no tool.
            t = pts_s
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
            # MEASURE EVERY BALL'S COLOUR UP FRONT (round 48), exactly as
            # FindIdEnsemble.detect does for the live path. The engine
            # calls strat._finder.detect directly - so it can run
            # identity on its own cadence with the fp16 finder - and that
            # bypassed the one place sample_colour runs. So offline,
            # measured_bgr was None on EVERY detection of EVERY clip, and
            # everything downstream that reasons about appearance was
            # dead: a track's mbgr_hist never filled, so blur recovery
            # could not select a single candidate (measured on the @85
            # bank: find() called 340 times in 340 frames, `lost` empty
            # every one), and the colour-consensus and settled-track
            # colour veto had nothing to read either. This is a
            # MEASUREMENT, not a verdict - nothing here names a ball.
            for d in found:
                try:
                    _ENSEMBLE.sample_colour(frame, d)
                except Exception:  # noqa: BLE001 - a colour read is never fatal
                    pass
            if fi % IDENT_EVERY == 0:
                ids = ident.detect(frame, calib) or []
                ident_by_pos = [(d.x, d.y, d.number) for d in ids
                                if getattr(d, "number", -1) >= 0]
            _pair_identities(found, ident_by_pos, frame)
            # THE shared stage: raw-frame -> rect space + every filter
            prepared = pipe.prepare_detections(found, calib, frame.shape,
                                               frame=frame,
                                               refresh_foreign=True)
            # Hand the tracker the DETECTIONS, not a flattened tuple
            # (round 47). The tuple carried x/y/r/number and dropped
            # measured_bgr, which is the only thing that fills a track's
            # mbgr_hist - and blur recovery requires 5 samples of it
            # before it will consider a track recoverable. So offline,
            # every track had an empty colour history, `lost` was empty
            # on every frame of every clip, and the recovery pass ran
            # 340 times in a 340-frame probe without once being able to
            # look. update() has taken either form since round 39; the
            # live path was already passing objects.
            rows = tracker.update(prepared, t)
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
            # kept so the SHOT stage can run over this same stream below
            ep_times.append(t)
            ep_frames.append([(r.id, r.x, r.y, r.radius, r.number,
                               r.cls.value, 1, r.coasting) for r in rows])
            ep_carried.append(sorted(carried))
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
        # THE SHOTS COME FROM THE MEASUREMENT, TOO (Joe, 2026-08-31: a
        # reprocess "should completely obliterate all sidecar data and
        # REPROCESS EVERYTHING as though it was just coming in as a raw
        # video. It should not be state dependent."). The engine used to
        # write only frames, so the shot LIST stayed whatever the
        # recording-time pass had decided - which is why a genuine stroke
        # was still labelled "rearranging" on his phone long after the
        # engine had measured it correctly.
        try:
            _write_shots(writer, ep_times, ep_frames, ep_carried, calib)
        except Exception:  # noqa: BLE001 - frames are the product; shots enrich
            log.exception("shot derivation failed")
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
