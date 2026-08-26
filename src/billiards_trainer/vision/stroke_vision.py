"""Visual stroke metrics: stay-down, backstroke, pause — from the overhead cam.

Joe (2026-08-24): "how long my cue shaft/tip stays still after I take the
shot... on misses there is often a tendency to have popped up quickly
immediately after or even during the shot... also backstroke and backstroke
pause." The Bluetooth cue sensor computes the same family from the IMU
(cue/analysis.py shot_metrics: back_depth, pause, finish) — this module is
the always-on camera-side twin; field names align so the sensor can later
cross-validate or supersede per shot.

Method (prototyped + verified frame-by-frame on session-20260824-012348,
9/10 shots HIGH confidence; evidence crops in the session scratchpad):
  - STRIKE: the sidecar shot start lags the true strike by 1.8-3.0s (the
    tracker reacquires the cue ball late). Re-detect it: track resting
    white circular blobs pre-start, strike = the vanish of the one whose
    departure is stroke-validated (an aim-consistent stick line near the
    rest point). White-pixel presence alone reads 0.35s late because the
    ferrule parks where the ball was — circularity + centre distance fix it.
  - STICK per frame: median-background fg mask; morphological opening
    finds thick arm/torso cores, subtracting them leaves thin structures;
    HoughLinesP fragments grouped by collinearity; the group whose line
    passes nearest the cue-ball rest point is the stick. Works through the
    arm-merge regime that defeats naive elongated-blob fitting.
  - STAY-DOWN: settle tip = median tip in [strike+0.15, strike+0.95]
    (absorbs follow-through); departure = first sustained run (8 frames)
    of tip >30px from settle, or 12 consecutive no-stick frames, backdated
    by the run; stay_down = departure - strike. popped_early = the stick
    never settles post-strike or departs within POP_EARLY_S.
  - BACKSTROKE/PAUSE: pre-strike tip positions projected on the stick
    axis give a 1D stroke signal; final backstroke = last rearward
    excursion before the strike; pause = dwell (|v| < PAUSE_V px/s) at the
    rearmost point; practice strokes = prior oscillation count.

Appended to the session sidecar as {type:'stroke_vision', v:N} records —
machine-derived, recomputable, keyed by shot start (never carried across
--force rebuilds; the backfill recomputes).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("vision.stroke")

STROKE_VISION_VERSION = 2   # v2: delivery anchored at forward-swing onset

PRE_S = 6.0          # window opens this long before the sidecar start
POST_S = 9.0         # ...and closes this long after
SETTLE_GATE_PX = 220
DEPART_PX = 30       # ~1.15 ball diameters
DEPART_RUN = 8       # consecutive frames past DEPART_PX
VANISH_RUN = 12      # consecutive no-stick frames = departure
POP_EARLY_S = 0.30
PAUSE_V = 60.0       # px/s: tip slower than this at the rearmost = pausing
BACK_MIN_PX = 25.0   # rearward excursion smaller than this isn't a backstroke
# Delivery sanity: a real final swing runs ~100-600ms; outside these bounds
# the onset was mispicked (occlusion, practice-cycle catch) — abstain.
DELIVER_MIN_MS = 66      # two frames at 30fps
DELIVER_MAX_MS = 1200
DELIVER_GAP_S = 0.8  # tip last seen further than this before the strike:
                     # the delivery happened entirely off-screen — abstain


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


class _Session:
    """Per-session context: background, felt polygon, homography."""

    def __init__(self, video: Path):
        self.video = Path(video)
        cap = cv2.VideoCapture(str(video))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, max(0, self.n - 1), 15).astype(int)
        frames = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                frames.append(f)
        cap.release()
        if not frames:
            raise RuntimeError("unreadable video")
        bg = np.median(np.stack(frames), axis=0).astype(np.uint8)
        self.bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.int16)
        self.felt = self._felt_polygon()
        self.k_open3 = np.ones((3, 3), np.uint8)
        self.k_core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        self.k_dil = np.ones((7, 7), np.uint8)
        self.k_close = np.ones((5, 5), np.uint8)

    def _felt_polygon(self):
        """Felt corners in visual coords: prefer the exported shots.json
        (transform + space, cached from a prior export), else map the rect
        warp's corners through _video_transform's hinv, else whole frame."""
        try:
            doc = json.loads(Path(str(self.video) + ".shots.json")
                             .read_text(encoding="utf-8"))
            hinv = np.array(doc["transform"]["hinv"])
            sp = doc["space"]
            cs = [(sp["x0"], sp["y0"]), (sp["x1"], sp["y0"]),
                  (sp["x1"], sp["y1"]), (sp["x0"], sp["y1"])]
        except Exception:  # noqa: BLE001 - fall through to the transform
            try:
                from .shots_export import _video_transform
                tr = _video_transform(self.video)
                hinv = np.array(tr["hinv"])
                cs = [(0, 0), (700, 0), (700, 1300), (0, 1300)]
            except Exception:  # noqa: BLE001 - felt gate is an optimisation
                h, w = self.bg_gray.shape
                return np.array([[0, 0], [w, 0], [w, h], [0, h]],
                                dtype=np.int32)
        out = []
        for c in cs:
            v = hinv @ np.array([c[0], c[1], 1.0])
            out.append([v[0] / v[2], v[1] / v[2]])
        return np.array(out, dtype=np.int32)

    # --- per-frame primitives (ported from the verified prototype) ----- #
    def fg_mask(self, fr):
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.int16)
        mask = ((np.abs(g - self.bg_gray) > 25).astype(np.uint8)) * 255
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k_open3)

    def thin_mask(self, mask):
        core = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k_core)
        thin = cv2.bitwise_and(mask, cv2.bitwise_not(cv2.dilate(core, self.k_dil)))
        return cv2.morphologyEx(thin, cv2.MORPH_CLOSE, self.k_close)

    def white_score(self, fr, pt, rad=16):
        x, y = int(pt[0]), int(pt[1])
        patch = fr[max(0, y - rad):y + rad, max(0, x - rad):x + rad]
        if patch.size == 0:
            return 0
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        return int(((hsv[:, :, 2] > 150) & (hsv[:, :, 1] < 80)).sum())

    def white_balls(self, fr, mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out = []
        for c in cnts:
            a = cv2.contourArea(c)
            if not (300 < a < 1600) or len(c) < 5:
                continue
            (cx, cy), (w, h), _ = cv2.fitEllipse(c)
            if max(w, h) / max(min(w, h), 1e-3) > 1.6 or max(w, h) > 45:
                continue
            if cv2.pointPolygonTest(self.felt, (cx, cy), True) < -5:
                continue
            if self.white_score(fr, (cx, cy)) > 200:
                out.append(np.array([cx, cy]))
        return out

    def ball_presence(self, fr, rest):
        """Distance of the nearest circular white blob to rest, or None.
        Circularity gate keeps the parked ferrule from impersonating the
        ball (naive white-count reads the strike 0.35s late)."""
        m = self.fg_mask(fr)
        best = None
        for b in self.white_balls(fr, m):
            d = float(np.linalg.norm(b - rest))
            if best is None or d < best:
                best = d
        return best

    def stick(self, fr, aim_pt):
        """Stick line via thin-mask + Hough fragments grouped collinear;
        pick the group nearest the aim point. dict(tip, butt, dist_aim,
        length) or None."""
        thin = self.thin_mask(self.fg_mask(fr))
        segs = cv2.HoughLinesP(thin, 1, np.pi / 180, 40,
                               minLineLength=60, maxLineGap=12)
        if segs is None:
            return None
        segs = segs.reshape(-1, 4).astype(np.float32)
        used = np.zeros(len(segs), bool)
        lens = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
        best = None
        for i in np.argsort(-lens):
            if used[i]:
                continue
            p0 = segs[i, :2]
            d = segs[i, 2:] - p0
            d = d / (np.linalg.norm(d) + 1e-9)
            grp = [i]
            used[i] = True
            for j in range(len(segs)):
                if used[j]:
                    continue
                dj = segs[j, 2:] - segs[j, :2]
                dj = dj / (np.linalg.norm(dj) + 1e-9)
                if abs(float(d @ dj)) < np.cos(np.deg2rad(8)):
                    continue
                mid = (segs[j, :2] + segs[j, 2:]) / 2
                if abs(float(_cross(d, mid - p0))) > 18:
                    continue
                grp.append(j)
                used[j] = True
            pts = np.vstack([segs[g].reshape(2, 2) for g in grp])
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            dvec = np.array([vx, vy])
            pl = np.array([x0, y0])
            proj = (pts - pl) @ dvec
            lo, hi = proj.min(), proj.max()
            if hi - lo < 100:
                continue
            e1, e2 = pl + dvec * lo, pl + dvec * hi
            dist_aim = abs(float(_cross(dvec, aim_pt - pl)))
            score = dist_aim - 0.2 * (hi - lo)
            if best is None or score < best[0]:
                best = (score, e1, e2, dist_aim, float(hi - lo), dvec)
        if best is None:
            return None
        _, e1, e2, dist_aim, length, dvec = best
        d1, d2 = np.linalg.norm(e1 - aim_pt), np.linalg.norm(e2 - aim_pt)
        tip, butt = (e1, e2) if d1 < d2 else (e2, e1)
        return {"tip": tip, "butt": butt, "dist_aim": dist_aim,
                "length": length, "dir": dvec}


class AbortMeasurement(Exception):
    """Raised mid-measurement when the caller's abort() goes true — the
    live worker uses it so no cv2 handle on the growing .part survives
    into the session-stop rename (an open handle fails the replace)."""


def _read_window(video: Path, t0: float, t1: float, fps: float, abort=None):
    """Stream frames [t0, t1] as (frame_index, time, frame) without holding
    the decoded window in memory (a 15s window of this footage is ~2.4GB)."""
    cap = cv2.VideoCapture(str(video))
    f0 = max(0, int(t0 * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    i = f0
    try:
        while True:
            if abort is not None and i % 30 == 0 and abort():
                raise AbortMeasurement
            ok, fr = cap.read()
            if not ok or i / fps > t1:
                break
            yield i, i / fps, fr
            i += 1
    finally:
        cap.release()


def measure_shot(sess: _Session, start: float, abort=None) -> dict:
    """All stroke metrics for the shot whose sidecar start is ``start``."""
    fps = sess.fps
    t_lo, t_hi = max(0.0, start - PRE_S), start + POST_S

    # pass 1: track resting white blobs (every 3rd frame) to find vanish
    # candidates; keep only light state, never frames.
    tracks: list[dict] = []
    for i, t, fr in _read_window(sess.video, t_lo, t_hi, fps, abort):
        if i % 3:
            continue
        m = sess.fg_mask(fr)
        for b in sess.white_balls(fr, m):
            hit = None
            for tr in tracks:
                if tr["gap"] <= 8 and np.linalg.norm(tr["pos"] - b) < 14:
                    hit = tr
                    break
            if hit is None:
                tracks.append({"pos": b, "first": t, "last": t, "gap": 0})
            else:
                hit["pos"] = 0.7 * hit["pos"] + 0.3 * b
                hit["last"] = t
        for tr in tracks:
            tr["gap"] = int(round((t - tr["last"]) * fps / 3))
    cands = [tr for tr in tracks if tr["last"] - tr["first"] >= 0.6]

    # pass 2: refine each candidate's vanish at full rate + stroke-validate
    events = []
    for tr in cands:
        rest = tr["pos"]
        pres = [(t, sess.ball_presence(fr, rest))
                for _, t, fr in _read_window(sess.video, tr["last"] - 0.4,
                                             tr["last"] + 1.0, fps, abort)]
        t_strike = None
        for k in range(len(pres) - 5):
            t, p = pres[k]
            if (p is not None and p <= 12
                    and all(p2 is None or p2 > 12 for _, p2 in pres[k + 1:k + 6])):
                t_strike = pres[k + 1][0]
        if t_strike is None or not (start - 5.0 <= t_strike <= start + 1.5):
            continue
        good = 0
        for i, _t, fr in _read_window(sess.video, t_strike - 0.3,
                                      t_strike + 0.5, fps, abort):
            if i % 3:
                continue
            s = sess.stick(fr, rest)
            if (s is not None and np.linalg.norm(s["tip"] - rest) < 300
                    and s["dist_aim"] < 45):
                good += 1
        if good >= 4:
            events.append((t_strike, rest))
    if not events:
        return {"confidence": "none",
                "reason": "no validated stroke (cue occluded or moved by hand)"}
    events.sort(key=lambda e: e[0])   # rest is an ndarray - never compare it
    t_strike, rest = events[-1]

    # pass 3: tip timeline, pre-strike through the window end
    tl = []   # (t, stick-or-None)
    for _, t, fr in _read_window(sess.video, t_strike - 4.0, t_hi, fps, abort):
        tl.append((t, sess.stick(fr, rest)))

    out = {"strike": round(t_strike, 2), "rest": [round(float(rest[0])),
                                                 round(float(rest[1]))]}
    out.update(_stay_down(tl, t_strike, tl[-1][0]))
    out.update(_backstroke(tl, t_strike, rest, fps))
    return out


def _stay_down(tl, t_strike, t_end) -> dict:
    win = [s for (t, s) in tl
           if t_strike + 0.15 <= t <= t_strike + 0.95 and s is not None]
    settle = [s["tip"] for s in win]
    if len(settle) < 5:
        # never settled after the strike: popped up during/immediately after
        dep = None
        for k in range(len(tl) - DEPART_RUN):
            t, s = tl[k]
            if t < t_strike + 0.1:
                continue
            if all(s2 is None for (_, s2) in tl[k:k + DEPART_RUN]):
                dep = t
                break
        stay = None if dep is None else round(dep - t_strike, 2)
        return {"stay_down_s": stay, "popped_early": True,
                "confidence": "low", "reason": "no post-strike settle"}
    p = np.median(np.array(settle), axis=0)
    dep = None
    run = miss = 0
    for t, s in tl:
        if t < t_strike + 0.95:
            continue
        if s is None:
            miss += 1
            run = 0
        elif np.linalg.norm(s["tip"] - p) > DEPART_PX:
            run += 1
            miss = 0
        else:
            run = miss = 0
        if run >= DEPART_RUN:
            dep = t - DEPART_RUN / 30.0
            break
        if miss >= VANISH_RUN:
            dep = t - VANISH_RUN / 30.0
            break
    stay = round((dep if dep is not None else t_end) - t_strike, 2)
    dets = [s for (t, s) in tl if t_strike <= t <= (dep or t_end)]
    frac = sum(1 for s in dets if s is not None) / max(1, len(dets))
    return {"stay_down_s": stay, "popped_early": stay < POP_EARLY_S,
            "capped": dep is None,
            "confidence": "high" if frac >= 0.7 else "low"}


def _backstroke(tl, t_strike, rest, fps) -> dict:
    """Backstroke depth/pause/practice strokes from the axial tip signal."""
    pre = [(t, s) for (t, s) in tl
           if t < t_strike and s is not None
           and np.linalg.norm(s["tip"] - rest) < 400 and s["dist_aim"] < 45]
    if len(pre) < int(fps * 0.8):
        return {"backstroke_conf": "none"}
    dirs = np.array([s["dir"] for _, s in pre[-int(fps):]])
    axis = dirs.mean(axis=0)
    axis /= (np.linalg.norm(axis) + 1e-9)
    if axis @ (rest - pre[-1][1]["tip"]) < 0:
        axis = -axis          # axis points tip-ward (toward the cue ball)
    ts = np.array([t for t, _ in pre])
    x = np.array([float(s["tip"] @ axis) for _, s in pre])
    # median-of-3 smoothing kills the bridge-hand tip flip-flop
    xs = np.copy(x)
    for i in range(1, len(x) - 1):
        xs[i] = np.median(x[i - 1:i + 2])
    # final backstroke: rearmost point in the last 2.5s before the strike
    m = ts >= t_strike - 2.5
    if m.sum() < 5:
        return {"backstroke_conf": "none"}
    ti, xi = ts[m], xs[m]
    k_min = int(np.argmin(xi))
    x_back, t_back = xi[k_min], ti[k_min]
    x_fwd_before = xi[:k_min + 1].max() if k_min > 0 else x_back
    depth = float(x_fwd_before - x_back)
    if depth < BACK_MIN_PX:
        return {"backstroke_conf": "none"}
    # pause: dwell around the rearmost point with axial speed < PAUSE_V
    v = np.gradient(xi, ti)
    lo = k_min
    while lo > 0 and abs(v[lo - 1]) < PAUSE_V:
        lo -= 1
    hi = k_min
    while hi < len(v) - 1 and abs(v[hi + 1]) < PAUSE_V:
        hi += 1
    pause_ms = int(round((ti[hi] - ti[lo]) * 1000))
    # DELIVERY: strike minus the FINAL FORWARD SWING's onset — not minus
    # t_back. The old anchor charged the whole post-rearmost dwell (and,
    # when argmin landed on a deeper practice stroke, an entire practice
    # cycle) to the delivery: measured library-wide it put 15.8% of
    # deliveries over 1500ms and correlated delivery with pause (0.29).
    # Walk back from the last visible pre-strike sample while the tip
    # moves forward; the sample before that run is the onset.
    j = len(v) - 1
    while j > 0 and v[j] > PAUSE_V:
        j -= 1
    delivery_ms = None
    gap_s = t_strike - ti[-1]        # tip often vanishes in the fast swing
    if gap_s <= DELIVER_GAP_S:
        d = int(round((t_strike - ti[j]) * 1000))
        if DELIVER_MIN_MS <= d <= DELIVER_MAX_MS:
            delivery_ms = d          # else: mispick/occlusion — abstain
    # practice strokes: rearward excursions > BACK_MIN_PX before the final one
    n_practice = 0
    down = False
    ref = xs[0]
    for val in xs[ts < t_back]:
        if not down and ref - val > BACK_MIN_PX:
            down = True
            n_practice += 1
        elif down and val > ref - BACK_MIN_PX / 2:
            down = False
        ref = max(ref, val) if not down else ref
    out = {"back_depth_px": round(depth, 1), "pause_ms": pause_ms,
           "practice_strokes": n_practice,
           "backstroke_conf": "high" if len(pre) > fps * 2 else "low"}
    if delivery_ms is not None:      # measured-or-abstained: no garbage
        out["delivery_ms"] = delivery_ms
    return out


def annotate_session(video, force: bool = False) -> int:
    """Append a {type:'stroke_vision'} record for every shot that lacks a
    current-version one. Recomputable machine data — review verdicts and
    corrections are never touched. Returns records appended."""
    from .analysis_cache import SidecarReader, sidecar_path
    video = Path(video)
    sc = sidecar_path(video)
    if not sc.is_file():
        return 0
    have: dict = {}
    for line in sc.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("type") == "stroke_vision":
            have[round(float(d.get("start", -1)), 2)] = int(d.get("v", 0))
    reader = SidecarReader(video)
    shots = list(getattr(reader, "shots", []) or [])
    todo = []
    for sh in shots:
        st = round(float(sh.get("start", 0.0)), 2)
        if not force and have.get(st, 0) >= STROKE_VISION_VERSION:
            continue
        todo.append(st)
    if not todo:
        return 0
    sess = _Session(video)
    n = 0
    with open(sc, "a", encoding="utf-8") as f:
        for st in todo:
            try:
                rec = measure_shot(sess, st)
            except Exception as exc:  # noqa: BLE001 - one bad shot never stops the pass
                log.warning("stroke_vision failed for %s@%s: %s",
                            video.name, st, exc)
                rec = {"confidence": "none", "reason": f"error: {exc}"}
            rec.update({"type": "stroke_vision", "v": STROKE_VISION_VERSION,
                        "start": st})
            f.write(json.dumps(rec) + "\n")
            n += 1
    log.info("stroke_vision: %d shot(s) annotated in %s", n, video.name)
    return n
