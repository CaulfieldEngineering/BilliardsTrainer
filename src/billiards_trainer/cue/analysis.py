"""Cue-motion analysis: envelope, cue-ball impact detection, fusion, metrics.

Ported VERBATIM from the pool-stroke-analyzer project (app/analysis.py) — the
science here is video-validated (2026-07-05 session: 14 confirmed hits from
1.7 g pokes to 8.3 g slams and 21 labelled handling events; the impact
signature scores 14/14 TP, 0 FP). Do not "clean up" thresholds without
re-validating against footage.

Units convention: acceleration in **g**, angular velocity in **degrees/second
(dps)**. Sensor decoding (protocol.py) already converts to these.

Pure Python (no deps) so the UI, the live worker, and tests share the exact
same logic.
"""

import math


def magnitude(a):
    return math.sqrt(a["x"] ** 2 + a["y"] ** 2 + a["z"] ** 2)


def envelope(accel, smooth=3):
    """(ts, env): motion energy = |a| deviation from 1g, smoothed — the timeline trace."""
    if not accel:
        return [], []
    ts = [a["t"] for a in accel]
    dev = [abs(magnitude(a) - 1.0) for a in accel]
    env = []
    for i in range(len(dev)):
        s = c = 0
        for j in range(-smooth, smooth + 1):
            k = i + j
            if 0 <= k < len(dev):
                s += dev[k]
                c += 1
        env.append(s / c)
    return ts, env


def detect_strokes(accel, gyro, sensitivity=0.2, min_dur=0.10, min_gap=0.12):
    """Find strokes (active motion segments) and the cue-ball hit within each.

    Returns dicts: { t0, t1, dur, hit_t, peak_g, peak_rate(dps), rot_deg }.
    """
    if not accel:
        return []
    ts, env = envelope(accel)
    strokes = []
    in_s = False
    s0 = 0
    n = len(env)
    for i in range(n):
        if not in_s and env[i] > sensitivity:
            in_s = True
            s0 = i
        elif in_s and env[i] <= sensitivity * 0.5:
            j = i
            bridged = False
            while j < n and accel[j]["t"] - accel[i]["t"] < min_gap:
                if env[j] > sensitivity:
                    bridged = True
                    break
                j += 1
            if not bridged:
                _emit(strokes, accel, gyro, s0, i)
                in_s = False
    if in_s:
        _emit(strokes, accel, gyro, s0, n - 1)
    return [s for s in strokes if s["dur"] >= min_dur]


def _emit(out, accel, gyro, i0, i1):
    seg = accel[i0:i1 + 1]
    if len(seg) < 2:
        return
    t0, t1 = seg[0]["t"], seg[-1]["t"]
    peak_g, hit_t = 0.0, t0
    for a in seg:
        m = magnitude(a)
        if m > peak_g:
            peak_g, hit_t = m, a["t"]
    rot_deg, peak_rate, last_t = 0.0, 0.0, None
    for g in gyro:                       # gyro already in dps
        if g["t"] < t0 or g["t"] > t1:
            continue
        rate = magnitude(g)
        peak_rate = max(peak_rate, rate)
        if last_t is not None:
            rot_deg += rate * (g["t"] - last_t)
        last_t = g["t"]
    out.append({"t0": t0, "t1": t1, "dur": t1 - t0, "hit_t": hit_t,
                "peak_g": peak_g, "peak_rate": peak_rate, "rot_deg": rot_deg})


def evaluate_impact(accel, gyro_mags_t, i, mag_floor=1.6, win=0.4):
    """Decide whether the |a| peak at accel[i] is a cue-ball strike.

    Video-validated on the 2026-07-05 session (14 confirmed hits from ~1.7 g
    soft pokes to 8.3 g slams, 21 labelled handling events — 14/14 TP, 0 FP).

    Physics of a real hit along the cue axis (+Z = butt→tip, baseline-relative):
      PATH A (driven stroke): a forward DRIVE phase (+z dev ≥ +1.0 in the 0.35 s
        before the dip) followed by the impact dip (z dev ≤ −0.9).
      PATH B (soft push, no distinct drive): deeper dip (≤ −1.5) with a genuine
        post-dip rebound (≥ +0.9) that is axial-dominant at the dip, and a
        stricter gyro cap.
    Common gates: cue near level (|gravity on Z| ≤ 0.35 over the baseline),
    rotation sane (fast flourish/spin ≠ shot), and no radial clipping (a ±8 g
    X/Y rail-out is a knock or jab, never ball contact).
    Returns (ok, feats) — feats has dip/drive/reb/bz for diagnostics.
    """
    T = [a["t"] for a in accel]
    n = len(T)
    t = T[i]
    import bisect as _b
    lo = _b.bisect_left(T, t - win)
    hi = _b.bisect_right(T, t + win) - 1
    b0 = _b.bisect_left(T, t - 0.95)
    b1 = _b.bisect_right(T, t - 0.35) - 1
    if b1 <= b0 or hi <= lo:
        return False, {}
    nb = b1 - b0 + 1
    bx = sum(accel[j]["x"] for j in range(b0, b1 + 1)) / nb
    by = sum(accel[j]["y"] for j in range(b0, b1 + 1)) / nb
    bz = sum(accel[j]["z"] for j in range(b0, b1 + 1)) / nb
    if abs(bz) > 0.35:                                   # cue tilted — jab/plant territory
        return False, {"bz": bz}
    devz = [accel[j]["z"] - bz for j in range(lo, hi + 1)]
    kd = min(range(len(devz)), key=lambda q: devz[q])
    dip = devz[kd]
    tdip = T[lo + kd]
    reb = max(devz[kd:kd + 9], default=0.0)              # rebound AFTER the dip
    d0 = _b.bisect_left(T, tdip - 0.38)
    d1 = _b.bisect_right(T, tdip - 0.03) - 1
    drive = max((accel[j]["z"] - bz for j in range(d0, d1 + 1)), default=0.0)
    rad_dip = math.sqrt((accel[lo + kd]["x"] - bx) ** 2 + (accel[lo + kd]["y"] - by) ** 2)
    dom = abs(dip) / max(rad_dip, 1e-6)
    if any(abs(accel[j]["x"]) >= 7.9 or abs(accel[j]["y"]) >= 7.9
           for j in range(lo, hi + 1)):                  # radial clip = knock/jab
        return False, {"bz": bz, "dip": dip, "clip": True}
    gy = max((m for tt, m in gyro_mags_t if t - win <= tt <= t + win), default=0.0)
    feats = {"bz": bz, "dip": dip, "reb": reb, "drive": drive, "dom": dom, "gy": gy}
    if gy > 450:
        return False, feats
    path_a = drive >= 1.0 and dip <= -0.9
    path_b = dip <= -1.5 and reb >= 0.9 and dom >= 0.65 and gy <= 350
    return (path_a or path_b), feats


def detect_impacts(accel, gyro, mag_thresh=1.6, refractory=1.2, win=0.4):
    """Cue-ball impact detector — see evaluate_impact for the validated
    signature. mag_thresh is the |a| candidate floor in g (soft pokes reach
    ~1.7 g at 25 Hz; anything lower drowns in stroke noise — that regime
    needs the planned audio channel).

    Returns the same dicts as detect_strokes: {t0, t1, dur, hit_t, peak_g,
    peak_rate, rot_deg} with t0/t1 = the ringing bounds.
    """
    if not accel:
        return []
    T = [a["t"] for a in accel]
    M = [magnitude(a) for a in accel]
    gyro_mt = [(g["t"], magnitude(g)) for g in gyro]
    n = len(T)
    out = []
    last = -1e9
    for i in range(n):
        t = T[i]
        if M[i] < mag_thresh or t - last < refractory:
            continue
        # only evaluate local maxima (the strike sample, not its shoulders)
        j0, j1 = i, i
        while j0 > 0 and t - T[j0 - 1] < 0.25:
            j0 -= 1
        while j1 < n - 1 and T[j1 + 1] - t < 0.25:
            j1 += 1
        if any(M[j] > M[i] for j in range(j0, j1 + 1)):
            continue
        ok, _ = evaluate_impact(accel, gyro_mt, i, mag_floor=mag_thresh, win=win)
        if not ok:
            continue
        lo, hi = j0, j1
        ring = [j for j in range(lo, hi + 1) if M[j] > mag_thresh] or [i]
        t0, t1 = T[ring[0]], T[ring[-1]]
        rot_deg, peak_rate, last_t = 0.0, 0.0, None
        for g in gyro:
            if g["t"] < t0 or g["t"] > t1:
                continue
            rate = magnitude(g)
            peak_rate = max(peak_rate, rate)
            if last_t is not None:
                rot_deg += rate * (g["t"] - last_t)
            last_t = g["t"]
        out.append({"t0": t0, "t1": t1, "dur": t1 - t0, "hit_t": t,
                    "peak_g": M[i], "peak_rate": peak_rate, "rot_deg": rot_deg})
        last = t
    return out


# ---------------- orientation fusion (Mahony complementary filter) ----------------
# Quaternion is (w, x, y, z), body->world. Gyro in dps, accel in g.

_D2R = math.pi / 180.0


def _step(q, gyro_dps, accel_g, dt, kp=1.5):
    """One fusion step; returns the new normalized quaternion."""
    if dt <= 0 or dt > 0.2:
        return q
    w, x, y, z = q
    wx = gyro_dps["x"] * _D2R
    wy = gyro_dps["y"] * _D2R
    wz = gyro_dps["z"] * _D2R
    if accel_g is not None:
        n = magnitude(accel_g)
        if 0.7 < n < 1.3:                # only correct near 1g (not mid-swing)
            ax, ay, az = accel_g["x"] / n, accel_g["y"] / n, accel_g["z"] / n
            ux = 2 * (x * z - w * y)     # world-up expressed in body frame
            uy = 2 * (y * z + w * x)
            uz = 1 - 2 * (x * x + y * y)
            wx += kp * (ay * uz - az * uy)
            wy += kp * (az * ux - ax * uz)
            wz += kp * (ax * uy - ay * ux)
    h = 0.5 * dt
    q = [w + h * (-x * wx - y * wy - z * wz),
         x + h * (w * wx + y * wz - z * wy),
         y + h * (w * wy - x * wz + z * wx),
         z + h * (w * wz + x * wy - y * wx)]
    nrm = math.sqrt(sum(v * v for v in q)) or 1.0
    return [v / nrm for v in q]


def fuse_track(accel, gyro):
    """Integrate an orientation track [{t, q}] over a whole recorded session."""
    merged = [(a["t"], 0, a) for a in accel] + [(g["t"], 1, g) for g in gyro]
    merged.sort(key=lambda m: m[0])
    q = [1.0, 0.0, 0.0, 0.0]
    last_a, last_t = None, None
    track = []
    for t, kind, d in merged:
        if kind == 0:
            last_a = d
        else:
            if last_t is not None:
                dt = min(t - last_t, 0.1)      # clamp gaps (between recording bursts) so fusion can't over-rotate
                q = _step(q, d, last_a, dt)
            last_t = t
            track.append({"t": t, "q": tuple(q)})
    return track


class Mahony:
    """Stateful live fusion. Feed accel with set_accel(); step with update(gyro, dt)."""

    def __init__(self, kp=1.5):
        self.q = [1.0, 0.0, 0.0, 0.0]
        self.kp = kp
        self._accel = None

    def set_accel(self, a):
        self._accel = a

    def update(self, gyro_dps, dt):
        self.q = _step(self.q, gyro_dps, self._accel, dt, self.kp)

    def level(self):
        """Snap orientation so measured gravity points to world +Z, yaw = 0."""
        if not self._accel:
            self.q = [1.0, 0.0, 0.0, 0.0]
            return
        n = magnitude(self._accel) or 1.0
        ux, uy, uz = self._accel["x"] / n, self._accel["y"] / n, self._accel["z"] / n
        # quaternion rotating body-up (ux,uy,uz) onto world +Z
        dot = uz
        if dot > 0.9999:
            self.q = [1.0, 0.0, 0.0, 0.0]
        elif dot < -0.9999:
            self.q = [0.0, 1.0, 0.0, 0.0]
        else:
            cx, cy = -uy, ux            # axis = up × Zworld  (Zworld=(0,0,1))
            s = math.sqrt((1 + dot) * 2)
            self.q = [s * 0.5, cx / s, cy / s, 0.0]
            nrm = math.sqrt(sum(v * v for v in self.q)) or 1.0
            self.q = [v / nrm for v in self.q]


def rotate_vec(q, v):
    """Rotate vector v=(x,y,z) by quaternion q=(w,x,y,z)."""
    w, x, y, z = q
    vx, vy, vz = v
    # t = 2 * cross(q_xyz, v)
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    # v' = v + w*t + cross(q_xyz, t)
    return (vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx))


# ---------------- position estimation (dead reckoning with drift control) ----------------
# A pool stroke is short and starts/ends at rest, so per-stroke double integration
# with zero-velocity boundary constraints (ZUPT) recovers a usable trajectory. The
# world frame has +Z up (gravity direction), matching the fusion output.

_G = 9.80665


def orient_at(track, t):
    """Nearest orientation quaternion in a fused track (binary search)."""
    if not track:
        return (1.0, 0.0, 0.0, 0.0)
    if t <= track[0]["t"]:
        return track[0]["q"]
    if t >= track[-1]["t"]:
        return track[-1]["q"]
    lo, hi = 0, len(track) - 1
    while lo < hi:
        m = (lo + hi) // 2
        if track[m]["t"] < t:
            lo = m + 1
        else:
            hi = m
    return track[lo]["q"]


def linear_world_accel(accel, orient_track):
    """Rotate measured accel to world frame and remove gravity. Returns m/s²."""
    out = []
    for a in accel:
        q = orient_at(orient_track, a["t"])
        wx, wy, wz = rotate_vec(q, (a["x"], a["y"], a["z"]))     # world accel, in g
        out.append({"t": a["t"], "ax": wx * _G, "ay": wy * _G, "az": (wz - 1.0) * _G})
    return out


def _integrate_segment(seg):
    """seg: [{t, ax, ay, az}] over one stroke -> [{t, px, py, pz}] (metres).

    Velocity is detrended to zero at both ends (rest→rest), which removes the
    dominant accel-bias drift; position is detrended so the motion oscillates
    about the origin (best for *seeing* the draw-back / push-forward).
    """
    n = len(seg)
    if n < 3:
        return None
    t = [s["t"] for s in seg]
    T = (t[-1] - t[0]) or 1.0
    vx = [0.0] * n; vy = [0.0] * n; vz = [0.0] * n
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        vx[i] = vx[i - 1] + 0.5 * (seg[i]["ax"] + seg[i - 1]["ax"]) * dt
        vy[i] = vy[i - 1] + 0.5 * (seg[i]["ay"] + seg[i - 1]["ay"]) * dt
        vz[i] = vz[i - 1] + 0.5 * (seg[i]["az"] + seg[i - 1]["az"]) * dt
    for i in range(n):                    # ZUPT: force v(end)=0
        f = (t[i] - t[0]) / T
        vx[i] -= vx[-1] * f; vy[i] -= vy[-1] * f; vz[i] -= vz[-1] * f
    px = [0.0] * n; py = [0.0] * n; pz = [0.0] * n
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        px[i] = px[i - 1] + 0.5 * (vx[i] + vx[i - 1]) * dt
        py[i] = py[i - 1] + 0.5 * (vy[i] + vy[i - 1]) * dt
        pz[i] = pz[i - 1] + 0.5 * (vz[i] + vz[i - 1]) * dt
    for i in range(n):                    # detrend position drift
        f = (t[i] - t[0]) / T
        px[i] -= px[-1] * f; py[i] -= py[-1] * f; pz[i] -= pz[-1] * f
    return [{"t": t[i], "px": px[i], "py": py[i], "pz": pz[i],
             "vx": vx[i], "vy": vy[i], "vz": vz[i]} for i in range(n)]


def position_track(accel, orient_track, strokes):
    """Per-stroke trajectory over the session; origin outside strokes. Metres."""
    if not accel:
        return []
    lin = linear_world_accel(accel, orient_track)
    track = [{"t": a["t"], "px": 0.0, "py": 0.0, "pz": 0.0} for a in accel]
    for s in strokes:
        idxs = [i for i, a in enumerate(accel) if s["t0"] <= a["t"] <= s["t1"]]
        if len(idxs) < 3:
            continue
        seg = _integrate_segment([lin[i] for i in idxs])
        if not seg:
            continue
        for k, i in enumerate(idxs):
            track[i] = {"t": accel[i]["t"], **{p: seg[k][p] for p in ("px", "py", "pz")}}
    return track


def _reversals(t, va, t_lo, t_hi, min_speed=0.08, min_gap=0.15, look=15):
    """Indices where along-cue velocity crosses zero with real motion on both
    sides — the turnaround points of practice strokes and the final backstroke.
    At each, cue velocity is physically zero (you can't reverse without stopping),
    giving free ZUPT anchors mid-window. `look` is generous (±0.6 s at 25 Hz) so
    a deliberate pause at the back — velocity dwelling near zero — still counts."""
    revs = []
    n = len(va)
    for i in range(1, n):
        if not (t_lo <= t[i] <= t_hi):
            continue
        if va[i - 1] * va[i] >= 0:
            continue
        lo_ok = any(abs(va[j]) >= min_speed for j in range(max(0, i - look), i))
        hi_ok = any(abs(va[j]) >= min_speed for j in range(i, min(n, i + look)))
        if lo_ok and hi_ok:
            if revs and t[i] - t[revs[-1]] < min_gap:
                continue
            revs.append(i)
    return revs


def _piecewise_integrate(seg, anchors):
    """Integrate accel → velocity → position with velocity forced to zero at
    every anchor index (rest/turnaround ZUPT). Velocity is linearly detrended
    within each span; position integrates continuously across spans with NO
    end detrend — net forward displacement through a shot is real and kept.
    Error grows ~t² per span, so each extra anchor sharply cuts drift."""
    n = len(seg)
    t = [s["t"] for s in seg]
    vx = [0.0] * n; vy = [0.0] * n; vz = [0.0] * n
    for a0, a1 in zip(anchors[:-1], anchors[1:]):
        cvx = cvy = cvz = 0.0
        raw = [(0.0, 0.0, 0.0)]
        for i in range(a0 + 1, a1 + 1):
            dt = t[i] - t[i - 1]
            cvx += 0.5 * (seg[i]["ax"] + seg[i - 1]["ax"]) * dt
            cvy += 0.5 * (seg[i]["ay"] + seg[i - 1]["ay"]) * dt
            cvz += 0.5 * (seg[i]["az"] + seg[i - 1]["az"]) * dt
            raw.append((cvx, cvy, cvz))
        T = (t[a1] - t[a0]) or 1.0
        for k, i in enumerate(range(a0, a1 + 1)):
            f = (t[i] - t[a0]) / T
            vx[i] = raw[k][0] - cvx * f
            vy[i] = raw[k][1] - cvy * f
            vz[i] = raw[k][2] - cvz * f
    px = [0.0] * n; py = [0.0] * n; pz = [0.0] * n
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        px[i] = px[i - 1] + 0.5 * (vx[i] + vx[i - 1]) * dt
        py[i] = py[i - 1] + 0.5 * (vy[i] + vy[i - 1]) * dt
        pz[i] = pz[i - 1] + 0.5 * (vz[i] + vz[i - 1]) * dt
    return px, py, pz, vx, vy, vz


def _quietest(env, ts, t_from, t_to):
    """Index of the calmest sample in [t_from, t_to] — the integration anchor."""
    import bisect as _b
    i0 = _b.bisect_left(ts, t_from)
    i1 = _b.bisect_right(ts, t_to) - 1
    if i1 < i0:
        return None
    return min(range(i0, i1 + 1), key=lambda i: env[i])


def _first_quiet(env, ts, t_from, t_to, thresh=0.06):
    """EARLIEST quiet sample in [t_from, t_to] — anchors the window at the address
    before the first practice stroke so the whole pre-shot routine is reconstructed
    (the quietest-only anchor skipped practice strokes older than the calm point).
    Falls back to the quietest sample when nothing dips under thresh."""
    import bisect as _b
    i0 = _b.bisect_left(ts, t_from)
    i1 = _b.bisect_right(ts, t_to) - 1
    if i1 < i0:
        return None
    for i in range(i0, i1 + 1):
        if env[i] < thresh:
            return i
    return min(range(i0, i1 + 1), key=lambda i: env[i])


def shot_kinematics(accel, orient_track, cue_axis, strokes, pre=2.5, post=1.5):
    """Butt trajectory reconstructed ONLY inside each shot's window.

    Session-long double integration drifts to tens of metres (ZUPT can't fire
    while the cue is handled) and a session-mean aim frame is meaningless when
    the cue points everywhere over a take — that combination is what made
    playback animation absurd. Per shot instead:

      - aim frame  = mean world cue direction over the address period
        [hit−1.2s, hit−0.2s] (video-validated: orientation is accurate there),
      - window     = anchored at the *calmest* samples within [hit−pre, hit−0.3]
        and [hit+0.3, hit+post] (the pause at address / after follow-through),
      - integration is piecewise ZUPT: besides the rest anchors at both window
        ends, velocity is forced to zero at every *turnaround* (along-cue
        velocity reversal — physically guaranteed zero). Practice strokes give
        one every half-cycle, so drift (~t² per span) stays in the few-cm range.

    Outside shot windows the butt sits at the origin — honest "no data" instead
    of integrated noise.

    Returns (track, shot_info): track = [{t, wx,wy,wz, along, lat, vert, speed}]
    aligned 1:1 with accel; shot_info = one dict per reconstructed shot:
    {hit_t, t_start, t_end, anchors: [times], fwd} for downstream metrics.
    """
    out = [{"t": a["t"], "wx": 0.0, "wy": 0.0, "wz": 0.0,
            "along": 0.0, "lat": 0.0, "vert": 0.0, "speed": 0.0} for a in accel]
    info = []
    if not accel or not strokes:
        return out, info
    lin = linear_world_accel(accel, orient_track)
    ts, env = envelope(accel)
    up = (0.0, 0.0, 1.0)
    prev_end = -1
    MAX_RANGE = 0.95        # a cue stroke never spans more than ~0.95 m — beyond is drift
    for s in strokes:
        hit = s["hit_t"]
        best = None
        # attempt 1: full viewing window from the first quiet address (animates the
        # whole practice routine); attempt 2: tight rest-anchored window (final
        # stroke only). Whichever stays physically plausible wins; neither → no
        # translation shown for this shot (honest, instead of fiction).
        for attempt, (p_, q_) in enumerate(((pre, post), (min(pre, 2.5), min(post, 1.5)))):
            if attempt == 0:
                i0 = _first_quiet(env, ts, hit - p_, hit - 0.3)
            else:
                i0 = _quietest(env, ts, hit - p_, hit - 0.3)
            i1 = _quietest(env, ts, hit + 0.3, hit + q_)
            if i0 is None or i1 is None or i1 - i0 < 3:
                continue
            i0 = max(i0, prev_end + 1)
            if i1 - i0 < 3:
                continue
            # per-shot aim frame from the address period
            sx = sy = sz = 0.0
            n = 0
            for a in accel[i0:i1 + 1]:
                if hit - 1.2 <= a["t"] <= hit - 0.2:
                    f = rotate_vec(orient_at(orient_track, a["t"]), cue_axis)
                    sx += f[0]; sy += f[1]; sz += f[2]
                    n += 1
            if not n:                                 # fall back to window mean
                for a in accel[i0:i1 + 1]:
                    f = rotate_vec(orient_at(orient_track, a["t"]), cue_axis)
                    sx += f[0]; sy += f[1]; sz += f[2]
            fwd = _n3((sx, sy, sz))
            lat = _cr3(up, fwd)
            if math.sqrt(_dt3(lat, lat)) < 1e-6:
                lat = (0.0, 1.0, 0.0)
            lat = _n3(lat)
            vert = _cr3(fwd, lat)
            seg = lin[i0:i1 + 1]
            segt = [x["t"] for x in seg]
            first = _integrate_segment(seg)           # pass 1: locate velocity reversals
            if not first:
                continue
            va = [_dt3((f["vx"], f["vy"], f["vz"]), fwd) for f in first]
            revs = _reversals(segt, va, segt[0] + 0.2, hit - 0.08)
            anchors = [0] + revs + [len(seg) - 1]
            px, py, pz, vx, vy, vz = _piecewise_integrate(seg, anchors)
            alongs = [_dt3((px[k], py[k], pz[k]), fwd) for k in range(len(seg))]
            rng = max(alongs) - min(alongs)
            best = (i0, i1, fwd, lat, vert, anchors, segt, px, py, pz, vx, vy, vz,
                    rng <= MAX_RANGE)
            if rng <= MAX_RANGE:
                break                                  # plausible — take it
        if best is None:
            continue
        i0, i1, fwd, lat, vert, anchors, segt, px, py, pz, vx, vy, vz, ok = best
        prev_end = i1
        if ok:
            for k, i in enumerate(range(i0, i1 + 1)):
                p = (px[k], py[k], pz[k])
                v = (vx[k], vy[k], vz[k])
                out[i] = {"t": accel[i]["t"], "wx": p[0], "wy": p[1], "wz": p[2],
                          "along": _dt3(p, fwd), "lat": _dt3(p, lat), "vert": _dt3(p, vert),
                          "speed": _dt3(v, fwd)}
        info.append({"hit_t": hit, "t_start": segt[0], "t_end": segt[-1],
                     "anchors": [segt[a] for a in anchors], "fwd": fwd,
                     "trans_ok": ok})
    return out, info


def shot_angles(frame_gyr, strokes, pre=4.0, post=2.0):
    """Integrate stroke-frame rates to angles inside each shot's window only,
    anchored to 0 at the window start (the address). Zero elsewhere — matches
    shot_kinematics so timeline lanes and playback agree on what is real."""
    out = [{"t": g["t"], "roll": 0.0, "yaw": 0.0, "pitch": 0.0} for g in frame_gyr]
    if not frame_gyr or not strokes:
        return out
    ts = [g["t"] for g in frame_gyr]
    import bisect as _b
    prev_end = -1
    for s in strokes:
        hit = s["hit_t"]
        i0 = max(_b.bisect_left(ts, hit - pre), prev_end + 1)
        i1 = _b.bisect_right(ts, hit + post) - 1
        if i1 <= i0:
            continue
        prev_end = i1
        roll = yaw = pitch = 0.0
        last = None
        for i in range(i0, i1 + 1):
            g = frame_gyr[i]
            if last is not None:
                dt = g["t"] - last
                if 0 < dt < 0.2:
                    roll += g["roll"] * dt
                    yaw += g["yaw"] * dt
                    pitch += g["pitch"] * dt
            last = g["t"]
            out[i] = {"t": g["t"], "roll": roll, "yaw": yaw, "pitch": pitch}
    return out


def shot_metrics(accel, strokes, kin, shot_info, ang=None):
    """Per-shot stroke metrics from full waveforms (DigiCue-inspired feature set,
    computed here rather than in sealed firmware).

    STEER comes from the GYRO (validated accurate): the cue's yaw swing over the
    final forward stroke (turnaround → impact). Drift-free over that ~0.3 s and
    it directly answers "did the cue rotate off line while delivering".
    Sign convention: +yaw ≈ tip toward the aim-frame's lat axis ("L") —
    provisional until one deliberately steered shot on video pins it down.

    The impact radial/axial shock ratio is kept as STRAIGHTNESS (cleanliness of
    contact) — its *direction* at a single aliased 25 Hz sample is phase noise
    (cue-flex whip), so no direction is claimed from it.

    Returns one dict per stroke (aligned with `strokes`):
    {interval, v_impact, stroke_len, back_depth, pause, steer_side, steer_deg,
     yaw_swing, steer_ratio, finish} — None where not computable.
    """
    out = []
    if not accel:
        return out
    T = [a["t"] for a in accel]
    ts, env = envelope(accel)
    kt = [k["t"] for k in kin] if kin else []
    at = [x["t"] for x in ang] if ang else []
    info_by_hit = {round(i["hit_t"], 3): i for i in shot_info}
    import bisect as _b
    prev_hit = None
    for s in strokes:
        hit = s["hit_t"]
        m = {"interval": (hit - prev_hit) if prev_hit is not None else None,
             "v_impact": None, "stroke_len": None, "back_depth": None,
             "pause": None, "steer_side": None, "steer_deg": None,
             "yaw_swing": None, "steer_ratio": None, "finish": None}
        prev_hit = hit
        # ---- impact vector (body frame, gravity-baseline subtracted) ----
        r0 = _b.bisect_left(T, s["t0"])
        r1 = _b.bisect_right(T, s["t1"]) - 1
        b0 = _b.bisect_left(T, s["t0"] - 0.65)
        b1 = _b.bisect_right(T, s["t0"] - 0.10) - 1
        if r1 >= r0 and b1 > b0:
            nb = b1 - b0 + 1
            bx = sum(accel[i]["x"] for i in range(b0, b1 + 1)) / nb
            by = sum(accel[i]["y"] for i in range(b0, b1 + 1)) / nb
            bz = sum(accel[i]["z"] for i in range(b0, b1 + 1)) / nb
            # transient AT the impact peak (the strike itself, before the rebound —
            # integrating the whole bipolar ring cancels the axial component)
            k = max(range(r0, r1 + 1),
                    key=lambda i: (accel[i]["x"] ** 2 + accel[i]["y"] ** 2 + accel[i]["z"] ** 2))
            rx, ry = accel[k]["x"] - bx, accel[k]["y"] - by
            az = accel[k]["z"] - bz
            if abs(az) > 0.3:                           # real axial hit
                rmag = math.sqrt(rx * rx + ry * ry)
                m["steer_ratio"] = rmag / abs(az)       # side-shock vs axial: contact cleanliness
        # ---- kinematic metrics (need the reconstruction) ----
        inf = info_by_hit.get(round(hit, 3))
        if inf and kt:
            ih = min(_b.bisect_left(kt, hit), len(kin) - 1)
            j0 = _b.bisect_left(kt, hit - 0.12)
            j1 = min(_b.bisect_right(kt, hit + 0.06), len(kin))
            if j1 > j0:
                m["v_impact"] = max(abs(kin[j]["speed"]) for j in range(j0, j1))
            w0 = _b.bisect_left(kt, inf["t_start"])
            m["back_depth"] = -min((kin[j]["along"] for j in range(w0, ih + 1)), default=0.0)
            turns = [a for a in inf["anchors"][1:-1] if a < hit]
            if turns:
                tv = turns[-1]
                jt = min(_b.bisect_left(kt, tv), len(kin) - 1)
                m["stroke_len"] = kin[ih]["along"] - kin[jt]["along"]
                # pause: contiguous |v_along| < 0.06 m/s around the turnaround
                lo = jt
                while lo > w0 and abs(kin[lo - 1]["speed"]) < 0.06:
                    lo -= 1
                hi = jt
                while hi < ih and abs(kin[hi + 1]["speed"]) < 0.06:
                    hi += 1
                m["pause"] = max(0.0, kt[hi] - kt[lo])
        # steer = gyro yaw swing over the final forward stroke (turnaround→impact)
        if at:
            ja = min(_b.bisect_left(at, hit), len(ang) - 1)
            tv = None
            if inf:
                turns = [a for a in inf["anchors"][1:-1] if a < hit]
                tv = turns[-1] if turns else None
            if tv is None:
                tv = hit - 0.35                          # fallback: final stroke duration
            jb = min(_b.bisect_left(at, tv), len(ang) - 1)
            swing = ang[ja]["yaw"] - ang[jb]["yaw"]
            m["yaw_swing"] = swing
            m["steer_deg"] = abs(swing)
            # +yaw ≈ tip toward lat axis ("L") — provisional, needs one calibration shot
            m["steer_side"] = "C" if abs(swing) < 0.8 else ("L" if swing > 0 else "R")
        # ---- finish: how long the cue stays quiet after the strike ----
        f0 = _b.bisect_right(ts, s["t1"])
        fin = None
        run = 0
        for i in range(f0, len(ts)):
            if env[i] > 0.08:
                run += 1
                if run >= 3:
                    fin = ts[max(f0, i - 2)] - s["t1"]
                    break
            else:
                run = 0
        if fin is None and f0 < len(ts):
            fin = ts[-1] - s["t1"]
        m["finish"] = fin
        out.append(m)
    return out


def continuous_kinematics(accel, orient_track, cue_axis, rest_thresh=0.04):
    """Continuous butt trajectory over the WHOLE recording (fills the timeline).

    Integrates gravity-removed acceleration to velocity/position, zeroing velocity
    whenever the sensor is at rest (ZUPT) so drift is reset between motions instead
    of accumulating. Decomposed in the aim frame (mean cue direction over the take).
    Returns [{t, wx,wy,wz, along, lat, vert, speed}] for every accel sample — no
    zeroed gaps; genuine rest simply reads ~0.
    """
    if not accel:
        return []
    lin = linear_world_accel(accel, orient_track)
    _, env = envelope(accel)
    ca = _n3(cue_axis)
    up = (0.0, 0.0, 1.0)
    sx = sy = sz = 0.0
    for a in accel:
        f = rotate_vec(orient_at(orient_track, a["t"]), ca)
        sx += f[0]; sy += f[1]; sz += f[2]
    fwd = _n3((sx, sy, sz))
    lat = _cr3(up, fwd)
    if math.sqrt(_dt3(lat, lat)) < 1e-6:
        lat = (0.0, 1.0, 0.0)
    lat = _n3(lat)
    vert = _cr3(fwd, lat)
    vx = vy = vz = 0.0
    px = py = pz = 0.0
    out = []
    prev = None
    for i, a in enumerate(accel):
        if prev is not None:
            dt = a["t"] - prev
            if 0 < dt < 0.2:
                li, lj = lin[i], lin[i - 1]
                vx += 0.5 * (li["ax"] + lj["ax"]) * dt
                vy += 0.5 * (li["ay"] + lj["ay"]) * dt
                vz += 0.5 * (li["az"] + lj["az"]) * dt
                if env[i] < rest_thresh:            # ZUPT: at rest, kill velocity
                    vx = vy = vz = 0.0
                px += vx * dt; py += vy * dt; pz += vz * dt
        prev = a["t"]
        wp, wv = (px, py, pz), (vx, vy, vz)
        out.append({"t": a["t"], "wx": px, "wy": py, "wz": pz,
                    "along": _dt3(wp, fwd), "lat": _dt3(wp, lat), "vert": _dt3(wp, vert),
                    "speed": _dt3(wv, fwd)})
    return out


def continuous_angles(frame_gyr, rest_rate=8.0):
    """Integrate stroke-frame rates to angles over the whole take, re-zeroed at each
    rest — so angle reads ~0 between motions and shows each stroke's rotation from
    its own address. Returns [{t, roll, yaw, pitch}] (degrees) for every sample."""
    out = []
    roll = yaw = pitch = 0.0
    rr = ry = rp = 0.0
    prev = None
    for g in frame_gyr:
        if prev is not None:
            dt = g["t"] - prev
            if 0 < dt < 0.2:
                roll += g["roll"] * dt
                yaw += g["yaw"] * dt
                pitch += g["pitch"] * dt
        prev = g["t"]
        if math.sqrt(g["roll"] ** 2 + g["yaw"] ** 2 + g["pitch"] ** 2) < rest_rate:
            rr, ry, rp = roll, yaw, pitch          # reset reference at rest
        out.append({"t": g["t"], "roll": roll - rr, "yaw": yaw - ry, "pitch": pitch - rp})
    return out


def stroke_angles(frame_gyr, strokes):
    """Integrate roll/yaw/pitch RATES (from stroke_frame) into ANGLES per stroke,
    anchored to 0 at the stroke's start. Degrees.

    These are the cue's true rotations — measured directly by the gyro and
    independent of where it pivots (rigid body), so steering (yaw) and wrist (roll)
    need no position estimate. Only 1 integration, so far more stable than position.
    """
    track = [{"t": g["t"], "roll": 0.0, "yaw": 0.0, "pitch": 0.0} for g in frame_gyr]
    for s in strokes:
        roll = yaw = pitch = 0.0
        last = None
        for i, g in enumerate(frame_gyr):
            if g["t"] < s["t0"] or g["t"] > s["t1"]:
                continue
            if last is not None:
                dt = g["t"] - last
                roll += g["roll"] * dt
                yaw += g["yaw"] * dt
                pitch += g["pitch"] * dt
            last = g["t"]
            track[i] = {"t": g["t"], "roll": roll, "yaw": yaw, "pitch": pitch}
    return track


def stroke_kinematics(accel, orient_track, cue_axis, strokes):
    """Per-stroke butt trajectory (world) + decomposition against the aim line.

    Steering has two parts and needs both signals: the cue's rotation (gyro,
    reliable) AND the sensor's lateral translation (accel, noisier) — together they
    give where the tip actually went relative to the aim line. So this reconstructs
    the full world position of the butt (double integration with a rest-to-rest
    zero-velocity constraint per stroke, anchored to the impact) and decomposes it
    in the aim frame fixed at the start of each stroke: fwd = cue direction at
    stroke start, up = world +Z, lat = up × fwd.

    Returns [{t, wx,wy,wz (world, m), along, lat, vert (aim frame, m), speed (m/s)}]
    aligned to `accel`; zero outside strokes. `along`/`speed` are the most reliable
    (1-D along the cue); `lat` (steering translation) is the least — drift-prone.
    """
    if not accel:
        return []
    lin = linear_world_accel(accel, orient_track)
    ca = _n3(cue_axis)
    up = (0.0, 0.0, 1.0)
    out = [{"t": a["t"], "wx": 0.0, "wy": 0.0, "wz": 0.0,
            "along": 0.0, "lat": 0.0, "vert": 0.0, "speed": 0.0} for a in accel]
    for s in strokes:
        idxs = [i for i, a in enumerate(accel) if s["t0"] <= a["t"] <= s["t1"]]
        if len(idxs) < 3:
            continue
        t = [accel[i]["t"] for i in idxs]
        n = len(idxs)
        T = (t[-1] - t[0]) or 1.0
        vx = [0.0] * n; vy = [0.0] * n; vz = [0.0] * n
        for k in range(1, n):
            dt = t[k] - t[k - 1]
            li, lj = lin[idxs[k]], lin[idxs[k - 1]]
            vx[k] = vx[k - 1] + 0.5 * (li["ax"] + lj["ax"]) * dt
            vy[k] = vy[k - 1] + 0.5 * (li["ay"] + lj["ay"]) * dt
            vz[k] = vz[k - 1] + 0.5 * (li["az"] + lj["az"]) * dt
        for k in range(n):                        # ZUPT
            f = (t[k] - t[0]) / T
            vx[k] -= vx[-1] * f; vy[k] -= vy[-1] * f; vz[k] -= vz[-1] * f
        px = [0.0] * n; py = [0.0] * n; pz = [0.0] * n
        for k in range(1, n):
            dt = t[k] - t[k - 1]
            px[k] = px[k - 1] + 0.5 * (vx[k] + vx[k - 1]) * dt
            py[k] = py[k - 1] + 0.5 * (vy[k] + vy[k - 1]) * dt
            pz[k] = pz[k - 1] + 0.5 * (vz[k] + vz[k - 1]) * dt
        hi = min(range(n), key=lambda k: abs(t[k] - s["hit_t"]))
        fwd = _n3(rotate_vec(orient_at(orient_track, t[0]), ca))   # aim line at address
        lat = _cr3(up, fwd)
        if math.sqrt(_dt3(lat, lat)) < 1e-6:
            lat = (0.0, 1.0, 0.0)
        lat = _n3(lat)
        vert = _cr3(fwd, lat)
        for k, gi in enumerate(idxs):
            wp = (px[k] - px[hi], py[k] - py[hi], pz[k] - pz[hi])
            wv = (vx[k], vy[k], vz[k])
            out[gi] = {"t": t[k], "wx": wp[0], "wy": wp[1], "wz": wp[2],
                       "along": _dt3(wp, fwd), "lat": _dt3(wp, lat), "vert": _dt3(wp, vert),
                       "speed": _dt3(wv, fwd)}
    return out


def position_scale(track, target_units=2.2):
    """Auto scale so the largest displacement maps to ~target_units (for display)."""
    mx = 0.0
    for p in track:
        d = math.sqrt(p["px"] ** 2 + p["py"] ** 2 + p["pz"] ** 2)
        mx = max(mx, d)
    return (target_units / mx) if mx > 1e-6 else 0.0


def _n3(v):
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) or 1.0
    return (v[0] / n, v[1] / n, v[2] / n)


def _cr3(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dt3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def stroke_frame(accel, gyro, orient_track, cue_axis):
    """Decompose motion into the cue's own frame (gravity removed).

    Acceleration -> {fwd, lat, vert} in g, signed: fwd+ is toward the tip, so a
    backstroke reads negative. Angular velocity -> {roll, yaw, pitch} in dps:
    roll about the cue (wrist), yaw about vertical (steering), pitch about lateral.
    """
    ca = _n3(cue_axis)
    up = (0.0, 0.0, 1.0)

    def frame_at(t):
        q = orient_at(orient_track, t)
        fwd = _n3(rotate_vec(q, ca))
        lat = _cr3(up, fwd)
        if math.sqrt(_dt3(lat, lat)) < 1e-6:
            lat = (0.0, 1.0, 0.0)
        lat = _n3(lat)
        return q, fwd, lat, _cr3(fwd, lat)

    acc = []
    for a in accel:
        q, fwd, lat, vert = frame_at(a["t"])
        w = rotate_vec(q, (a["x"], a["y"], a["z"]))
        lin = (w[0], w[1], w[2] - 1.0)              # remove gravity (world +Z), g
        acc.append({"t": a["t"], "fwd": _dt3(lin, fwd), "lat": _dt3(lin, lat), "vert": _dt3(lin, vert)})
    gyr = []
    for g in gyro:
        q, fwd, lat, vert = frame_at(g["t"])
        wg = rotate_vec(q, (g["x"], g["y"], g["z"]))
        gyr.append({"t": g["t"], "roll": _dt3((g["x"], g["y"], g["z"]), ca),
                    "yaw": _dt3(wg, vert), "pitch": _dt3(wg, lat)})
    return acc, gyr


def pos_at(track, t):
    """Interpolated position (px,py,pz) at time t, or (0,0,0)."""
    if not track:
        return (0.0, 0.0, 0.0)
    if t <= track[0]["t"]:
        p = track[0]
    elif t >= track[-1]["t"]:
        p = track[-1]
    else:
        lo, hi = 0, len(track) - 1
        while lo < hi:
            m = (lo + hi) // 2
            if track[m]["t"] < t:
                lo = m + 1
            else:
                hi = m
        a, b = track[max(0, lo - 1)], track[lo]
        f = (t - a["t"]) / ((b["t"] - a["t"]) or 1)
        return (a["px"] + (b["px"] - a["px"]) * f,
                a["py"] + (b["py"] - a["py"]) * f,
                a["pz"] + (b["pz"] - a["pz"]) * f)
    return (p["px"], p["py"], p["pz"])
