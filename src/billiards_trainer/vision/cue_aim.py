"""Cue-stick aim detection — where is the cue POINTING?

Joe's first analysis tool: "if I miss a ball, I can see if my cue is just
pointing wrong." The aim line is computed HERE, once, during analysis —
desktop and phone both render the same stored geometry, so the two can
never disagree (his hard requirement).

Method (frame-verified on session-20260820-005647 address frames): the
stick is a long thin bright structure against the felt whose extension
passes through the cue ball. Hough segments on the non-felt edge map,
keep those whose infinite line passes near the cue centre, then take the
DENSEST 3-degree angle cluster — the stick's two parallel edges agree
with each other; a forearm edge that sneaks past the distance filter
does not (it tilted the naive weighted mean visibly on the @100s frame).
Aim direction points AWAY from the stick's mass (tip through ball).
"""

import math

import cv2
import numpy as np

#: minimum Hough segment length (rect px) — arms produce short curly
#: edges at this threshold; the stick produces 100-400px runs
_MIN_LEN = 90.0
#: a candidate's infinite line must pass within this many cue radii
_NEAR = 2.0
#: angle cluster half-width — stick edges agree well inside this
_CLUSTER_DEG = 3.0


def detect_cue_aim(rect_bgr: np.ndarray, cue_xy: tuple, cue_r: float,
                   ) -> tuple[float, float] | None:
    """(aim angle radians, quality 0..1) in rect space, or None.

    ``rect_bgr`` is the rectified table image; ``cue_xy`` the cue ball
    centre in the same coordinates. Quality reflects how much collinear
    stick evidence backs the angle — callers should demand a few
    consistent samples across the address before trusting one.
    """
    if rect_bgr is None or rect_bgr.size == 0:
        return None
    cx, cy = float(cue_xy[0]), float(cue_xy[1])
    hsv = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    ch = float(np.median(hsv[..., 0]))
    cs = float(np.median(hsv[..., 1]))
    cv_ = float(np.median(hsv[..., 2]))
    dh = np.abs(hsv[..., 0] - ch)
    dh = np.minimum(dh, 180 - dh)
    dist = (2.0 * dh + 0.8 * np.abs(hsv[..., 1] - cs)
            + 0.4 * np.abs(hsv[..., 2] - cv_))
    mask = (dist > 90).astype(np.uint8) * 255
    edges = cv2.Canny(mask, 60, 140)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=60,
                            minLineLength=int(_MIN_LEN), maxLineGap=12)
    if lines is None:
        return None
    near = max(6.0, _NEAR * cue_r)
    cand = []                      # (length, undirected angle, mid x, mid y)
    for x1, y1, x2, y2 in lines[:, 0]:
        length = math.hypot(x2 - x1, y2 - y1)
        if length < _MIN_LEN:
            continue
        d = (abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1)
             / max(1e-6, length))
        if d > near:
            continue
        ang = math.atan2(y2 - y1, x2 - x1) % math.pi   # undirected
        cand.append((length, ang, (x1 + x2) / 2.0, (y1 + y2) / 2.0))
    if not cand:
        return None
    # densest angle cluster by total length (circular over half-turn)
    half = math.radians(_CLUSTER_DEG)
    best, best_w = None, 0.0
    for _, a0, _, _ in cand:
        w = sum(length for length, a, _, _ in cand
                if min(abs(a - a0), math.pi - abs(a - a0)) <= half)
        if w > best_w:
            best, best_w = a0, w
    members = [(length, a, mx, my) for length, a, mx, my in cand
               if min(abs(a - best), math.pi - abs(a - best)) <= half]
    # circular mean over the half-turn domain (double-angle trick)
    sx = sum(length * math.cos(2 * a) for length, a, _, _ in members)
    sy = sum(length * math.sin(2 * a) for length, a, _, _ in members)
    ang = 0.5 * math.atan2(sy, sx) % math.pi
    # SIGN: the stick's mass lies behind the ball; aim points away from it
    mvx, mvy = math.cos(ang), math.sin(ang)
    proj = sum(length * ((mx - cx) * mvx + (my - cy) * mvy)
               for length, _, mx, my in members)
    if proj > 0:                   # mass is on the +ang side: aim is -ang
        ang += math.pi
    total = sum(length for length, _, _, _ in cand)
    quality = min(1.0, best_w / max(1e-6, total)) * min(1.0, best_w / 400.0)
    return ang % (2 * math.pi), quality


def aim_ray_end(cx: float, cy: float, ang: float,
                bounds: tuple[float, float, float, float]) -> tuple:
    """Where the aim ray first leaves the bed rectangle (one segment, no
    reflections — v1 keeps the overlay honest and simple)."""
    x0, y0, x1, y1 = bounds
    vx, vy = math.cos(ang), math.sin(ang)
    ts = []
    if vx > 1e-9:
        ts.append((x1 - cx) / vx)
    elif vx < -1e-9:
        ts.append((x0 - cx) / vx)
    if vy > 1e-9:
        ts.append((y1 - cy) / vy)
    elif vy < -1e-9:
        ts.append((y0 - cy) / vy)
    t = min((t for t in ts if t > 0), default=0.0)
    return cx + vx * t, cy + vy * t
