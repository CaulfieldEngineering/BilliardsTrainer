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
                   ) -> tuple[float, float, tuple[float, float]] | None:
    """(aim angle radians, quality 0..1, anchor point) in rect space.

    The anchor lies ON the stick's fitted axis — the extension is drawn
    through the CUE (Joe's spec), never re-anchored at the ball centre:
    forcing the ray through the ball would hide exactly the error being
    hunted (a stick that isn't actually lined up through the ball).
    ``cue_xy`` is used only to FIND the stick (a stick being aimed passes
    near the ball). Quality reflects how much collinear stick evidence
    backs the angle — callers should demand a few consistent samples
    across the address before trusting one.
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
    # ANCHOR on the stick's axis: length-weighted centroid of the cluster
    # segments. Rendering extends the line THROUGH this point — the
    # stick's true axis — so any lateral offset from the ball shows.
    wsum = sum(length for length, _, _, _ in members)
    ax = sum(length * mx for length, _, mx, _ in members) / wsum
    ay = sum(length * my for length, _, _, my in members) / wsum
    # SUB-PIXEL CENTERLINE (Joe: "aligned perfectly with the cue"): the
    # Hough cluster averages EDGE segments, which sits a hair off when
    # glare favours one edge. Walk the axis, take a perpendicular
    # profile every few px, keep thin bright bands (the glove and the
    # ball fail the width test and drop out), and refit the line
    # through the band centres.
    refined = _refine_centerline(mask, ang, ax, ay)
    if refined is not None:
        ang, ax, ay = refined
    total = sum(length for length, _, _, _ in cand)
    quality = min(1.0, best_w / max(1e-6, total)) * min(1.0, best_w / 400.0)
    return ang % (2 * math.pi), quality, (ax, ay)


def _refine_centerline(mask: np.ndarray, ang: float, ax: float, ay: float,
                       span: float = 260.0, step: float = 7.0,
                       ) -> tuple[float, float, float] | None:
    """Refit (angle, anchor) through the stick band's per-profile centres.
    Returns None when too few clean profiles exist (keep the Hough fit)."""
    h, w = mask.shape[:2]
    vx, vy = math.cos(ang), math.sin(ang)
    nx, ny = -vy, vx
    pts = []
    s = -span
    while s <= span:
        sx, sy = ax + vx * s, ay + vy * s
        offs, vals = [], []
        d = -14.0
        while d <= 14.0:
            px, py = int(sx + nx * d), int(sy + ny * d)
            if 0 <= px < w and 0 <= py < h and mask[py, px]:
                offs.append(d)
                vals.append(1.0)
            d += 1.0
        # a clean stick profile is a THIN contiguous band; hands, balls
        # and rail wood are wide, gaps mean clutter
        if 3 <= len(offs) <= 13 and (offs[-1] - offs[0]) <= 13.0:
            c = sum(offs) / len(offs)
            pts.append((sx + nx * c, sy + ny * c))
        s += step
    if len(pts) < 8:
        return None
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    mx_, my_ = float(xs.mean()), float(ys.mean())
    # total least squares via PCA; keep the aim's forward sign
    u = np.stack([xs - mx_, ys - my_])
    cov = u @ u.T
    evals, evecs = np.linalg.eigh(cov)
    dx, dy = float(evecs[0, 1]), float(evecs[1, 1])
    new_ang = math.atan2(dy, dx)
    d = (new_ang - ang) % (2 * math.pi)
    if min(d, 2 * math.pi - d) > math.pi / 2:
        new_ang += math.pi
    # sanity: refinement must agree with the cluster within a few degrees
    d = (new_ang - ang) % (2 * math.pi)
    if min(d, 2 * math.pi - d) > math.radians(4.0):
        return None
    return new_ang % (2 * math.pi), mx_, my_


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
