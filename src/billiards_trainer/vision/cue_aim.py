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

#: Bump when the aim GEOMETRY changes. Stored aim records carry this, and
#: the exporter recomputes any record stamped with an older one -- without
#: it, the 8.8px -> 4.7px full-shaft fix would have sat in the code while
#: every session kept serving the cached tip-weighted line.
#:   1 = tip-weighted mean (exp(-db/300))
#:   2 = full-shaft robust total-least-squares (Joe: "use as much of the
#:       visible cue as possible")
AIM_VERSION = 2
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
    refined = _refine_centerline(mask, ang, ax, ay, ball_xy=(cx, cy))
    if refined is not None:
        ang, ax, ay = refined
    total = sum(length for length, _, _, _ in cand)
    quality = min(1.0, best_w / max(1e-6, total)) * min(1.0, best_w / 400.0)
    return ang % (2 * math.pi), quality, (ax, ay)


def _refine_centerline(mask: np.ndarray, ang: float, ax: float, ay: float,
                       step: float = 5.0,
                       ball_xy: tuple | None = None,
                       ) -> tuple[float, float, float] | None:
    """Refit (angle, anchor) through the stick band's per-profile centres.

    Uses the ENTIRE visible shaft (Joe, twice: "use as much of the
    visible cue as possible"). An earlier version weighted samples
    toward the tip on the theory that the raised butt bends the stick's
    image — that reasoning was WRONG: a straight 3D line projects to a
    straight line under a homography, so the shaft's rect-space image is
    straight and every sample is equally valid. What actually corrupted
    the long fit was the forearm and glove sneaking into the band, so
    the answer is OUTLIER REJECTION, not a short lever: fit all clean
    profiles by total least squares, drop the samples furthest from that
    line, refit. Longer lever + robust fit beats a tip-only fit on both
    counts. Returns None when too few clean profiles exist."""
    h, w = mask.shape[:2]

    def _profiles(a: float, x0: float, y0: float) -> list:
        vx, vy = math.cos(a), math.sin(a)
        nx, ny = -vy, vx
        pts = []
        for sgn in (1.0, -1.0):
            # tolerate long invalid runs: the BRIDGE GLOVE is ~20
            # consecutive wide profiles and sits between the anchor and
            # the tip — stopping at it starved the fit of exactly the
            # tip samples the weighting exists for
            misses = 0
            s = step * sgn
            while abs(s) <= 1200.0 and misses < 40:
                sx, sy = x0 + vx * s, y0 + vy * s
                offs = []
                d = -14.0
                while d <= 14.0:
                    px, py = int(sx + nx * d), int(sy + ny * d)
                    if 0 <= px < w and 0 <= py < h and mask[py, px]:
                        offs.append(d)
                    d += 1.0
                # a clean stick profile is a THIN contiguous band; hands,
                # balls and rail wood are wide, gaps mean clutter
                if (3 <= len(offs) <= 13
                        and (offs[-1] - offs[0]) <= 13.0):
                    c = sum(offs) / len(offs)
                    pts.append((sx + nx * c, sy + ny * c))
                    misses = 0
                else:
                    misses += 1
                s += step * sgn
        return pts

    cur_ang, cx_, cy_ = ang, ax, ay
    for _ in range(2):
        pts = _profiles(cur_ang, cx_, cy_)
        if len(pts) < 10:
            return None
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        # ROBUST total-least-squares over the WHOLE shaft: fit, measure
        # each sample's perpendicular distance, drop the worst tail
        # (arm/glove), refit. Two passes settle it.
        keep = np.ones(len(xs), dtype=bool)
        mx_ = my_ = 0.0
        dx = dy = 0.0
        for _pass in range(3):
            kx, ky = xs[keep], ys[keep]
            if len(kx) < 8:
                keep = np.ones(len(xs), dtype=bool)
                kx, ky = xs, ys
            mx_, my_ = float(kx.mean()), float(ky.mean())
            u = np.stack([kx - mx_, ky - my_])
            evals, evecs = np.linalg.eigh(u @ u.T)
            dx, dy = float(evecs[0, 1]), float(evecs[1, 1])
            perp = np.abs(-dy * (xs - mx_) + dx * (ys - my_))
            med = float(np.median(perp[keep])) if keep.any() else 0.0
            cut = max(1.5, 3.0 * max(med, 0.35))
            nk = perp <= cut
            if nk.sum() >= 8:
                keep = nk
        new_ang = math.atan2(dy, dx)
        d = (new_ang - cur_ang) % (2 * math.pi)
        if min(d, 2 * math.pi - d) > math.pi / 2:
            new_ang += math.pi
        cur_ang, cx_, cy_ = new_ang % (2 * math.pi), mx_, my_
    # sanity: refinement must agree with the Hough cluster within a few
    # degrees — a runaway fit (arm, rail) must not replace the stick
    d = (cur_ang - ang) % (2 * math.pi)
    if min(d, 2 * math.pi - d) > math.radians(4.0):
        return None
    return cur_ang, cx_, cy_


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


def infer_target(cue_xy: tuple, ang: float, balls: list, ball_r: float,
                 ) -> tuple[int, tuple[float, float], float] | None:
    """First ball in the cue's collision path — Joe's target inference.

    Capsule sweep: the cue ball's centre travels along ``ang`` from
    ``cue_xy``; any ball whose centre lies within 2r of the path line
    (and ahead of the cue) can be struck; the FIRST contact wins,
    assuming no kick. ``balls`` is [(num, x, y)] in PLANE space (physics
    uses contact points; render via visual_from_plane afterwards).
    Returns (ball num, ghost-ball centre at contact, travel px) or None
    when nothing is in the path (kick shot / safety).
    """
    vx, vy = math.cos(ang), math.sin(ang)
    reach = 2.0 * float(ball_r)
    best = None
    for num, bx, by in balls:
        rx, ry = bx - cue_xy[0], by - cue_xy[1]
        t_par = rx * vx + ry * vy
        if t_par <= 0:
            continue                      # behind the cue ball
        d_perp = abs(rx * vy - ry * vx)
        if d_perp >= reach:
            continue                      # path misses it
        t_hit = t_par - math.sqrt(reach * reach - d_perp * d_perp)
        if t_hit <= 0:
            continue                      # already overlapping: ignore
        if best is None or t_hit < best[2]:
            best = (int(num),
                    (cue_xy[0] + vx * t_hit, cue_xy[1] + vy * t_hit),
                    float(t_hit))
    return best
