"""Piecewise-straight trajectory fit: one model over ALL observations.

The design docs/design/FUSION.md builds toward, and the replacement for
per-sample verdict heuristics (four of them broke in four days; the last
flipped its answer with sampling phase). A pool ball rolls in straight
segments broken only by collisions — with a cushion (known geometry) or
another ball. So the honest model is a small number of straight segments,
fitted to every observation at once, with residuals that say how much to
trust it.

v1 fits ONE or TWO segments:
  * one  — the ball rolled and stopped (or was only seen on one leg)
  * two  — one collision inside the window; if the breakpoint lies near a
           cushion and the velocity component normal to it flips, the
           collision IS that cushion, named
Verdicts read off the fit:
  * departure direction  — segment 1's direction (longest usable baseline)
  * first rail           — the named cushion at the breakpoint, if any
  * residual             — worst perpendicular distance of any observation
                           from its segment; the trust signal

Deliberately NOT in v1: deceleration modelling (direction verdicts do not
need speed), 3+ segments (rare inside one shot window, and two covers the
miss-rattle-return shape that dominates the library).
"""

from __future__ import annotations

import math

__all__ = ["fit_shot", "Fit"]


class Fit:
    """Result of a piecewise fit. Positions in rect px, times in seconds."""

    def __init__(self, segments, residual, rail, break_t):
        #: list of (t0, t1, (x0, y0), (ux, uy)) — start time/end time,
        #: start point, unit direction — one per straight leg
        self.segments = segments
        #: worst perpendicular residual (px) of any observation vs its leg
        self.residual = residual
        #: "top" | "bottom" | "left" | "right" | None — the cushion at the
        #: first breakpoint, when the geometry names one
        self.rail = rail
        #: time of the first breakpoint (None for a single leg)
        self.break_t = break_t

    @property
    def departure(self):
        """Unit direction of the first leg."""
        return self.segments[0][3]


def _fit_line(pts):
    """Total-least-squares line through pts [(t,x,y)]. Returns
    ((mx, my), (ux, uy), worst_residual). Direction is oriented by time."""
    n = len(pts)
    mx = sum(p[1] for p in pts) / n
    my = sum(p[2] for p in pts) / n
    sxx = sum((p[1] - mx) ** 2 for p in pts)
    syy = sum((p[2] - my) ** 2 for p in pts)
    sxy = sum((p[1] - mx) * (p[2] - my) for p in pts)
    # principal axis of the covariance (closed form, no numpy needed)
    theta = 0.5 * math.atan2(2 * sxy, sxx - syy)
    ux, uy = math.cos(theta), math.sin(theta)
    # orient along time: later points should have larger projection
    if n >= 2:
        proj_first = (pts[0][1] - mx) * ux + (pts[0][2] - my) * uy
        proj_last = (pts[-1][1] - mx) * ux + (pts[-1][2] - my) * uy
        if proj_last < proj_first:
            ux, uy = -ux, -uy
    worst = max(abs(-uy * (p[1] - mx) + ux * (p[2] - my)) for p in pts)
    return (mx, my), (ux, uy), worst


def _intersect(ma, da, mb, db):
    """Intersection of two lines (point, direction), or None if parallel."""
    det = da[0] * (-db[1]) - da[1] * (-db[0])
    if abs(det) < 1e-9:
        return None
    rx, ry = mb[0] - ma[0], mb[1] - ma[1]
    t = (rx * (-db[1]) - ry * (-db[0])) / det
    return (ma[0] + da[0] * t, ma[1] + da[1] * t)


def fit_shot(obs, table, ball_r):
    """Fit [(t, x, y)] observations of one ball across one shot.

    ``table`` needs x0/y0/x1/y1 (the bed rect). Returns a Fit, or None
    when there are not enough observations to say anything (< 3, or < 2
    after the ball actually moves).
    """
    if len(obs) < 3:
        return None
    # drop the leading at-rest samples: they are one repeated point and
    # would dominate any least-squares fit with zero information
    moved = 0
    x0, y0 = obs[0][1], obs[0][2]
    for i, p in enumerate(obs):
        if math.hypot(p[1] - x0, p[2] - y0) > 0.5 * ball_r:
            moved = i
            break
    else:
        return None
    pts = [obs[max(0, moved - 1)]] + list(obs[moved:])
    if len(pts) < 3:
        return None

    # --- single leg
    _m1, d1, r1 = _fit_line(pts)
    best = Fit([(pts[0][0], pts[-1][0], (pts[0][1], pts[0][2]), d1)],
               r1, None, None)
    if len(pts) < 5:
        return best

    # --- two legs: try every interior breakpoint, keep the best total
    for k in range(2, len(pts) - 2):
        a, b = pts[: k + 1], pts[k:]
        _ma, da, ra = _fit_line(a)
        _mb, db, rb = _fit_line(b)
        worst = max(ra, rb)
        if worst < best.residual - 0.15 * ball_r:   # must clearly beat one leg
            bt = pts[k][0]
            # The TRUE corner is where the two fitted legs INTERSECT — the
            # chosen breakpoint sample can sit one observation up the
            # return leg, next to the wrong cushion (measured on @233: the
            # sample named "left", the intersection names "bottom").
            bx, by = _intersect(_ma, da, _mb, db) or (pts[k][1], pts[k][2])
            # Which cushion: the axis whose velocity component FLIPS names
            # it — a horizontal cushion flips vy, a vertical one flips vx.
            # A pocket-jaw rattle can flip both; the DOMINANT incoming
            # component decides. The named cushion must then actually be
            # where the corner is; otherwise it was a ball, not a rail.
            flip_x = da[0] * db[0] < 0
            flip_y = da[1] * db[1] < 0
            rail = None
            if flip_y and (not flip_x or abs(da[1]) >= abs(da[0])):
                rail = "bottom" if by > (table.y0 + table.y1) / 2 else "top"
                edge = (table.y1 - by) if rail == "bottom" else (by - table.y0)
            elif flip_x:
                rail = "right" if bx > (table.x0 + table.x1) / 2 else "left"
                edge = (table.x1 - bx) if rail == "right" else (bx - table.x0)
            if rail is not None and edge > 3.5 * ball_r:
                rail = None                          # bent mid-felt: a ball
            best = Fit([(a[0][0], bt, (a[0][1], a[0][2]), da),
                        (bt, b[-1][0], (bx, by), db)],
                       worst, rail, bt)
    return best
