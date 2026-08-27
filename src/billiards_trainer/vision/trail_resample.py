"""Dense, video-locked trail resampling (Joe: "the trail should be pixel
perfectly in sync with the cue ball's motion").

The tracker samples at ~10Hz and goes blind for ~1.5-2s at takeoff
(motion blur defeats the detector; rest-frozen identity holds the last
position). No interpolation over 150ms-to-2s holes can track the real
ball. But at REVIEW time the problem is easy: within a shot window the
only things moving on the table are the shot's balls, so frame
differencing against a pre-strike reference finds them at full frame
rate, in video pixels, on the video clock - the exact space the phone
draws in.

Guardrails (measured-or-abstained): each ball's dense path must agree
with the tracker's own sparse samples where both exist; disagreement
abstains and the sparse trail ships unchanged. Player arm/cue blobs are
rejected by area; contested frames (merged blobs) are skipped - a 33ms
hole is invisible.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger("vision.trail_resample")

# blob acceptance, in units of expected ball AREA (radius from meta)
_AREA_LO, _AREA_HI = 0.25, 9.0     # blur streaks elongate; arms are >9x
_AGREE_FRAC = 0.035                # sparse sample must lie this close to the
                                   # dense path (fraction of frame width)
_DIFF_THRESH = 26


def _ball_blobs(mask, area: float, r_scaled: float) -> list:
    """Ball-sized moving blobs in a diff mask, rejecting fragments glued
    to oversized movers (the player's arm/cue shed ball-sized pieces that
    hijacked tracks)."""
    n_lbl, _lbl, stats, cents = cv2.connectedComponentsWithStats(mask)
    # oversized movers as RECTANGLES: an arm is long, so centroid distance
    # under-vetoes exactly where balls hide (at the fingertips)
    big = [stats[i] for i in range(1, n_lbl)
           if stats[i][4] > _AREA_HI * area]

    def near_big(bx, by):
        for (gx, gy, gw, gh, _a) in big:
            dx = max(gx - bx, 0, bx - (gx + gw))
            dy = max(gy - by, 0, by - (gy + gh))
            if (dx * dx + dy * dy) ** 0.5 < 5.0 * r_scaled:
                return True
        return False

    blobs = []
    for i in range(1, n_lbl):
        if not (_AREA_LO * area <= stats[i][4] <= _AREA_HI * area):
            continue
        bx, by = cents[i][0], cents[i][1]
        if near_big(bx, by):
            continue
        blobs.append((bx, by, stats[i][4]))
    return blobs


def resample_shot(video: str, t0: float, t1: float,
                  seeds: dict, ball_r_px: float) -> dict:
    """seeds: {ball_n: (x_norm, y_norm)} at flight start. Returns
    {ball_n: [[t_video, x_norm, y_norm], ...]} for balls that tracked
    cleanly; missing keys = abstained."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {}
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not (w and h):
            return {}
        scale = 0.5                          # half-res: blobs, not pixels
        area = np.pi * (ball_r_px * scale) ** 2

        # reference: settled table just before the strike
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, (t0 - 0.25)) * 1000)
        ok, ref = cap.read()
        if not ok:
            return {}
        ref = cv2.resize(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None,
                         fx=scale, fy=scale)

        # per ball: last position, last time, velocity (px/s), locked?
        st = {n: {"p": (sx * w * scale, sy * h * scale), "t": None,
                  "v": (0.0, 0.0), "lock": False}
              for n, (sx, sy) in seeds.items()}
        out: dict = {n: [] for n in seeds}
        vmax = 1.3 * w * scale               # plausible ball speed, px/s

        cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
        while True:
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            ok, frame = cap.read()
            if not ok or t > t1:
                break
            g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None,
                           fx=scale, fy=scale)
            d = cv2.absdiff(g, ref)
            _, m = cv2.threshold(d, _DIFF_THRESH, 255, cv2.THRESH_BINARY)
            m = cv2.dilate(m, np.ones((3, 3), np.uint8))
            blobs = _ball_blobs(m, area, ball_r_px * scale)
            claimed: dict = {}
            for n, s2 in st.items():
                if not blobs:
                    continue
                if s2["lock"] and s2["t"] is not None:
                    dt = max(1e-3, t - s2["t"])
                    px = s2["p"][0] + s2["v"][0] * dt
                    py = s2["p"][1] + s2["v"][1] * dt
                    rad = max(3.0 * ball_r_px * scale, 1.6 * vmax * dt * 0.3)
                else:
                    # acquisition: the ball is within vmax*(elapsed) of the
                    # seed - the disc grows until the takeoff blob separates
                    # from the arm/cue cluster and gets caught mid-flight
                    px, py = s2["p"]
                    rad = 2.0 * ball_r_px * scale + vmax * max(0.0, t - t0)
                best, bd = None, rad
                for bi, (bx, by, _a) in enumerate(blobs):
                    dd = ((bx - px) ** 2 + (by - py) ** 2) ** 0.5
                    if dd < bd:
                        best, bd = bi, dd
                if best is not None:
                    claimed.setdefault(best, []).append(n)
            for bi, ns in claimed.items():
                if len(ns) != 1:
                    continue                 # merged/contested: skip frame
                n = ns[0]
                s2 = st[n]
                bx, by, _a = blobs[bi]
                if s2["t"] is not None:
                    dt = max(1e-3, t - s2["t"])
                    s2["v"] = ((bx - s2["p"][0]) / dt, (by - s2["p"][1]) / dt)
                    s2["lock"] = True
                s2["p"], s2["t"] = (bx, by), t
                out[n].append([round(t, 3),
                               round(bx / (w * scale), 4),
                               round(by / (h * scale), 4)])
        return {n: pts for n, pts in out.items() if len(pts) >= 5}
    finally:
        cap.release()


def agrees(dense: list, sparse: list) -> bool:
    """GEOMETRIC audit, deliberately time-free: sidecar and video clocks
    drift per shot (the whole one-clock saga), so comparing positions at
    matched timestamps rejects good paths for clock reasons. Instead:
    every MOVING sparse sample must lie spatially near the dense
    polyline, and both paths must end in the same place (rest)."""
    if not dense or len(sparse) < 3:
        return False
    moving = []
    x0, y0 = sparse[0][1], sparse[0][2]
    for (_st, sx, sy) in sparse:
        if ((sx - x0) ** 2 + (sy - y0) ** 2) ** 0.5 > 0.008:
            moving.append((sx, sy))
    if len(moving) < 2:
        return False
    for (sx, sy) in moving:
        dd = min(((dx - sx) ** 2 + (dy - sy) ** 2) ** 0.5
                 for (_dt, dx, dy) in dense)
        if dd > _AGREE_FRAC:
            return False
    ex, ey = dense[-1][1], dense[-1][2]
    lx, ly = sparse[-1][1], sparse[-1][2]
    return ((ex - lx) ** 2 + (ey - ly) ** 2) ** 0.5 < 0.05
