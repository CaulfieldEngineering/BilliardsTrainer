"""Felt-mask-prior local Hough (CueDetat-style) — pure OpenCV, on the RAW frame.

Idea (from HereLiesAz/CueDetat, reimplemented; no code copied): the felt is a
strong prior. A ball is a ball-sized NON-felt blob *surrounded by* felt. So:
  1. build the felt mask on the raw frame (within the table polygon)
  2. morphological-CLOSE it with a ball-sized kernel — this fills the holes the
     balls punch in the felt
  3. (closed felt) − (felt) = exactly those filled holes = ball candidates
  4. for each candidate blob, run HoughCircles *locally* in its crop to refine
     the centre/radius (Hough on a tiny ROI is fast + robust, vs. blind global)
This restricts detection to balls on the cloth and ignores rails/hands/background.
"""

import cv2
import numpy as np

from ..vision.types import Detection
from . import (
    DetectorStrategy,
    ball_radius_raw,
    classify_crop,
    felt_mask_raw,
    table_polygon_mask,
)


class FeltMaskHoughStrategy(DetectorStrategy):
    name = "felt_mask_hough"
    description = "Felt-mask prior -> close -> subtract -> local Hough per blob (raw frame, CueDetat-style)."

    def detect(self, frame_bgr, calib):
        table = table_polygon_mask(frame_bgr.shape, calib)
        felt = cv2.bitwise_and(felt_mask_raw(frame_bgr, calib), table)
        r = ball_radius_raw(calib, frame_bgr.shape)
        k = max(3, int(r))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        sealed = cv2.morphologyEx(felt, cv2.MORPH_CLOSE, ker)
        # holes the close() filled = ball candidates (non-felt surrounded by felt)
        cand = cv2.subtract(sealed, felt)
        cand = cv2.bitwise_and(cand, table)
        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        gray = cv2.medianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), 3)
        H, W = frame_bgr.shape[:2]
        min_a, max_a = np.pi * (0.4 * r) ** 2, np.pi * (2.6 * r) ** 2
        out: list[Detection] = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (min_a <= area <= max_a):
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            pad = int(r)
            x0, y0 = max(0, bx - pad), max(0, by - pad)
            x1, y1 = min(W, bx + bw + pad), min(H, by + bh + pad)
            crop = gray[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            circles = cv2.HoughCircles(crop, cv2.HOUGH_GRADIENT, dp=1.2,
                                       minDist=int(r * 1.2), param1=110, param2=14,
                                       minRadius=int(r * 0.5), maxRadius=int(r * 2.2))
            if circles is not None:
                cx, cy, cr = circles[0][0]
                gx, gy, gr = x0 + float(cx), y0 + float(cy), float(cr)
            else:  # fallback: enclosing circle of the candidate blob
                (mc, mcy), mr = cv2.minEnclosingCircle(c)
                gx, gy, gr = float(mc), float(mcy), float(mr)
                if not (r * 0.5 <= gr <= r * 2.2):
                    continue
            cls, bgr = classify_crop(frame_bgr, gx, gy, gr)
            out.append(Detection(gx, gy, gr, bgr, cls, 0.9))
        return _dedupe(out, r * 1.1)


def _dedupe(dets, min_dist):
    kept = []
    for d in sorted(dets, key=lambda d: -d.score):
        if all(np.hypot(d.x - k.x, d.y - k.y) > min_dist for k in kept):
            kept.append(d)
    return kept


STRATEGIES = [FeltMaskHoughStrategy()]
