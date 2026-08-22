"""Dedicated cue-ball detector — RAW frame (M2: persistent cue-ball tracking).

The cue ball is the only PURE-white object on the table. A wide saturation/value
band finds white-ish blobs robustly (no knife-edge threshold to pump the size),
then a per-blob HSV-purity test separates the cue ball from STRIPED balls — a
stripe has a white equator that passes a naive white mask but is mostly coloured,
so it fails the purity check. Finally, single-instance enforcement: there is
exactly one cue ball, so we return only the whitest surviving candidate. This
fixes "stripes read as the cue ball" and "two cue balls at once".

Returns [] rather than raising on any frame (self-healing contract).
"""

import cv2
import numpy as np

from ..core.types import BallClass, Detection
from . import DetectorStrategy, ball_radius_raw, table_polygon_mask


class CueBallWhiteStrategy(DetectorStrategy):
    name = "cue_ball_white"
    description = "Whitest pure-white blob on the bed (HSV-purity gated, single instance) = the cue ball."

    # Shape mask: deliberately LOOSE so the bright cue ball is found even under
    # warm/tinted lighting (on real footage "white" reads as bright-but-saturated,
    # meanS ~50-75, not pure S<30 — an absolute white gate kills recall).
    S_MAX = 80
    V_MIN = 185
    MIN_CIRCULARITY = 0.6
    # Stripe/solid rejector: a real coloured ball has a meaningfully SATURATED
    # region (a stripe's band, a solid's whole face). The cue does not.
    COLOR_S = 110      # "clearly saturated colour" if S above this ...
    COLOR_V = 70       # ... and V above this
    MAX_COLOR_FRAC = 0.35   # reject blobs with this much saturated-colour area
    MAX_MEAN_S = 115        # and blobs that are too saturated overall to be white

    def _stats(self, hsv, cx, cy, rad, shape):
        """(mean saturation, % saturated-colour pixels) inside the candidate circle.
        The cue ball is the least-saturated bright blob with ~no colour."""
        h, w = shape[:2]
        x0, y0 = max(0, int(cx - rad)), max(0, int(cy - rad))
        x1, y1 = min(w, int(cx + rad) + 1), min(h, int(cy + rad) + 1)
        if x1 <= x0 or y1 <= y0:
            return 255.0, 1.0
        patch = hsv[y0:y1, x0:x1]
        yy, xx = np.ogrid[y0:y1, x0:x1]
        cmask = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad * rad
        sel = patch[cmask]
        if sel.size == 0:
            return 255.0, 1.0
        s, v = sel[:, 1], sel[:, 2]
        mean_s = float(s.mean())
        color = float(((s > self.COLOR_S) & (v > self.COLOR_V)).mean())
        return mean_s, color

    def detect(self, frame_bgr, calib):
        table = table_polygon_mask(frame_bgr.shape, calib)
        table = cv2.erode(table, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (0, 0, self.V_MIN), (180, self.S_MAX, 255))
        white = cv2.bitwise_and(white, table)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k)
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, k)

        r = ball_radius_raw(calib, frame_bgr.shape)
        min_area = float(np.pi * (0.5 * r) ** 2)
        max_area = float(np.pi * (2.2 * r) ** 2)

        cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue
            perim = cv2.arcLength(c, True)
            if perim <= 0:
                continue
            circ = 4.0 * np.pi * area / (perim * perim)
            if circ < self.MIN_CIRCULARITY:
                continue
            (cx, cy), rad = cv2.minEnclosingCircle(c)
            mean_s, color_frac = self._stats(hsv, cx, cy, rad, frame_bgr.shape)
            # the stripe/solid rejector: a coloured ball carries saturated colour;
            # the cue does not, and is the least-saturated bright blob.
            if color_frac > self.MAX_COLOR_FRAC or mean_s > self.MAX_MEAN_S:
                continue
            cands.append((mean_s, float(cx), float(cy), float(rad)))

        if not cands:
            return []
        # Single-instance: exactly one cue ball — keep the LEAST-saturated (whitest)
        # candidate, so a stripe that slips the gate still loses to the true cue.
        cands.sort(key=lambda t: t[0])
        mean_s, cx, cy, rad = cands[0]
        score = float(min(0.99, max(0.5, 1.0 - mean_s / 200.0)))  # whiter -> higher
        return [Detection(cx, cy, max(3.0, rad), (245, 245, 245), BallClass.CUE, score)]


STRATEGIES = [CueBallWhiteStrategy()]
