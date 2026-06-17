"""Dedicated cue-ball detector — RAW frame (M2: persistent cue-ball tracking).

The cue ball is the only pure-white object on the table, and white-on-felt is the
cleanest signal there is. Crucially this is structurally MORE stable than
SimpleBlobDetector for the white ball: a wide saturation/value BAND (not a single
binarization threshold) means a few grains of sensor noise can't push edge pixels
across a knife-edge and pump the blob's size frame-to-frame — the exact "ball
keeps changing size / appearing+disappearing on a perfectly still scene" symptom.

Method: HSV white mask (high V, low S) restricted to inside the table polygon,
morphologically cleaned, then the round connected components near the expected ball
radius are returned as CUE detections (largest/roundest first). Returns [] rather
than raising on any frame (self-healing contract).
"""

import cv2
import numpy as np

from ..vision.types import BallClass, Detection
from . import DetectorStrategy, ball_radius_raw, table_polygon_mask


class CueBallWhiteStrategy(DetectorStrategy):
    name = "cue_ball_white"
    description = "White-on-felt HSV band inside the table → the cue ball (stable, no threshold edge)."

    # HSV white band. Wide on H (white has no hue), low S, high V. Deliberately
    # generous so noise can't flip pixels in/out at the boundary.
    S_MAX = 70
    V_MIN = 175
    MIN_CIRCULARITY = 0.6

    def detect(self, frame_bgr, calib):
        table = table_polygon_mask(frame_bgr.shape, calib)
        # erode the table mask a touch so bright rails/diamonds just outside the
        # bed don't leak in as "white"
        er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        table = cv2.erode(table, er)

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (0, 0, self.V_MIN), (180, self.S_MAX, 255))
        white = cv2.bitwise_and(white, table)
        # close small gaps (glare specks inside the ball) then open speckle
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
            circ = 4.0 * np.pi * area / (perim * perim)  # 1.0 == perfect circle
            if circ < self.MIN_CIRCULARITY:
                continue
            (cx, cy), rad = cv2.minEnclosingCircle(c)
            # score: prefer round + radius close to the expected ball radius
            radius_fit = 1.0 - min(1.0, abs(rad - r) / max(r, 1.0))
            cands.append((circ * 0.6 + radius_fit * 0.4, float(cx), float(cy), float(rad), float(circ)))

        cands.sort(key=lambda t: t[0], reverse=True)
        out: list[Detection] = []
        # The cue ball is singular; return the best few candidates (the tracker
        # settles on the persistent one). White → classify as CUE.
        for score, cx, cy, rad, _circ in cands[:3]:
            out.append(Detection(cx, cy, max(3.0, rad), (245, 245, 245),
                                 BallClass.CUE, float(min(0.99, 0.6 + 0.4 * score))))
        return out


STRATEGIES = [CueBallWhiteStrategy()]
