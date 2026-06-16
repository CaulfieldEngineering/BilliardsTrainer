"""SimpleBlobDetector strategy (per Danthewaann/snooker-ball-tracker) — RAW frame.

Instead of HoughCircles, use cv2.SimpleBlobDetector with circularity / inertia /
convexity / area filters on the non-felt regions inside the table. A more robust
classical alternative to Hough for round-blob detection.
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


class SimpleBlobStrategy(DetectorStrategy):
    name = "simple_blob"
    description = "SimpleBlobDetector (circularity/inertia/area) on non-felt blobs within the table (raw frame)."

    def detect(self, frame_bgr, calib):
        table = table_polygon_mask(frame_bgr.shape, calib)
        felt = felt_mask_raw(frame_bgr, calib)
        r = ball_radius_raw(calib, frame_bgr.shape)
        # balls = non-felt inside the table; clean up speckle
        nonfelt = cv2.bitwise_and(cv2.bitwise_not(felt), table)
        nonfelt = cv2.morphologyEx(nonfelt, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        p = cv2.SimpleBlobDetector_Params()
        p.filterByColor = True
        p.blobColor = 255           # white blobs (the non-felt mask)
        p.minThreshold, p.maxThreshold, p.thresholdStep = 50, 220, 30
        p.filterByArea = True
        p.minArea = float(np.pi * (0.45 * r) ** 2)
        p.maxArea = float(np.pi * (2.4 * r) ** 2)
        p.filterByCircularity = True
        p.minCircularity = 0.55
        p.filterByInertia = True
        p.minInertiaRatio = 0.3
        p.filterByConvexity = True
        p.minConvexity = 0.7
        detector = cv2.SimpleBlobDetector_create(p)
        kps = detector.detect(nonfelt)

        out: list[Detection] = []
        for kp in kps:
            cx, cy = kp.pt
            rad = max(3.0, kp.size / 2.0)
            cls, bgr = classify_crop(frame_bgr, cx, cy, rad)
            out.append(Detection(float(cx), float(cy), float(rad), bgr, cls, 0.85))
        return out


STRATEGIES = [SimpleBlobStrategy()]
