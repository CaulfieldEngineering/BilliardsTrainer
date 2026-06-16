"""Baseline strategy: the CURRENT shipped detector (Hough+HSV on the rectified
bird's-eye), with its detections projected back to raw coords. This is the
Phase-0 reference row (recall ~2.6%) every variant is compared against. It is the
one strategy that detects on the *rectified* image — a known limitation kept
deliberately as the baseline; all other strategies detect on the raw frame.
"""

import cv2

from ..config import Settings
from ..vision.balls import ClassicalBallDetector
from . import DetectorStrategy, project_rect_dets_to_raw


class ClassicalRectifiedStrategy(DetectorStrategy):
    name = "classical_rectified"
    description = "Shipped Hough+HSV on the rectified view, projected to raw (Phase-0 baseline)."

    def __init__(self):
        self._balls = Settings().balls

    def detect(self, frame_bgr, calib):
        if calib is None or getattr(calib, "H", None) is None:
            return []
        rect = cv2.warpPerspective(frame_bgr, calib.H, tuple(calib.dst_size))
        det = ClassicalBallDetector(self._balls, calib.felt)
        dets = det.detect(rect, calib.rect_mask, calib.table)
        return project_rect_dets_to_raw(dets, calib)


STRATEGIES = [ClassicalRectifiedStrategy()]
