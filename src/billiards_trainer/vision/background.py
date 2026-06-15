"""Background subtraction (MOG2) + optical-flow magnitude.

These are the two extra "modalities" the evidence fusion uses alongside the
pixel-change motion energy:

* ``BackgroundModel`` learns the static table over the first seconds and reports
  the *foreground fraction* of the playing area — how much differs from the
  learned baseline. Robust to gradual lighting drift (it adapts), so a global
  brightness flicker doesn't read as activity.

* ``flow_activity`` runs Farneback dense optical flow on a downscaled ROI and
  returns the fraction of pixels with coherent displacement — the signal that
  separates a real moving ball (coherent translation) from a flickering specular
  highlight (brightness change, no coherent motion).

Both operate on the rectified playing-area ROI and are cheap (downscaled).
"""

import cv2
import numpy as np


class BackgroundModel:
    def __init__(self, history: int = 400, var_threshold: float = 30.0):
        self._mog = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False)
        self._frames = 0
        self._warmup = 60  # ~2 s @ 30 fps before the model is trustworthy

    @property
    def ready(self) -> bool:
        return self._frames >= self._warmup

    def reset(self) -> None:
        self._mog = cv2.createBackgroundSubtractorMOG2(
            history=400, varThreshold=30.0, detectShadows=False)
        self._frames = 0

    def update(self, small_gray: np.ndarray) -> float:
        """Feed the (downscaled) playing-area ROI; return foreground fraction."""
        fg = self._mog.apply(small_gray)
        self._frames += 1
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        if not self.ready:
            return 0.0
        return float((fg > 0).mean())


def downscale(gray: np.ndarray, max_w: int = 160) -> np.ndarray:
    h, w = gray.shape[:2]
    if w <= max_w:
        return gray
    scale = max_w / w
    return cv2.resize(gray, (max_w, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def flow_activity(prev_small: np.ndarray | None, small: np.ndarray) -> float:
    """Fraction of (downscaled) ROI pixels with coherent optical-flow displacement.

    Near-zero for incoherent brightness flicker; meaningful for a translating
    ball."""
    if prev_small is None or prev_small.shape != small.shape:
        return 0.0
    flow = cv2.calcOpticalFlowFarneback(
        prev_small, small, None, pyr_scale=0.5, levels=2, winsize=11,
        iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
    mag = np.hypot(flow[..., 0], flow[..., 1])
    return float((mag > 1.0).mean())
