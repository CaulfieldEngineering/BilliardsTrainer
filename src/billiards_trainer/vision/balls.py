"""Ball appearance classification.

Once a detection exists, this turns its crop into a ball class (cue / solid /
stripe / eight) and a mean colour for the bird's-eye render. It is the only piece
of the old classical-detection module still in use — detection itself is now done
by the trained model (or the cue-ball heuristic) in ``detector_strategies``.

The standalone Hough/blob ``ClassicalBallDetector`` + the legacy ``make_detector``
backend factory were removed: a trained model beats classical CV on a real
overhead camera, and keeping two parallel detector stacks was the "detector vs
backend" confusion. ``detector_strategies.onnx_model`` is the single detector now.
"""

import cv2
import numpy as np

from .types import BallClass


def classify_ball(patch_bgr: np.ndarray, mask: np.ndarray | None = None) -> tuple[BallClass, tuple[int, int, int]]:
    """Classify a ball patch into cue/solid/stripe/eight and return its mean BGR.

    Heuristics on the circular interior:
      * mostly bright + desaturated         -> cue (white)
      * mostly very dark                    -> eight (black)
      * meaningful white fraction + colour  -> stripe
      * otherwise                           -> solid
    """
    if patch_bgr.size == 0:
        return BallClass.UNKNOWN, (200, 200, 200)
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    if mask is not None:
        sel = mask > 0
        if not np.any(sel):
            sel = np.ones(patch_bgr.shape[:2], bool)
    else:
        sel = np.ones(patch_bgr.shape[:2], bool)

    px = patch_bgr[sel].reshape(-1, 3)
    s = hsv[sel][:, 1].astype(np.float32)
    v = hsv[sel][:, 2].astype(np.float32)
    mean_bgr = tuple(int(c) for c in px.mean(axis=0))

    white_frac = float(np.mean((s < 55) & (v > 165)))
    dark_frac = float(np.mean(v < 65))
    med_v = float(np.median(v))
    med_s = float(np.median(s))

    if dark_frac > 0.55:
        return BallClass.EIGHT, mean_bgr
    if white_frac > 0.75:
        return BallClass.CUE, mean_bgr
    # Stripes show a white band alongside a saturated colour.
    if white_frac > 0.25 and med_s > 70:
        return BallClass.STRIPE, mean_bgr
    if med_s > 60 or med_v < 200:
        return BallClass.SOLID, mean_bgr
    return BallClass.CUE if med_v > 180 else BallClass.SOLID, mean_bgr
