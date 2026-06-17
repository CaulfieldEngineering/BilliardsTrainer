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


# --------------------------------------------------------------------------- #
# Pool-ball identification: cue + 1..15 with canonical colours
# --------------------------------------------------------------------------- #
# Canonical colours (BGR) for a clean overhead render — the real ball colours,
# not the muddy per-crop mean. Stripes 9..15 reuse the 1..7 hues.
_SOLID_BGR = {
    1: (40, 200, 235),   # yellow
    2: (200, 75, 25),    # blue
    3: (40, 40, 210),    # red
    4: (110, 35, 95),    # purple
    5: (20, 110, 240),   # orange
    6: (55, 150, 55),    # green
    7: (45, 45, 120),    # maroon
    8: (25, 25, 25),     # black
}
# Reference hue (OpenCV H, 0..180) for the non-red solids (red/maroon wrap at 0/180
# and are handled separately by darkness).
_HUE_REF = ((1, 27.0), (5, 14.0), (2, 110.0), (6, 68.0), (4, 145.0))


def pool_ball_bgr(number: int) -> tuple[int, int, int]:
    """Canonical BGR for a ball number (0 = cue)."""
    if number <= 0:
        return (245, 245, 245)
    base = number if number <= 8 else number - 8
    return _SOLID_BGR.get(base, (200, 200, 200))


def _hue_to_base(hue: float, val: float) -> int:
    if hue <= 10.0 or hue >= 168.0:          # red family (wraps)
        return 7 if val < 110 else 3         # darker => maroon(7), else red(3)
    best, best_d = 3, 1e9
    for num, ref in _HUE_REF:
        d = abs(hue - ref)
        if d < best_d:
            best_d, best = d, num
    return best


def classify_pool_ball(patch_bgr: np.ndarray, mask: np.ndarray | None = None
                       ) -> tuple[BallClass, int, tuple[int, int, int]]:
    """Identify a ball crop as cue / 1..15 and return (class, number, canonical BGR).

    number: 0 = cue, 1..8 solids (8 = black), 9..15 stripes, -1 = unknown.

    The cue is separated from the 9-ball (white body + yellow stripe) by the
    fraction of clearly-coloured pixels: the cue has ~none, the 9 has a saturated
    band. A tight inner-circle sample avoids neighbour/felt contamination so balls
    in a cluster (rack) still classify.
    """
    if patch_bgr is None or patch_bgr.size == 0:
        return BallClass.UNKNOWN, -1, (200, 200, 200)
    h, w = patch_bgr.shape[:2]
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    if mask is not None and np.any(mask > 0):
        sel = mask > 0
    else:
        yy, xx = np.ogrid[:h, :w]
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        rr = 0.62 * min(h, w) / 2.0          # inner 62% — dodge edges/neighbours
        sel = (xx - cx) ** 2 + (yy - cy) ** 2 <= rr * rr
    if not np.any(sel):
        return BallClass.UNKNOWN, -1, (200, 200, 200)
    s = hsv[:, :, 1][sel].astype(np.float32)
    v = hsv[:, :, 2][sel].astype(np.float32)
    hh = hsv[:, :, 0][sel].astype(np.float32)

    white_frac = float(np.mean((s < 60) & (v > 150)))
    dark_frac = float(np.mean(v < 55))
    colored = (s > 90) & (v > 60)
    colored_frac = float(np.mean(colored))

    # CUE: essentially no saturated colour and bright. (The 9-ball's yellow band
    # pushes colored_frac well past this, so it lands as a stripe, not a 2nd cue.)
    if colored_frac < 0.10 and white_frac > 0.40:
        return BallClass.CUE, 0, (245, 245, 245)
    # 8-BALL: mostly dark with little colour.
    if dark_frac > 0.45 and colored_frac < 0.30:
        return BallClass.EIGHT, 8, _SOLID_BGR[8]
    if colored_frac < 0.05:
        return BallClass.UNKNOWN, -1, (190, 190, 190)

    ch = hh[colored]
    cvv = v[colored]
    near0 = float(np.mean(ch < 15)) + float(np.mean(ch > 165))
    if near0 > 0.5:                          # red region straddles the hue wrap
        base = 7 if float(np.median(cvv)) < 110 else 3
    else:
        base = _hue_to_base(float(np.median(ch)), float(np.median(cvv)))
    is_stripe = white_frac > 0.22 and base != 8
    number = base + 8 if is_stripe else base
    cls = BallClass.STRIPE if is_stripe else BallClass.SOLID
    return cls, number, pool_ball_bgr(number)
