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


def number_to_class(number: int) -> BallClass:
    """Map a ball number to its class. 0=cue, 8=eight, 1..7=solid, 9..15=stripe."""
    if number == 0:
        return BallClass.CUE
    if number == 8:
        return BallClass.EIGHT
    if 1 <= number <= 7:
        return BallClass.SOLID
    if 9 <= number <= 15:
        return BallClass.STRIPE
    return BallClass.UNKNOWN


def _hue_to_base(hue: float, val: float) -> int:
    if hue <= 10.0 or hue >= 168.0:          # red family (wraps)
        return 7 if val < 110 else 3         # darker => maroon(7), else red(3)
    best, best_d = 3, 1e9
    for num, ref in _HUE_REF:
        d = abs(hue - ref)
        if d < best_d:
            best_d, best = d, num
    return best


def _inner_disc(patch_bgr, mask):
    """Boolean selection of the ball's inner 62% — dodges felt and neighbours."""
    h, w = patch_bgr.shape[:2]
    if mask is not None and np.any(mask > 0):
        return mask > 0
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rr = 0.62 * min(h, w) / 2.0
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= rr * rr


def stripe_reading(patch_bgr: np.ndarray, mask: np.ndarray | None = None,
                   solid_below: float = 0.20, stripe_above: float = 0.45):
    """Is this crop a STRIPE (True), a SOLID (False), or too close to call (None)?

    Deliberately abstains in the middle. The 16-class model gets a ball's HUE
    right and its stripe bit wrong (measured on session-20260729: the purple 4
    read as the 12, the yellow 9 read as the 1 — one error in each direction),
    and stripe == solid + 8, so a confident pixel reading repairs the number by
    +/-8. Abstaining between the thresholds means an ambiguous crop never
    OVERWRITES a model answer that may well be right.

    Top-down, a stripe shows a lot of white — the poles are white and the
    colour band only crosses the middle — while a solid is saturated except for
    its small number circle and a glare spot.
    """
    if patch_bgr is None or patch_bgr.size == 0:
        return None
    sel = _inner_disc(patch_bgr, mask)
    if not np.any(sel):
        return None
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1][sel].astype(np.float32)
    v = hsv[:, :, 2][sel].astype(np.float32)
    white_frac = float(np.mean((s < 60) & (v > 150)))
    if white_frac < solid_below:
        return False
    if white_frac > stripe_above:
        return True
    return None


def classify_pool_ball(patch_bgr: np.ndarray, mask: np.ndarray | None = None,
                       felt_hsv: tuple[int, int, int] | None = None
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
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    sel = _inner_disc(patch_bgr, mask)   # inner 62% — dodge edges/neighbours
    if not np.any(sel):
        return BallClass.UNKNOWN, -1, (200, 200, 200)
    s = hsv[:, :, 1][sel].astype(np.float32)
    v = hsv[:, :, 2][sel].astype(np.float32)
    hh = hsv[:, :, 0][sel].astype(np.float32)

    white_frac = float(np.mean((s < 60) & (v > 150)))
    dark_frac = float(np.mean(v < 55))
    colored = (s > 90) & (v > 60)
    # Felt subtraction: on a BLUE table the felt itself is a legal ball colour,
    # so any felt in the crop votes "blue ball" and everything drifts toward
    # 2/10. Pixels matching the calibrated felt hue are background, not ball —
    # drop them from the colour vote. A REAL blue ball is then recovered below
    # by its gloss: it matches the felt hue but not the felt brightness.
    felt_like = None
    if felt_hsv is not None:
        fh = float(felt_hsv[0])
        dh = np.abs(hh - fh)
        dh = np.minimum(dh, 180.0 - dh)
        felt_like = (dh < 14) & (s > 60)
        colored = colored & ~felt_like
    colored_frac = float(np.mean(colored))

    # CUE: essentially no saturated colour and bright. (The 9-ball's yellow band
    # pushes colored_frac well past this, so it lands as a stripe, not a 2nd cue.)
    if colored_frac < 0.10 and white_frac > 0.40:
        return BallClass.CUE, 0, (245, 245, 245)
    # 8-BALL: mostly dark AND near-colourless. The maroon 7 is also dark but carries
    # a real red — so it must have little colour to read as the 8, else it falls
    # through to the hue path and lands as maroon (7). (8 vs 7 is a known hard pair.)
    if dark_frac > 0.5 and colored_frac < 0.15:
        return BallClass.EIGHT, 8, _SOLID_BGR[8]
    if colored_frac < 0.05:
        # Nothing colourful besides (possibly) felt-hue pixels. A genuine blue
        # ball on blue felt also lands here — same hue as the cloth — but so do
        # shadowed felt patches, and on a bright felt (V pegged at 255) gloss
        # and shadow are indistinguishable by brightness. UNKNOWN ('?') is the
        # honest answer; blue-on-blue identity is the 16-class model's job.
        return BallClass.UNKNOWN, -1, (190, 190, 190)

    ch = hh[colored]
    cvv = v[colored]
    near0 = float(np.mean(ch < 15)) + float(np.mean(ch > 165))
    if near0 > 0.5:                          # red region straddles the hue wrap
        base = 7 if float(np.median(cvv)) < 110 else 3
    else:
        base = _hue_to_base(float(np.median(ch)), float(np.median(cvv)))
        # On BLUE felt, true-blue ball pixels were excluded with the felt above,
        # so a surviving blue-family vote is really the purple 4/12 (measured
        # hue ~112-125 on this camera — right at the blue reference).
        if base == 2 and felt_hsv is not None and abs(float(felt_hsv[0]) - 110.0) < 20.0:
            base = 4
    # Stripe (9..15) vs solid (1..7): a stripe's white band is a LARGE fraction of
    # the ball; a solid's glare is a small spot. Bias toward solid (raise the bar)
    # so glare doesn't turn the 1 into a 9. (solid/stripe is a known hard pair.)
    is_stripe = white_frac > 0.30 and base != 8
    number = base + 8 if is_stripe else base
    cls = BallClass.STRIPE if is_stripe else BallClass.SOLID
    return cls, number, pool_ball_bgr(number)
