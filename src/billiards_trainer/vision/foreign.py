"""Foreign-object (arm/hand/cue-carried-ball) presence on the playing surface.

Why: Joe's sessions are drill-heavy — shoot, walk, gather balls, replace,
shoot again. Balls carried by hand travel table lengths, trip the motion
gates and "vanish" when picked up, so the shot detector logged
rearrangement as shots (a hand-placed pair was scored as a SCRATCH in the
first audio-validated baseline). The discriminator between shooting and
rearranging is dwell: a stroke puts the arm over the felt for a moment and
withdraws while the balls roll; rearranging keeps it there for the whole
"shot". This module supplies the signal: what fraction of the bed is
covered by something that is neither felt nor ball.

Method: downscaled rectified frame, distance from the calibrated cloth
colour in HSV, morphological cleanup, and a size floor that ignores
ball-sized blobs (balls are foreign to the felt too, but they are the
game). Deliberately coarse — it answers "is an arm over the table", not
"where exactly is the wrist"."""

import cv2
import numpy as np

#: Downscale target for the mask computation; coarse is the point.
_W = 160

#: A blob must exceed this fraction of the bed to count as foreign — a ball
#: at the downscaled size is ~0.15%, a HAND placing a ball ~0.7-1.5%, a
#: reaching arm 2-6% (measured on session-20260812: the first floor of 1.2%
#: made the placing hand invisible, which was the whole point of the signal).
_MIN_BLOB_FRAC = 0.004


def foreign_mask(rect_bgr: np.ndarray,
                 cloth_hsv: tuple[float, float, float] | None
                 ) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """(covered fraction, blob mask, raw non-felt mask).

    ``cloth_hsv`` is the calibrated cloth colour (median H, S, V); without it
    the estimate falls back to the frame's own median, which is the cloth on
    any frame that isn't mostly covered already. Masks are in the input's
    own (downscaled) geometry — callers map coordinates by scale.

    The blob mask keeps only hand/arm-scale regions (the size floor exists
    so lone balls never read as "arm over the table"). The RAW mask is the
    same non-felt test BEFORE the floor: it answers "is there anything at
    this spot at all" — ball, stick, hand — which is a different question
    with a different customer (spot-occupancy in vacancy pruning).
    """
    if rect_bgr is None or rect_bgr.size == 0:
        return 0.0, None, None
    h = max(1, int(rect_bgr.shape[0] * _W / max(1, rect_bgr.shape[1])))
    small = cv2.resize(rect_bgr, (_W, h), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.int16)
    if cloth_hsv is None:
        ch = float(np.median(hsv[..., 0]))
        cs = float(np.median(hsv[..., 1]))
        cv_ = float(np.median(hsv[..., 2]))
    else:
        ch, cs, cv_ = cloth_hsv
    # Hue is circular; distance wraps at 180 (OpenCV hue range).
    dh = np.abs(hsv[..., 0] - ch)
    dh = np.minimum(dh, 180 - dh)
    ds = np.abs(hsv[..., 1] - cs)
    dv = np.abs(hsv[..., 2] - cv_)
    # Weighted colour distance: hue dominates (skin/wood/cloth all separate
    # by hue), saturation next, value least (shadows are still felt).
    dist = 2.0 * dh + 0.8 * ds + 0.4 * dv
    mask = (dist > 90).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    total = mask.size
    floor = _MIN_BLOB_FRAC * total
    keep = np.zeros_like(mask)
    covered = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= floor:
            keep[labels == i] = 1
            covered += area
    return covered / total, keep, mask


def foreign_fraction(rect_bgr: np.ndarray,
                     cloth_hsv: tuple[float, float, float] | None) -> float:
    """Fraction only — see :func:`foreign_mask`."""
    return foreign_mask(rect_bgr, cloth_hsv)[0]
