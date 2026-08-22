"""Felt / table-surface detection.

A faithful Python port of the C++ ``FeltDetection`` module (the strongest part
of the original prototype). Algorithm, unchanged in spirit:

    HSV inRange (with hue-wrap) -> morphology close/open -> keep largest
    connected component -> fill holes -> scan the 4 extreme edges -> reject
    pocket-curve outliers by median filtering -> robust Huber line fit per
    edge -> intersect adjacent lines -> 4 ordered corners (TL, TR, BR, BL).

The per-column / per-row edge scans that were ``O(W*H)`` nested loops in C++ are
vectorised here with ``np.argmax``, which is both the numpy idiom and far faster.
"""


import cv2
import numpy as np

from ..config import FeltSettings
from ..core.types import FeltResult


def _build_mask(bgr: np.ndarray, felt: FeltSettings) -> np.ndarray:
    """HSV threshold (handling hue wrap) + morphological cleanup -> binary mask."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo_s, lo_v = felt.s_min, felt.v_min
    hi_s, hi_v = felt.s_max, felt.v_max

    if felt.h_min <= felt.h_max:
        mask = cv2.inRange(
            hsv,
            np.array([felt.h_min, lo_s, lo_v], np.uint8),
            np.array([felt.h_max, hi_s, hi_v], np.uint8),
        )
    else:
        # Wrapped hue range: [0..h_max] U [h_min..180]
        a = cv2.inRange(hsv, np.array([0, lo_s, lo_v], np.uint8),
                        np.array([felt.h_max, hi_s, hi_v], np.uint8))
        b = cv2.inRange(hsv, np.array([felt.h_min, lo_s, lo_v], np.uint8),
                        np.array([180, hi_s, hi_v], np.uint8))
        mask = cv2.bitwise_or(a, b)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    # index 0 is background; pick the largest of the rest
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask)
    out[labels == largest] = 255
    return out


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Flood-fill hole filling: fill the outside of the inverted mask, then any
    remaining holes are interior regions not connected to the border."""
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    inv = cv2.bitwise_not(mask)
    ff = inv.copy()
    h, w = mask.shape[:2]
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, ffmask, (0, 0), 255)
    holes = cv2.bitwise_and(inv, cv2.bitwise_not(ff))
    return cv2.bitwise_or(mask, holes)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL (sort by y, then x within pairs)."""
    pts = pts[np.argsort(pts[:, 1])]  # by y
    top, bottom = pts[:2], pts[2:]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _filter_outliers(values: np.ndarray, coords: np.ndarray, keep_larger: bool,
                     threshold: int = 15) -> np.ndarray:
    """Reject pocket-curve points: keep coords whose scan value stays near the
    median edge position (pockets pull the edge ~10-30 px inward)."""
    if values.size == 0:
        return coords
    median = int(np.median(values))
    if keep_larger:
        keep = values >= median - threshold
    else:
        keep = values <= median + threshold
    return coords[keep]


def _fit_line(points: np.ndarray) -> np.ndarray | None:
    if points.shape[0] < 10:
        return None
    line = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
    return line.reshape(-1)  # [vx, vy, x0, y0]


def _line_intersection(l1: np.ndarray, l2: np.ndarray) -> tuple[float, float]:
    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2
    det = vx1 * vy2 - vy1 * vx2
    if abs(det) < 1e-6:
        return (float((x1 + x2) / 2), float((y1 + y2) / 2))
    dx, dy = x2 - x1, y2 - y1
    t = (dx * vy2 - dy * vx2) / det
    return (float(x1 + t * vx1), float(y1 + t * vy1))


def detect_felt(bgr: np.ndarray, felt: FeltSettings) -> FeltResult:
    """Detect the felt surface and return a cleaned mask + 4 ordered corners."""
    result = FeltResult()
    if bgr is None or bgr.size == 0:
        return result

    mask = _build_mask(bgr, felt)
    mask = _keep_largest_component(mask)
    mask = _fill_holes(mask)
    result.mask = mask

    white = int(cv2.countNonZero(mask))
    area = mask.shape[0] * mask.shape[1]
    result.area_ratio = white / area if area else 0.0
    if result.area_ratio > 0.95 or white == 0:
        result.mask = np.zeros_like(mask)
        return result

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return result
    result.contour = max(contours, key=cv2.contourArea)
    result.ok = True

    # --- vectorised extreme-edge scans -------------------------------------
    binary = mask > 0
    h, w = binary.shape
    cols_have = binary.any(axis=0)        # which columns contain felt
    rows_have = binary.any(axis=1)        # which rows contain felt
    xs = np.arange(w)
    ys = np.arange(h)

    top_y = np.argmax(binary, axis=0)                    # first white row per col
    bottom_y = (h - 1) - np.argmax(binary[::-1], axis=0)  # last white row per col
    left_x = np.argmax(binary, axis=1)                   # first white col per row
    right_x = (w - 1) - np.argmax(binary[:, ::-1], axis=1)  # last white col per row

    top_pts = np.column_stack([xs[cols_have], top_y[cols_have]])
    bottom_pts = np.column_stack([xs[cols_have], bottom_y[cols_have]])
    left_pts = np.column_stack([left_x[rows_have], ys[rows_have]])
    right_pts = np.column_stack([right_x[rows_have], ys[rows_have]])

    top_pts = _filter_outliers(top_pts[:, 1], top_pts, keep_larger=False)
    bottom_pts = _filter_outliers(bottom_pts[:, 1], bottom_pts, keep_larger=True)
    left_pts = _filter_outliers(left_pts[:, 0], left_pts, keep_larger=False)
    right_pts = _filter_outliers(right_pts[:, 0], right_pts, keep_larger=True)

    if min(len(top_pts), len(bottom_pts), len(left_pts), len(right_pts)) < 10:
        return result  # ok=True but no corners

    top_l = _fit_line(top_pts)
    bottom_l = _fit_line(bottom_pts)
    left_l = _fit_line(left_pts)
    right_l = _fit_line(right_pts)
    if any(v is None for v in (top_l, bottom_l, left_l, right_l)):
        return result

    tl = _line_intersection(top_l, left_l)
    tr = _line_intersection(top_l, right_l)
    br = _line_intersection(bottom_l, right_l)
    bl = _line_intersection(bottom_l, left_l)

    result.corners = _order_corners(np.array([tl, tr, br, bl], dtype=np.float32))
    result.has_corners = True
    return result


def estimate_felt_settings(frame: np.ndarray, base: FeltSettings) -> FeltSettings:
    """Auto-estimate the felt colour from the frame centre.

    For a centred/overhead table the middle of the frame is reliably felt, so the
    dominant saturated hue there gives a good colour key without any manual
    tuning. Used as a calibration fallback when the configured range matches too
    little of the frame (different felt shade / lighting than the defaults).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]
    region = hsv[int(h * 0.30):int(h * 0.70), int(w * 0.30):int(w * 0.70)].reshape(-1, 3)
    sat = region[(region[:, 1] > 40) & (region[:, 2] > 40)]
    if len(sat) < 50:
        return base
    peak = int(np.argmax(np.bincount(sat[:, 0], minlength=180)))
    near = sat[np.abs(sat[:, 0].astype(int) - peak) <= 12]
    s_med = int(np.median(near[:, 1]))
    v_med = int(np.median(near[:, 2]))
    return FeltSettings(
        h_min=(peak - 18) % 180, h_max=(peak + 18) % 180,
        s_min=max(0, s_med - 90), s_max=255,
        v_min=max(0, v_med - 120), v_max=255,
        sensitivity=base.sensitivity, picked_hsv=[peak, s_med, v_med],
    )


def felt_from_point(frame: np.ndarray, x: int, y: int, sensitivity: int = 82,
                    patch: int = 12) -> FeltSettings:
    """Build felt settings by sampling the colour the user clicked on.

    Samples a small patch around (x, y) on the original frame, takes the median
    HSV, and expands it into a range by ``sensitivity``. This is the click-to-pick
    tuning helper for when the auto-estimate or defaults don't match a table.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    x0, x1 = max(0, x - patch), min(w, x + patch + 1)
    y0, y1 = max(0, y - patch), min(h, y + patch + 1)
    region = hsv[y0:y1, x0:x1].reshape(-1, 3)
    med = np.median(region, axis=0)
    picked = (int(med[0]), int(med[1]), int(med[2]))
    rng = derive_hsv_range(picked, sensitivity)
    return FeltSettings(
        h_min=rng["h_min"], h_max=rng["h_max"],
        s_min=rng["s_min"], s_max=rng["s_max"],
        v_min=rng["v_min"], v_max=rng["v_max"],
        sensitivity=sensitivity, picked_hsv=list(picked),
    )


def derive_hsv_range(picked_hsv: tuple[int, int, int], sensitivity: int) -> dict:
    """Translate a sampled felt colour + a 0..100 sensitivity into an HSV range.

    Mirrors the original UI concept: higher sensitivity widens the accepted band.
    """
    h, s, v = picked_hsv
    sens = max(0, min(100, sensitivity)) / 100.0
    h_span = int(8 + sens * 22)     # +/- 8..30 hue
    s_span = int(40 + sens * 120)
    v_span = int(40 + sens * 120)
    return {
        "h_min": (h - h_span) % 180,
        "h_max": (h + h_span) % 180,
        "s_min": max(0, s - s_span),
        "s_max": min(255, s + s_span),
        "v_min": max(0, v - v_span),
        "v_max": min(255, v + v_span),
    }
