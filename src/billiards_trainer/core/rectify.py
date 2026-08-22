"""Perspective rectification to a bird's-eye (top-down) view.

Python port of the C++ ``Rectification`` module. The base step maps the four
felt corners to a forced 2:1 portrait rectangle. An optional refinement pass
detects rail edges (Hough) in the roughly-rectified image and composes a small
correction homography to "square up" the rectangle. Refinement is best-effort:
any failure leaves the base homography untouched.
"""

import cv2
import numpy as np

from .types import RectifyResult


def _line_intersection_pts(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    a1, b1, c1 = (y2 - y1), (x1 - x2), (x2 * y1 - x1 * y2)
    a2, b2, c2 = (y4 - y3), (x3 - x4), (x4 * y3 - x3 * y4)
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return (-1.0, -1.0)
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    return (float(x), float(y))


def _line_angle(line):
    dx = float(line[2] - line[0])
    dy = float(line[3] - line[1])
    return np.degrees(np.arctan2(abs(dy), abs(dx)))


def _refine_homography(rect_bgr, rect_mask, dst_size) -> np.ndarray:
    """Best-effort correction homography from rail-edge Hough lines.

    Returns identity if refinement cannot be confidently computed.
    """
    eye = np.eye(3, dtype=np.float64)
    try:
        band_width = 40
        dilated = cv2.dilate(rect_mask, None, iterations=band_width)
        eroded = cv2.erode(rect_mask, None, iterations=band_width)
        band = cv2.subtract(dilated, eroded)

        gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY) if rect_bgr.ndim == 3 else rect_bgr
        gray_masked = cv2.bitwise_and(gray, gray, mask=band)
        edges = cv2.Canny(gray_masked, 50, 150, apertureSize=3)
        edges = cv2.bitwise_and(edges, band)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=80,
                                minLineLength=100, maxLineGap=20)
        if lines is None or len(lines) < 4:
            return eye
        # OpenCV 4 returns (N, 1, 4); OpenCV 5 returns (N, 4) — normalise both
        lines = lines.reshape(-1, 4)

        horiz = [ln for ln in lines if _line_angle(ln) < 15.0]
        vert = [ln for ln in lines if _line_angle(ln) > 75.0]
        if len(horiz) < 2 or len(vert) < 2:
            return eye

        x, y, w, h = cv2.boundingRect(rect_mask)
        felt_top, felt_bottom = float(y), float(y + h)
        felt_left, felt_right = float(x), float(x + w)

        top_ys, bot_ys = [], []
        for ln in horiz:
            yc = (ln[1] + ln[3]) / 2.0
            if abs(yc - felt_top) < band_width * 2:
                top_ys.append(yc)
            elif abs(yc - felt_bottom) < band_width * 2:
                bot_ys.append(yc)
        left_xs, right_xs = [], []
        for ln in vert:
            xc = (ln[0] + ln[2]) / 2.0
            if abs(xc - felt_left) < band_width * 2:
                left_xs.append(xc)
            elif abs(xc - felt_right) < band_width * 2:
                right_xs.append(xc)
        if not (top_ys and bot_ys and left_xs and right_xs):
            return eye

        top_y = sorted(top_ys)[len(top_ys) // 2]
        bot_y = sorted(bot_ys)[len(bot_ys) // 2]
        left_x = sorted(left_xs)[len(left_xs) // 2]
        right_x = sorted(right_xs)[len(right_xs) // 2]

        def closest(cands, key_fn, target):
            return min(cands, key=lambda ln: abs(key_fn(ln) - target))

        top_l = closest(horiz, lambda ln: (ln[1] + ln[3]) / 2.0, top_y)
        bot_l = closest(horiz, lambda ln: (ln[1] + ln[3]) / 2.0, bot_y)
        left_l = closest(vert, lambda ln: (ln[0] + ln[2]) / 2.0, left_x)
        right_l = closest(vert, lambda ln: (ln[0] + ln[2]) / 2.0, right_x)

        tl = _line_intersection_pts(top_l, left_l)
        tr = _line_intersection_pts(top_l, right_l)
        br = _line_intersection_pts(bot_l, right_l)
        bl = _line_intersection_pts(bot_l, left_l)
        quad = np.array([tl, tr, br, bl], dtype=np.float32)

        dw, dh = dst_size
        if np.any(quad < 0) or np.any(quad[:, 0] >= dw) or np.any(quad[:, 1] >= dh):
            return eye
        if not np.all(np.isfinite(quad)):
            return eye

        min_x, max_x = quad[:, 0].min(), quad[:, 0].max()
        min_y, max_y = quad[:, 1].min(), quad[:, 1].max()
        det_w, det_h = max_x - min_x, max_y - min_y
        if det_w <= 0 or det_h <= 0:
            return eye

        aspect_err = abs((det_w / det_h) - 2.0) / 2.0
        if aspect_err > 0.20:
            return eye

        avg = (det_w + det_h) / 2.0
        tgt_h = avg
        tgt_w = tgt_h * 2.0
        if tgt_w > det_w * 1.5:
            tgt_w = det_w * 1.5
            tgt_h = tgt_w / 2.0
        if tgt_h > det_h * 1.5:
            tgt_h = det_h * 1.5
            tgt_w = tgt_h * 2.0

        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        hw, hh = tgt_w / 2.0, tgt_h / 2.0
        margin = 10.0
        cx = max(hw + margin, min(dw - hw - margin, cx))
        cy = max(hh + margin, min(dh - hh - margin, cy))
        target = np.array([
            [cx - hw, cy - hh], [cx + hw, cy - hh],
            [cx + hw, cy + hh], [cx - hw, cy + hh],
        ], dtype=np.float32)

        h_corr = cv2.getPerspectiveTransform(quad, target)
        det_c = h_corr[0, 0] * h_corr[1, 1] - h_corr[0, 1] * h_corr[1, 0]
        if abs(det_c) < 1e-6:
            return eye
        return h_corr
    except cv2.error:
        return eye


def rectify_tabletop(bgr: np.ndarray, felt_mask: np.ndarray, corners: np.ndarray,
                     pad_px: int = 40, aspect: float = 2.0,
                     refine: bool = True) -> RectifyResult:
    """Warp the table to a top-down rectangle of forced ``aspect`` (long:short)."""
    result = RectifyResult()
    if bgr is None or felt_mask is None or corners is None:
        return result
    corners = np.asarray(corners, dtype=np.float32)
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)) or pad_px < 0:
        return result
    result.src_quad = corners

    tl, tr, br, bl = corners
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)
    natural_w = max(w_top, w_bot)
    natural_h = max(h_left, h_right)

    # Force portrait: the long dimension becomes height, width = height / aspect.
    dst_h = max(natural_w, natural_h)
    dst_w = dst_h / aspect

    base_w = int(round(dst_w)) + 2 * pad_px
    base_h = int(round(dst_h)) + 2 * pad_px
    result.dst_size = (base_w, base_h)

    p = float(pad_px)
    dst_quad = np.array([
        [p, p], [base_w - 1 - p, p],
        [base_w - 1 - p, base_h - 1 - p], [p, base_h - 1 - p],
    ], dtype=np.float32)
    result.dst_quad = dst_quad

    H = cv2.getPerspectiveTransform(corners, dst_quad)
    if H is None or H.shape != (3, 3):
        return result
    if abs(H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]) < 1e-6:
        return result

    rect_bgr = cv2.warpPerspective(bgr, H, result.dst_size, flags=cv2.INTER_LINEAR)
    rect_mask = cv2.warpPerspective(felt_mask, H, result.dst_size, flags=cv2.INTER_NEAREST)

    h_corr = np.eye(3, dtype=np.float64)
    if refine and rect_bgr is not None and rect_mask is not None:
        h_corr = _refine_homography(rect_bgr, rect_mask, result.dst_size)

    H_refined = h_corr @ H
    result.rectified_bgr = cv2.warpPerspective(bgr, H_refined, result.dst_size,
                                               flags=cv2.INTER_LINEAR)
    result.rectified_mask = cv2.warpPerspective(felt_mask, H_refined, result.dst_size,
                                                flags=cv2.INTER_NEAREST)
    result.H = H_refined
    try:
        result.Hinv = np.linalg.inv(H_refined)
    except np.linalg.LinAlgError:
        return result

    if result.rectified_bgr is None or result.rectified_mask is None:
        return result
    result.ok = True
    return result


def project_points(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply a homography to an (N,2) array of points -> (N,2)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)


def estimate_camera_position(Hinv: np.ndarray,
                             image_size: tuple[int, int]) -> np.ndarray | None:
    """Camera centre in RECTIFIED-plane coordinates (x, y, z; z = height above
    the cloth, all in rect px) from the rect→image homography.

    Why: the homography maps the CLOTH plane, but a ball's centre sits one ball
    radius ABOVE it, so an oblique camera projects every ball centre radially
    outward from itself — up to a couple of ball diameters at the far rail,
    which is exactly the "balls sitting in the rail" artefact on the overhead.
    Knowing where the camera is lets the pipeline slide each projected point
    back along the camera ray to the true contact point.

    Standard plane-pose decomposition with assumed intrinsics (principal point
    at the image centre, square pixels): Hinv ∝ K·[r1 r2 t], the focal length
    solved from r1⊥r2 and |r1|=|r2|, camera centre C = -Rᵀt. Returns None when
    the decomposition is degenerate or lands somewhere physically absurd —
    callers must treat correction as best-effort.
    """
    G = np.asarray(Hinv, dtype=np.float64)
    if G.shape != (3, 3) or not np.all(np.isfinite(G)):
        return None
    w, h = float(image_size[0]), float(image_size[1])
    cx, cy = w / 2.0, h / 2.0
    # remove the principal point: rows of K⁻¹·G with f still unknown
    u = G[0] - cx * G[2]
    v = G[1] - cy * G[2]
    wv = G[2]
    # focal from the two rotation-column constraints (least-squares-ish: use
    # both estimates when available, geometric mean)
    f_sq = []
    denom = wv[0] * wv[1]
    if abs(denom) > 1e-12:
        est = -(u[0] * u[1] + v[0] * v[1]) / denom
        if est > 0:
            f_sq.append(est)
    denom = wv[1] ** 2 - wv[0] ** 2
    if abs(denom) > 1e-12:
        est = ((u[0] ** 2 + v[0] ** 2) - (u[1] ** 2 + v[1] ** 2)) / denom
        if est > 0:
            f_sq.append(est)
    if not f_sq:
        return None
    f = float(np.sqrt(np.exp(np.mean(np.log(f_sq)))))
    if not (0.2 * w <= f <= 20.0 * w):        # nonsense focal -> bail
        return None
    B = np.vstack([u / f, v / f, wv])          # K⁻¹·G, columns ∝ [r1 r2 t]
    n1, n2 = np.linalg.norm(B[:, 0]), np.linalg.norm(B[:, 1])
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    s = 2.0 / (n1 + n2)
    if B[2, 2] * s < 0:                        # camera must be in front of the plane
        s = -s
    r1, r2, t = s * B[:, 0], s * B[:, 1], s * B[:, 2]
    r3 = np.cross(r1, r2)
    R = np.column_stack([r1, r2, r3])
    C = -R.T @ t                               # camera centre, rect-plane coords
    if not np.all(np.isfinite(C)) or abs(C[2]) < 1e-6:
        return None
    # The rect frame is y-DOWN (image convention), so its right-handed z axis
    # points INTO the table — a camera physically above the cloth decomposes to
    # NEGATIVE z (and the planar two-fold ambiguity can flip it again). The
    # ground point C[:2] is identical for both mirror solutions and the ball is
    # always on the camera's side of the plane, so report height as |z|.
    C[2] = abs(C[2])
    return C


def visual_from_plane(x: float, y: float, cam, r_ball: float
                      ) -> tuple[float, float]:
    """Invert the ball-height parallax correction for DISPLAY.

    Stored track positions are the felt-CONTACT point (physics space);
    imagery shows the ball's DISC, displaced radially away from the
    camera by the ball's height. Any overlay drawn on video/rect frames
    must map ball positions through this or the graphic sits visibly
    off the ball (Joe's screenshot, first overlay render: cyan ring
    low-left of the cue ball)."""
    shrink = max(1e-6, 1.0 - float(r_ball) / float(cam[2]))
    return (float(cam[0]) + (x - float(cam[0])) / shrink,
            float(cam[1]) + (y - float(cam[1])) / shrink)
