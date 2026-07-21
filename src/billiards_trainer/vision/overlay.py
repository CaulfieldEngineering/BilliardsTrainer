"""Overlay rendering (OpenCV) for the live and bird's-eye views.

Drawing the annotations into the numpy frame keeps the Qt widgets dumb (just
blit a QImage) and keeps the live + rectified overlays visually consistent.
"""

import cv2
import numpy as np

from .geometry import TableModel
from .rectify import project_points
from .types import BallClass, Track

_CLS_COLOR = {
    BallClass.CUE: (255, 255, 255),
    BallClass.EIGHT: (40, 40, 40),
    BallClass.SOLID: (60, 200, 255),
    BallClass.STRIPE: (80, 255, 180),
    BallClass.UNKNOWN: (200, 200, 200),
}

_UNCERTAIN_GREY = (150, 150, 150)


def _ball_label(tr) -> str:
    """The ball's identity label: 'C' for the cue, '1'..'15' for a numbered ball,
    '' when unknown."""
    n = getattr(tr, "number", -1)
    if n == 0:
        return "C"
    if n and n > 0:
        return str(n)
    return ""


def _contrast_text(bgr) -> tuple[int, int, int]:
    """Black or white text, whichever reads on the given fill colour."""
    b, g, r = bgr
    lum = 0.114 * b + 0.587 * g + 0.299 * r
    return (20, 20, 20) if lum > 140 else (245, 245, 245)


def _draw_centered(img, text, c, fill_bgr, r) -> None:
    scale = max(0.32, r / 20.0)
    col = _contrast_text(fill_bgr)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.putText(img, text, (c[0] - tw // 2, c[1] + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def ball_color(tr, measured: bool = True) -> tuple[tuple[int, int, int], bool]:
    """Return (BGR, uncertain) for a tracked ball.

    With ``measured`` on (default) a ball is drawn in its *measured* mean colour
    so it actually looks like itself — a blue ball blue, a red ball red. The cue
    and 8 are forced to their canonical white/black (those classes are reliable),
    and an UNKNOWN class is drawn neutral grey + a "?" rather than a confident,
    wrong colour. With ``measured`` off we fall back to the fixed per-class
    palette (the old behaviour, kept behind a flag).
    """
    if not measured:
        return _CLS_COLOR.get(tr.cls, _UNCERTAIN_GREY), False
    if tr.cls == BallClass.CUE:
        return (255, 255, 255), False
    if tr.cls == BallClass.EIGHT:
        return (28, 28, 28), False
    if tr.cls == BallClass.UNKNOWN:
        return _UNCERTAIN_GREY, True
    bgr = getattr(tr, "bgr", None) or _UNCERTAIN_GREY
    return (int(bgr[0]), int(bgr[1]), int(bgr[2])), False


_SHIFT = 4          # sub-pixel drawing: 1/16th-px precision kills integer snap
_S = 1 << _SHIFT


def _smooth_path(pts: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Chaikin corner-cutting: turns the jagged frame-to-frame tracked path
    into a smooth curve (each pass replaces every corner with two points at
    1/4 and 3/4 of its edges)."""
    p = np.asarray(pts, np.float32)
    if len(p) > 1:  # collapse runs of identical points (a resting ball)
        keep = np.concatenate(([True], np.any(np.diff(p, axis=0) != 0, axis=1)))
        p = p[keep]
    for _ in range(iterations):
        if len(p) < 3:
            break
        q = p[:-1] * 0.75 + p[1:] * 0.25
        r = p[:-1] * 0.25 + p[1:] * 0.75
        inter = np.empty((2 * (len(p) - 1), 2), np.float32)
        inter[0::2] = q
        inter[1::2] = r
        p = np.vstack([p[:1], inter, p[-1:]])
    return p


def _draw_trail(img, history, color) -> None:
    """Fading comet tail: the smoothed path drawn in chunks that brighten and
    thicken toward the ball, so motion reads as a flowing stroke instead of a
    jagged wire."""
    p = _smooth_path(np.asarray(history, np.float32))
    n = len(p)
    if n < 2:
        return
    ip = np.round(p * _S).astype(np.int32)
    chunks = min(6, n - 1)
    for c in range(chunks):
        i0 = n * c // chunks
        i1 = min(n - 1, n * (c + 1) // chunks)
        if i1 <= i0:
            continue
        f = (c + 1) / chunks            # oldest chunk dimmest, newest brightest
        col = tuple(int(v * (0.25 + 0.75 * f)) for v in color)
        cv2.polylines(img, [ip[i0:i1 + 1].reshape(-1, 1, 2)], False, col,
                      2 if f > 0.7 else 1, cv2.LINE_AA, shift=_SHIFT)


def _accent_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return (151, 220, 61)  # mint default in BGR
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def draw_rectified(rect_bgr: np.ndarray, tracks: list[Track], table: TableModel,
                   show_traj: bool = True, show_ids: bool = True,
                   accent: str = "#3DDC97", measured_colors: bool = True,
                   fixed_radius: float | None = None) -> np.ndarray:
    img = rect_bgr.copy()
    acc = _accent_bgr(accent)

    # playing-surface rectangle
    cv2.rectangle(img, (int(table.x0), int(table.y0)), (int(table.x1), int(table.y1)),
                  (90, 110, 90), 1, cv2.LINE_AA)
    # pockets
    for p in table.pockets:
        cv2.circle(img, (int(p.x), int(p.y)), int(table.pocket_radius), (30, 30, 30), -1, cv2.LINE_AA)
        cv2.circle(img, (int(p.x), int(p.y)), int(table.pocket_radius), (70, 70, 70), 1, cv2.LINE_AA)
    # diamonds
    for (dx, dy) in table.diamonds():
        cv2.circle(img, (int(dx), int(dy)), 2, (200, 200, 160), -1, cv2.LINE_AA)

    for tr in tracks:
        color, _uncertain = ball_color(tr, measured_colors)
        c = (int(tr.x), int(tr.y))
        cs = (int(round(tr.x * _S)), int(round(tr.y * _S)))
        r = max(4, int(fixed_radius if fixed_radius else tr.radius))
        if show_traj and len(tr.history) > 1:
            _draw_trail(img, tr.history, acc)
        cv2.circle(img, cs, r * _S, color, 2, cv2.LINE_AA, shift=_SHIFT)
        cv2.circle(img, cs, 2 * _S, color, -1, cv2.LINE_AA, shift=_SHIFT)
        # velocity vector
        if tr.speed > 2.0:
            tip = (int(tr.x + tr.vx * 3), int(tr.y + tr.vy * 3))
            cv2.arrowedLine(img, c, tip, acc, 1, cv2.LINE_AA, tipLength=0.3)
        if show_ids:
            label = _ball_label(tr) or str(tr.id)
            cv2.putText(img, label, (c[0] + r + 2, c[1] - r),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img


_SCHEM_BASE: tuple | None = None  # (key, image) — static table art, cached


def _schematic_base(table: TableModel) -> np.ndarray:
    """The static part of the schematic (rails, felt, gradient, spots, diamonds,
    pockets). Identical every frame for a locked table, so it's drawn once and
    cached; render_schematic copies it and adds only the dynamic layer."""
    global _SCHEM_BASE
    h, w = table.height, table.width
    # exact-float key: any geometry change, however small, must redraw the base
    key = (w, h, table.x0, table.y0, table.x1, table.y1, table.pocket_radius)
    if _SCHEM_BASE is not None and _SCHEM_BASE[0] == key:
        return _SCHEM_BASE[1].copy()

    img = np.full((h, w, 3), (34, 38, 44), np.uint8)        # slate "rails"/frame
    x0, y0, x1, y1 = int(table.x0), int(table.y0), int(table.x1), int(table.y1)
    # felt with a soft inner gradient for depth
    cv2.rectangle(img, (x0, y0), (x1, y1), (58, 120, 72), -1, cv2.LINE_AA)
    inner = img[y0:y1, x0:x1]
    if inner.size:
        vign = np.zeros_like(inner)
        cv2.rectangle(vign, (0, 0), (inner.shape[1], inner.shape[0]), (16, 30, 18), -1)
        cv2.rectangle(vign, (12, 12), (inner.shape[1] - 12, inner.shape[0] - 12), (0, 0, 0), -1)
        img[y0:y1, x0:x1] = cv2.subtract(inner, vign)
    cv2.rectangle(img, (x0, y0), (x1, y1), (40, 80, 50), 2, cv2.LINE_AA)

    # head/foot spots (subtle)
    cx = (x0 + x1) // 2
    for fy in (0.25, 0.75):
        sy = int(y0 + (y1 - y0) * fy)
        cv2.circle(img, (cx, sy), 3, (90, 150, 105), -1, cv2.LINE_AA)

    # diamonds
    for (dx, dy) in table.diamonds():
        cv2.circle(img, (int(dx), int(dy)), 3, (210, 215, 180), -1, cv2.LINE_AA)

    # pockets
    for p in table.pockets:
        cv2.circle(img, (int(p.x), int(p.y)), int(table.pocket_radius), (12, 14, 16), -1, cv2.LINE_AA)
        cv2.circle(img, (int(p.x), int(p.y)), int(table.pocket_radius), (70, 78, 86), 1, cv2.LINE_AA)
    _SCHEM_BASE = (key, img)
    return img.copy()


def render_schematic(table: TableModel, tracks: list[Track], accent: str = "#3DDC97",
                     show_traj: bool = True, show_ids: bool = True,
                     debug: bool = False, detections=None, diag=None,
                     measured_colors: bool = True,
                     fixed_radius: float | None = None) -> np.ndarray:
    """Render a clean, proportional top-down table from the game state — felt,
    rails, pockets, diamonds, and balls as circles — instead of the warped camera
    image. Ball positions are the rectified (already-proportional) coordinates."""
    h = table.height
    img = _schematic_base(table)
    acc = _accent_bgr(accent)

    # raw detections (debug)
    if debug and detections:
        for d in detections:
            cv2.circle(img, (int(d.x), int(d.y)), int(d.radius), (0, 180, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"{d.score:.2f}", (int(d.x) + 4, int(d.y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 180, 255), 1, cv2.LINE_AA)

    # balls — drawn at the KNOWN physical radius when fixed_radius is given, so the
    # overhead shows uniform regulation balls instead of the detector's per-frame wobble
    for tr in tracks:
        color, uncertain = ball_color(tr, measured_colors)
        c = (int(tr.x), int(tr.y))
        cs = (int(round(tr.x * _S)), int(round(tr.y * _S)))
        r = max(6, int(fixed_radius if fixed_radius else tr.radius))
        rs = r * _S
        if show_traj and len(tr.history) > 1:
            _draw_trail(img, tr.history, acc)
        # sub-pixel centres: a ball whose track sits at x.5 no longer snaps
        # between neighbouring pixels frame to frame
        cv2.circle(img, cs, rs, color, -1, cv2.LINE_AA, shift=_SHIFT)
        # Stripe: a white equatorial band over the base colour, so 9..15 read as
        # stripes at a glance (the base colour still identifies the number).
        if tr.cls == BallClass.STRIPE:
            band_h = max(2 * _S, int(rs * 0.55))
            cv2.rectangle(img, (cs[0] - rs, cs[1] - band_h // 2),
                          (cs[0] + rs, cs[1] + band_h // 2),
                          (245, 245, 245), -1, cv2.LINE_AA, shift=_SHIFT)
            cv2.circle(img, cs, rs, color, 2, cv2.LINE_AA, shift=_SHIFT)  # restore rim
        cv2.circle(img, cs, rs, (20, 22, 26), 1, cv2.LINE_AA, shift=_SHIFT)
        label = _ball_label(tr)
        if uncertain and not label:
            cv2.putText(img, "?", (c[0] - r // 2, c[1] + r // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.3, r / 18.0), (40, 40, 40), 1,
                        cv2.LINE_AA)
        elif show_ids and label:
            # the ball NUMBER, centred on the ball (white-band centre for stripes)
            _draw_centered(img, label, c,
                           (245, 245, 245) if tr.cls == BallClass.STRIPE else color, r)
        elif not uncertain:
            cv2.circle(img, (cs[0] - rs // 3, cs[1] - rs // 3), max(2 * _S, rs // 4),
                       (255, 255, 255), -1, cv2.LINE_AA, shift=_SHIFT)
        if tr.speed > 2.0:
            tip = (int(tr.x + tr.vx * 3), int(tr.y + tr.vy * 3))
            cv2.arrowedLine(img, c, tip, acc, 1, cv2.LINE_AA, tipLength=0.3)

    if debug and diag:
        line1 = (f"state={diag.get('state')} cue={diag.get('cue')} "
                 f"travel={diag.get('travel')} pot={diag.get('potted')} "
                 f"fps={diag.get('fps', 0)}")
        line2 = (f"motion={diag.get('motion')} flow={diag.get('flow')} "
                 f"fg={diag.get('fg')} -> fused={diag.get('fused')}")
        cv2.putText(img, line1, (8, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (200, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(img, line2, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (180, 255, 200), 1, cv2.LINE_AA)
        stages = diag.get("stages") or {}
        if stages:
            line3 = " ".join(f"{k}={v:.1f}" for k, v in stages.items())
            cv2.putText(img, line3, (8, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                        (255, 210, 170), 1, cv2.LINE_AA)
    return img


def draw_perspective(frame: np.ndarray, corners: np.ndarray | None,
                     tracks: list[Track], Hinv: np.ndarray | None,
                     accent: str = "#3DDC97", table=None) -> np.ndarray:
    img = frame.copy()
    acc = _accent_bgr(accent)
    if corners is not None:
        pts = corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, acc, 2, cv2.LINE_AA)
        for (cx, cy) in corners.astype(int):
            cv2.circle(img, (int(cx), int(cy)), 5, acc, -1, cv2.LINE_AA)
    # Cushion-nose line: the playing boundary the app actually uses, projected
    # back onto the camera. The calibration aid for Settings -> Cushion inset:
    # tune the inches until this line sits ON the cushion noses.
    if Hinv is not None and table is not None:
        nose = np.array([[table.x0, table.y0], [table.x1, table.y0],
                         [table.x1, table.y1], [table.x0, table.y1]], np.float64)
        cam = project_points(nose, Hinv).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [cam], True, (60, 200, 255), 1, cv2.LINE_AA)
    if Hinv is not None and tracks:
        rect_pts = np.array([[tr.x, tr.y] for tr in tracks], np.float64)
        orig = project_points(rect_pts, Hinv)
        for tr, (ox, oy) in zip(tracks, orig, strict=False):
            color = _CLS_COLOR.get(tr.cls, (200, 200, 200))
            cv2.circle(img, (int(ox), int(oy)), 6, color, 2, cv2.LINE_AA)
    return img
