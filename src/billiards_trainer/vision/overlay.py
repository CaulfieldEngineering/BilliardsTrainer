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


def _accent_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return (151, 220, 61)  # mint default in BGR
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def draw_rectified(rect_bgr: np.ndarray, tracks: list[Track], table: TableModel,
                   show_traj: bool = True, show_ids: bool = True,
                   accent: str = "#3DDC97") -> np.ndarray:
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
        color = _CLS_COLOR.get(tr.cls, (200, 200, 200))
        c = (int(tr.x), int(tr.y))
        r = max(4, int(tr.radius))
        if show_traj and len(tr.history) > 1:
            pts = np.array(tr.history, np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, acc, 1, cv2.LINE_AA)
        cv2.circle(img, c, r, color, 2, cv2.LINE_AA)
        cv2.circle(img, c, 2, color, -1, cv2.LINE_AA)
        # velocity vector
        if tr.speed > 2.0:
            tip = (int(tr.x + tr.vx * 3), int(tr.y + tr.vy * 3))
            cv2.arrowedLine(img, c, tip, acc, 1, cv2.LINE_AA, tipLength=0.3)
        if show_ids:
            cv2.putText(img, str(tr.id), (c[0] + r + 2, c[1] - r),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img


def render_schematic(table: TableModel, tracks: list[Track], accent: str = "#3DDC97",
                     show_traj: bool = True, show_ids: bool = True,
                     debug: bool = False, detections=None, diag=None) -> np.ndarray:
    """Render a clean, proportional top-down table from the game state — felt,
    rails, pockets, diamonds, and balls as circles — instead of the warped camera
    image. Ball positions are the rectified (already-proportional) coordinates."""
    h, w = table.height, table.width
    img = np.full((h, w, 3), (34, 38, 44), np.uint8)        # slate "rails"/frame
    acc = _accent_bgr(accent)

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

    # raw detections (debug)
    if debug and detections:
        for d in detections:
            cv2.circle(img, (int(d.x), int(d.y)), int(d.radius), (0, 180, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"{d.score:.2f}", (int(d.x) + 4, int(d.y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 180, 255), 1, cv2.LINE_AA)

    # balls
    for tr in tracks:
        color = _CLS_COLOR.get(tr.cls, (200, 200, 200))
        c = (int(tr.x), int(tr.y))
        r = max(6, int(tr.radius))
        if show_traj and len(tr.history) > 1:
            pts = np.array(tr.history, np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, acc, 1, cv2.LINE_AA)
        cv2.circle(img, c, r, color, -1, cv2.LINE_AA)
        cv2.circle(img, c, r, (20, 22, 26), 1, cv2.LINE_AA)
        cv2.circle(img, (c[0] - r // 3, c[1] - r // 3), max(2, r // 4), (255, 255, 255), -1, cv2.LINE_AA)
        if tr.speed > 2.0:
            tip = (int(tr.x + tr.vx * 3), int(tr.y + tr.vy * 3))
            cv2.arrowedLine(img, c, tip, acc, 1, cv2.LINE_AA, tipLength=0.3)
        if show_ids:
            cv2.putText(img, str(tr.id), (c[0] + r + 2, c[1] - r),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (235, 235, 235), 1, cv2.LINE_AA)

    if debug and diag:
        txt = (f"state={diag.get('state')} cue={diag.get('cue')} "
               f"spd={diag.get('max_speed')} travel={diag.get('max_travel')} "
               f"pot={diag.get('potted')}")
        cv2.putText(img, txt, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (200, 220, 255), 1, cv2.LINE_AA)
    return img


def draw_perspective(frame: np.ndarray, corners: np.ndarray | None,
                     tracks: list[Track], Hinv: np.ndarray | None,
                     accent: str = "#3DDC97") -> np.ndarray:
    img = frame.copy()
    acc = _accent_bgr(accent)
    if corners is not None:
        pts = corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, acc, 2, cv2.LINE_AA)
        for (cx, cy) in corners.astype(int):
            cv2.circle(img, (int(cx), int(cy)), 5, acc, -1, cv2.LINE_AA)
    if Hinv is not None and tracks:
        rect_pts = np.array([[tr.x, tr.y] for tr in tracks], np.float64)
        orig = project_points(rect_pts, Hinv)
        for tr, (ox, oy) in zip(tracks, orig, strict=False):
            color = _CLS_COLOR.get(tr.cls, (200, 200, 200))
            cv2.circle(img, (int(ox), int(oy)), 6, color, 2, cv2.LINE_AA)
    return img
