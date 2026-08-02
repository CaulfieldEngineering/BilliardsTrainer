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
    # A ball at rest must not keep wearing its old path: position history stays
    # in the deque for ~64 frames, which left stale streaks across the table
    # seconds after a shot. If the ball hasn't moved lately, draw no trail.
    recent = np.asarray(history[-12:], np.float32)
    if len(recent) >= 2 and np.abs(np.diff(recent, axis=0)).sum() < 6.0:
        return
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
_BUILD_SS = 2                       # supersample the static art at build time only


def _rounded(img, x0, y0, x1, y1, r, color) -> None:
    r = int(r)
    cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1, cv2.LINE_AA)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def _cloth(pw: int, ph: int) -> np.ndarray:
    """Tournament-blue cloth: a top-lit vertical gradient, a soft radial edge
    vignette, and a faint woven texture — returns a float BGR (ph, pw, 3)."""
    top = np.array([168, 108, 46], float)      # BGR of a lit tournament blue
    bot = np.array([120, 66, 24], float)         # darker toward the foot rail
    grad = np.linspace(0, 1, ph)[:, None]
    felt = np.tile((top * (1 - grad) + bot * grad)[:, None, :], (1, pw, 1))
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    cx, cy = pw / 2.0, ph / 2.0
    d = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    felt *= np.clip(1.0 - 0.30 * np.clip(d - 0.35, 0, 1.4), 0.58, 1.0)[:, :, None]
    weave = (np.sin(xx * 0.9) + np.sin(yy * 0.9)) * 2.0
    grain = np.random.default_rng(7).standard_normal((ph, pw)).astype(np.float32) * 2.4
    felt += (weave + grain)[:, :, None]
    return felt


def _sphere(img, cx: float, cy: float, r: float, bgr) -> None:
    """A glossy shaded sphere: a dark-rim→lit-centre radial gradient built from
    concentric circles offset toward the light (top-left)."""
    bgr = np.array(bgr, float)
    dark = np.clip(bgr * 0.40, 0, 255)
    light = np.clip(bgr * 1.32 + 42, 0, 255)
    steps = 22
    for i in range(steps):
        t = i / (steps - 1)
        col = (dark * (1 - t) + light * t).tolist()
        rr = r * (1.0 - 0.9 * t)
        off = r * 0.32 * t
        cv2.circle(img, (int(round((cx - off) * _S)), int(round((cy - off) * _S))),
                   max(1, int(round(rr * _S))), col, -1, cv2.LINE_AA, shift=_SHIFT)


def _highlight(img, cx: float, cy: float, r: float) -> None:
    cv2.circle(img, (int(round((cx - r * 0.34) * _S)), int(round((cy - r * 0.42) * _S))),
               max(1, int(round(r * 0.20 * _S))), (255, 255, 255), -1, cv2.LINE_AA, shift=_SHIFT)


def _stripe(img, cx: float, cy: float, r: float, bgr) -> None:
    """A white ball with a coloured equatorial band, both keeping sphere shading."""
    pad = int(r) + 2
    x0, y0, x1, y1 = int(cx) - pad, int(cy) - pad, int(cx) + pad, int(cy) + pad
    ih, iw = img.shape[:2]
    if x0 < 0 or y0 < 0 or x1 > iw or y1 > ih:
        _sphere(img, cx, cy, r, bgr)
        return
    lcx, lcy = cx - x0, cy - y0
    white = img[y0:y1, x0:x1].copy()
    colour = white.copy()
    _sphere(white, lcx, lcy, r, (240, 240, 242))
    _sphere(colour, lcx, lcy, r, bgr)
    band = r * 0.66
    mask = np.zeros(white.shape[:2], np.float32)
    mask[max(0, int(lcy - band)):int(lcy + band), :] = 1.0
    mask = cv2.GaussianBlur(mask, (0, 0), 0.8)[:, :, None]
    img[y0:y1, x0:x1] = (white * (1 - mask) + colour * mask).astype(np.uint8)


def _schematic_base(table: TableModel) -> np.ndarray:
    """The static part of the schematic (rail, tournament-blue cloth with a
    top-lit gradient + woven texture, inlaid diamonds, recessed pockets). Built
    once at 2x and downscaled for crisp edges, then cached — render_schematic
    copies it and adds only the dynamic ball layer."""
    global _SCHEM_BASE
    h, w = table.height, table.width
    key = (w, h, table.x0, table.y0, table.x1, table.y1, table.pocket_radius)
    if _SCHEM_BASE is not None and _SCHEM_BASE[0] == key:
        return _SCHEM_BASE[1].copy()

    ss = _BUILD_SS
    bw, bh = w * ss, h * ss
    x0, y0 = int(table.x0 * ss), int(table.y0 * ss)
    x1, y1 = int(table.x1 * ss), int(table.y1 * ss)
    img = np.full((bh, bw, 3), (30, 26, 24), np.uint8)

    m = int(table.pad * ss * 0.30)
    _rounded(img, m, m, bw - m, bh - m, table.pad * ss * 0.95, (40, 36, 34))       # rail
    _rounded(img, m, m, bw - m, (y0 + y1) // 2, table.pad * ss * 0.95, (50, 45, 42))  # top bevel
    _rounded(img, x0 - 4 * ss, y0 - 4 * ss, x1 + 4 * ss, y1 + 4 * ss, 7 * ss, (22, 19, 18))  # inner shadow

    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = np.clip(_cloth(x1 - x0, y1 - y0), 0, 255).astype(np.uint8)
    cv2.rectangle(img, (x0, y0), (x1, y1), (150, 96, 40), 3 * ss, cv2.LINE_AA)      # cushion shade
    cv2.rectangle(img, (x0, y0), (x1, y1), (200, 150, 96), max(1, ss), cv2.LINE_AA)  # nose highlight

    cxm = (x0 + x1) // 2
    cv2.circle(img, (cxm, int(y0 + (y1 - y0) * 0.75)), int(3 * ss), (180, 150, 110), -1, cv2.LINE_AA)

    dsz = table.short_side * ss * 0.017
    for (dx, dy) in table.diamonds():
        c = np.array([dx * ss, dy * ss])
        quad = np.array([[0, -dsz], [dsz, 0], [0, dsz], [-dsz, 0]], np.float32)
        cv2.fillConvexPoly(img, (c + quad + [0, ss]).astype(np.int32), (14, 12, 11), cv2.LINE_AA)
        cv2.fillConvexPoly(img, (c + quad).astype(np.int32), (232, 226, 214), cv2.LINE_AA)

    for p in table.pockets:
        pc = (int(p.x * ss), int(p.y * ss))
        pr = int(table.pocket_radius * ss * 1.08)
        for i in range(pr, 0, -1):
            t = i / pr
            cv2.circle(img, pc, i, (int(12 + 26 * t), int(11 + 22 * t), int(10 + 20 * t)),
                       -1, cv2.LINE_AA)
        cv2.circle(img, pc, pr, (96, 86, 80), max(1, ss), cv2.LINE_AA)

    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    _SCHEM_BASE = (key, img)
    return img.copy()


def _draw_play_paths(img, play_paths) -> None:
    """Persistent per-play shot paths: white outlined stroke for the cue ball,
    each object ball's path in its own colour — the broadcast trace."""
    for e in play_paths.values():
        pts = e.get("pts") or []
        if len(pts) < 2:
            continue
        p = _smooth_path(np.asarray(pts, np.float32))
        ip = np.round(p * _S).astype(np.int32).reshape(-1, 1, 2)
        if e.get("cue"):
            cv2.polylines(img, [ip], False, (25, 28, 32), 4, cv2.LINE_AA, shift=_SHIFT)
            cv2.polylines(img, [ip], False, (250, 250, 250), 2, cv2.LINE_AA, shift=_SHIFT)
        else:
            col = tuple(int(v) for v in e.get("bgr", (200, 200, 200)))
            cv2.polylines(img, [ip], False, col, 2, cv2.LINE_AA, shift=_SHIFT)


def render_schematic(table: TableModel, tracks: list[Track], accent: str = "#3DDC97",
                     show_traj: bool = True, show_ids: bool = True,
                     debug: bool = False, detections=None, diag=None,
                     measured_colors: bool = True,
                     fixed_radius: float | None = None,
                     play_paths: dict | None = None,
                     paths_alpha: float = 1.0) -> np.ndarray:
    """Render a clean, modern top-down table from the game state — tournament-blue
    cloth, rail, pockets, diamonds, and balls as glossy shaded spheres — instead of
    the warped camera image. Ball positions are the rectified (proportional)
    coordinates; identity reads from colour + stripe, never numbers."""
    h = table.height
    img = _schematic_base(table)
    acc = _accent_bgr(accent)

    # raw detections (debug)
    if debug and detections:
        for d in detections:
            cv2.circle(img, (int(d.x), int(d.y)), int(d.radius), (0, 180, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"{d.score:.2f}", (int(d.x) + 4, int(d.y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 180, 255), 1, cv2.LINE_AA)

    # shot paths under the balls: the whole play's traces, cue in white.
    # After the table settles the traces linger 3s, then fade over ~1s.
    if play_paths and paths_alpha > 0.0:
        if paths_alpha >= 1.0:
            _draw_play_paths(img, play_paths)
        else:
            base = img.copy()
            _draw_play_paths(img, play_paths)
            cv2.addWeighted(img, paths_alpha, base, 1.0 - paths_alpha, 0, dst=img)

    # soft contact shadows for every ball — one blurred pass multiplied into the
    # felt, so the balls read as sitting ON the cloth (depth) instead of pasted on.
    if tracks:
        shadow = np.zeros(img.shape[:2], np.float32)
        for tr in tracks:
            r = max(6, int(fixed_radius if fixed_radius else tr.radius))
            cv2.circle(shadow, (int(tr.x + r * 0.16), int(tr.y + r * 0.26)),
                       int(r * 1.03), 1.0, -1, cv2.LINE_AA)
        shadow = cv2.GaussianBlur(shadow, (0, 0), max(1.5, (fixed_radius or 10) * 0.28))
        img = (img.astype(np.float32) * (1.0 - 0.45 * shadow[:, :, None])).astype(np.uint8)

    # balls — glossy shaded spheres at the KNOWN physical radius (uniform overhead,
    # not the detector's per-frame wobble). Identity is COLOUR-only: a solid is a
    # full colour sphere, a stripe is a white ball with a coloured band, the cue is
    # white and the 8 is black. No numbers — deliberately clean and legible.
    for tr in tracks:
        color, _uncertain = ball_color(tr, measured_colors)
        cx, cy = tr.x, tr.y
        r = max(6, int(fixed_radius if fixed_radius else tr.radius))
        if show_traj and play_paths is None and len(tr.history) > 1:
            _draw_trail(img, tr.history, acc)
        if tr.cls == BallClass.STRIPE:
            _stripe(img, cx, cy, r, color)
        else:
            _sphere(img, cx, cy, r, color)
        _highlight(img, cx, cy, r)
        cv2.circle(img, (int(round(cx * _S)), int(round(cy * _S))), int(round(r * _S)),
                   (16, 15, 14), 1, cv2.LINE_AA, shift=_SHIFT)
        if tr.speed > 2.0:
            tip = (int(cx + tr.vx * 3), int(cy + tr.vy * 3))
            cv2.arrowedLine(img, (int(cx), int(cy)), tip, acc, 2, cv2.LINE_AA, tipLength=0.3)

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
