"""Pluggable, raw-frame detector strategies (Phase 1 experimentation).

Per the architectural pivot, every strategy detects on the **raw oblique camera
frame** and returns detections in **raw-frame pixel coordinates** (the bird's-eye
is display-only). A strategy gets the raw BGR frame + the table ``Calibration``
(corners/H/Hinv/table/felt) and returns ``list[Detection]`` (reusing the vision
Detection type; x/y/radius are raw pixels).

Adding a variant is a ONE-FILE change: drop ``<name>.py`` in this package that
defines a module-level ``STRATEGIES = [MyStrategy(...), ...]``. ``discover()``
auto-imports every module and collects them. These modules are ADDITIVE — nothing
here is wired into the shipped pipeline, so the default detector is unchanged.
"""

import importlib
import logging
import pkgutil

import cv2
import numpy as np

from ..vision.rectify import project_points
from ..vision.types import BallClass, Detection

log = logging.getLogger("detector_strategies")


class DetectorStrategy:
    name = "base"
    description = ""
    # Model-based detectors (YOLO/ONNX) already validate "this is a ball" with high
    # confidence, so the pipeline's physical-size prior — which exists to reject
    # classical blob noise (pocket shadows etc.) — should NOT cull them.
    model_based = False

    def detect(self, frame_bgr: np.ndarray, calib) -> list[Detection]:
        """frame_bgr: RAW camera frame. calib: Calibration | None. Returns raw-px
        detections. MUST NOT raise on a normal frame (return [] if it can't run)."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Shared raw-frame helpers
# --------------------------------------------------------------------------- #
def table_polygon_mask(shape, calib) -> np.ndarray:
    """uint8 mask of the table playing area in RAW coords (from calib.corners)."""
    h, w = shape[:2]
    mask = np.zeros((h, w), np.uint8)
    if calib is None or getattr(calib, "corners", None) is None:
        mask[:] = 255
        return mask
    cv2.fillConvexPoly(mask, np.asarray(calib.corners, np.int32).reshape(-1, 2), 255)
    return mask


def felt_mask_raw(frame_bgr, calib) -> np.ndarray:
    """HSV felt mask on the RAW frame using calib.felt (handles hue wrap)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    f = getattr(calib, "felt", None)
    if f is None:
        # fall back: estimate felt as the dominant hue inside the table polygon
        return _auto_felt(hsv, calib)
    lo_s, hi_s, lo_v, hi_v = f.s_min, f.s_max, f.v_min, f.v_max
    if f.h_min <= f.h_max:
        m = cv2.inRange(hsv, (f.h_min, lo_s, lo_v), (f.h_max, hi_s, hi_v))
    else:
        a = cv2.inRange(hsv, (0, lo_s, lo_v), (f.h_max, hi_s, hi_v))
        b = cv2.inRange(hsv, (f.h_min, lo_s, lo_v), (180, hi_s, hi_v))
        m = cv2.bitwise_or(a, b)
    return m


def _auto_felt(hsv, calib) -> np.ndarray:
    poly = table_polygon_mask(hsv.shape, calib)
    sel = hsv[poly > 0]
    if sel.size == 0:
        return np.zeros(hsv.shape[:2], np.uint8)
    h = int(np.median(sel[:, 0]))
    return cv2.inRange(hsv, (max(0, h - 18), 40, 40), (min(180, h + 18), 255, 255))


def ball_radius_raw(calib, frame_shape) -> float:
    """Rough expected ball radius in RAW pixels. On an oblique view ball size
    varies across the table, so this is just a scale anchor; callers use a wide
    band around it."""
    if calib is not None and getattr(calib, "corners", None) is not None:
        c = np.asarray(calib.corners, np.float64).reshape(-1, 2)
        # mean edge length ~ table side in px; short side / ~44 ≈ ball radius
        edges = [np.hypot(*(c[i] - c[(i + 1) % 4])) for i in range(4)]
        short = min(edges)
        return max(4.0, short * 0.0225)
    return max(4.0, frame_shape[1] * 0.012)


def project_rect_dets_to_raw(dets, calib) -> list[Detection]:
    """Map rectified-space detections back to RAW coords via calib.Hinv."""
    if not dets or calib is None:
        return []
    pts = np.array([[d.x, d.y] for d in dets], np.float64)
    raw = project_points(pts, calib.Hinv)
    off = np.array([[d.x + max(d.radius, 1.0), d.y] for d in dets], np.float64)
    raw_off = project_points(off, calib.Hinv)
    out = []
    for d, (rx, ry), (ox, oy) in zip(dets, raw, raw_off, strict=False):
        out.append(Detection(float(rx), float(ry), float(np.hypot(ox - rx, oy - ry)),
                              d.bgr, d.cls, d.score))
    return out


def classify_crop(frame_bgr, cx, cy, r) -> tuple[BallClass, tuple]:
    """Colour/type of a ball from its raw-frame crop (reuses the classical logic)."""
    from ..vision.balls import classify_ball
    h, w = frame_bgr.shape[:2]
    x0, y0 = max(0, int(cx - r)), max(0, int(cy - r))
    x1, y1 = min(w, int(cx + r) + 1), min(h, int(cy + r) + 1)
    patch = frame_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return BallClass.UNKNOWN, (200, 200, 200)
    return classify_ball(patch)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover(params: dict | None = None) -> dict:
    """Import every strategy module and collect its module-level STRATEGIES.

    FROZEN-SAFE. The core strategies (``simple_blob``/``felt_mask_hough``/
    ``classical``) are imported STATICALLY at the bottom of this module, so
    PyInstaller's module graph bundles them and they are *always* available.
    ``pkgutil.iter_modules`` is used only as an ADDITIVE convenience to pick up
    extra drop-in strategy files on disk during dev.

    This matters: in a PyInstaller onefile, ``iter_modules(__path__)`` finds
    nothing (the modules live in the bundle, not on disk), so the previous
    iter_modules-ONLY discovery returned ``{}`` in every shipped build — which
    silently collapsed both the live detector and the Settings "Live detector"
    dropdown to ``legacy``. The static core below is the fix.
    """
    out: dict[str, DetectorStrategy] = {}
    modules = list(_CORE_MODULES)
    seen = {m.__name__ for m in modules}
    try:
        for m in pkgutil.iter_modules(__path__):
            if m.name.startswith("_"):
                continue
            full = f"{__name__}.{m.name}"
            if full in seen:
                continue
            try:
                modules.append(importlib.import_module(full))
                seen.add(full)
            except Exception as exc:  # noqa: BLE001 - a broken/optional strategy file shouldn't kill discovery
                log.warning("strategy module %s failed to import: %s", m.name, exc)
    except Exception as exc:  # noqa: BLE001 - iter_modules can be unavailable in a frozen build
        log.debug("iter_modules unavailable (%s); using static core only", exc)
    for mod in modules:
        for s in getattr(mod, "STRATEGIES", []):
            out[s.name] = s
    return out


# Static imports of the core detectors MUST come last: each module does
# ``from . import <helpers>`` from THIS package, so it can only be imported after
# the helpers above are defined (otherwise a partial-init circular import). These
# explicit imports (a) make PyInstaller bundle the modules and (b) make discover()
# work identically in frozen and source builds.
#
# Two detectors now: the trained YOLO model (onnx_model — the default/primary) and
# the cue-ball white heuristic (cue_ball_white — the no-model fallback). The old
# classical blob "strategy zoo" (classical/simple_blob/felt_mask_hough) and the
# torch-based ultralytics_pt were removed: a trained model beats them all and they
# only added the "which detector?" confusion.
from . import cue_ball_white, onnx_model  # noqa: E402

_CORE_MODULES = (cue_ball_white, onnx_model)
