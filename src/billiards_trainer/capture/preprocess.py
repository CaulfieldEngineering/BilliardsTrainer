"""Raw-frame preprocessing applied at ingest: rotation + colour correction.

Both run on the BGR frame the moment it arrives from the source and *before* the
pipeline sees it, so detection, felt-picking, the live preview, recording and
training-capture all share one consistent orientation and colour space. Kept as
pure functions (no Qt, no capture state) so they're unit-testable headless and
cheap enough to run every frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import CameraSettings, ColorCorrectionSettings

# OpenCV rotate codes keyed by clockwise degrees. 0 is a passthrough (no copy).
_ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def rotate_frame(frame: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate a frame clockwise by 0/90/180/270 degrees.

    90/270 swap width and height (landscape -> portrait). Any other value is
    treated as 0 (no rotation) so a bad setting can never raise on the hot path.
    """
    code = _ROTATE_CODES.get(int(degrees) % 360)
    return frame if code is None else cv2.rotate(frame, code)


def _white_patch_gains(frame: np.ndarray, top_frac: float = 0.02) -> tuple[float, float, float]:
    """Per-channel gains that neutralise the brightest ~``top_frac`` of pixels.

    White-patch (a.k.a. Retinex) white balance: the brightest pixels on a pool
    table are the white cue ball and the ceiling lights, which *should* be neutral.
    We scale each channel so their mean lands on a common grey, giving felt-safe
    correction (grey-world would instead fight the dominant green). Gains are
    clamped to [0.5, 2.0] so a frame with no bright neutral can't blow up colour.
    """
    # Luminance threshold at the (1 - top_frac) percentile.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = float(np.quantile(gray, 1.0 - top_frac))
    mask = gray >= max(thresh, 1.0)
    if int(mask.sum()) < 25:  # too few bright pixels to trust — leave unchanged
        return 1.0, 1.0, 1.0
    means = frame[mask].reshape(-1, 3).mean(axis=0)  # B, G, R means of bright pixels
    means = np.clip(means, 1.0, None)
    target = float(means.mean())
    gains = target / means
    gains = np.clip(gains, 0.5, 2.0)
    return float(gains[0]), float(gains[1]), float(gains[2])


def _apply_gains(frame: np.ndarray, gb: float, gg: float, gr: float) -> np.ndarray:
    if gb == 1.0 and gg == 1.0 and gr == 1.0:
        return frame
    out = frame.astype(np.float32)
    out[:, :, 0] *= gb
    out[:, :, 1] *= gg
    out[:, :, 2] *= gr
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_saturation(frame: np.ndarray, sat: float) -> np.ndarray:
    if abs(sat - 1.0) < 1e-3:
        return frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def color_correct(frame: np.ndarray, cc: ColorCorrectionSettings) -> np.ndarray:
    """Apply colour correction per the settings mode (auto | manual | off)."""
    mode = (cc.mode or "off").lower()
    if mode == "off":
        return frame
    if mode == "manual":
        out = _apply_gains(frame, cc.gain_b, cc.gain_g, cc.gain_r)
        return _apply_saturation(out, cc.saturation)
    # auto: felt-safe white-patch balance, blended by auto_strength.
    gb, gg, gr = _white_patch_gains(frame)
    strength = float(np.clip(cc.auto_strength, 0.0, 1.0))
    if strength < 1.0:  # blend each gain toward 1.0 (no correction)
        gb = 1.0 + (gb - 1.0) * strength
        gg = 1.0 + (gg - 1.0) * strength
        gr = 1.0 + (gr - 1.0) * strength
    return _apply_gains(frame, gb, gg, gr)


def flip_frame(frame: np.ndarray, flip_h: bool, flip_v: bool) -> np.ndarray:
    """Mirror a frame: flip_h left/right, flip_v top/bottom (either, both, none)."""
    if flip_h and flip_v:
        return cv2.flip(frame, -1)
    if flip_h:
        return cv2.flip(frame, 1)
    if flip_v:
        return cv2.flip(frame, 0)
    return frame


def preprocess_frame(frame: np.ndarray, cam: CameraSettings) -> np.ndarray:
    """Rotate, flip, then colour-correct a raw frame per the camera settings.

    Order matters: geometry (rotate + flip) first so colour stats aren't
    recomputed on a soon-to-be discarded orientation. Returns the frame
    unchanged when nothing is configured.
    """
    if frame is None:
        return frame
    frame = rotate_frame(frame, cam.rotation)
    frame = flip_frame(frame, cam.flip_h, cam.flip_v)
    frame = color_correct(frame, cam.color)
    return frame
