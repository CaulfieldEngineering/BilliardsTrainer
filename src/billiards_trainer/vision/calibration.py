"""One-shot table calibration + deviation watchdog.

The core principle from the code review: the table does not move during a
session, so detect it *once*, lock the homography, and spend the per-frame
budget on balls. Re-running felt detection every frame (as the C++ prototype
did) wastes time and adds corner jitter.

``CalibrationManager`` runs the expensive felt+rectify pipeline on demand, caches
H/Hinv and the table model, then cheaply warps subsequent frames with the locked
homography. A lightweight watchdog periodically re-checks the corners and raises
``deviated`` if the table appears to have shifted.
"""

import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import numpy as np

from ..config import FeltSettings, Settings
from .felt import detect_felt, estimate_felt_settings
from .geometry import TableModel
from .rectify import rectify_tabletop

log = logging.getLogger("vision.calibration")


@dataclass
class Calibration:
    corners: np.ndarray
    H: np.ndarray
    Hinv: np.ndarray
    dst_size: tuple[int, int]
    table: TableModel
    rect_mask: np.ndarray | None  # None when restored from disk (detectors recompute)
    felt: FeltSettings  # effective felt colour key (possibly auto-estimated)


class CalibrationManager:
    def __init__(self, deviation_px: float = 30.0, deviation_frames: int = 12,
                 settle_px: float = 12.0, corner_ema: float = 0.06):
        self.calib: Calibration | None = None
        self.deviated: bool = False
        # The table/camera don't move during play, so the lock is sticky and
        # SELF-CORRECTING rather than twitchy: a re-detect that closely agrees eases
        # the locked corners toward it (averages out detection noise, fixes tiny
        # initial-lock error); a poor/absent re-detect (a hand/cue/person over a
        # corner) is ignored, never lost; only a LARGE, SUSTAINED disagreement —
        # the table genuinely moved — flags a relock.
        self._deviation_px = deviation_px        # px RMSE that counts as "moved"
        self._deviation_frames = deviation_frames  # consecutive checks before relock
        self._settle_px = settle_px              # RMSE under which we average-in
        self._corner_ema = corner_ema            # how fast locked corners ease over
        self._consecutive = 0

    @property
    def is_calibrated(self) -> bool:
        return self.calib is not None

    def calibrate(self, frame: np.ndarray, settings: Settings) -> bool:
        """Run full felt + rectify detection and lock the result. Returns success."""
        if frame is None or frame.size == 0:
            return False
        felt_settings = settings.felt
        felt = detect_felt(frame, felt_settings)
        # Fallback: if the configured colour matches too little of the frame,
        # auto-estimate the felt colour from the centre and retry. This makes
        # calibration work across tables/lighting without manual tuning.
        if not felt.has_corners or felt.area_ratio < 0.04:
            est = estimate_felt_settings(frame, settings.felt)
            felt_est = detect_felt(frame, est)
            if felt_est.has_corners and felt_est.area_ratio > felt.area_ratio:
                log.info("Auto-estimated felt colour (hue~%d); area %.3f -> %.3f",
                         est.picked_hsv[0], felt.area_ratio, felt_est.area_ratio)
                felt, felt_settings = felt_est, est
        if not felt.has_corners:
            log.info("Calibration failed: no felt corners")
            return False
        # refine=False: the base homography (clean felt corners -> forced 2:1
        # rectangle) is stable and correct. The Hough-line "square-up" refinement
        # fires intermittently and, on a bad frame, skews the rectangle
        # non-uniformly (fat far rail, egg-shaped balls, pockets off their marks).
        # Fewer moving parts = a deterministic, undistorted bird's-eye. (review: calib-5)
        rect = rectify_tabletop(frame, felt.mask, felt.corners,
                                pad_px=settings.rectify.pad_px,
                                aspect=settings.rectify.aspect, refine=False)
        if not rect.ok:
            log.info("Calibration failed: rectification not ok")
            return False
        table = TableModel.from_rect(rect.dst_size, settings.rectify.pad_px,
                                     settings.table.pocket_radius_frac,
                                     nose_inset_frac=settings.table.nose_inset_frac)
        self.calib = Calibration(
            corners=felt.corners, H=rect.H, Hinv=rect.Hinv,
            dst_size=rect.dst_size, table=table, rect_mask=rect.rectified_mask,
            felt=felt_settings,
        )
        self.deviated = False
        self._consecutive = 0
        log.info("Calibrated: dst_size=%s, %d pockets", rect.dst_size, len(table.pockets))
        return True

    def rectify(self, frame: np.ndarray) -> np.ndarray | None:
        """Cheaply warp a frame to bird's-eye using the locked homography."""
        if self.calib is None:
            return None
        return cv2.warpPerspective(frame, self.calib.H, self.calib.dst_size,
                                   flags=cv2.INTER_LINEAR)

    def check_deviation(self, frame: np.ndarray, settings: Settings) -> float:
        """Watchdog: re-detect felt corners and reconcile with the locked ones.

        The table/camera don't move during play, so this is conservative and
        self-correcting, not twitchy:
          - a poor/absent re-detect (a hand/cue/person over a corner) is IGNORED —
            it's not evidence the table moved, so the lock is never lost to it;
          - a close re-detect eases the locked corners toward it (averages out
            detection noise, fixes tiny initial-lock error) — no jumps;
          - only a LARGE, SUSTAINED disagreement flags ``deviated`` for a relock.
        Returns RMSE px (0.0 when the re-detect was unusable)."""
        if self.calib is None:
            return 0.0
        felt = detect_felt(frame, self.calib.felt)
        # Occlusion guard: no/weak felt this frame => something's over the table.
        # Keep the lock, don't count it, don't average garbage in.
        if not felt.has_corners or felt.area_ratio < 0.04:
            self._consecutive = max(0, self._consecutive - 1)
            return 0.0
        rmse = float(np.sqrt(np.mean(np.sum((felt.corners - self.calib.corners) ** 2, axis=1))))
        if rmse <= self._settle_px:
            # agrees with the lock -> gently average the corners toward it
            self._consecutive = 0
            self.deviated = False
            self._ema_corners(felt.corners, settings)
        elif rmse > self._deviation_px:
            self._consecutive += 1
        else:
            # moderate disagreement (likely partial occlusion) -> decay, don't trip
            self._consecutive = max(0, self._consecutive - 1)
        if self._consecutive >= self._deviation_frames:
            if not self.deviated:
                log.info("Calibration deviated (table moved?): RMSE %.1f px", rmse)
            self.deviated = True
        return rmse

    def _ema_corners(self, detected: np.ndarray, settings: Settings) -> None:
        """Ease the locked corners toward a trusted re-detection and recompute the
        homography. Slow (corner_ema) so the lock is stable; the dst rectangle is
        unchanged, so the table model/size stay put."""
        try:
            old = np.asarray(self.calib.corners, np.float32)
            new = ((1.0 - self._corner_ema) * old
                   + self._corner_ema * np.asarray(detected, np.float32)).astype(np.float32)
            w, h = self.calib.dst_size
            p = float(settings.rectify.pad_px)
            dst_quad = np.array([[p, p], [w - 1 - p, p],
                                 [w - 1 - p, h - 1 - p], [p, h - 1 - p]], np.float32)
            H = cv2.getPerspectiveTransform(new, dst_quad)
            Hinv = np.linalg.inv(H)
        except (cv2.error, np.linalg.LinAlgError):
            return
        self.calib.corners = new
        self.calib.H = H
        self.calib.Hinv = Hinv

    def clear(self) -> None:
        self.calib = None
        self.deviated = False
        self._consecutive = 0

    # ------------------------------------------------------------------ #
    # Persistence — reuse the locked table across launches
    # ------------------------------------------------------------------ #
    def save(self, path: Path, source: str, frame_shape: tuple, settings: Settings) -> None:
        if self.calib is None:
            return
        h, w = frame_shape[:2]
        payload = {
            "source": source,
            "frame_w": int(w), "frame_h": int(h),
            "pad": settings.rectify.pad_px,
            "pocket_frac": settings.table.pocket_radius_frac,
            "dst_size": list(self.calib.dst_size),
            "corners": self.calib.corners.tolist(),
            "H": self.calib.H.tolist(),
            "Hinv": self.calib.Hinv.tolist(),
            "felt": asdict(self.calib.felt),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            log.warning("Could not save calibration: %s", exc)

    def try_load(self, path: Path, source: str, frame_shape: tuple,
                 settings: Settings | None = None) -> bool:
        """Restore a saved calibration if it matches this source + resolution."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        h, w = frame_shape[:2]
        if data.get("source") != source or data.get("frame_w") != int(w) \
                or data.get("frame_h") != int(h):
            return False
        try:
            dst_size = tuple(data["dst_size"])
            felt_keys = {f.name for f in fields(FeltSettings)}
            felt = FeltSettings(**{k: v for k, v in data["felt"].items() if k in felt_keys})
            inset = settings.table.nose_inset_frac if settings is not None else 0.0
            table = TableModel.from_rect(dst_size, data["pad"], data["pocket_frac"],
                                         nose_inset_frac=inset)
            self.calib = Calibration(
                corners=np.array(data["corners"], dtype=np.float32),
                H=np.array(data["H"], dtype=np.float64),
                Hinv=np.array(data["Hinv"], dtype=np.float64),
                dst_size=dst_size, table=table, rect_mask=None, felt=felt,
            )
            self.deviated = False
            self._consecutive = 0
            log.info("Restored saved calibration for source %s (%dx%d)", source, w, h)
            return True
        except (KeyError, ValueError, TypeError) as exc:
            log.warning("Saved calibration invalid: %s", exc)
            return False
