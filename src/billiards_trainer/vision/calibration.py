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

import logging
from dataclasses import dataclass

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
    rect_mask: np.ndarray
    felt: FeltSettings  # effective felt colour key (possibly auto-estimated)


class CalibrationManager:
    def __init__(self, deviation_px: float = 18.0, deviation_frames: int = 8):
        self.calib: Calibration | None = None
        self.deviated: bool = False
        self._deviation_px = deviation_px
        self._deviation_frames = deviation_frames
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
        rect = rectify_tabletop(frame, felt.mask, felt.corners,
                                pad_px=settings.rectify.pad_px,
                                aspect=settings.rectify.aspect, refine=True)
        if not rect.ok:
            log.info("Calibration failed: rectification not ok")
            return False
        table = TableModel.from_rect(rect.dst_size, settings.rectify.pad_px,
                                     settings.table.pocket_radius_frac)
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
        """Re-detect felt corners and compare to the locked ones. Updates the
        ``deviated`` flag using a consecutive-frame debounce. Returns RMSE px."""
        if self.calib is None:
            return 0.0
        felt = detect_felt(frame, self.calib.felt)
        if not felt.has_corners:
            return 0.0
        rmse = float(np.sqrt(np.mean(np.sum((felt.corners - self.calib.corners) ** 2, axis=1))))
        if rmse > self._deviation_px:
            self._consecutive += 1
        else:
            self._consecutive = 0
        if self._consecutive >= self._deviation_frames:
            if not self.deviated:
                log.info("Calibration deviated: RMSE %.1f px", rmse)
            self.deviated = True
        return rmse

    def clear(self) -> None:
        self.calib = None
        self.deviated = False
        self._consecutive = 0
