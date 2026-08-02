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


# Billiard cloth is strongly saturated; the things that get laid over a table
# are not. Measured on the rig: felt S=184, the grey vinyl table cover S=61
# (p90 68). 90 sits clear of both.
_CLOTH_MIN_SAT = 90


def cloth_saturation(frame: np.ndarray, mask: np.ndarray | None) -> float:
    """Median HSV saturation of the region a felt detection claims is cloth."""
    if frame is None or frame.size == 0 or mask is None or mask.size == 0:
        return 0.0
    sel = mask > 0
    if not np.any(sel):
        return 0.0
    sat = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
    return float(np.median(sat[sel]))


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
        self._pending: list[np.ndarray] = []     # corner sets gathering consensus
        self._refused_not_cloth = False          # log the cover transition once

    @property
    def is_calibrated(self) -> bool:
        return self.calib is not None

    def calibrate(self, frame: np.ndarray, settings: Settings) -> bool:
        """Run felt detection and lock the table. Returns success.

        The lock is a MULTI-FRAME CONSENSUS (table.calib_consensus_frames): the
        per-corner median over the last N successful detections, with outlier
        frames rejected. A single frame with a person leaning over the table
        (which pulls a corner tens of px inward) can therefore never become the
        session's homography — the exact failure seen on real footage."""
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
        # A COVERED TABLE still detects "a big flat quadrilateral", and because
        # the felt colour is auto-estimated from the frame it happily estimates
        # "grey cloth" and locks onto the cover — then SAVES that over a lock
        # that was correct all day. Seen twice on the rig. Cloth is saturated and
        # covers are not, so refuse anything that isn't cloth-coloured and keep
        # whatever lock we already had.
        sat = cloth_saturation(frame, felt.mask)
        if sat < _CLOTH_MIN_SAT:
            # Log the TRANSITION only. calibrate() is retried every frame while
            # unlocked, so logging per-frame buried the log at ~9 lines/second.
            if not self._refused_not_cloth:
                log.info("Calibration refused: region is not cloth (median "
                         "saturation %.0f < %d) — table covered or lights off?",
                         sat, _CLOTH_MIN_SAT)
                self._refused_not_cloth = True
            return False
        if self._refused_not_cloth:
            log.info("Cloth is visible again (saturation %.0f) — calibrating", sat)
            self._refused_not_cloth = False

        corners = self._consensus(felt.corners, settings)
        if corners is None:
            return False  # still gathering frames — caller retries next frame
        # refine=False: the base homography (clean felt corners -> forced 2:1
        # rectangle) is stable and correct. The Hough-line "square-up" refinement
        # fires intermittently and, on a bad frame, skews the rectangle
        # non-uniformly (fat far rail, egg-shaped balls, pockets off their marks).
        # Fewer moving parts = a deterministic, undistorted bird's-eye. (review: calib-5)
        rect = rectify_tabletop(frame, felt.mask, corners,
                                pad_px=settings.rectify.pad_px,
                                aspect=settings.rectify.aspect, refine=False)
        if not rect.ok:
            log.info("Calibration failed: rectification not ok")
            return False
        table = TableModel.from_rect(rect.dst_size, settings.rectify.pad_px,
                                     settings.table.pocket_radius_frac,
                                     nose_inset_frac=settings.table.computed_nose_inset_frac())
        self.calib = Calibration(
            corners=corners, H=rect.H, Hinv=rect.Hinv,
            dst_size=rect.dst_size, table=table, rect_mask=rect.rectified_mask,
            felt=felt_settings,
        )
        self.deviated = False
        self._consecutive = 0
        log.info("Calibrated: dst_size=%s, %d pockets", rect.dst_size, len(table.pockets))
        return True

    def _consensus(self, corners: np.ndarray, settings: Settings) -> np.ndarray | None:
        """Accumulate corner detections; return the outlier-robust per-corner
        median once enough agreeing frames are in, else None (keep gathering)."""
        n = max(1, int(getattr(settings.table, "calib_consensus_frames", 1)))
        if n == 1:
            return np.asarray(corners, np.float32)
        self._pending.append(np.asarray(corners, np.float32))
        if len(self._pending) > n:
            self._pending.pop(0)
        if len(self._pending) < n:
            return None
        stack = np.stack(self._pending)
        med = np.median(stack, axis=0)
        errs = np.sqrt(np.mean(np.sum((stack - med) ** 2, axis=2), axis=1))
        good = stack[errs <= 20.0]
        if len(good) < n // 2 + 1:
            # no stable majority (someone moving over the table) — drop the
            # oldest and keep gathering
            self._pending.pop(0)
            return None
        self._pending.clear()
        return np.median(good, axis=0).astype(np.float32)

    def validate_against(self, frame: np.ndarray, settings: Settings) -> bool:
        """One-shot check that the CURRENT lock still matches what's actually in
        the frame — used right after a restore, so a stale saved lock (moved
        camera, or a lock from a different scene) is rejected immediately
        instead of after ~12 watchdog checks. Returns True to keep the lock."""
        if self.calib is None:
            return False
        felt = detect_felt(frame, self.calib.felt)
        if not felt.has_corners or felt.area_ratio < 0.04:
            return True  # can't judge (table occluded right now) — watchdog guards
        rmse = float(np.sqrt(np.mean(np.sum(
            (felt.corners - self.calib.corners) ** 2, axis=1))))
        if rmse > self._deviation_px:
            log.info("Restored calibration failed live validation (RMSE %.1f px)", rmse)
            return False
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
        # Keep the lock, don't count it, don't average garbage in. The saturation
        # test covers the whole-table case (a cover laid over it), which detects
        # perfectly well as a quadrilateral and would otherwise drag the corners.
        if (not felt.has_corners or felt.area_ratio < 0.04
                or cloth_saturation(frame, felt.mask) < _CLOTH_MIN_SAT):
            self._consecutive = max(0, self._consecutive - 1)
            return 0.0
        dist = np.linalg.norm(felt.corners - self.calib.corners, axis=1)
        rmse = float(np.sqrt(np.mean(dist ** 2)))
        # RMSE over four corners HIDES a single badly-placed one: a rail mis-fitted
        # by 33px alongside three good corners averages down to RMSE ~17. Judge
        # agreement on the WORST corner, escalation on the RMSE.
        worst = float(dist.max())
        if worst <= self._settle_px:
            # agrees with the lock -> gently average the corners toward it
            self._consecutive = 0
            self.deviated = False
            self._ema_corners(felt.corners, settings)
        elif rmse > self._deviation_px:
            # a GROSSLY wrong lock (not "the table drifted" but "this lock is
            # nonsense") counts triple, so it's abandoned in ~4 checks (~4 s)
            # instead of 12 — the user shouldn't watch a wrong overlay for 12 s
            self._consecutive += 3 if rmse > 4 * self._deviation_px else 1
        else:
            # THE DEAD BAND — too far to average in, not far enough to relock.
            # This used to do nothing at all, so a lock wrong by ~30px on ONE
            # corner sat there forever: "stable, but wrong". Measured on the rig,
            # a top-right corner 33px inside the cushion gave RMSE 17, landing
            # squarely between settle (12) and deviation (30) and freezing.
            # Ease toward the re-detect at half rate: it converges over a few
            # seconds without the twitchiness a full-rate correction would bring.
            self._consecutive = max(0, self._consecutive - 1)
            self._ema_corners(felt.corners, settings, rate=self._corner_ema * 0.5)
        if self._consecutive >= self._deviation_frames:
            if not self.deviated:
                log.info("Calibration deviated (table moved?): RMSE %.1f px", rmse)
            self.deviated = True
        return rmse

    def _ema_corners(self, detected: np.ndarray, settings: Settings,
                     rate: float | None = None) -> None:
        """Ease the locked corners toward a trusted re-detection and recompute the
        homography. Slow (corner_ema) so the lock is stable; the dst rectangle is
        unchanged, so the table model/size stay put. ``rate`` overrides the EMA
        weight — the dead-band caller uses a slower one."""
        k = self._corner_ema if rate is None else rate
        try:
            old = np.asarray(self.calib.corners, np.float32)
            new = ((1.0 - k) * old
                   + k * np.asarray(detected, np.float32)).astype(np.float32)
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
        self._pending.clear()

    # ------------------------------------------------------------------ #
    # Persistence — reuse the locked table across launches
    # ------------------------------------------------------------------ #
    def save(self, path: Path, source: str, frame_shape: tuple, settings: Settings) -> None:
        """Persist the lock, keyed BY SOURCE. One shared document used to hold a
        single entry, so starting the laptop camera overwrote the pool table's
        lock — the app then restored a lock for a completely different scene."""
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
        entries = self._read_entries(path)
        entries.pop(source, None)   # re-insert at the end (newest-last order)
        entries[source] = payload
        while len(entries) > 8:     # cap growth; drop the oldest source
            entries.pop(next(iter(entries)))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"version": 2, "entries": entries}, indent=2),
                            encoding="utf-8")
        except OSError as exc:
            log.warning("Could not save calibration: %s", exc)

    @staticmethod
    def _read_entries(path: Path) -> dict:
        """Read the per-source entry map; migrate a legacy single-entry file."""
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(doc, dict):
            return {}
        entries = doc.get("entries")
        if isinstance(entries, dict):
            return entries
        # legacy v1: the whole doc is one entry — keep it under its source key
        if doc.get("source") and doc.get("corners"):
            return {str(doc["source"]): doc}
        return {}

    def try_load(self, path: Path, source: str, frame_shape: tuple,
                 settings: Settings | None = None) -> bool:
        """Restore a saved calibration if it matches this source + resolution."""
        data = self._read_entries(path).get(source)
        if data is None:
            return False
        h, w = frame_shape[:2]
        if data.get("frame_w") != int(w) or data.get("frame_h") != int(h):
            return False
        try:
            dst_size = tuple(data["dst_size"])
            felt_keys = {f.name for f in fields(FeltSettings)}
            felt = FeltSettings(**{k: v for k, v in data["felt"].items() if k in felt_keys})
            inset = settings.table.computed_nose_inset_frac() if settings is not None else 0.0
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
