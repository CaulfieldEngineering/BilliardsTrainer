"""Regression tests for the v0.1.5 reliability work: calibration persistence,
shot-clock non-interference, felt picking, and tracker robustness."""

import cv2
import numpy as np

from billiards_trainer.capture.camera import DemoSource
from billiards_trainer.config import FeltSettings, Settings, ShotClockSettings
from billiards_trainer.game.shot_clock import ShotClock
from billiards_trainer.vision.calibration import CalibrationManager
from billiards_trainer.vision.felt import felt_from_point
from billiards_trainer.vision.tracking import BallTracker
from billiards_trainer.vision.types import BallClass, Detection


# ---- calibration persistence --------------------------------------------- #
def test_calibration_save_and_restore(tmp_path):
    frame = DemoSource().read()
    settings = Settings()
    mgr = CalibrationManager()
    assert mgr.calibrate(frame, settings)
    path = tmp_path / "calibration.json"
    mgr.save(path, "demo", frame.shape, settings)
    assert path.exists()

    restored = CalibrationManager()
    assert restored.try_load(path, "demo", frame.shape)
    assert restored.is_calibrated
    assert np.allclose(restored.calib.corners, mgr.calib.corners, atol=0.5)
    assert restored.calib.dst_size == mgr.calib.dst_size


def test_calibration_rejects_mismatched_source(tmp_path):
    frame = DemoSource().read()
    settings = Settings()
    mgr = CalibrationManager()
    mgr.calibrate(frame, settings)
    path = tmp_path / "calibration.json"
    mgr.save(path, "demo", frame.shape, settings)

    other = CalibrationManager()
    assert not other.try_load(path, "0", frame.shape)            # different source
    assert not other.try_load(path, "demo", (10, 10, 3))         # different resolution


# ---- shot clock must not interfere with sandbox -------------------------- #
def test_shot_clock_disabled_is_noop():
    clock = ShotClock(ShotClockSettings(enabled=False, seconds=30))
    clock.start(0.0)
    assert not clock.running
    assert clock.remaining(100.0) == 30.0      # never counts down
    assert clock.poll(100.0) == ""             # never fires warn/expire
    assert not clock.is_warning(100.0)


def test_shot_clock_enabled_counts_down():
    clock = ShotClock(ShotClockSettings(enabled=True, seconds=10, warn_seconds=3))
    clock.start(0.0)
    assert clock.running
    assert clock.remaining(4.0) == 6.0
    assert clock.poll(8.0) == "warn"
    assert clock.poll(11.0) == "expired"


# ---- click-to-pick felt -------------------------------------------------- #
def test_felt_from_point_keys_on_clicked_pixel():
    hsv = np.full((300, 300, 3), (100, 150, 200), np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    fs = felt_from_point(bgr, 150, 150, sensitivity=82)
    assert isinstance(fs, FeltSettings)
    assert abs(fs.picked_hsv[0] - 100) <= 2
    assert fs.h_min <= 100 <= fs.h_max


# ---- tracker robustness -------------------------------------------------- #
def test_tracker_handles_off_frame_detection():
    tr = BallTracker(min_hits=2)
    # negative / out-of-frame coordinates must not crash association
    tr.update([Detection(-40, -40, 10, cls=BallClass.SOLID)], 400)
    out = tr.update([Detection(-40, -40, 10, cls=BallClass.SOLID)], 400)
    assert isinstance(out, list)
    # then it vanishes — tracker ages it out without error once the keep-alive
    # budget (max_misses, raised to survive ~1s occlusion) is exceeded.
    for _ in range(tr.max_misses + 2):
        tr.update([], 400)
    assert tr.tracks == []
