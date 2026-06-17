"""Phase-1 detector-strategy framework tests (CI-safe — uses the committed demo clip).

Covers: strategies are discovered, the no-dep strategies run on a raw frame
without crashing and return raw-coord Detections, and the framework is robust to
a strategy that raises (self-healing contract).
"""

from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "eval" / "demo_clip.mp4"


def test_discovery_includes_nodep_strategies():
    from billiards_trainer.detector_strategies import discover
    names = set(discover())
    assert {"classical_rectified", "felt_mask_hough", "simple_blob"} <= names


def test_discovery_is_frozen_safe(monkeypatch):
    """REGRESSION: in a PyInstaller onefile, ``pkgutil.iter_modules(__path__)``
    finds nothing on disk. The old iter_modules-only discovery then returned {},
    which silently collapsed the live detector and the Settings dropdown to
    'legacy' in EVERY shipped build. The core strategies must come from static
    imports, so discovery still works when iter_modules yields nothing.
    """
    import pkgutil

    from billiards_trainer.detector_strategies import discover

    monkeypatch.setattr(pkgutil, "iter_modules", lambda *a, **k: iter(()))
    names = set(discover())
    # simple_blob is the shipped default — it MUST survive a frozen build.
    assert {"simple_blob", "felt_mask_hough", "classical_rectified", "cue_ball_white"} <= names


def test_cue_ball_white_detects_cue_on_synthetic_table():
    """The M2 cue-ball detector finds a white ball on felt and labels it CUE."""
    import numpy as np

    from billiards_trainer.detector_strategies import discover
    from billiards_trainer.vision.types import BallClass

    img = np.full((360, 640, 3), (110, 90, 40), np.uint8)  # bluish felt
    cv2.circle(img, (320, 180), 13, (245, 245, 245), -1, cv2.LINE_AA)  # cue ball

    dets = discover()["cue_ball_white"].detect(img, None)
    assert dets, "cue ball not detected"
    cue = min(dets, key=lambda d: (d.x - 320) ** 2 + (d.y - 180) ** 2)
    assert cue.cls == BallClass.CUE
    assert abs(cue.x - 320) < 6 and abs(cue.y - 180) < 6
    assert 8 < cue.radius < 20


def test_cue_ball_white_rejects_stripes_and_is_single_instance():
    """REGRESSION: a striped ball (white body + saturated colour band) must not be
    read as the cue, and exactly one cue is returned (no 'two cue balls at once')."""
    import cv2
    import numpy as np

    from billiards_trainer.detector_strategies import discover

    img = np.full((360, 640, 3), (110, 90, 40), np.uint8)
    cv2.circle(img, (160, 180), 14, (245, 245, 245), -1, cv2.LINE_AA)   # cue (pure white)
    cv2.circle(img, (420, 180), 14, (248, 248, 248), -1, cv2.LINE_AA)   # striped ball body
    band = np.zeros_like(img)
    cv2.rectangle(band, (406, 173), (434, 187), (40, 40, 210), -1)      # red stripe (BGR)
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(m, (420, 180), 14, 255, -1)
    img[(m > 0) & (band[:, :, 2] > 0)] = (40, 40, 210)

    dets = discover()["cue_ball_white"].detect(img, None)
    assert len(dets) == 1, "single-instance: exactly one cue ball"
    assert abs(dets[0].x - 160) < 12, "must pick the pure-white cue, not the stripe"


@pytest.mark.skipif(not FIXTURE.exists(), reason="demo fixture missing")
def test_nodep_strategies_run_on_raw_frame():
    from billiards_trainer.config import Settings
    from billiards_trainer.detector_strategies import discover
    from billiards_trainer.vision.calibration import CalibrationManager
    from billiards_trainer.vision.types import Detection

    cap = cv2.VideoCapture(str(FIXTURE))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
    ok, frame = cap.read()
    cap.release()
    assert ok

    mgr = CalibrationManager()
    calib = mgr.calib if mgr.calibrate(frame, Settings()) else None

    strategies = discover()
    for name in ("classical_rectified", "felt_mask_hough", "simple_blob"):
        dets = strategies[name].detect(frame, calib)
        assert isinstance(dets, list)
        assert all(isinstance(d, Detection) for d in dets)
        # raw-coord sanity: any detection sits within the frame
        h, w = frame.shape[:2]
        assert all(-5 <= d.x <= w + 5 and -5 <= d.y <= h + 5 for d in dets)


def test_strategy_that_raises_does_not_break_discovery():
    """A strategy raising in detect() is the harness's problem to catch — discovery
    itself must still succeed (self-healing contract at the framework level)."""
    from billiards_trainer.detector_strategies import DetectorStrategy, discover

    class Boom(DetectorStrategy):
        name = "boom"

        def detect(self, frame_bgr, calib):
            raise RuntimeError("kaboom")

    # discovery is independent of any one strategy's runtime behaviour
    assert "classical_rectified" in discover()
    with pytest.raises(RuntimeError):
        Boom().detect(None, None)
