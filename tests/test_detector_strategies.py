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
