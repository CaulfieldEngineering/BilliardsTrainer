"""End-to-end pipeline test on the synthetic demo source (no Qt, CI-safe).

Drives the deterministic demo table through the real pipeline and asserts the
scripted make-shot is detected at the bottom-right pocket. This is the
regression guard for the whole vision→event chain working together.
"""

from billiards_trainer.capture.camera import DemoSource
from billiards_trainer.config import Settings
from billiards_trainer.events.shot_detector import ShotOutcome
from billiards_trainer.vision.pipeline import Pipeline


def test_demo_calibrates_and_detects_make():
    settings = Settings()
    settings.detection.warmup_seconds = 0.5   # don't wait the full warm-up in tests
    settings.detection.cooldown_seconds = 0.5
    src = DemoSource()
    pipe = Pipeline(settings)

    makes = []
    calibrated = False
    for i in range(480):  # ~2 full cycles past warm-up
        frame = src.read()
        res = pipe.process(frame, t=i * 0.033)
        calibrated = calibrated or pipe.calib.is_calibrated
        if res.shot_event and res.shot_event.outcome == ShotOutcome.MAKE:
            makes.append(res.shot_event)

    assert calibrated, "demo table should calibrate on the first frame"
    assert len(makes) >= 1, "scripted make-shot should be detected"
    assert makes[0].target_pocket == "bottom-right"
    assert makes[0].num_pocketed == 1


def test_demo_idle_produces_no_false_shots():
    """The core fix: a static scene (no real motion) must never count a shot."""
    import numpy as np

    settings = Settings()
    settings.detection.warmup_seconds = 0.0
    src = DemoSource()
    pipe = Pipeline(settings)
    # calibrate on a demo frame, then feed the SAME frame repeatedly (zero motion)
    frame = src.read()
    pipe.process(frame, t=0.0)
    static = np.array(frame)
    events = 0
    for i in range(120):
        res = pipe.process(static, t=1.0 + i * 0.033)
        if res.shot_event is not None:
            events += 1
    assert events == 0, "a frozen frame must not generate shots"


def test_demo_tracks_balls():
    settings = Settings()
    src = DemoSource()
    pipe = Pipeline(settings)
    seen = 0
    for i in range(30):
        res = pipe.process(src.read(), t=i * 0.033)
        seen = max(seen, res.n_balls)
    assert seen >= 1, "should track at least the cue/object balls during settle"
