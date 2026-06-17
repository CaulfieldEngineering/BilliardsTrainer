"""End-to-end pipeline test on the synthetic demo source (no Qt, CI-safe).

Drives the deterministic demo table through the real pipeline and asserts the
scripted make-shot is detected at the bottom-right pocket. This is the
regression guard for the whole vision→event chain working together.
"""

import numpy as np

from billiards_trainer.capture.camera import DemoSource
from billiards_trainer.config import Settings
from billiards_trainer.events.shot_detector import ShotOutcome
from billiards_trainer.vision.pipeline import Pipeline


def test_temporal_median_stabilize():
    """The noise-suppression preprocessor passes frames through until its window
    fills, then returns the per-pixel median (so a single outlier frame can't
    move the detector); disabling it is a clean passthrough."""
    s = Settings()
    s.balls.temporal_median = True
    s.balls.temporal_median_frames = 3
    p = Pipeline(s)
    a = np.full((8, 8, 3), 100, np.uint8)
    assert np.array_equal(p._stabilize(a), a)   # buffer filling -> passthrough
    assert np.array_equal(p._stabilize(a), a)
    assert np.array_equal(p._stabilize(a), a)   # median of 3 identical == a
    b = np.full((8, 8, 3), 200, np.uint8)
    med = p._stabilize(b)                        # window [a,a,b] -> median 100
    assert int(med.mean()) == 100               # the outlier is rejected
    s.balls.temporal_median = False
    assert np.array_equal(p._stabilize(b), b)    # disabled -> passthrough
    assert p._frame_ring is None


def test_demo_calibrates_and_detects_make():
    settings = Settings()
    # This regression guards the vision->event (make/miss) chain on the synthetic
    # demo, which was authored for the classical rectified detector. Pin it to the
    # 'legacy' detector; the new raw-frame strategies (simple_blob default) are
    # validated on real footage via the eval harness, not the synthetic make.
    settings.balls.live_strategy = "legacy"
    settings.detection.warmup_seconds = 3.0   # > MOG2 bg warm-up so fusion's fg is ready
    settings.detection.cooldown_seconds = 0.5
    src = DemoSource()
    pipe = Pipeline(settings)

    makes = []
    calibrated = False
    for i in range(560):  # a couple of cycles past warm-up
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
