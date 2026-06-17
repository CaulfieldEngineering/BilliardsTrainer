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


def test_median_frames_is_exact_and_fast_path():
    """The fast median-of-3 (max/min identity, ~18x faster than np.median at 1080p)
    must be bit-exact vs np.median, and handle the n=1/2/4 cases."""
    from billiards_trainer.vision.pipeline import _median_frames

    rng = np.random.default_rng(3)
    for _ in range(8):
        fs = [rng.integers(0, 256, (24, 24, 3), dtype=np.uint8) for _ in range(3)]
        ref = np.median(np.stack(fs, 0), 0).astype(np.uint8)
        assert np.array_equal(_median_frames(fs), ref)
    a = np.full((4, 4, 3), 10, np.uint8)
    assert np.array_equal(_median_frames([a]), a)                 # n=1 passthrough
    b = np.full((4, 4, 3), 20, np.uint8)
    assert int(_median_frames([a, b]).mean()) == 15               # n=2 average
    c = np.full((4, 4, 3), 200, np.uint8)
    five = [a, a, a, b, c]                                          # n=5 -> np.median
    assert np.array_equal(_median_frames(five),
                          np.median(np.stack(five, 0), 0).astype(np.uint8))


def test_detect_false_skips_detection_and_reuses_tracks():
    """A display-only cadence frame must NOT call the strategy detector and must
    reuse the existing tracks (this is what lets playback outrun detection)."""
    settings = Settings()
    settings.balls.live_strategy = "simple_blob"
    src = DemoSource()
    pipe = Pipeline(settings)
    pipe.detect_enabled = True
    # warm up: calibrate + populate tracks
    for _ in range(40):
        pipe.process(src.read(), 0.0, detect=True)
    before = [(t.id, round(t.x, 3), round(t.y, 3)) for t in pipe.tracker.tracks]

    calls = {"n": 0}
    orig = pipe._strategy.detect
    pipe._strategy.detect = lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or orig(*a, **k)
    res = pipe.process(src.read(), 0.0, detect=False)
    assert calls["n"] == 0, "strategy.detect must not run on a display-only frame"
    after = [(t.id, round(t.x, 3), round(t.y, 3)) for t in res.tracks]
    assert after == before, "tracks should be reused unchanged on a skip frame"


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
