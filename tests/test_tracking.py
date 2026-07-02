"""Tracker tests using synthetic detections (deterministic, detector-independent)."""

from billiards_trainer.vision.tracking import BallTracker
from billiards_trainer.vision.types import BallClass, Detection


def det(x, y, cls=BallClass.SOLID, r=10):
    return Detection(x=x, y=y, radius=r, cls=cls)


def test_track_confirms_after_min_hits():
    tr = BallTracker(min_hits=3)
    short = 400
    for _ in range(2):
        out = tr.update([det(100, 100)], short)
        assert out == []          # not yet confirmed
    out = tr.update([det(100, 100)], short)
    assert len(out) == 1          # confirmed on the 3rd hit


def test_radius_outlier_is_rejected_on_confirmed_track():
    """Once a track is established, a single wildly-different detection radius
    (sensor-noise outlier) must NOT pump the track's size — the cause of the
    'ball keeps changing size on a still scene' complaint."""
    tr = BallTracker(min_hits=3)
    short = 400
    for _ in range(4):                       # establish a stable radius-10 track
        tr.update([det(100, 100, r=10)], short)
    out = tr.update([det(100, 100, r=22)], short)   # +120% radius spike
    assert out[0].radius < 13                # held near 10, not blown up to ~22
    # a moderate change is still allowed to drift in slowly
    for _ in range(8):
        out = tr.update([det(100, 100, r=12)], short)
    assert 10 < out[0].radius <= 12.5


def test_adaptive_smoothing_beats_fixed_alpha_on_both_axes():
    """The adaptive position alpha must keep stillness jitter as low as a heavily-
    smoothed tracker AND follow fast motion with much less lag — the fix for the
    'cue ball lags a foot' report without sacrificing the still-scene tuning."""
    import statistics as st

    import numpy as np

    def lag(tr, vx=8.0):
        short = 400.0
        tx, ty = 100.0, 200.0
        out = []
        for _ in range(12):
            tr.update([det(tx, ty)], short)
        for _ in range(25):
            tx += vx
            o = tr.update([det(tx, ty)], short)
            if o:
                out.append(((o[0].x - tx) ** 2 + (o[0].y - ty) ** 2) ** 0.5)
        return st.mean(out[-8:])

    def jitter(tr):
        rng = np.random.default_rng(0)
        short = 400.0
        pos = []
        for i in range(40):
            o = tr.update([det(100 + rng.normal(0, 1.2), 200 + rng.normal(0, 1.2))], short)
            if o and i > 10:
                pos.append((o[0].x, o[0].y))
        xs = [p[0] for p in pos]
        ys = [p[1] for p in pos]
        return (st.pstdev(xs) ** 2 + st.pstdev(ys) ** 2) ** 0.5

    def mk(slow, fast):
        return BallTracker(min_hits=2, pos_alpha_slow=slow, pos_alpha_fast=fast)

    smooth_lag, smooth_jit = lag(mk(0.15, 0.15)), jitter(mk(0.15, 0.15))
    adapt_lag, adapt_jit = lag(mk(0.15, 0.85)), jitter(mk(0.15, 0.85))
    # adaptive keeps the smooth tracker's low jitter ...
    assert adapt_jit <= smooth_jit + 0.05
    # ... but follows motion with much less lag
    assert adapt_lag < smooth_lag * 0.7


def test_id_persists_across_motion():
    tr = BallTracker(min_hits=2)
    short = 400
    tr.update([det(100, 100)], short)
    out = tr.update([det(112, 100)], short)
    assert len(out) == 1
    tid = out[0].id
    out = tr.update([det(124, 100)], short)
    assert out[0].id == tid       # same identity
    assert out[0].vx > 5          # velocity picked up rightward motion


def test_two_balls_keep_separate_ids():
    tr = BallTracker(min_hits=2)
    short = 400
    tr.update([det(100, 100), det(300, 300)], short)
    out = tr.update([det(105, 100), det(305, 300)], short)
    ids = sorted(t.id for t in out)
    assert len(ids) == 2 and ids[0] != ids[1]


def test_track_survives_brief_occlusion():
    tr = BallTracker(min_hits=2, max_misses=10)
    short = 400
    tr.update([det(100, 100)], short)
    out = tr.update([det(110, 100)], short)
    tid = out[0].id
    # 3 frames with no detection (occluded) — track should coast, not die
    for _ in range(3):
        tr.update([], short)
    out = tr.update([det(150, 100)], short)
    assert any(t.id == tid for t in out)


def test_track_dies_after_max_misses():
    tr = BallTracker(min_hits=2, max_misses=3)
    short = 400
    tr.update([det(100, 100)], short)
    tr.update([det(100, 100)], short)
    for _ in range(5):
        tr.update([], short)
    assert tr.tracks == []


def test_number_unique_across_tracks():
    """One ball number must never be committed on two live tracks — the
    weaker claimant renders as unknown (measured on real footage: duplicate
    numbers were alive in ~80% of frames before arbitration)."""
    tr = BallTracker(min_hits=2)
    short = 400
    # two well-separated balls both repeatedly detected as number 5; the one
    # at (100,100) accumulates more evidence first
    for _ in range(4):
        tr.update([Detection(100, 100, 10, cls=BallClass.SOLID, number=5)], short)
    for _ in range(3):
        out = tr.update([
            Detection(100, 100, 10, cls=BallClass.SOLID, number=5),
            Detection(300, 300, 10, cls=BallClass.SOLID, number=5),
        ], short)
    fives = [t for t in out if t.number == 5]
    assert len(fives) == 1, f"number 5 alive on {len(fives)} tracks"
    assert len(out) == 2, "the losing track must survive (as unknown), not die"
    assert fives[0].x < 200, "the stronger-evidence track must keep the number"
