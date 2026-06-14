"""Shot/make/miss state-machine tests with hand-built track sequences."""

from billiards_trainer.config import BallSettings
from billiards_trainer.events.shot_detector import ShotDetector, ShotOutcome
from billiards_trainer.vision.geometry import TableModel
from billiards_trainer.vision.types import BallClass, Track

TABLE = TableModel.from_rect((488, 895), 40)
# bottom-right pocket is at (x1, y1) = (448, 855)
BR = next(p for p in TABLE.pockets if p.name == "bottom-right")


def trk(tid, x, y, vx=0.0, vy=0.0, cls=BallClass.SOLID):
    return Track(id=tid, x=x, y=y, radius=10, vx=vx, vy=vy, cls=cls)


def feed(det, frames, t0=0.0, dt=0.033):
    event = None
    for i, tracks in enumerate(frames):
        ev = det.update(tracks, TABLE, t0 + i * dt)
        if ev is not None:
            event = ev
    return event


def test_make_when_object_ball_pocketed():
    d = ShotDetector(BallSettings(), settle_frames=4, min_shot_frames=2)
    cue = lambda: trk(1, 200, 400, cls=BallClass.CUE)  # noqa: E731
    frames = [
        [cue(), trk(2, 250, 420)],                       # settled
        [cue(), trk(2, 300, 520, vx=20, vy=20)],         # moving
        [cue(), trk(2, 380, 650, vx=20, vy=20)],
        [cue(), trk(2, 446, 853, vx=20, vy=20)],         # inside BR pocket zone
    ]
    frames += [[cue()] for _ in range(6)]                # object gone, cue stopped
    event = feed(d, frames)
    assert event is not None
    assert event.outcome == ShotOutcome.MAKE
    assert event.num_pocketed == 1
    assert event.target_pocket == "bottom-right"


def test_miss_when_nothing_pocketed():
    d = ShotDetector(BallSettings(), settle_frames=4, min_shot_frames=2)
    cue = lambda: trk(1, 200, 400, cls=BallClass.CUE)  # noqa: E731
    frames = [
        [cue(), trk(2, 250, 420)],
        [cue(), trk(2, 250, 480, vx=0, vy=18)],
        [cue(), trk(2, 250, 560, vx=0, vy=18)],
        [cue(), trk(2, 250, 600)],                       # stopped mid-table
    ]
    frames += [[cue(), trk(2, 250, 600)] for _ in range(6)]
    event = feed(d, frames)
    assert event is not None
    assert event.outcome == ShotOutcome.MISS
    assert event.num_pocketed == 0


def test_scratch_when_cue_pocketed():
    d = ShotDetector(BallSettings(), settle_frames=4, min_shot_frames=2)
    frames = [
        [trk(1, 300, 700, cls=BallClass.CUE), trk(2, 250, 420)],
        [trk(1, 360, 760, vx=20, vy=20, cls=BallClass.CUE), trk(2, 250, 420)],
        [trk(1, 446, 853, vx=20, vy=20, cls=BallClass.CUE), trk(2, 250, 420)],  # cue in pocket
    ]
    frames += [[trk(2, 250, 420)] for _ in range(6)]     # cue gone, object stationary
    event = feed(d, frames)
    assert event is not None
    assert event.outcome == ShotOutcome.SCRATCH
    assert event.cue_scratch is True


def test_no_event_while_settled():
    d = ShotDetector(BallSettings(), settle_frames=4)
    cue = trk(1, 200, 400, cls=BallClass.CUE)
    for _ in range(10):
        ev = d.update([cue, trk(2, 250, 420)], TABLE, 0.0)
        assert ev is None
    assert d.state == "settled"
