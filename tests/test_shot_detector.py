"""Shot/make/miss state-machine tests for the motion-gated detector.

These drive the hardened gates directly: motion energy (start/stop), cue
presence, ball travel, and pocket approach+vanish for the outcome.
"""

from billiards_trainer.config import BallSettings, DetectionSettings
from billiards_trainer.core.geometry import TableModel
from billiards_trainer.core.types import BallClass, Track
from billiards_trainer.events.shot_detector import ShotDetector, ShotOutcome

TABLE = TableModel.from_rect((488, 895), 40)
BR = next(p for p in TABLE.pockets if p.name == "bottom-right")  # (448, 855)


def det() -> ShotDetector:
    # fusion off here so these gate tests can drive plain motion energy; the
    # multi-modal fusion is validated separately (test_fusion_*).
    d = DetectionSettings(warmup_seconds=0.0, cooldown_seconds=0.0, use_fusion=False,
                          motion_active=0.4, motion_quiet=0.2, strike_frames=3,
                          min_travel_px=50.0, pocket_frames=5)
    # time floors zeroed: these short synthetic sequences test outcome logic;
    # the physical-duration floors are covered in test_shot_arm_gate.py
    return ShotDetector(d, BallSettings(), settle_frames=3, min_shot_frames=2,
                        min_shot_s=0.0, settle_s=0.0)


def trk(tid, x, y, cls=BallClass.SOLID):
    return Track(id=tid, x=x, y=y, radius=10, cls=cls)


def cue(x=200, y=400):
    return trk(1, x, y, cls=BallClass.CUE)


def feed(d, frames):
    """frames: list of (tracks, motion)."""
    event = None
    for i, (tracks, motion) in enumerate(frames):
        ev = d.update(tracks, TABLE, t=i * 0.033, motion=motion)
        if ev is not None:
            event = ev
    return event


def test_make_when_object_approaches_pocket_and_vanishes():
    d = det()
    frames = []
    # settle (establish rest) — quiet
    for _ in range(3):
        frames.append(([cue(), trk(2, 250, 420)], 0.0))
    # active: object travels to the bottom-right pocket (motion high)
    for x, y in [(300, 520), (360, 640), (410, 760), (446, 853)]:
        frames.append(([cue(), trk(2, x, y)], 1.0))
    # object drops in (gone); table goes quiet
    for _ in range(5):
        frames.append(([cue()], 0.0))
    event = feed(d, frames)
    assert event is not None
    assert event.outcome == ShotOutcome.MAKE
    assert event.num_pocketed == 1
    assert event.target_pocket == "bottom-right"
    assert event.max_travel >= 50


def test_no_shot_when_table_is_idle():
    d = det()
    # balls present but no motion at all (Joe's "counter creeps while I sit") —
    # must never fire.
    for _ in range(60):
        ev = d.update([cue(), trk(2, 250, 420)], TABLE, t=0.0, motion=0.05)
        assert ev is None
    assert d.state == "settled"


def test_discard_when_no_real_travel():
    d = det()
    frames = [([cue(), trk(2, 250, 420)], 0.0)] * 3
    # motion spikes (flicker) but the ball barely moves (<50px) and never pockets
    for _ in range(5):
        frames.append(([cue(), trk(2, 255, 425)], 1.0))
    for _ in range(5):
        frames.append(([cue(), trk(2, 255, 425)], 0.0))
    event = feed(d, frames)
    assert event is None  # no meaningful travel -> not a shot


def test_no_detection_without_cue_when_required():
    d = det()
    d.det.require_cue = True
    frames = [([trk(2, 250, 420)], 0.0)] * 3
    for x, y in [(300, 520), (360, 640), (410, 760), (446, 853)]:
        frames.append(([trk(2, x, y)], 1.0))   # no cue ball present
    frames += [([], 0.0)] * 5
    assert feed(d, frames) is None


def test_miss_when_motion_but_no_pocket():
    d = det()
    frames = [([cue(), trk(2, 250, 420)], 0.0)] * 3
    # object travels far across the table but stops in open play (no pocket)
    for x, y in [(250, 480), (250, 560), (250, 640), (250, 700)]:
        frames.append(([cue(), trk(2, x, y)], 1.0))
    for _ in range(5):
        frames.append(([cue(), trk(2, 250, 700)], 0.0))
    event = feed(d, frames)
    assert event is not None
    assert event.outcome == ShotOutcome.MISS
