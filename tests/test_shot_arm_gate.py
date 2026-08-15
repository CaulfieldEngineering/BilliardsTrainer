"""Carried-ball discrimination: hand-moved balls must not count as shots.

Audio validation caught the failure this pins down: between drill reps Joe
gathers and replaces balls by hand; the balls travel table lengths under his
hand and 'vanish' when lifted, which the detector scored as shots (one was
logged as a SCRATCH). The working discriminator is per-ball foreign
ADJACENCY: a carried ball is next to the hand for its whole displacement,
a struck ball leaves the stick within a frame or two — so only free
(non-adjacent) motion counts toward travel and pocket credit.

(A first attempt used whole-table arm dwell and failed in both directions:
the cue stick hovers over the felt through every real shot, and a placing
hand is too small to move the covered fraction. These tests encode the
survivor.)
"""

from billiards_trainer.config import Settings
from billiards_trainer.events.shot_detector import ShotDetector
from billiards_trainer.vision.geometry import TableModel
from billiards_trainer.vision.types import BallClass, Track


def _table() -> TableModel:
    return TableModel.from_rect((675, 1271), pad=0)


def _mk(tid, x, y, cls=BallClass.SOLID, number=3, active=True, vx=0.0):
    t = Track(id=tid, x=x, y=y, radius=15.0, cls=cls, number=number, vx=vx)
    t.active = active
    return t


def _detector() -> ShotDetector:
    s = Settings()
    s.detection.require_cue = False       # keep the test about the carried gate
    s.detection.use_fusion = False
    s.detection.warmup_seconds = 0.0
    s.detection.cooldown_seconds = 0.0
    return ShotDetector(s.detection, s.balls, settle_frames=3, min_shot_frames=2)


def _run(det, frames):
    """frames: list of (tracks, motion, carried_ids). Returns emitted events."""
    events = []
    t = 0.0
    for tracks, motion, carried in frames:
        ev = det.update(tracks, _table(), t, motion,
                        {"motion": motion, "arm": 0.0, "carried_ids": carried})
        if ev is not None:
            events.append(ev)
        t += 1.0 / 30.0
    return events


def _sequence(carried: bool):
    """Settle, move a ball two feet, settle. ``carried`` keeps the moving ball
    foreign-adjacent for its whole displacement (a hand carrying it)."""
    frames = []
    ball = _mk(1, 300.0, 300.0)
    for _ in range(10):                                   # settle + rest snapshot
        frames.append(([ball], 0.0, set()))
    x = 300.0
    for _ in range(12):                                   # motion
        x += 40.0
        ids = {1} if carried else set()
        frames.append(([_mk(1, x, 300.0, vx=40.0)], 5.0, ids))
    for _ in range(10):                                   # settle again
        frames.append(([_mk(1, x, 300.0)], 0.0, set()))
    return frames


class TestCarriedBallGate:
    def test_hand_carried_ball_never_resolves_as_shot(self):
        det = _detector()
        assert _run(det, _sequence(carried=True)) == [], \
            "hand-carried travel must not resolve as a shot"

    def test_struck_ball_still_resolves(self):
        det = _detector()
        events = _run(det, _sequence(carried=False))
        assert len(events) == 1
        assert events[0].outcome.value in ("miss", "make")

    def test_stick_contact_at_strike_is_fine(self):
        """The cue stick is adjacent to the cue ball for the first frames of
        every real shot — that must not gate."""
        det = _detector()
        frames = []
        ball = _mk(1, 300.0, 300.0)
        for _ in range(10):
            frames.append(([ball], 0.0, set()))
        x = 300.0
        for i in range(12):
            x += 40.0
            ids = {1} if i < 2 else set()                 # stick contact, then free
            frames.append(([_mk(1, x, 300.0, vx=40.0)], 5.0, ids))
        for _ in range(10):
            frames.append(([_mk(1, x, 300.0)], 0.0, set()))
        events = _run(det, frames)
        assert len(events) == 1, "brief stick adjacency must not gate a shot"

    def test_picked_up_ball_is_not_a_pot(self):
        """A ball lifted by hand near a pocket must not earn pocket credit."""
        det = _detector()
        table = _table()
        pocket = table.pockets[0]
        frames = []
        mover = _mk(1, 300.0, 300.0)
        victim = _mk(2, pocket.x + 30.0, pocket.y + 30.0, number=5)
        for _ in range(10):
            frames.append(([mover, victim], 0.0, set()))
        # a real ball rolls (free), while the victim is picked up by hand
        x = 300.0
        for i in range(12):
            x += 40.0
            tracks = [_mk(1, x, 300.0, vx=40.0)]
            if i < 4:      # victim carried toward the pocket, then gone
                tracks.append(_mk(2, pocket.x + 20.0 - i * 5, pocket.y + 20.0 - i * 5,
                                  number=5, vx=5.0))
            frames.append((tracks, 5.0, {2}))
        for _ in range(10):
            frames.append(([_mk(1, x, 300.0)], 0.0, set()))
        events = _run(det, frames)
        assert len(events) == 1
        assert events[0].num_pocketed == 0, "picked-up ball must not be a pot"
