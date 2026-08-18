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
    # time floors zeroed: these tests exercise the gates in tiny synthetic
    # sequences; the physical-duration floors get their own dedicated test
    return ShotDetector(s.detection, s.balls, settle_frames=3, min_shot_frames=2,
                        min_shot_s=0.0, settle_s=0.0)


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


class TestBankRecencyGate:
    def test_recently_carried_ball_does_not_bank(self):
        """A just-placed ball's post-release wobble must not bank into a shot
        (the wobble plus banked steps once resurrected a false scratch)."""
        det = _detector()
        frames = []
        ball = _mk(1, 300.0, 300.0)
        for _ in range(10):
            frames.append(([ball], 0.0, set()))
        # hand carries the ball across the table
        x = 300.0
        for _ in range(8):
            x += 40.0
            frames.append(([_mk(1, x, 300.0, vx=40.0)], 5.0, {1}))
        # released: wobbles freely while motion still reads active
        for _ in range(6):
            x += 8.0
            frames.append(([_mk(1, x, 300.0, vx=8.0)], 5.0, set()))
        for _ in range(10):
            frames.append(([_mk(1, x, 300.0)], 0.0, set()))
        assert _run(det, frames) == [], "placement wobble must not become a shot"

    def test_never_carried_ball_still_banks(self):
        """The fast-pot case: pre-shot steps of a struck ball must count."""
        det = _detector()
        frames = []
        ball = _mk(1, 300.0, 300.0)
        for _ in range(10):
            frames.append(([ball], 0.0, set()))
        x = 300.0
        for _ in range(12):
            x += 40.0
            frames.append(([_mk(1, x, 300.0, vx=40.0)], 5.0, set()))
        for _ in range(10):
            frames.append(([_mk(1, x, 300.0)], 0.0, set()))
        assert len(_run(det, frames)) == 1


class TestBallMotionArming:
    """G5 recall fix: the whole-table motion signal of ONE fast ball hovers
    around the activity threshold and a single sub-threshold frame used to
    kill the arm run (011928 @ 44.5s: died at run 5/6, armed 1.2s late).
    A demonstrably struck ball — banked free frames + travel — arms the
    detector directly, and the shot start is backdated to the strike."""

    def test_stroke_arms_on_ball_motion_alone(self):
        det = _detector()
        frames = []
        for _ in range(10):
            frames.append(([_mk(1, 300.0, 300.0)], 0.0, set()))
        x = 300.0
        for _ in range(12):                    # struck ball, motion signal DEAD
            x += 40.0
            frames.append(([_mk(1, x, 300.0, vx=40.0)], 0.0, set()))
        for _ in range(10):
            frames.append(([_mk(1, x, 300.0)], 0.0, set()))
        events = _run(det, frames)
        assert len(events) == 1, "struck ball must arm without table motion"

    def test_shot_start_backdated_to_strike(self):
        det = _detector()
        frames = []
        for _ in range(10):
            frames.append(([_mk(1, 300.0, 300.0)], 0.0, set()))
        x = 300.0
        for _ in range(12):
            x += 40.0
            frames.append(([_mk(1, x, 300.0, vx=40.0)], 0.0, set()))
        for _ in range(10):
            frames.append(([_mk(1, x, 300.0)], 0.0, set()))
        events = _run(det, frames)
        strike_t = 10 / 30.0                   # first moving frame
        assert events and abs(events[0].start_t - strike_t) < 3 / 30.0, \
            f"start {events[0].start_t:.3f} should be ~{strike_t:.3f}"

    def test_release_wobble_does_not_arm(self):
        det = _detector()
        frames = []
        for _ in range(10):
            frames.append(([_mk(1, 300.0, 300.0)], 0.0, set()))
        x = 300.0
        for _ in range(4):                     # small free wobble, no motion
            x += 3.0
            frames.append(([_mk(1, x, 300.0)], 0.0, set()))
        for _ in range(12):
            frames.append(([_mk(1, x, 300.0)], 0.0, set()))
        assert _run(det, frames) == [], "a release wobble must not arm a shot"


class TestTimeFloorsAndCarriedDwell:
    def _det(self, **kw):
        s = Settings()
        s.detection.require_cue = False
        s.detection.use_fusion = False
        s.detection.warmup_seconds = 0.0
        s.detection.cooldown_seconds = 0.0
        return ShotDetector(s.detection, s.balls, settle_frames=3,
                            min_shot_frames=2, **kw)

    def test_shot_stays_open_for_the_blurred_flyer(self):
        """011510 @ 72.2s: the struck ball blurs out of tracking; the shot
        resolved in 0.23s with ~35px of tracked travel and was discarded.
        With the time floor the shot is still open when the flyer's track
        revives far away — its snap displacement crosses the travel gate."""
        det = self._det(min_shot_s=1.2, settle_s=0.5)
        frames = []
        cue = _mk(1, 300.0, 300.0)
        flyer = _mk(2, 320.0, 300.0)
        for _ in range(10):
            frames.append(([cue, flyer], 0.0, set()))
        # strike: cue nudges 3 frames (arms the detector), flyer VANISHES
        x = 300.0
        for _ in range(3):
            x += 12.0
            frames.append(([_mk(1, x, 300.0, vx=12.0)], 0.3, set()))
        # 0.6s of nothing tracked moving (flyer in blurred flight)
        for _ in range(18):
            frames.append(([_mk(1, x, 300.0)], 0.1, set()))
        # flyer's track revives 500px away, rolls a touch, then all settle
        fx = 820.0
        for _ in range(4):
            fx += 6.0
            frames.append(([_mk(1, x, 300.0), _mk(2, fx, 300.0, vx=6.0)], 0.2, set()))
        for _ in range(40):
            frames.append(([_mk(1, x, 300.0), _mk(2, fx, 300.0)], 0.0, set()))
        events = _run(det, frames)
        assert len(events) == 1, "flyer's revival must be credited to the shot"
        assert events[0].max_travel >= 450.0

    def test_gathering_with_hands_on_movers_is_discarded(self):
        """The 19 'silent shots with hands involved': gathering whose
        intermittent free frames beat the travel gate. Moving updates with
        a hand ON the mover dominate -> discarded at resolve."""
        det = self._det(min_shot_s=0.0, settle_s=0.0)
        frames = []
        ball = _mk(1, 300.0, 300.0)
        for _ in range(10):
            frames.append(([ball], 0.0, set()))
        x = 300.0
        # sweep: ball moves 24 frames; hand adjacent on ~70% of them, but
        # with free gaps long enough to arm and accrue travel
        for i in range(24):
            x += 20.0
            ids = {1} if (i % 10) < 7 else set()
            frames.append(([_mk(1, x, 300.0, vx=20.0)], 2.0, ids))
        for _ in range(10):
            frames.append(([_mk(1, x, 300.0)], 0.0, set()))
        assert _run(det, frames) == [], "hand-dominated movement must not resolve"


class TestVanishedFlyerCredit:
    def _det(self):
        s = Settings()
        s.detection.require_cue = False
        s.detection.use_fusion = False
        s.detection.warmup_seconds = 0.0
        s.detection.cooldown_seconds = 0.0
        return ShotDetector(s.detection, s.balls, settle_frames=3,
                            min_shot_frames=2, min_shot_s=0.0, settle_s=0.0)

    def test_potted_flyer_with_tiny_tracked_travel_still_a_shot(self):
        """011510 @ 72.2s: the object ball's whole flight is blur, it drops
        in a pocket unseen; tracked travel is a ~35px cue nudge. The free
        ball's disappearance must carry the shot past the travel gate."""
        det = self._det()
        frames = []
        cue = _mk(1, 300.0, 300.0)
        obj = _mk(2, 340.0, 300.0, number=5)
        for _ in range(10):
            frames.append(([cue, obj], 0.0, set()))
        # strike: object ball takes 3 slow tracked steps (arming bank), cue still
        x = 340.0
        for _ in range(3):
            x += 14.0
            frames.append(([cue, _mk(2, x, 300.0, vx=14.0)], 0.3, set()))
        # then the flyer is GONE for good (blur -> pocket); table calm
        for _ in range(20):
            frames.append(([cue], 0.0, set()))
        events = _run(det, frames)
        assert len(events) == 1, "vanished free ball must carry the shot"

    def test_static_ghost_dying_does_not_carry_a_shot(self):
        """A phantom track expiring mid-shot has no free motion — the travel
        gate must still discard."""
        det = self._det()
        frames = []
        cue = _mk(1, 300.0, 300.0)
        ghost = _mk(9, 500.0, 500.0)
        for _ in range(10):
            frames.append(([cue, ghost], 0.0, set()))
        x = 300.0
        for _ in range(3):                      # tiny cue wobble arms nothing much
            x += 14.0
            frames.append(([_mk(1, x, 300.0, vx=14.0), ghost], 0.5, set()))
        for i in range(20):                     # ghost dies, never moved
            tracks = [_mk(1, x, 300.0)] if i > 2 else [_mk(1, x, 300.0), ghost]
            frames.append((tracks, 0.0, set()))
        assert _run(det, frames) == [], "static ghost death must not make a shot"
