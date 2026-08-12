"""Frame-level detection hygiene: the size prior for model detectors, and
per-frame identity uniqueness.

Backstory these tests pin down: Joe's felt carries drill position markers
(donut stickers, chalk dots) that the ball model confidently detects as balls.
Being static they confirm, settle, and become near-immortal tracks — the
invariant scorer measured 10.9 balls/frame against 6 real ones. Markers
project to well under 0.7x the geometric ball radius; real balls stay above
it. If someone re-loosens the band or re-exempts model detectors, these fail.
"""

from billiards_trainer.config import Settings
from billiards_trainer.vision.geometry import expected_ball_radius_px
from billiards_trainer.vision.tracking import BallTracker
from billiards_trainer.vision.types import BallClass, Detection


def _mk(x, y, r, cls=BallClass.UNKNOWN, number=-1, score=0.8):
    return Detection(x, y, r, cls=cls, number=number, score=score)


class TestModelSizePrior:
    def test_defaults_split_markers_from_balls(self):
        """The band must separate the measured populations: markers <=0.66x,
        real balls >=0.89x of geometric radius."""
        s = Settings()
        assert s.balls.model_size_lo > 0.66, "band floor must be above marker size"
        assert s.balls.model_size_lo < 0.89, "band floor must be below real balls"
        assert s.balls.model_size_hi >= 1.4

    def test_marker_sized_detection_rejected_ball_kept(self):
        """End-to-end through _apply_detections with a model-based strategy."""
        from billiards_trainer.vision.pipeline import Pipeline
        s = Settings()
        p = Pipeline(s)

        class FakeStrategy:
            model_based = True
        p._strategy = FakeStrategy()

        calib = _fake_calib()
        exp = expected_ball_radius_px(calib.table, s.table.size)
        raw = [
            _mk(200, 200, exp * 1.0, cls=BallClass.SOLID, number=3, score=0.9),
            _mk(400, 400, exp * 0.5, cls=BallClass.SOLID, number=7, score=0.88),
        ]
        # identity projection (H=I): raw coords are already rect coords
        dets, _tracks = _run_apply(p, calib, raw)
        radii = sorted(d.radius for d in dets)
        assert len(dets) == 1, f"marker survived: {radii}"
        assert abs(dets[0].radius - exp) < 1.0


class TestFrameUniqueness:
    def test_weaker_duplicate_number_demoted(self):
        from billiards_trainer.vision.pipeline import Pipeline
        s = Settings()
        p = Pipeline(s)

        class FakeStrategy:
            model_based = True
        p._strategy = FakeStrategy()
        calib = _fake_calib()
        exp = expected_ball_radius_px(calib.table, s.table.size)
        raw = [
            _mk(200, 200, exp, cls=BallClass.SOLID, number=2, score=0.82),
            _mk(500, 500, exp, cls=BallClass.SOLID, number=2, score=0.52),
        ]
        dets, _ = _run_apply(p, calib, raw)
        nums = sorted(d.number for d in dets)
        assert nums == [-1, 2], f"duplicate #2 must be demoted, got {nums}"
        keeper = next(d for d in dets if d.number == 2)
        assert keeper.score == 0.82

    def test_weaker_cue_demoted_to_unknown(self):
        from billiards_trainer.vision.pipeline import Pipeline
        s = Settings()
        p = Pipeline(s)

        class FakeStrategy:
            model_based = True
        p._strategy = FakeStrategy()
        calib = _fake_calib()
        exp = expected_ball_radius_px(calib.table, s.table.size)
        raw = [
            _mk(200, 200, exp, cls=BallClass.CUE, number=0, score=0.9),
            _mk(600, 300, exp, cls=BallClass.CUE, number=0, score=0.55),
        ]
        dets, _ = _run_apply(p, calib, raw)
        cues = [d for d in dets if d.cls == BallClass.CUE]
        assert len(cues) == 1
        assert cues[0].score == 0.9


class TestSettledIdentityLock:
    def test_settled_ball_identity_frozen_against_challenger(self):
        """A resting ball must not become a different ball, however loud the
        misreads get. Feed a settled track a long run of #7 votes: committed
        stays #3 until it moves."""
        tr = BallTracker(min_hits=2, still_frames=3)
        d3 = _mk(300, 300, 15, cls=BallClass.SOLID, number=3, score=0.9)
        for _ in range(10):   # confirm + settle as #3
            tr.update([d3], short_side=675.0)
        assert tr.tracks[0].number == 3

        d7 = _mk(300, 300, 15, cls=BallClass.SOLID, number=7, score=0.9)
        for _ in range(30):   # challenger dominates the vote window
            tr.update([d7], short_side=675.0)
        assert tr.tracks[0].number == 3, "settled identity must not flip"

    def test_moving_ball_can_still_be_corrected(self):
        """The lock is rest-only: identity established during motion blur must
        remain correctable once the vote flips."""
        tr = BallTracker(min_hits=2, still_frames=3)
        x = 300.0
        for _ in range(6):    # confirm as #3 while MOVING (never settles)
            x += 12.0
            tr.update([_mk(x, 300, 15, cls=BallClass.SOLID, number=3)],
                      short_side=675.0)
        for _ in range(40):   # keep moving; votes now say #7
            x += 12.0
            tr.update([_mk(x, 300, 15, cls=BallClass.SOLID, number=7)],
                      short_side=675.0)
        assert tr.tracks[0].number == 7, "moving-ball correction must work"


# --------------------------------------------------------------------------- #
# Minimal calibration stand-in: identity homography over a 675x1271 table.
# --------------------------------------------------------------------------- #
def _fake_calib():
    import numpy as np

    from billiards_trainer.vision.geometry import TableModel

    class C:
        H = np.eye(3)
        table = TableModel.from_rect((675, 1271), pad=0)
    return C()


def _run_apply(p, calib, raw):
    dets_seen = {}
    orig = p.tracker.update

    def spy(dets, *a, **k):
        dets_seen["dets"] = dets
        return orig(dets, *a, **k)
    p.tracker.update = spy
    # parallax needs a real camera pose — irrelevant here
    p.settings.balls.parallax_correction = False
    result = p._apply_detections(raw, calib, (1271, 675, 3))
    return dets_seen["dets"], result


class TestTrackLevelCueUniqueness:
    def test_two_cue_voting_tracks_render_as_one_cue(self):
        """Det-level dedup can't stop two EXISTING tracks from both rendering
        as cue (a demoted track's class-vote history is still full of CUE).
        The public view must arbitrate: one cue, best evidence wins."""
        tr = BallTracker(min_hits=2, still_frames=3)
        a = _mk(200, 200, 15, cls=BallClass.CUE, number=0, score=0.9)
        b = _mk(600, 600, 15, cls=BallClass.CUE, number=0, score=0.9)
        for _ in range(10):
            tr.update([a, b], short_side=675.0)
        cues = [t for t in tr.tracks if t.cls == BallClass.CUE]
        assert len(cues) == 1, f"{len(cues)} cue balls rendered"

    def test_real_cue_keeps_identity_when_marker_suppressed(self):
        tr = BallTracker(min_hits=2, still_frames=3)
        real = _mk(200, 200, 15, cls=BallClass.CUE, number=0, score=0.9)
        for _ in range(10):                      # real cue builds history alone
            tr.update([real], short_side=675.0)
        ghost = _mk(600, 600, 15, cls=BallClass.CUE, number=-1, score=0.6)
        for _ in range(6):
            tr.update([real, ghost], short_side=675.0)
        cues = [t for t in tr.tracks if t.cls == BallClass.CUE]
        assert len(cues) == 1
        assert abs(cues[0].x - 200) < 5, "the evidence-rich cue must win"
