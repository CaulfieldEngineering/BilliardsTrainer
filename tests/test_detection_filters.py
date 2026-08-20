"""Frame-level detection hygiene: the size prior for model detectors, and
per-frame identity uniqueness.

Backstory these tests pin down: Joe's felt carries drill position markers
(donut stickers, chalk dots) that the ball model confidently detects as balls.
Being static they confirm, settle, and become near-immortal tracks — the
invariant scorer measured 10.9 balls/frame against 6 real ones. Markers
project to well under 0.7x the geometric ball radius; real balls stay above
it. If someone re-loosens the band or re-exempts model detectors, these fail.
"""

from pathlib import Path

from billiards_trainer.config import Settings
from billiards_trainer.vision.geometry import expected_ball_radius_px
from billiards_trainer.vision.tracking import BallTracker
from billiards_trainer.vision.types import BallClass, Detection


def _mk(x, y, r, cls=BallClass.UNKNOWN, number=-1, score=0.8):
    return Detection(x, y, r, cls=cls, number=number, score=score)


#: Committed snapshot of the measured per-table colour references. Tests must
#: NEVER read _train/colour_refs.json — it is gitignored (per-table measured
#: data, Joe's machine only), and depending on it made every CI push red for a
#: day while the local suite stayed green.
_REFS_FIXTURE = Path(__file__).parent / "fixtures" / "colour_refs.json"


def _use_fixture_refs(monkeypatch, tmp_path):
    """Point the measured-colour layer at the fixture snapshot (hermetic on
    CI). monkeypatch restores APP_DIR and the refs cache on teardown."""
    import shutil

    import billiards_trainer.config as cfg
    from billiards_trainer.vision import balls
    shutil.copy2(_REFS_FIXTURE, tmp_path / "colour_refs.json")
    monkeypatch.setattr(cfg, "APP_DIR", tmp_path)
    monkeypatch.setattr(balls, "_MEASURED_REFS", None)   # drop cache
    return balls


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


class TestColourReassignment:
    """The 'two sevens' fix: an arbitration loser gets its next-best FREE
    identity from its own colour, instead of rendering anonymously in a
    measured colour that LOOKS like the winner (Joe: 'two sevens, one of
    which is actually a 4')."""

    def test_duplicate_loser_reassigned_by_colour(self, monkeypatch, tmp_path):
        _use_fixture_refs(monkeypatch, tmp_path)
        tr = BallTracker(min_hits=2, still_frames=3)
        # real 7 (maroon) builds stronger evidence first
        real7 = _mk(200, 200, 15, cls=BallClass.SOLID, number=7, score=0.9)
        real7.bgr = (45, 45, 120)                  # maroon
        for _ in range(6):
            tr.update([real7], short_side=675.0)
        # purple 4, misread as 7 by the model
        fake7 = _mk(500, 500, 15, cls=BallClass.SOLID, number=7, score=0.8)
        fake7.bgr = (110, 35, 95)                  # purple — the 4's colour
        for _ in range(8):
            tr.update([real7, fake7], short_side=675.0)
        nums = sorted(t.number for t in tr.tracks)
        assert nums == [4, 7], f"loser must become the 4, got {nums}"

    def test_loser_stays_unknown_when_its_colour_is_taken(self, monkeypatch, tmp_path):
        _use_fixture_refs(monkeypatch, tmp_path)
        tr = BallTracker(min_hits=2, still_frames=3)
        a = _mk(200, 200, 15, cls=BallClass.SOLID, number=7, score=0.9)
        a.bgr = (45, 45, 120)
        b = _mk(500, 500, 15, cls=BallClass.SOLID, number=7, score=0.8)
        b.bgr = (45, 45, 120)                      # ALSO maroon: no free match
        c = _mk(350, 600, 15, cls=BallClass.SOLID, number=15, score=0.8)
        for _ in range(8):
            tr.update([a, b, c], short_side=675.0)
        nums = sorted(t.number for t in tr.tracks if t.number > 0)
        # 7 kept once; the same-coloured loser gets 15's slot? no — 15 is
        # taken by track c, so the loser must stay unknown
        assert nums.count(7) == 1
        assert nums.count(15) == 1

    def test_white_ball_never_guessed(self):
        from billiards_trainer.vision.tracking import BallTracker as BT
        t = _mk(1, 1, 15)
        class Fake:
            bgr = (250, 250, 250)
            committed_cls = BallClass.UNKNOWN
            cls_hist = []
        assert BT._colour_identity(Fake(), set()) == -1


class TestMeasuredColourFix:
    """The measured-reference identity layer: this table's colours, not
    canonical ones. Under Joe's light the purple 4 measures NAVY, which is
    why canonical classification called it a 7."""

    def _use_repo_refs(self, monkeypatch, tmp_path):
        return _use_fixture_refs(monkeypatch, tmp_path)

    def test_measured_navy_resolves_to_4(self, monkeypatch, tmp_path):
        balls = self._use_repo_refs(monkeypatch, tmp_path)
        # the 4's MEASURED colour on Joe's table (navy, not canonical purple)
        assert balls.measured_identity((142, 26, 36)) == 4

    def test_measured_maroon_resolves_to_7(self, monkeypatch, tmp_path):
        balls = self._use_repo_refs(monkeypatch, tmp_path)
        assert balls.measured_identity((30, 14, 80)) == 7

    def test_taken_identity_excluded(self, monkeypatch, tmp_path):
        balls = self._use_repo_refs(monkeypatch, tmp_path)
        got = balls.measured_identity((142, 26, 36), taken={4})
        assert got != 4

    def test_white_never_guessed(self, monkeypatch, tmp_path):
        balls = self._use_repo_refs(monkeypatch, tmp_path)
        assert balls.measured_identity((250, 250, 250)) == -1

    def test_junk_colour_rejected(self, monkeypatch, tmp_path):
        balls = self._use_repo_refs(monkeypatch, tmp_path)
        # turquoise felt colour: far from every ball reference
        assert balls.measured_identity((200, 200, 60), max_dist=32.0) == -1


class TestDarkTrioCorrection:
    """4/7/8 are THE confusion cluster under warm light (7v8 refs sit 41 Lab
    units apart). The correction is a RELATIVE margin — another reference
    must beat the claim decisively — and the 8 participates."""

    def _fix(self, monkeypatch, tmp_path, bgr, claimed):
        import numpy as np

        from billiards_trainer.detector_strategies.ensemble import FindIdEnsemble
        _use_fixture_refs(monkeypatch, tmp_path)
        frame = np.zeros((60, 60, 3), np.uint8)
        frame[:] = bgr
        f = Detection(30, 30, 14, cls=BallClass.SOLID, number=claimed, score=0.9)
        FindIdEnsemble._fix_colour(frame, f)
        return f.number

    def test_eight_misread_as_seven_corrected(self, monkeypatch, tmp_path):
        # the 8's measured colour on Joe's table, claimed as 7
        assert self._fix(monkeypatch, tmp_path, (39, 21, 16), claimed=7) == 8

    def test_true_seven_left_alone(self, monkeypatch, tmp_path):
        assert self._fix(monkeypatch, tmp_path, (30, 14, 80), claimed=7) == 7

    def test_navy_four_claimed_as_seven_corrected(self, monkeypatch, tmp_path):
        assert self._fix(monkeypatch, tmp_path, (142, 26, 36), claimed=7) == 4

    def test_ambiguous_midpoint_trusts_model(self, monkeypatch, tmp_path):
        # halfway between 7 and 8 references: no decisive winner -> no change
        assert self._fix(monkeypatch, tmp_path, (34, 18, 48), claimed=7) == 7


class TestForeignHandFilter:
    def test_detection_inside_hand_blob_is_dropped(self):
        """A gloved bridge hand resting on the cushion read as a resting ball
        (#4 + two ghosts on session-20260802-173553). Detections whose centre
        sits inside a kept foreign (hand-scale) blob must never be ingested;
        detections on open felt are untouched."""
        import numpy as np

        from billiards_trainer.vision.pipeline import Pipeline
        s = Settings()
        p = Pipeline(s)

        class FakeStrategy:
            model_based = True
        p._strategy = FakeStrategy()
        calib = _fake_calib()
        tbl = calib.table
        exp = expected_ball_radius_px(tbl, s.table.size)
        fs = 160.0 / (tbl.x1 - tbl.x0)
        mh = max(1, int((tbl.y1 - tbl.y0) * fs))
        mask = np.zeros((mh, 160), np.uint8)
        hx, hy = int((200 - tbl.x0) * fs), int((200 - tbl.y0) * fs)
        mask[max(0, hy - 8):hy + 8, max(0, hx - 8):hx + 8] = 1
        p._foreign_last = (0.05, mask, fs, tbl.x0, tbl.y0)
        raw = [
            _mk(200, 200, exp, cls=BallClass.SOLID, number=4, score=0.9),
            _mk(400, 900, exp, cls=BallClass.SOLID, number=2, score=0.9),
        ]
        dets, _ = _run_apply(p, calib, raw)
        assert len(dets) == 1, f"glove detection survived: {[(d.x, d.y) for d in dets]}"
        assert dets[0].number == 2

    def test_no_foreign_state_means_no_filtering(self):
        from billiards_trainer.vision.pipeline import Pipeline
        s = Settings()
        p = Pipeline(s)

        class FakeStrategy:
            model_based = True
        p._strategy = FakeStrategy()
        calib = _fake_calib()
        exp = expected_ball_radius_px(calib.table, s.table.size)
        raw = [_mk(200, 200, exp, cls=BallClass.SOLID, number=4, score=0.9)]
        dets, _ = _run_apply(p, calib, raw)
        assert len(dets) == 1


class TestLiveSchematicFreshness:
    def test_ingest_invalidates_cached_schematic(self):
        """The live async split sends every display frame with detect=False,
        so the schematic cache must be invalidated when detections are
        INGESTED — or the bird's-eye renders once at startup (empty) and
        freezes forever while the tracker publishes balls (Joe's thrice-
        reported empty schematic)."""
        import numpy as np

        from billiards_trainer.vision.pipeline import Pipeline
        s = Settings()
        s.ui.schematic_birdseye = True
        p = Pipeline(s)

        class FakeStrategy:
            model_based = True
        p._strategy = FakeStrategy()
        base = _fake_calib()
        exp = expected_ball_radius_px(base.table, s.table.size)

        class C2:
            H = base.H
            Hinv = np.eye(3)
            table = base.table
            corners = np.float32([[0, 0], [675, 0], [675, 1271], [0, 1271]])
            felt = None
        calib = C2()
        p.calib.calib = calib
        s.ui.show_overlays = False        # keep the test off the overlay path

        frame = np.zeros((1271, 675, 3), np.uint8)
        r1 = p.process(frame, 1.0, detect=False)
        assert r1.rect_bgr is not None
        before = r1.rect_bgr.copy()
        # async worker delivers a ball; display frames stay detect=False
        for i in range(8):
            p.ingest_raw_detections(
                [_mk(200 + i * 4, 300, exp, cls=BallClass.SOLID,
                     number=3, score=0.9)], frame.shape)
            r = p.process(frame, 1.0 + i * 0.03, detect=False)
        assert r.rect_bgr is not None
        assert not np.array_equal(before, r.rect_bgr), \
            "schematic must re-render after ingested detections"


class TestFullShotTrails:
    def test_pre_arm_trail_survives_the_arm_transition(self):
        """Joe: 'the trail is only the second half of the movement.' Arming
        lags the strike, and the settled->moving transition wiped the paths —
        including the struck ball's opening steps. Fresh entries must survive
        the transition; stale (previous-play) entries must not."""
        from billiards_trainer.vision.pipeline import Pipeline
        s = Settings()
        p = Pipeline(s)
        p.calib.calib = None      # move threshold falls back to a constant

        class T:
            def __init__(self, tid, x, y, vx=8.0):
                self.id, self.x, self.y = tid, x, y
                self.vx, self.vy = vx, 0.0
                self.number, self.bgr = 3, (10, 10, 10)
                self.misses = 0
                self.history = [(x - 40, y)] * 2 + [(x, y)]

        # previous play's leftover: entry that last moved long ago
        p._play_paths[99] = {"pts": [(1.0, 1.0)], "bgr": (1, 1, 1),
                             "cue": False, "t_last": 5.0}
        p._prev_shot_state = "settled"
        # strike happens; ball accumulates pre-arm points at t=10.0-10.4
        for i, t in enumerate((10.0, 10.2, 10.4)):
            p._update_play_paths([T(1, 100 + i * 20, 300)], "settled", t)
        assert 1 in p._play_paths and len(p._play_paths[1]["pts"]) >= 2
        before = list(p._play_paths[1]["pts"])
        # the detector arms NOW (settled -> moving)
        p._update_play_paths([T(1, 160, 300)], "moving", 10.5)
        assert 99 not in p._play_paths, "stale previous-play entry must clear"
        assert 1 in p._play_paths, "the stroke's opening must survive arming"
        assert p._play_paths[1]["pts"][:len(before)] == before


class TestNameUnknownByMeasuredColour:
    """Detection-level naming for balls neither model could identify (the
    green 6 on turquoise felt): tight absolute Lab bar + decisive margin —
    the real ball passes with 2x headroom, bare felt (64+) and half-felt
    crops (25+) fail. Probed on real footage before the thresholds were set."""

    FELT = (250, 210, 75)      # this table's bright turquoise (measured)
    GREEN6 = (126, 152, 11)    # the 6's tight-crop median (measured)

    def _det(self, x=30.0, y=30.0, r=10.0):
        from billiards_trainer.vision.types import BallClass, Detection
        return Detection(x=x, y=y, radius=r, cls=BallClass.UNKNOWN, number=-1)

    def _frame(self, ball_bgr, felt_bgr=None, ball_r=10):
        import numpy as np
        f = np.zeros((60, 60, 3), np.uint8)
        f[:] = felt_bgr or self.FELT
        if ball_bgr is not None:
            import cv2
            cv2.circle(f, (30, 30), ball_r, ball_bgr, -1)
        return f

    def test_green6_on_felt_is_named(self, monkeypatch, tmp_path):
        _use_fixture_refs(monkeypatch, tmp_path)
        from billiards_trainer.detector_strategies.ensemble import FindIdEnsemble
        d = self._det()
        FindIdEnsemble._name_unknown(self._frame(self.GREEN6), d)
        assert d.number == 6

    def test_bare_felt_is_never_named(self, monkeypatch, tmp_path):
        _use_fixture_refs(monkeypatch, tmp_path)
        from billiards_trainer.detector_strategies.ensemble import FindIdEnsemble
        d = self._det()
        FindIdEnsemble._name_unknown(self._frame(None), d)
        assert d.number == -1

    def test_half_felt_crop_is_rejected(self, monkeypatch, tmp_path):
        """An off-centre or oversized crop mixes felt into the median —
        the tight absolute bar must reject it rather than guess."""
        _use_fixture_refs(monkeypatch, tmp_path)
        from billiards_trainer.detector_strategies.ensemble import FindIdEnsemble
        d = self._det(x=40.0, y=40.0)          # centred off the ball
        FindIdEnsemble._name_unknown(self._frame(self.GREEN6, ball_r=7), d)
        assert d.number == -1

    def test_white_is_never_named(self, monkeypatch, tmp_path):
        _use_fixture_refs(monkeypatch, tmp_path)
        from billiards_trainer.detector_strategies.ensemble import FindIdEnsemble
        d = self._det()
        FindIdEnsemble._name_unknown(self._frame((250, 250, 250)), d)
        assert d.number == -1

    def test_two_similar_blobs_one_number_per_frame(self, monkeypatch, tmp_path):
        """Pin for frame-level uniqueness: two green blobs in one frame must
        not BOTH be named 6 (parallel vote streams would commit two tracks
        to one number and the at-rest contest demotes the real ball)."""
        import numpy as np, cv2
        _use_fixture_refs(monkeypatch, tmp_path)
        from billiards_trainer.detector_strategies.ensemble import FindIdEnsemble
        f = np.zeros((60, 120, 3), np.uint8)
        f[:] = self.FELT
        cv2.circle(f, (30, 30), 10, self.GREEN6, -1)
        cv2.circle(f, (90, 30), 10, self.GREEN6, -1)   # reflection twin
        d1, d2 = self._det(30.0, 30.0), self._det(90.0, 30.0)
        taken = set()
        for d in (d1, d2):
            FindIdEnsemble._name_unknown(f, d, taken)
            if d.number > 0:
                taken.add(d.number)
        assert sorted(x.number for x in (d1, d2)) == [-1, 6]

    def test_margin_fails_closed_on_sparse_refs(self, monkeypatch, tmp_path):
        """Pin for fail-closed: with only ONE loaded reference there is no
        runner-up to measure a margin against — the gate must abstain, not
        wave the naming through (a sparse regenerated refs file must never
        widen the gate)."""
        import json
        balls = _use_fixture_refs(monkeypatch, tmp_path)
        refs = json.loads((tmp_path / "colour_refs.json").read_text())
        (tmp_path / "colour_refs.json").write_text(json.dumps({"6": refs["6"]}))
        balls._MEASURED_REFS = None                    # drop cache
        from billiards_trainer.detector_strategies.ensemble import FindIdEnsemble
        d = self._det()
        FindIdEnsemble._name_unknown(self._frame(self.GREEN6), d)
        assert d.number == -1

    def test_named_detection_carries_measured_colour(self, monkeypatch, tmp_path):
        """The tracker's colour consensus samples measured_bgr ONLY — a
        naming that didn't set it would silently fall out of the evidence."""
        _use_fixture_refs(monkeypatch, tmp_path)
        from billiards_trainer.detector_strategies.ensemble import FindIdEnsemble
        d = self._det()
        FindIdEnsemble._name_unknown(self._frame(self.GREEN6), d)
        assert d.number == 6 and d.measured_bgr is not None

    def test_unmatched_heuristic_guess_is_colour_corrected(self, monkeypatch, tmp_path):
        """The purple 4 measures NAVY under warm light and the finder's
        heuristic guesses blue 2 — unchecked, it voted '2' every frame of
        a session and stayed nameless (duplicate stripped by arbitration).
        Unmatched numbered finds must get the same measured-colour
        correction matched pairs get."""
        import numpy as np, cv2
        _use_fixture_refs(monkeypatch, tmp_path)
        from billiards_trainer.detector_strategies.ensemble import FindIdEnsemble
        from billiards_trainer.vision.types import BallClass, Detection

        class _Finder:
            name = "stub"
            far_rail_rescan = False
            def detect(self, frame, calib, rescan=None):
                return [Detection(x=30.0, y=30.0, radius=10.0,
                                  cls=BallClass.SOLID, number=2,
                                  bgr=(200, 75, 25))]   # guessed blue 2

        class _Ident:
            name = "stub_id"
            far_rail_rescan = False
            def detect(self, frame, calib, rescan=None):
                return []                                # identifier blind

        f = np.zeros((60, 60, 3), np.uint8)
        f[:] = self.FELT
        cv2.circle(f, (30, 30), 10, (142, 26, 36), -1)   # navy = the real 4
        ens = FindIdEnsemble(_Finder(), _Ident())
        out = ens.detect(f, None)
        assert out[0].number == 4 and out[0].measured_bgr is not None


class TestSizePriorCornerInflation:
    """The rectifying warp inflates ball discs toward the corners (balls
    have height; only the table plane rectifies uniformly). The 4-ball
    measured 1.59x expected radius in the top-right corner and the old
    1.55 model-band ceiling discarded it every frame of a session. The
    band must admit corner inflation and still reject merged-pair blobs."""

    def _band(self):
        from billiards_trainer.config import Settings
        from billiards_trainer.vision.pipeline import Pipeline
        pipe = Pipeline.__new__(Pipeline)
        pipe.settings = Settings.load()
        lo = getattr(pipe.settings.balls, "model_size_lo", 0.72)
        hi = getattr(pipe.settings.balls, "model_size_hi", 1.75)
        return lo, hi

    def test_corner_inflated_ball_survives(self):
        lo, hi = self._band()
        assert 1.59 <= hi, f"measured corner inflation 1.59x exceeds cap {hi}"

    def test_merged_pair_blob_still_rejected(self):
        lo, hi = self._band()
        assert hi < 1.95, f"cap {hi} would admit a merged two-ball blob"

    def test_marker_floor_unchanged(self):
        lo, hi = self._band()
        assert lo >= 0.7, "sticker/marker floor must hold"


class TestVacancyPruning:
    """Joe: 'false positive cue balls and lingering cue ball assumed
    positions.' A STILL track whose spot is plainly visible and empty (no
    detection near it, no hand covering it) must die quickly instead of
    coasting on the minutes-long occlusion budget."""

    def _pipeline(self):
        from billiards_trainer.vision.pipeline import Pipeline
        s = Settings()
        p = Pipeline(s)

        class FakeStrategy:
            model_based = True
        p._strategy = FakeStrategy()
        p.settings.balls.parallax_correction = False
        return p

    def _feed(self, p, calib, raw, n):
        tracks = None
        for _ in range(n):
            tracks, _dets = p._apply_detections(raw, calib, (1271, 675, 3))
        return tracks

    def test_lingering_numbered_track_dies_when_spot_is_bare(self):
        p = self._pipeline()
        calib = _fake_calib()
        exp = expected_ball_radius_px(calib.table, p.settings.table.size)
        ball = _mk(200, 200, exp, cls=BallClass.CUE, number=0, score=0.9)
        self._feed(p, calib, [ball], 12)              # establish + settle
        tracks = self._feed(p, calib, [], 59)         # picked up by hand...
        assert any(t.number == 0 for t in tracks), "died too early"
        tracks = self._feed(p, calib, [], 2)          # ...60th bare frame
        assert not any(t.number == 0 for t in tracks), \
            "lingering assumed position survived a visibly empty spot"

    def test_unnumbered_blob_dies_faster(self):
        p = self._pipeline()
        calib = _fake_calib()
        exp = expected_ball_radius_px(calib.table, p.settings.table.size)
        blob = _mk(400, 600, exp, cls=BallClass.UNKNOWN, number=-1, score=0.8)
        self._feed(p, calib, [blob], 12)
        tracks = self._feed(p, calib, [], 9)
        assert not tracks, "unnumbered ghost survived 9 bare frames"

    def test_hand_cover_protects_the_occluded_ball(self):
        import numpy as np
        p = self._pipeline()
        calib = _fake_calib()
        exp = expected_ball_radius_px(calib.table, p.settings.table.size)
        ball = _mk(200, 200, exp, cls=BallClass.CUE, number=0, score=0.9)
        self._feed(p, calib, [ball], 12)
        # a foreign blob covers the spot: full-frame mask at scale 1
        mask = np.zeros((1271, 675), np.uint8)
        mask[150:260, 150:260] = 1
        p._foreign_last = (0.05, mask, 1.0, 0.0, 0.0)
        tracks = self._feed(p, calib, [], 40)         # long occlusion
        assert any(t.number == 0 for t in tracks), \
            "occluded ball was killed — the budget exists for exactly this"
