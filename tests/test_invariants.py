"""The invariant scorer is the foundation every future vision change is judged
by, so its own correctness is load-bearing: a scorer that misses violations
lets regressions ship, and one that invents them blocks real progress.

Layout mirrors the laws: uniqueness, rigid bodies, containment, temporal
continuity, and — most important — the anti-gaming guards, because the first
thing any optimiser will discover is that detecting nothing breaks no laws.
"""

import math

from billiards_trainer.eval.invariants import (
    SequenceScorer,
    bed_and_pockets,
    check_frame,
    check_sequence,
)
from billiards_trainer.vision.types import BallClass, Track

R = 15.0          # ball radius used throughout; diameter 30
D = 2 * R


def ball(tid, x, y, number=-1, cls=BallClass.UNKNOWN, r=R):
    return Track(id=tid, x=x, y=y, radius=r, number=number, cls=cls)


def spread(n, y=300.0):
    """n balls in a row, safely separated (3 diameters apart)."""
    return [ball(i, 100 + 3 * D * i, y, number=i + 1) for i in range(n)]


# --------------------------------------------------------------------------- #
# Frame laws
# --------------------------------------------------------------------------- #
class TestUniqueness:
    def test_two_three_balls_is_impossible(self):
        tracks = [ball(1, 100, 100, number=3), ball(2, 500, 500, number=3)]
        kinds = [v.kind for v in check_frame(tracks, 0)]
        assert "duplicate_number" in kinds

    def test_two_cue_balls_flagged_even_without_numbers(self):
        tracks = [ball(1, 100, 100, cls=BallClass.CUE),
                  ball(2, 500, 500, cls=BallClass.CUE)]
        kinds = [v.kind for v in check_frame(tracks, 0)]
        assert "multiple_cue_balls" in kinds

    def test_ball_16_does_not_exist(self):
        kinds = [v.kind for v in check_frame([ball(1, 100, 100, number=16)], 0)]
        assert "invalid_number" in kinds

    def test_ball_12_in_nine_ball_is_implausible_not_impossible(self):
        vs = check_frame([ball(1, 100, 100, number=12)], 0, max_number=9)
        v = next(v for v in vs if v.kind == "out_of_game_number")
        assert v.severity == "implausible"

    def test_distinct_numbers_are_clean(self):
        assert check_frame(spread(9), 0) == []


class TestRigidBodies:
    def test_interpenetrating_balls_flagged(self):
        tracks = [ball(1, 100, 100), ball(2, 100 + 0.5 * D, 100)]
        kinds = [v.kind for v in check_frame(tracks, 0)]
        assert "overlapping_balls" in kinds

    def test_touching_balls_are_legal(self):
        # frozen balls: centres exactly one diameter apart
        tracks = [ball(1, 100, 100), ball(2, 100 + D, 100)]
        assert [v for v in check_frame(tracks, 0)
                if v.kind == "overlapping_balls"] == []


class TestContainment:
    def test_ball_far_outside_bed_flagged(self):
        bed = (0, 0, 1000, 500)
        vs = check_frame([ball(1, 1400, 250)], 0, bed=bed)
        assert any(v.kind == "off_table" for v in vs)

    def test_ball_against_cushion_is_legal(self):
        bed = (0, 0, 1000, 500)
        # centre half a diameter inside the rail line
        vs = check_frame([ball(1, R, 250)], 0, bed=bed)
        assert [v for v in vs if v.kind == "off_table"] == []


# --------------------------------------------------------------------------- #
# Temporal laws
# --------------------------------------------------------------------------- #
class TestTemporal:
    def test_teleport_flagged(self):
        s = SequenceScorer()
        s.add([ball(1, 100, 100)], 0)
        s.add([ball(1, 100 + 12 * D, 100)], 1)       # 12 diameters in one frame
        assert any(v.kind == "teleport" for v in s.report.violations)

    def test_break_speed_ball_is_legal(self):
        s = SequenceScorer()
        s.add([ball(1, 100, 100)], 0)
        s.add([ball(1, 100 + 6 * D, 100)], 1)        # fast but physical
        assert not any(v.kind == "teleport" for v in s.report.violations)

    def test_resting_ball_changing_number_flagged(self):
        s = SequenceScorer()
        s.add([ball(1, 100, 100, number=3)], 0)
        s.add([ball(1, 100.5, 100, number=7)], 1)    # sub-jitter movement
        assert any(v.kind == "id_flicker" for v in s.report.violations)

    def test_moving_ball_relabelled_is_not_flicker(self):
        # While moving, a genuine correction (blur cleared) is expected.
        s = SequenceScorer()
        s.add([ball(1, 100, 100, number=3)], 0)
        s.add([ball(1, 100 + 2 * D, 100, number=7)], 1)
        assert not any(v.kind == "id_flicker" for v in s.report.violations)

    def test_class_flicker_at_rest_flagged(self):
        s = SequenceScorer()
        s.add([ball(1, 100, 100, cls=BallClass.SOLID)], 0)
        s.add([ball(1, 100, 100, cls=BallClass.STRIPE)], 1)
        assert any(v.kind == "class_flicker" for v in s.report.violations)

    def test_vanish_mid_table_flagged_only_with_known_pockets(self):
        bed, pockets = bed_and_pockets(1000, 500)
        s = SequenceScorer(bed=bed, pockets=pockets)
        s.add([ball(1, 500, 250), ball(2, 800, 250)], 0)
        s.add([ball(2, 800, 250)], 1)                 # ball 1 gone, centre table
        assert any(v.kind == "vanished_mid_table" for v in s.report.violations)

    def test_vanish_at_pocket_is_a_pot_not_a_bug(self):
        bed, pockets = bed_and_pockets(1000, 500)
        s = SequenceScorer(bed=bed, pockets=pockets)
        s.add([ball(1, D, D), ball(2, 800, 250)], 0)  # ball 1 by the TL pocket
        s.add([ball(2, 800, 250)], 1)
        assert not any(v.kind == "vanished_mid_table" for v in s.report.violations)

    def test_no_pocket_geometry_never_accuses(self):
        s = SequenceScorer()                           # pockets unknown
        s.add([ball(1, 500, 250)], 0)
        s.add([], 1)
        assert not any(v.kind == "vanished_mid_table" for v in s.report.violations)


# --------------------------------------------------------------------------- #
# Anti-gaming: the empty detector must never win
# --------------------------------------------------------------------------- #
class TestDegenerateGuard:
    def test_empty_run_is_degenerate_with_infinite_rate(self):
        rep = check_sequence([[], [], []])
        assert rep.degenerate
        assert rep.rate() == float("inf")

    def test_sparse_run_is_degenerate(self):
        frames = [[ball(1, 100, 100)] if i % 4 == 0 else [] for i in range(40)]
        assert check_sequence(frames).degenerate

    def test_healthy_run_is_not_degenerate(self):
        rep = check_sequence([spread(9) for _ in range(30)])
        assert not rep.degenerate
        assert rep.impossible == 0
        assert rep.coverage == 9.0

    def test_rate_normalises_by_ball_frames_not_frames(self):
        # Same single violation; the run tracking MORE balls scores BETTER.
        few = [[ball(1, 100, 100, number=3), ball(2, 100 + 3 * D, 100, number=3)]]
        many = [spread(9) + [ball(99, 2000, 300, number=3)]]
        assert check_sequence(many).rate() < check_sequence(few).rate()

    def test_calibration_frames_do_not_dilute(self):
        s = SequenceScorer()
        for i in range(100):
            s.add([], i, tracking=False)
        s.add(spread(9), 100)
        assert s.report.tracking_frames == 1
        assert s.report.frames == 101


# --------------------------------------------------------------------------- #
# Report ergonomics — what the harness and promotion gate consume
# --------------------------------------------------------------------------- #
class TestReport:
    def test_count_stability_reflects_flicker(self):
        steady = check_sequence([spread(9) for _ in range(20)])
        flicker = check_sequence([spread(9 if i % 2 else 8) for i in range(20)])
        assert steady.count_stability == 1.0
        assert flicker.count_stability < steady.count_stability

    def test_summary_names_degeneracy(self):
        assert "DEGENERATE" in check_sequence([[]]).summary()

    def test_summary_ranks_kinds_by_frequency(self):
        rep = check_sequence([[ball(1, 100, 100, number=3),
                               ball(2, 100 + 3 * D, 100, number=3)]] * 3)
        assert "duplicate_number" in rep.summary()

    def test_violation_str_is_greppable(self):
        rep = check_sequence([[ball(1, 100, 100, number=16)]])
        assert "invalid_number" in str(rep.violations[0])

    def test_scale_independence(self):
        """The same scene at 2x resolution must score identically."""
        def scene(scale):
            return [[Track(id=i, x=scale * (100 + 3 * D * i), y=scale * 300,
                           radius=scale * R, number=i + 1) for i in range(5)]
                    for _ in range(10)]
        a, b = check_sequence(scene(1.0)), check_sequence(scene(2.0))
        assert len(a.violations) == len(b.violations) == 0
        assert math.isclose(a.coverage, b.coverage)


class TestKnownDiameter:
    def test_known_diameter_overrides_biased_boxes(self):
        """Inflated box radii must not turn legitimately-touching balls into
        violations when the true diameter is supplied."""
        true_d = 30.0
        # boxes report r=19 (1.27x true) but centres sit exactly 1 true
        # diameter apart — touching, legal
        frames = [[Track(id=1, x=100, y=100, radius=19.0, number=1),
                   Track(id=2, x=100 + true_d, y=100, radius=19.0, number=2)]
                  for _ in range(10)]
        biased = check_sequence(frames)                       # box-derived ruler
        s = SequenceScorer(diameter=true_d)                   # honest ruler
        for i, tr in enumerate(frames):
            s.add(tr, i)
        assert any(v.kind == "overlapping_balls" for v in biased.violations), \
            "sanity: the biased ruler flags this"
        assert not any(v.kind == "overlapping_balls" for v in s.report.violations)

    def test_zero_or_none_diameter_falls_back(self):
        frames = [[Track(id=1, x=100, y=100, radius=15.0, number=1)]]
        for d in (None, 0.0):
            s = SequenceScorer(diameter=d)
            s.add(frames[0], 0)
            assert s.report.frames == 1
