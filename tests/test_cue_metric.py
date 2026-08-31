"""The cue metric must not score frames where the cue ball is in a pocket.

Round 72: the cold clip's cue read 95.4% against a 99 target, and
essentially all of the deficit was ONE 7-second window where the cue
ball had been potted - it sits in the jaws at 101.4s, drops, and Joe
reaches in and replaces it at ~108.6s. The app correctly had no cue
track throughout, and the metric counted all 210 frames as failures. It
was penalising the engine for refusing to hallucinate a ball.

Absence comes from the naming truth (pixel-derived, eye-checked) and
only a RUN of consecutive samples counts, because a lone missing sample
may be the yardstick ABSTAINING - and an abstention must never excuse
the engine.
"""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


@pytest.fixture()
def sc(monkeypatch):
    import scorecard
    return scorecard


def _truth(tmp_path, missing_ts, step=1.0, n=20):
    """A naming truth whose cue is missing at the given sample times."""
    samples = []
    for k in range(n):
        t = round(k * step, 2)
        balls = [[1, 100.0, 100.0]]
        if t not in missing_ts:
            balls.append([0, 200.0, 200.0])
        samples.append({"t": t, "balls": balls})
    p = tmp_path / "nt.json"
    p.write_text(json.dumps({"session": "x.mp4", "samples": samples}),
                 encoding="utf-8")
    return p


def test_a_run_of_missing_cue_samples_is_an_absence(sc, tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "NAMING_TRUTH", _truth(tmp_path, {5.0, 6.0, 7.0}))
    wins = sc._cue_absent_windows()
    assert len(wins) == 1
    a, b = wins[0]
    assert a < 5.0 and b > 7.0, "the window must cover the missing samples"
    assert a > 4.0 and b < 8.0, "and must not swallow neighbouring samples"


def test_a_single_missing_sample_is_not_an_absence(sc, tmp_path, monkeypatch):
    """It may be the yardstick abstaining, which must not excuse the app."""
    monkeypatch.setattr(sc, "NAMING_TRUTH", _truth(tmp_path, {5.0}))
    assert sc._cue_absent_windows() == [], (
        "one missing sample became an excuse - abstention is not absence")


def test_no_missing_samples_means_no_windows(sc, tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "NAMING_TRUTH", _truth(tmp_path, set()))
    assert sc._cue_absent_windows() == []


def test_the_bench_cue_is_never_absent(sc, monkeypatch):
    """Its cue appears in all 221 samples, so nothing may be skipped -
    the exclusion must not quietly widen to the pinned session."""
    monkeypatch.setattr(sc, "NAMING_TRUTH",
                        ROOT / "docs" / "bench_naming_truth.json")
    assert sc._cue_absent_windows() == []


def test_the_cold_window_is_the_measured_one(sc, monkeypatch):
    monkeypatch.setattr(
        sc, "NAMING_TRUTH",
        ROOT / "docs" / "cold_naming_truth_20260823-185550.json")
    wins = sc._cue_absent_windows()
    assert len(wins) == 1, f"expected exactly the potted-cue window, got {wins}"
    a, b = wins[0]
    # a pixel sweep put the ball off the bed 101.5 -> 108.5s
    assert 101.0 <= a <= 102.0 and 108.0 <= b <= 109.0, wins


def test_a_coasted_row_is_not_a_moving_ball(sc):
    """Round 78: it was, and it made a real fix look like a regression.

    The moving-ball metric counted every ACTIVE row, estimates included,
    so a coasting ghost's prediction drift registered as a ball in
    flight. When round 77 correctly stopped that ghost holding a real
    ball's number, its rows became "moving and unnamed" and dragged the
    figure 99.3 -> 98.7%. All 11 contested bench cases were one coasting
    track at 18.81-19.11s, clocked at up to 2,052 units/s while sitting
    on empty cloth. Excluding estimates: 99.4%, above where it started.

    The cue metric has demanded a real sighting since 2026-08-28; this
    one had not caught up.
    """
    import inspect
    src = inspect.getsource(sc.score)
    i = src.find("moving_named += 1")
    assert i > 0, "the moving-ball metric vanished"
    window = src[max(0, i - 900):i]
    assert re.search(r"len\(tr\) > 7 and tr\[7\]", window), (
        "coasted rows are being counted as moving balls again - a "
        "prediction is not a sighting")


def test_naming_credit_on_estimates_is_reported_separately(sc):
    """Round 79's audit: how much of "named correctly" is a guess?

    The headline credits a correct name even when the nearest track is a
    COASTED estimate rather than a sighting. Audited across both clips:
    7 of the bench's 1,094 correct verdicts and 2 of the cold clip's
    1,185 sit on estimates - and ALL SEVEN bench cases are the same
    thing, track 11 holding the red 3 through the seconds Joe stands
    over it, which is round 71's occlusion fix working exactly as
    designed. The ball is really there and the estimate is within 9px.

    So this is not a defect to remove; it is a fact to SHOW. Joe's own
    precedent governs - when he asked "what does it mean to correctly
    name 99.6% of balls", the answer was to expose the stricter figure
    beside the headline, not to quietly move it.
    """
    import inspect
    src = inspect.getsource(sc._naming_correctness)
    assert "right_coast" in src, (
        "the estimate-backed portion of the naming score is no longer "
        "counted; the headline silently banks predictions as sightings")
    assert "name_right_seen_pct" in src, "the stricter figure is not returned"
    printed = inspect.getsource(sc)
    assert "ACTUALLY SEEN" in printed, (
        "the stricter figure is computed but never shown - a number that "
        "only lives in JSON is not a number Joe sees")


def test_the_skip_count_is_reported(sc):
    """A metric that drops frames from its denominator must say so."""
    import inspect
    src = inspect.getsource(sc)
    assert "cue_frames_skipped" in src and "cue_absent_windows" in src
    assert "frames skipped" in src, (
        "the skip must be printed, not hidden inside the JSON")


class TestWhoNamedTheBall:
    """Round 82: make the colour dependency a standing line.

    Round 81 measured that the identity model emits NOTHING at the dark
    balls - the cold table's black 8 is read 0 times in 92 - so those
    balls are named entirely by the measured-colour path. Colour is not
    a backstop there, it is the floor, and nothing on the scorecard said
    so. Detection.identified -> Track.id_read -> sidecar element 9 ->
    "...NAMED BY COLOUR" carries it, the same shape as round 68's
    Track.read.
    """

    def test_the_flag_survives_the_projection(self):
        """THE BUG THIS ROUND ACTUALLY HIT. prepare_detections rebuilds
        every Detection, and a rebuilt object loses any field the
        constructor is not told about. On the first cut the scorecard
        claimed ALL 1,094 bench names were colour-only - which round 81
        had already disproved - because the flag died at projection, the
        same way measured_bgr once did."""
        import inspect

        from billiards_trainer.vision import pipeline as pl
        src = inspect.getsource(pl.Pipeline._project_raw_to_rect)
        assert "identified=" in src, (
            "the projection drops Detection.identified again; every ball "
            "will look colour-named")

    def test_the_sidecar_round_trips_it(self):
        from billiards_trainer.vision.analysis_cache import SidecarReader
        t = SidecarReader._to_track(
            [7, 1.0, 2.0, 13.0, 8, "eight", True, False, 8, True])
        assert t.id_read is True
        t2 = SidecarReader._to_track(
            [7, 1.0, 2.0, 13.0, 8, "eight", True, False, 8, False])
        assert t2.id_read is False
        old = SidecarReader._to_track([7, 1.0, 2.0, 13.0, 8, "eight", True])
        assert old.id_read is False, "old sidecars must still load"

    def test_both_paths_set_it(self):
        import inspect

        from billiards_trainer.detector_strategies import ensemble as ens
        from billiards_trainer.measure import engine as eng
        assert "identified" in inspect.getsource(eng._pair_identities), (
            "the offline engine no longer records who named the ball")
        assert "f.identified" in inspect.getsource(ens.FindIdEnsemble.detect), (
            "the live path no longer records who named the ball")

    def test_the_scorecard_shows_it(self):
        import inspect

        import scorecard as sc
        src = inspect.getsource(sc)
        assert "name_right_by_colour" in src
        assert "NAMED BY COLOUR" in src, (
            "computed but never printed - a number that only lives in "
            "JSON is not a number Joe sees")


class TestBenchPaletteProvenance:
    """Round 83: the bench truth finally has an independent derivation.

    Its naming truth was hand-fitted colour windows and its truth file
    recorded no ball colours at all, while the truth files have been the
    defect four times (25, 58, 63, 69). The pot order gives a derivation
    that uses NO colour judgement, and rebuilding the naming truth from
    it agreed with the hand-fitted one on 1082 of 1082 shared samples.
    """

    def test_the_palette_exists_and_covers_the_rack(self):
        import json
        pal = json.loads(
            (ROOT / "docs" / "bench_palette_20260824-220247.json")
            .read_text(encoding="utf-8"))
        assert set(pal["balls"]) == {"0", "1", "2", "3", "4", "9"}
        assert pal["balls"]["9"]["stripe"] is True
        assert all(not pal["balls"][k]["stripe"]
                   for k in ("0", "1", "2", "3", "4"))

    def test_the_one_and_the_nine_share_a_colour(self):
        """The structural fact rounds 27-33 paid for, re-derived: the 9 IS
        the 1 with a white band, so colour cannot separate them and only
        the white fraction can. If this ever stops being true the palette
        was built wrong."""
        import json
        import math
        pal = json.loads(
            (ROOT / "docs" / "bench_palette_20260824-220247.json")
            .read_text(encoding="utf-8"))
        one, nine = pal["balls"]["1"], pal["balls"]["9"]
        d = math.dist(one["lab"], nine["lab"])
        assert d < 15.0, (
            f"the 1 and the 9 measure {d:.1f} Lab apart; they are the same "
            f"colour and something has gone wrong with the derivation")
        assert nine["white_frac"] - one["white_frac"] > 0.15, (
            "white fraction is the ONLY thing separating them and the gap "
            "has closed")

    def test_the_truth_file_records_its_colours(self):
        import json
        d = json.loads((ROOT / "docs" / "bench_truth.json")
                       .read_text(encoding="utf-8"))
        assert d.get("ball_colours_observed"), (
            "the bench truth stopped recording what its balls look like - "
            "that absence is what made it uncheckable for 83 rounds")
        assert set(d["ball_colours_observed"]) == {
            "0", "1", "2", "3", "4", "9"}


class TestColdPaletteProvenance:
    """Round 84: the cold clip's five POTTED balls are pot-order confirmed.

    A ball that vanishes at a pot IS the ball that was potted - no colour
    reference consulted, each ball matched only to itself across the four
    seconds spanning the pot. The 1 needed the stripe test because its
    colour twin the 9 stays on the table and masks the disappearance.

    6, 7 and 8 are never potted, so the method cannot reach them - and
    round 82 showed the 7 and 8 are the two named ENTIRELY by colour.
    That gap is the point of this test: it must stay visible.
    """

    @staticmethod
    def _pal():
        import json
        return json.loads(
            (ROOT / "docs" / "cold_palette_20260823-185550.json")
            .read_text(encoding="utf-8"))

    def test_the_potted_balls_are_confirmed(self):
        pal = self._pal()
        for n in ("1", "2", "3", "4", "5"):
            src = pal["balls"][n].get("source", "")
            assert "pot-order" in src, (
                f"ball {n} lost its independent derivation; the potted "
                f"balls are the only ones the pot order can confirm")

    def test_the_unreachable_balls_never_claim_the_pot_order(self):
        """If someone quietly marks these derived without doing the work,
        the campaign loses the one honest record of what is unchecked.

        Round 85 gave them what support it could - the averaged colours
        are unambiguous and force the assignment ON A STANDARD SET - but
        that is an assumption about Joe's equipment, not a measurement,
        and it must never be dressed up as the pot order."""
        pal = self._pal()
        for n in ("6", "7", "8"):
            src = pal["balls"][n].get("source", "")
            assert "pot-order" not in src, (
                f"ball {n} is never potted - the pot order cannot confirm "
                f"it, so claiming it does is false provenance")
            assert "canonical-set" in src, (
                f"ball {n} lost the record of HOW it is supported")

    def test_the_rejected_digit_method_stays_recorded(self):
        """Round 85 tried to read the printed numbers off 400-frame
        averages. The control failed - ball 2, confirmed by its own pot,
        does not read as a '2' - so the method was rejected. Keeping that
        written down stops it being retried as though it were new."""
        pal = self._pal()
        note = pal.get("unreached_round85", "")
        assert "REJECTED" in note and "control" in note, (
            "the rejected digit-reading attempt is no longer recorded; "
            "someone will spend a round rediscovering that it fails")

    def test_the_gap_is_written_down(self):
        pal = self._pal()
        note = pal.get("validated_round84", "")
        assert "6, 7 and 8" in note and "never potted" in note, (
            "the limit of the method must stay recorded - the two most "
            "load-bearing references are the two it cannot check")


class TestBorrowedRefsStaySignposted:
    """Three sessions are named from a palette they did not earn.

    Rounds 88, 89 and 90 lent verified colour references to sessions
    whose felt reads within a couple of units of a verified clip's -
    the criterion round 87 bought by finding that Joe has ONE table and
    what varies between sessions is which lamps were on. It works: the
    frames showing an impossible ball fell 99%, 98% and 98%.

    But a borrowed palette is NOT a derived one. Neither borrower has
    ground truth, so the honest claim is "far fewer self-contradictions",
    never "correct". The only thing standing between that distinction
    and a future round quietly promoting these to evidence is the note
    inside each file - so the note is a tested artefact, not a comment.

    Round 86 is what the promotion costs: naming 64.7% -> 86.5% on a
    session with NO lighting match looked like a clean win and was
    reference-sensitive noise.
    """

    BORROWED = {
        "session-20260823-191319": "session-20260823-185550",
        "session-20260823-194542": "session-20260823-185550",
        "session-20260824-220740": "session-20260824-220247",
    }

    def test_every_borrowed_palette_names_its_lender(self):
        for stem, lender in self.BORROWED.items():
            p = ROOT / "docs" / f"colour_refs_{stem}.json"
            assert p.is_file(), f"{p.name} vanished"
            note = json.loads(p.read_text(encoding="utf-8")).get("_note", "")
            assert "REUSED" in note, (
                f"{p.name} no longer declares itself borrowed - it will be "
                f"read as an independently derived palette")
            assert lender in note, (
                f"{p.name} does not say which session it borrowed from; "
                f"without the lender the justification cannot be rechecked")

    def test_every_borrowed_palette_carries_its_felt_measurement(self):
        """The felt reading IS the justification, so it is a field, not
        prose.

        Round 89 wrote "within 2.3 units" into a note with no reading to
        recheck - which is a sentence, not a criterion. Round 90 measured
        all three pairs the same way and put the numbers in the files:
            191319 <- 185550   1.12
            194542 <- 185550   2.74
            220740 <- 220247   2.29
        Round 89's survey found every reusable session at <= 4.1 and
        every session needing its own palette at >= 15; 8 is the line
        through that gap.
        """
        for stem, lender in self.BORROWED.items():
            p = ROOT / "docs" / f"colour_refs_{stem}.json"
            felt = json.loads(p.read_text(encoding="utf-8")).get("_felt")
            assert felt, (
                f"{p.name} lost the felt measurement that justifies the "
                f"loan; round 86 shows what borrowing without one costs")
            assert felt["lender_session"] == lender
            assert felt["distance"] < 8.0, (
                f"{p.name} claims a lighting match at distance "
                f"{felt['distance']} - past the gap round 89 measured, so "
                f"this palette describes a different lighting")
            got = [round(a - b, 1)
                   for a, b in zip(felt["borrower"], felt["lender"])]
            assert any(abs(v) <= 8 for v in got), (
                "the recorded readings do not support the recorded distance")
