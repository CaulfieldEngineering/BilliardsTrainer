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
