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


def test_the_skip_count_is_reported(sc):
    """A metric that drops frames from its denominator must say so."""
    import inspect
    src = inspect.getsource(sc)
    assert "cue_frames_skipped" in src and "cue_absent_windows" in src
    assert "frames skipped" in src, (
        "the skip must be printed, not hidden inside the JSON")
