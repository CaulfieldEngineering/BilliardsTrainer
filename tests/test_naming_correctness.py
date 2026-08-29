"""The scorecard must be able to see a ball called by the WRONG name.

Round 27 case law. A change lifted every naming figure on the scorecard while
renaming the red 3 to "1" in 1843 frames, and nothing caught it: the metrics
were "does this ball have a name" and "is that name on the inventory", and a
wrong-but-valid name passes both. These pin the metric that closes that hole -
if it ever stops distinguishing right from wrong, this fails.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

scorecard = pytest.importorskip("scorecard")


class _Reader:
    """Minimal stand-in for SidecarReader: identity rect->video mapping."""
    meta = {"hinv": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}


def _row(track_id, x, y, number):
    # (id, x, y, radius, number, cls, active)
    return (track_id, x, y, 13.0, number, 2, 1)


def _run(tmp_path, monkeypatch, truth_balls, rows):
    doc = {"session": "t.mp4", "samples": [{"t": 1.0, "balls": truth_balls}]}
    p = tmp_path / "naming_truth.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setattr(scorecard, "NAMING_TRUTH", p)
    return scorecard._naming_correctness(_Reader(), [1.0], [rows])


class TestSeesWrongNames:
    def test_correct_name_scores_right(self, tmp_path, monkeypatch):
        out = _run(tmp_path, monkeypatch, [[3, 100.0, 100.0]],
                   [_row(1, 100.0, 100.0, 3)])
        assert out["name_right_pct"] == 100.0
        assert out["name_wrong_frames"] == 0

    def test_ball_called_by_another_balls_name_is_WRONG(self, tmp_path, monkeypatch):
        """THE round-27 case: the red 3 answered to '1'."""
        out = _run(tmp_path, monkeypatch, [[3, 100.0, 100.0]],
                   [_row(1, 100.0, 100.0, 1)])
        assert out["name_right_pct"] == 0.0
        assert out["name_wrong_frames"] == 1
        assert out["name_confusions"] == {"3->1": 1}

    def test_wrong_name_is_not_hidden_by_correct_ones(self, tmp_path, monkeypatch):
        """An average must not launder a ball that is never right."""
        out = _run(tmp_path, monkeypatch,
                   [[3, 100.0, 100.0], [2, 300.0, 300.0], [9, 500.0, 500.0]],
                   [_row(1, 100.0, 100.0, 3), _row(2, 300.0, 300.0, 2),
                    _row(3, 500.0, 500.0, 1)])
        assert out["name_confusions"] == {"9->1": 1}
        assert out["name_per_ball"]["9"]["right"] == 0
        assert out["name_per_ball"]["9"]["wrong"] == 1

    def test_unnamed_is_counted_apart_from_wrong(self, tmp_path, monkeypatch):
        out = _run(tmp_path, monkeypatch, [[3, 100.0, 100.0]],
                   [_row(1, 100.0, 100.0, -1)])
        assert out["name_unnamed_frames"] == 1
        assert out["name_wrong_frames"] == 0

    def test_no_track_where_truth_has_a_ball(self, tmp_path, monkeypatch):
        out = _run(tmp_path, monkeypatch, [[3, 100.0, 100.0]],
                   [_row(1, 900.0, 900.0, 3)])
        assert out["name_missing_frames"] == 1

    def test_a_far_away_track_never_answers_for_a_ball(self, tmp_path, monkeypatch):
        """Matching is by position: a correct name somewhere else is not
        credit for the ball truth is asking about."""
        out = _run(tmp_path, monkeypatch, [[3, 100.0, 100.0]],
                   [_row(1, 100.0 + scorecard.NAME_TOL_PX + 5, 100.0, 3)])
        assert out["name_missing_frames"] == 1
        assert out["name_right_pct"] == 0.0
