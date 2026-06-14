"""Repository tests on an in-memory SQLite DB."""

from pathlib import Path

import pytest

from billiards_trainer.db.repository import Repository


@pytest.fixture
def repo():
    return Repository(db_path=Path(":memory:"))


def test_session_and_shots_roundtrip(repo):
    sid = repo.start_session(mode="practice", table_size="9ft")
    repo.record_shot(sid, "make", num_pocketed=1, target_pocket="top-left", duration_s=2.0)
    repo.record_shot(sid, "make", num_pocketed=1, target_pocket="top-left")
    repo.record_shot(sid, "miss")
    summary = repo.session_summary(sid)
    assert summary["shots"] == 3
    assert summary["makes"] == 2
    assert round(summary["make_pct"]) == 67


def test_streak_tracking(repo):
    sid = repo.start_session()
    for outcome in ["make", "make", "make", "miss", "make"]:
        repo.record_shot(sid, outcome, num_pocketed=1 if outcome == "make" else 0,
                         target_pocket="left-side" if outcome == "make" else None)
    summary = repo.session_summary(sid)
    assert summary["best_streak"] == 3
    assert summary["current_streak"] == 1


def test_by_pocket_distribution(repo):
    sid = repo.start_session()
    repo.record_shot(sid, "make", target_pocket="top-left")
    repo.record_shot(sid, "make", target_pocket="top-left")
    repo.record_shot(sid, "make", target_pocket="bottom-right")
    summary = repo.session_summary(sid)
    pockets = {row[0]: row[1] for row in summary["by_pocket"]}
    assert pockets["top-left"] == 2
    assert pockets["bottom-right"] == 1


def test_global_summary_counts_sessions(repo):
    s1 = repo.start_session()
    repo.record_shot(s1, "make", target_pocket="top-left")
    s2 = repo.start_session()
    repo.record_shot(s2, "miss")
    g = repo.global_summary()
    assert g["sessions"] == 2
    assert g["shots"] == 2
    assert len(g["recent_sessions"]) == 2


def test_csv_export(repo, tmp_path):
    sid = repo.start_session()
    repo.record_shot(sid, "make", num_pocketed=1, target_pocket="top-left")
    dest = repo.export_csv(tmp_path / "out.csv")
    text = dest.read_text()
    assert "outcome" in text and "make" in text


def test_json_export(repo, tmp_path):
    sid = repo.start_session(mode="drill", drill_key="straight_in")
    repo.record_shot(sid, "make", target_pocket="top-left")
    dest = repo.export_json(tmp_path / "out.json")
    import json
    data = json.loads(dest.read_text())
    assert data[0]["mode"] == "drill"
    assert data[0]["shots"][0]["outcome"] == "make"
