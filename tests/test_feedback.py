"""Feedback storage + the DevNote seed + sync helpers."""

from pathlib import Path

import pytest

from billiards_trainer.db.repository import Repository


@pytest.fixture
def repo():
    return Repository(db_path=Path(":memory:"))


def test_devnote_147_seeded(repo):
    items = {f["id"]: f for f in repo.recent_feedback()}
    assert 147 in items
    assert items[147]["kind"] == "devnote"
    assert "feedback" in items[147]["title"].lower()


def test_add_and_list_feedback(repo):
    fid = repo.add_feedback(kind="bug", title="Cue ball not detected", severity="high",
                            description="under my lamp", app_version="0.1.6",
                            system_info="Windows 11")
    assert fid > 147  # numbering continues past the seed
    rows = {f["id"]: f for f in repo.recent_feedback()}
    assert rows[fid]["kind"] == "bug"
    assert rows[fid]["severity"] == "high"
    assert rows[fid]["title"] == "Cue ball not detected"


def test_feedback_sync_flags(repo):
    fid = repo.add_feedback(kind="feature", title="Add a timer")
    unsynced = {f["id"] for f in repo.unsynced_feedback()}
    assert fid in unsynced and 147 in unsynced  # the dev notes seed as unsynced too
    repo.mark_feedback_synced(list(unsynced))
    assert repo.unsynced_feedback() == []


def test_attach_to_feedback(repo):
    fid = repo.add_feedback(kind="bug", title="x")
    repo.mark_feedback_synced([fid])
    repo.attach_to_feedback(fid, "C:/clip.mp4")
    row = {f["id"]: f for f in repo.recent_feedback()}[fid]
    assert "C:/clip.mp4" in row["attachments"]
    assert row["synced"] is False  # re-queued for sync after change


def test_session_shot_sync_helpers(repo):
    sid = repo.start_session(mode="free_play")
    repo.record_shot(sid, "make", target_pocket="top-left")
    assert any(s["id"] == sid for s in repo.unsynced_sessions())
    assert len(repo.unsynced_shots()) == 1
    repo.mark_synced("sessions", [sid])
    assert all(s["id"] != sid for s in repo.unsynced_sessions())
