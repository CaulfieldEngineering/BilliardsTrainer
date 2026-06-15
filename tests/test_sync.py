"""Supabase sync skeleton: disabled by default, functional with a client."""

from pathlib import Path

import pytest

from billiards_trainer.db.repository import Repository
from billiards_trainer.sync.supabase import SyncManager


class FakeClient:
    def __init__(self):
        self.calls = []  # (table, n_rows)

    def upsert(self, table, rows):
        self.calls.append((table, len(rows)))
        return True


@pytest.fixture(scope="module")
def app():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_disabled_without_credentials(app, monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.setattr("billiards_trainer.sync.supabase.load_supabase_config", lambda: None)
    mgr = SyncManager(Repository(db_path=Path(":memory:")))
    assert mgr.enabled is False
    mgr.sync_now()  # must not raise


def test_pushes_unsynced_rows_with_client(app):
    repo = Repository(db_path=Path(":memory:"))
    sid = repo.start_session(mode="free_play")
    repo.record_shot(sid, "make", target_pocket="top-left")
    repo.add_feedback(kind="bug", title="something")

    fake = FakeClient()
    mgr = SyncManager(repo, client=fake)
    assert mgr.enabled is True
    mgr.sync_now()

    pushed_tables = {t for t, _ in fake.calls}
    assert {"feedback", "sessions", "shots"} <= pushed_tables
    # everything is now marked synced -> a second sync pushes nothing new
    fake.calls.clear()
    mgr.sync_now()
    assert fake.calls == []


def test_feedback_payload_strips_local_only_fields(app):
    repo = Repository(db_path=Path(":memory:"))
    repo.add_feedback(kind="bug", title="t", attachments=["a.png", "b.mp4"])
    row = repo.unsynced_feedback()[-1]
    payload = SyncManager._feedback_payload(row)
    assert "synced" not in payload
    assert "attachments" not in payload
    assert payload["attachment_count"] == 2
