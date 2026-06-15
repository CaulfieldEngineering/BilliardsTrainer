"""Supabase (PostgREST) sync.

``SupabaseClient`` upserts rows to a table via the REST API. ``SyncManager`` lives
on its own QThread, periodically (and on close) pushing unsynced feedback /
sessions / shots and flipping their local ``synced`` flag. If no credentials are
configured the manager is inert — every method returns immediately.

Why upsert-by-id: the local autoincrement id is the primary key on both sides, so
re-pushing a row is idempotent (``Prefer: resolution=merge-duplicates``).
"""

import logging

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot

from ..config import load_supabase_config
from ..db.repository import Repository

log = logging.getLogger("sync")

_SYNC_INTERVAL_MS = 5 * 60 * 1000  # every 5 minutes


def sync_status() -> str:
    return "configured" if load_supabase_config() else "not configured"


class SupabaseClient:
    """Thin PostgREST upsert client (requests-based)."""

    def __init__(self, url: str, key: str):
        self.base = url.rstrip("/")
        self.key = key

    def upsert(self, table: str, rows: list[dict]) -> bool:
        if not rows:
            return True
        import requests
        endpoint = f"{self.base}/rest/v1/{table}?on_conflict=id"
        headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=minimal",
        }
        resp = requests.post(endpoint, json=rows, headers=headers, timeout=20)
        resp.raise_for_status()
        return True


class SyncManager(QObject):
    status = Signal(str)   # human-readable status for the UI

    def __init__(self, repository: Repository, client: SupabaseClient | None = None):
        super().__init__()
        self._repo = repository
        cfg = load_supabase_config()
        if client is not None:
            self._client = client
        elif cfg:
            self._client = SupabaseClient(cfg["url"], cfg["key"])
        else:
            self._client = None
        self._timer: QTimer | None = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    @Slot()
    def on_started(self) -> None:
        self._timer = QTimer()
        self._timer.setInterval(_SYNC_INTERVAL_MS)
        self._timer.timeout.connect(self.sync_now)
        if self.enabled:
            self._timer.start()
            # one initial push shortly after launch
            QTimer.singleShot(8000, self.sync_now)

    @Slot()
    def sync_now(self) -> None:
        """Push all unsynced rows. No-op + status when not configured."""
        if not self.enabled:
            self.status.emit("Supabase not configured")
            return
        pushed = 0
        try:
            pushed += self._push("feedback", self._repo.unsynced_feedback(),
                                 self._feedback_payload)
            pushed += self._push("sessions", self._repo.unsynced_sessions(), lambda r: r)
            pushed += self._push("shots", self._repo.unsynced_shots(), lambda r: r)
        except Exception as exc:  # noqa: BLE001 - network/HTTP best effort
            log.warning("Supabase sync failed: %s", exc)
            self.status.emit(f"Sync failed: {exc}")
            return
        self.status.emit(f"Synced {pushed} row(s)" if pushed else "Up to date")

    def _push(self, table: str, rows: list[dict], to_payload) -> int:
        if not rows:
            return 0
        payload = [to_payload(r) for r in rows]
        self._client.upsert(table, payload)
        self._repo.mark_synced(table, [r["id"] for r in rows])
        log.info("Synced %d row(s) to %s", len(rows), table)
        return len(rows)

    @staticmethod
    def _feedback_payload(r: dict) -> dict:
        # attachments are local file paths; keep them as a count for the backup
        out = dict(r)
        out.pop("synced", None)
        out["attachment_count"] = len(out.pop("attachments", []) or [])
        return out


def make_sync_thread(manager: SyncManager) -> QThread:
    thread = QThread()
    thread.setObjectName("sync")
    manager.moveToThread(thread)
    thread.started.connect(manager.on_started)
    thread.start()
    return thread


def trigger_sync(manager: SyncManager) -> None:
    from PySide6.QtCore import QMetaObject
    QMetaObject.invokeMethod(manager, "sync_now", Qt.QueuedConnection)
