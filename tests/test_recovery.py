"""Self-update integrity (SHA256) + rollback recovery sentinels."""

import hashlib

from billiards_trainer.update import recovery, updater


def test_sha256_file(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"billiards")
    assert updater.sha256_file(str(p)) == hashlib.sha256(b"billiards").hexdigest()


def test_update_info_parses_sha256():
    info = updater.UpdateInfo.from_manifest(
        {"version": "0.2.0", "url": "u", "sha256": "ABC123DEF"})
    assert info.sha256 == "abc123def"  # normalised to lowercase


def test_consume_update_failed(monkeypatch, tmp_path):
    monkeypatch.setattr(recovery, "UPDATE_FAILED", tmp_path / "failed.flag")
    monkeypatch.setattr(recovery, "UPDATE_PENDING", tmp_path / "pending.json")
    assert recovery.consume_update_failed() is False
    (tmp_path / "failed.flag").write_text("failed")
    assert recovery.consume_update_failed() is True       # detected
    assert recovery.consume_update_failed() is False      # and cleared


def test_mark_launched_ok(monkeypatch, tmp_path):
    flag = tmp_path / "launched_ok.flag"
    monkeypatch.setattr(recovery, "APP_DIR", tmp_path)
    monkeypatch.setattr(recovery, "LAUNCHED_OK", flag)
    recovery.mark_launched_ok()
    assert flag.exists()


def test_verify_integrity_noop_when_not_frozen():
    # not a frozen build -> nothing to verify
    assert recovery.verify_frozen_integrity() is None


class _FakeResp:
    def __init__(self, data):
        self._data = data
        self.headers = {"content-length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self._data


def test_download_worker_rejects_checksum_mismatch(monkeypatch):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    monkeypatch.setattr(updater.requests, "get", lambda *a, **k: _FakeResp(b"payload-bytes"))
    captured = {}
    worker = updater.DownloadWorker("http://x/BilliardsTrainer-0.2.0.exe",
                                    expected_sha="deadbeef")  # deliberately wrong
    worker.failed.connect(lambda m: captured.setdefault("failed", m))
    worker.finished.connect(lambda p: captured.setdefault("finished", p))
    worker.run()
    assert "failed" in captured and "corrupt" in captured["failed"].lower()
    assert "finished" not in captured


def test_download_worker_accepts_matching_checksum(monkeypatch):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    payload = b"good-payload"
    good = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(updater.requests, "get", lambda *a, **k: _FakeResp(payload))
    captured = {}
    worker = updater.DownloadWorker("http://x/BilliardsTrainer-0.2.0.exe", expected_sha=good)
    worker.failed.connect(lambda m: captured.setdefault("failed", m))
    worker.finished.connect(lambda p: captured.setdefault("finished", p))
    worker.run()
    assert "finished" in captured and "failed" not in captured
