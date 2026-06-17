"""Model auto-fetch: atomic write + sha verification (no network)."""

import hashlib

import pytest

from billiards_trainer.detector_strategies import model_fetch


class _FakeResp:
    def __init__(self, payload):
        self._p = payload
        self.headers = {"content-length": str(len(payload))}

    def raise_for_status(self):
        pass

    def iter_content(self, n):
        for i in range(0, len(self._p), n):
            yield self._p[i:i + n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _wire(monkeypatch, tmp_path, payload, sha):
    monkeypatch.setattr(model_fetch, "MODELS_DIR", tmp_path)
    monkeypatch.setitem(model_fetch.FETCHABLE["pool_yolo11"], "sha256", sha)
    monkeypatch.setattr(model_fetch.requests, "get", lambda *a, **k: _FakeResp(payload))


def test_fetch_writes_atomically_and_reports_progress(tmp_path, monkeypatch):
    payload = b"onnx-bytes" * 5000
    _wire(monkeypatch, tmp_path, payload, hashlib.sha256(payload).hexdigest())

    fracs = []
    dest = model_fetch.fetch_model("pool_yolo11", progress=fracs.append)

    assert dest.exists() and dest.read_bytes() == payload
    assert fracs and fracs[-1] == 1.0
    assert not (tmp_path / "pool_yolo11.onnx.part").exists()   # temp cleaned up
    assert model_fetch.is_present("pool_yolo11")


def test_fetch_rejects_checksum_mismatch(tmp_path, monkeypatch):
    payload = b"corrupted"
    _wire(monkeypatch, tmp_path, payload, "deadbeef" * 8)      # wrong sha

    with pytest.raises(ValueError, match="checksum"):
        model_fetch.fetch_model("pool_yolo11")
    assert not model_fetch.is_present("pool_yolo11")           # nothing left behind
    assert not (tmp_path / "pool_yolo11.onnx.part").exists()
