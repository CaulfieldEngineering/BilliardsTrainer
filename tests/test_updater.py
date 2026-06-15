"""Updater fetch/compare path — incl. the previously-silent failure modes."""

from billiards_trainer.update import updater


class FakeResp:
    def __init__(self, json_data, status=200):
        self._j = json_data
        self.status_code = status
        self.url = "https://example/version.json"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise updater.requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._j


def test_fetch_manifest_success(monkeypatch):
    monkeypatch.setattr(updater.requests, "get",
                        lambda *a, **k: FakeResp({"version": "0.1.9", "url": "u"}))
    info = updater.fetch_manifest("http://x")
    assert info is not None and info.version == "0.1.9"


def test_fetch_manifest_ssl_error_returns_none_not_raise(monkeypatch):
    def boom(*a, **k):
        raise updater.requests.exceptions.SSLError("bad CA bundle")
    monkeypatch.setattr(updater.requests, "get", boom)
    # must be swallowed (and logged), never propagate
    assert updater.fetch_manifest("http://x") is None


def test_fetch_manifest_network_error_returns_none(monkeypatch):
    def boom(*a, **k):
        raise updater.requests.ConnectionError("offline")
    monkeypatch.setattr(updater.requests, "get", boom)
    assert updater.fetch_manifest("http://x") is None


def test_check_for_update_offers_only_when_newer(monkeypatch):
    monkeypatch.setattr(updater.requests, "get",
                        lambda *a, **k: FakeResp({"version": "0.2.0", "url": "u"}))
    assert updater.check_for_update("0.1.5") is not None

    monkeypatch.setattr(updater.requests, "get",
                        lambda *a, **k: FakeResp({"version": "0.1.5", "url": "u"}))
    assert updater.check_for_update("0.1.5") is None


def test_check_logs_decision(monkeypatch, caplog):
    import logging
    monkeypatch.setattr(updater.requests, "get",
                        lambda *a, **k: FakeResp({"version": "0.9.9", "url": "u"}))
    with caplog.at_level(logging.INFO, logger="updater"):
        updater.check_for_update("0.1.5")
    assert any("OFFER UPDATE" in r.message for r in caplog.records)
