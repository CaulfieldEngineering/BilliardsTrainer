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


def test_manifest_platform_assets():
    """The manifest carries per-platform binaries; each platform gets its own
    (url, sha) — and platforms without a binary get an empty url, never a
    wrong-OS download. Pre-Mac manifests (no mac fields) parse fine."""
    info = updater.UpdateInfo.from_manifest({
        "version": "0.3.0",
        "url": "https://x/BilliardsTrainer-0.3.0.exe", "sha256": "AA",
        "mac_url": "https://x/BilliardsTrainer-mac-0.3.0.zip", "mac_sha256": "BB",
    })
    assert info.asset_for_platform("win32") == (info.url, "aa")
    assert info.asset_for_platform("darwin") == (info.mac_url, "bb")
    assert info.asset_for_platform("linux") == ("", "")

    legacy = updater.UpdateInfo.from_manifest({"version": "0.2.60", "url": "u"})
    assert legacy.asset_for_platform("darwin") == ("", "")


def test_check_logs_decision(monkeypatch, caplog):
    import logging
    monkeypatch.setattr(updater.requests, "get",
                        lambda *a, **k: FakeResp({"version": "0.9.9", "url": "u"}))
    with caplog.at_level(logging.INFO, logger="updater"):
        updater.check_for_update("0.1.5")
    assert any("OFFER UPDATE" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# Self-update spawn safety (regression for the stuck/respawning CMD window)
# --------------------------------------------------------------------------- #
def test_swap_creationflags_are_no_window_not_detached():
    """INVARIANT: the swap is spawned hidden (CREATE_NO_WINDOW) and never with
    DETACHED_PROCESS, which is what made the PID-poll's piped `find` open a
    visible console and block/respawn forever."""
    assert updater.SWAP_CREATIONFLAGS & 0x08000000, "CREATE_NO_WINDOW must be set"
    assert not (updater.SWAP_CREATIONFLAGS & 0x00000008), "DETACHED_PROCESS must NOT be set"


def test_install_and_relaunch_spawns_hidden_with_devnull(monkeypatch, tmp_path):
    """The detached swap process must be launched with CREATE_NO_WINDOW and all
    three std handles set to DEVNULL — no console window can ever appear."""
    import subprocess as sp
    import sys as _sys

    # Pretend we're a frozen Windows exe so the real swap path runs.
    monkeypatch.setattr(_sys, "platform", "win32", raising=False)
    monkeypatch.setattr(_sys, "frozen", True, raising=False)
    exe = tmp_path / "BilliardsTrainer.exe"
    exe.write_bytes(b"OLD")
    monkeypatch.setattr(_sys, "executable", str(exe))
    downloaded = tmp_path / "new.exe"
    downloaded.write_bytes(b"NEW")

    captured = {}

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(sp, "Popen", fake_popen)

    updater.install_and_relaunch(str(downloaded))

    kw = captured["kwargs"]
    assert kw["creationflags"] & 0x08000000, "CREATE_NO_WINDOW must be set"
    assert not (kw["creationflags"] & 0x00000008), "DETACHED_PROCESS must NOT be set"
    assert kw["stdin"] is sp.DEVNULL
    assert kw["stdout"] is sp.DEVNULL
    assert kw["stderr"] is sp.DEVNULL
