"""The proxy renderer's recording guard.

The proxy encode uses the same AMD hardware encoder as the recorder, so
the one rule that must never regress: a render NEVER runs (and a running
render dies) while a recording is live. Joe outranks proxies.
"""

import types

import billiards_trainer.capture.audio as audio
import billiards_trainer.config as config
from billiards_trainer.vision import proxy_render


def _video(tmp_path):
    v = tmp_path / "session-p.mp4"
    v.write_bytes(b"x" * 26_000_000)   # above MIN_SOURCE_BYTES
    return v


class TestProxyGuard:
    def test_existing_proxy_short_circuits(self, tmp_path, monkeypatch):
        v = _video(tmp_path)
        p = proxy_render.proxy_path(v)
        p.parent.mkdir()
        p.write_bytes(b"x" * 2_000_000)
        # ffmpeg lookup must not even happen
        monkeypatch.setattr(audio, "find_ffmpeg",
                            lambda: (_ for _ in ()).throw(AssertionError))
        assert proxy_render.render_proxy(v) is True

    def test_deferred_when_recording_live(self, tmp_path, monkeypatch):
        v = _video(tmp_path)
        rec = tmp_path / "rec"
        rec.mkdir()
        (rec / ".session-live.part.mp4").write_bytes(b"x")
        monkeypatch.setattr(config, "EXPORTS_DIR", rec)
        monkeypatch.setattr(audio, "find_ffmpeg", lambda: "ffmpeg")
        popened = []
        monkeypatch.setattr(proxy_render.subprocess, "Popen",
                            lambda *a, **k: popened.append(a))
        assert proxy_render.render_proxy(v) is False
        assert not popened, "encode must not start while a recording is live"

    def test_killed_when_recording_starts_midrender(self, tmp_path, monkeypatch):
        v = _video(tmp_path)
        rec = tmp_path / "rec"
        rec.mkdir()
        monkeypatch.setattr(config, "EXPORTS_DIR", rec)
        monkeypatch.setattr(audio, "find_ffmpeg", lambda: "ffmpeg")
        monkeypatch.setattr(proxy_render.time, "sleep", lambda s: None)

        killed = []

        class FakeProc:
            returncode = None
            def poll(self):
                # first poll: still encoding; a recording appears now
                (rec / ".session-live.part.mp4").write_bytes(b"x")
                return None
            def kill(self):
                killed.append(True)
            def wait(self, timeout=None):
                return 0

        tmp = proxy_render.proxy_path(v).with_suffix(".part.mp4")
        def fake_popen(*a, **k):
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(b"partial")
            return FakeProc()
        monkeypatch.setattr(proxy_render.subprocess, "Popen", fake_popen)
        assert proxy_render.render_proxy(v) is False
        assert killed, "live recording must kill the encode"
        assert not tmp.exists(), "partial output must be deleted"
