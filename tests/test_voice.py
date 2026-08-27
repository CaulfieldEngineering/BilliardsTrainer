"""The spoken-cue module's contract (render chain + cache + mute)."""

from billiards_trainer.ui import voice


class TestVoice:
    def test_slug_and_path(self):
        assert voice._slug("Ball in hand") == "ball-in-hand"
        assert voice.wav_path("Ten").name == "ten.wav"

    def test_zero_volume_never_spawns_work(self, monkeypatch):
        spawned = []
        monkeypatch.setattr(voice.threading, "Thread",
                            lambda **k: spawned.append(k))
        voice.say("Ten", volume=0)
        assert not spawned

    def test_cached_phrase_skips_renderers(self, tmp_path, monkeypatch):
        monkeypatch.setattr(voice, "VOICE_DIR", tmp_path)
        wav = tmp_path / "ten.wav"
        wav.write_bytes(b"x" * 2000)
        monkeypatch.setattr(voice, "_render_edge",
                            lambda *a: (_ for _ in ()).throw(AssertionError))
        monkeypatch.setattr(voice, "_render_sapi",
                            lambda *a: (_ for _ in ()).throw(AssertionError))
        assert voice.ensure("Ten") == wav

    def test_renderer_chain_falls_back_to_sapi(self, tmp_path, monkeypatch):
        monkeypatch.setattr(voice, "VOICE_DIR", tmp_path)
        calls = []

        def edge(phrase, out):
            calls.append("edge")
            return False                     # offline / package missing

        def sapi(phrase, out):
            calls.append("sapi")
            out.write_bytes(b"x" * 2000)
            return True
        monkeypatch.setattr(voice, "_render_edge", edge)
        monkeypatch.setattr(voice, "_render_sapi", sapi)
        assert voice.ensure("Scratch") is not None
        assert calls == ["edge", "sapi"]

    def test_both_renderers_failing_yields_none_not_crash(self, tmp_path,
                                                          monkeypatch):
        monkeypatch.setattr(voice, "VOICE_DIR", tmp_path)
        monkeypatch.setattr(voice, "_render_edge", lambda *a: False)
        monkeypatch.setattr(voice, "_render_sapi", lambda *a: False)
        assert voice.ensure("Nope") is None
