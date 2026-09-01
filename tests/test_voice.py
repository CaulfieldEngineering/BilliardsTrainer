"""The spoken-cue module's contract (render chain + cache + mute)."""

from billiards_trainer.ui import voice


class TestVoice:
    def test_slug_and_path(self):
        assert voice._slug("Ball in hand") == "ball-in-hand"
        # the voice is part of the cache key (2026-08-31): keying on the
        # phrase alone made the 28 Aug switch to Christopher inert, because
        # ensure() kept returning the WAV already rendered as Guy
        assert voice.wav_path("Ten").name == f"ten-{voice._voice_tag()}.wav"

    def test_zero_volume_never_spawns_work(self, monkeypatch):
        spawned = []
        monkeypatch.setattr(voice.threading, "Thread",
                            lambda **k: spawned.append(k))
        voice.say("Ten", volume=0)
        assert not spawned

    def test_cached_phrase_skips_renderers(self, tmp_path, monkeypatch):
        monkeypatch.setattr(voice, "VOICE_DIR", tmp_path)
        wav = tmp_path / f"ten-{voice._voice_tag()}.wav"
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


class TestPerceptualVolume:
    """Joe: 'almost down and voice is still loud' - the voice ignored
    volume entirely (winsound has no level control), and the cue curve
    was linear so it stayed loud until the bottom."""

    def test_curve_is_perceptual_not_linear(self):
        from billiards_trainer.ui.sounds import gain_for
        assert gain_for(100) == 1.0
        assert gain_for(0) == 0.0
        assert gain_for(50) < 0.2, "half the slider must be a quarter as loud"
        assert gain_for(25) < 0.05
        assert gain_for(75) < gain_for(90) < gain_for(100)

    def test_voice_writes_a_quieter_copy(self, tmp_path, monkeypatch):
        import wave

        import numpy as np

        from billiards_trainer.ui import voice
        src = tmp_path / "phrase.wav"
        with wave.open(str(src), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(22050)
            w.writeframes((np.ones(2205, dtype=np.int16) * 8000).tobytes())
        out = voice._at_volume(src, 40)
        assert out is not None and out != src
        with wave.open(str(out), "rb") as w:
            quiet = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        assert abs(int(quiet.max())) < 8000 * 0.2

    def test_full_volume_uses_the_original(self, tmp_path):
        from billiards_trainer.ui import voice
        src = tmp_path / "x.wav"
        src.write_bytes(b"RIFF")
        assert voice._at_volume(src, 100) == src
        assert voice._at_volume(src, 0) is None
