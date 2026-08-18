"""Sidebar session summaries: duration + shots, cached, stub detection."""

import json

from billiards_trainer.ui import session_summaries as ss


def _tiny_video(tmp_path, seconds=3):
    import cv2
    import numpy as np
    p = tmp_path / "session-x.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for _ in range(30 * seconds):
        w.write(np.zeros((48, 64, 3), "uint8"))
    w.release()
    return p


class TestSummaries:
    def test_duration_and_shots(self, tmp_path):
        v = _tiny_video(tmp_path)
        sc = tmp_path / "session-x.mp4.analysis.jsonl"
        sc.write_text('{"type":"meta","v":1}\n'
                      '{"type":"f","t":1,"tracks":[]}\n'
                      '{"type":"shot","start":1,"end":2,"outcome":"make","pocketed":1}\n'
                      '{"type":"shot","start":5,"end":7,"outcome":"miss","pocketed":0}\n')
        cache = {}
        s = ss.summarize(v, cache)
        assert 2.5 <= s["dur_s"] <= 3.5
        assert s["shots"] == 2
        assert len(cache) == 1

    def test_cache_hit_skips_recompute(self, tmp_path, monkeypatch):
        v = _tiny_video(tmp_path)
        cache = {}
        ss.summarize(v, cache)
        monkeypatch.setattr(ss, "_video_duration_s",
                            lambda p: (_ for _ in ()).throw(AssertionError("recomputed")))
        s2 = ss.summarize(v, cache)          # same size+mtime -> cache hit
        assert s2["shots"] is None or "dur_s" in s2

    def test_no_sidecar_means_none_shots(self, tmp_path):
        v = _tiny_video(tmp_path)
        s = ss.summarize(v, {})
        assert s["shots"] is None

    def test_row_text_and_stub(self):
        assert "42m" in ss.row_text("Aug 17  21:59", 119.0,
                                    {"dur_s": 2520.0, "shots": 12})
        assert "12 shots" in ss.row_text("Aug 17  21:59", 119.0,
                                         {"dur_s": 2520.0, "shots": 12})
        assert "MB" in ss.row_text("Aug 17  21:59", 13.0, None)
        assert ss.is_stub({"dur_s": 8.0, "shots": 0}, 13.0)
        assert not ss.is_stub({"dur_s": 300.0, "shots": 4}, 55.0)
        assert "1h05" in ss.row_text("x", 1.0, {"dur_s": 3920.0, "shots": None})
