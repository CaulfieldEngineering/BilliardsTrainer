"""Companion server: sessions API, shot markers, and iOS-critical ranges."""

import http.client
import json
import threading

import pytest

from billiards_trainer.companion import server as srv


@pytest.fixture()
def companion(tmp_path, monkeypatch):
    # a tiny fake recordings dir: one "video" + sidecar with 2 shots + a correction
    video = tmp_path / "session-20260818-x.mp4"
    video.write_bytes(bytes(range(256)) * 40)          # 10240 bytes, seekable
    (tmp_path / "session-20260818-x.mp4.analysis.jsonl").write_text(
        '{"type":"meta","v":2,"fps":30}\n'
        '{"type":"f","t":1,"tracks":[]}\n'
        '{"type":"shot","start":5.0,"end":9.0,"outcome":"miss","pocketed":0}\n'
        '{"type":"shot","start":20.0,"end":24.5,"outcome":"make","pocketed":1}\n'
        '{"type":"correction","start":5.0,"outcome":"make"}\n')
    monkeypatch.setattr(srv, "_recordings_dir", lambda: tmp_path)
    # summaries touch cv2 on a fake mp4; stub them out for the API shape test
    import billiards_trainer.ui.session_summaries as ss
    monkeypatch.setattr(ss, "summarize",
                        lambda p, c: {"dur_s": 60.0, "shots": 2})
    httpd = srv.serve(0)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield port
    httpd.shutdown()


def _get(port, path, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.request("GET", path, headers=headers or {})
    r = c.getresponse()
    body = r.read()
    c.close()
    return r, body


class TestCompanion:
    def test_sessions_lists_with_shot_counts(self, companion):
        r, body = _get(companion, "/api/sessions")
        assert r.status == 200
        rows = json.loads(body)
        assert rows and rows[0]["name"] == "session-20260818-x.mp4"
        assert rows[0]["shots"] == 2

    def test_shots_endpoint_applies_corrections(self, companion):
        r, body = _get(companion, "/api/session/session-20260818-x.mp4/shots")
        shots = json.loads(body)
        assert len(shots) == 2
        assert shots[0]["outcome"] == "make" and shots[0]["corrected"] is True
        assert shots[1]["outcome"] == "make"

    def test_media_range_request_ios_semantics(self, companion):
        r, body = _get(companion, "/media/session-20260818-x.mp4",
                       {"Range": "bytes=100-199"})
        assert r.status == 206
        assert r.getheader("Content-Range") == "bytes 100-199/10240"
        assert len(body) == 100
        assert body[0] == 100 and body[-1] == 199

    def test_media_open_ended_and_full(self, companion):
        r, body = _get(companion, "/media/session-20260818-x.mp4",
                       {"Range": "bytes=10200-"})
        assert r.status == 206 and len(body) == 40
        r, body = _get(companion, "/media/session-20260818-x.mp4")
        assert r.status == 200 and len(body) == 10240
        assert r.getheader("Accept-Ranges") == "bytes"

    def test_traversal_refused(self, companion):
        r, _ = _get(companion, "/media/..%5Csettings.json")
        assert r.status == 404

    def test_index_served_with_cors(self, companion):
        r, body = _get(companion, "/")
        assert r.status == 200 and b"Billiards Trainer" in body
        assert r.getheader("Access-Control-Allow-Origin") == "*"
