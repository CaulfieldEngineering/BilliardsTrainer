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


class TestCorrectionsWatcher:
    """Phone verdicts arrive as JSON files via Dropbox sync; the watcher
    applies them as REVIEW-ranked records and archives the file."""

    def _session(self, tmp_path):
        import json
        vid = tmp_path / "session-test.mp4"
        vid.write_bytes(b"0")
        with open(str(vid) + ".analysis.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps({"type": "meta", "v": 1, "fps": 30}) + "\n")
            t = 0.0
            while t <= 20.0:
                f.write(json.dumps({"type": "f", "t": round(t, 2), "tracks":
                                    [[1, 100.0, 100.0, 12.0, 0, "cue", True]]}) + "\n")
                t += 0.5
            f.write(json.dumps({"type": "shot", "start": 8.0, "end": 12.0,
                                "outcome": "miss", "pocketed": 0}) + "\n")
        return vid

    def test_verdict_applied_review_ranked_and_archived(self, tmp_path):
        import json
        from billiards_trainer.companion.corrections_watcher import scan_once
        from billiards_trainer.vision.actions import classify_and_mark
        from billiards_trainer.vision.analysis_cache import SidecarReader
        from billiards_trainer.vision.shots_export import summary_path
        vid = self._session(tmp_path)
        box = tmp_path / "corrections"
        box.mkdir()
        (box / "abc.json").write_text(json.dumps(
            {"session": "session-test.mp4", "start": 8.0,
             "outcome": "make", "action": "rearrange"}))
        assert scan_once(tmp_path) == 1
        s = SidecarReader(vid).shots[0]
        assert s["outcome"] == "make" and s["action"] == "rearrange"
        assert s["corrected"] and s["action_corrected"]
        assert (box / "done" / "abc.json").is_file()
        assert summary_path(vid).is_file()          # phone summary refreshed
        # derived re-runs must not undo the human verdict
        classify_and_mark(vid)
        s = SidecarReader(vid).shots[0]
        assert s["action"] == "rearrange"

    def test_garbage_and_traversal_archived_harmlessly(self, tmp_path):
        import json
        from billiards_trainer.companion.corrections_watcher import scan_once
        box = tmp_path / "corrections"
        box.mkdir()
        (box / "bad.json").write_text("{not json")
        (box / "evil.json").write_text(json.dumps(
            {"session": "../../secrets.mp4", "start": 1, "outcome": "make"}))
        assert scan_once(tmp_path) == 2
        assert not list(box.glob("*.json"))

    def test_note_flows_into_sidecar_and_summary(self, tmp_path):
        import json
        from billiards_trainer.companion.corrections_watcher import scan_once
        from billiards_trainer.vision.analysis_cache import SidecarReader
        from billiards_trainer.vision.shots_export import summary_path
        vid = self._session(tmp_path)
        box = tmp_path / "corrections"
        box.mkdir()
        (box / "n.json").write_text(json.dumps(
            {"session": "session-test.mp4", "start": 8.0,
             "action": "rearrange", "note": "was racking for the next drill"}))
        assert scan_once(tmp_path) == 1
        s = SidecarReader(vid).shots[0]
        assert s["note"] == "was racking for the next drill"
        doc = json.loads(summary_path(vid).read_text())
        assert doc["shots"][0]["note"].startswith("was racking")
        assert doc["shots"][0]["corrected"] is True

    def test_clear_restores_derived_authority(self, tmp_path):
        import json
        from billiards_trainer.companion.corrections_watcher import scan_once
        from billiards_trainer.vision.analysis_cache import SidecarReader
        vid = self._session(tmp_path)
        box = tmp_path / "corrections"
        box.mkdir()
        (box / "v.json").write_text(json.dumps(
            {"session": "session-test.mp4", "start": 8.0,
             "outcome": "make", "note": "oops wrong"}))
        assert scan_once(tmp_path) == 1
        assert SidecarReader(vid).shots[0]["outcome"] == "make"
        (box / "clear.json").write_text(json.dumps(
            {"session": "session-test.mp4", "start": 8.0, "clear": True}))
        assert scan_once(tmp_path) == 1
        s = SidecarReader(vid).shots[0]
        assert not s.get("corrected") and not s.get("note")
        # post-clear the watcher re-derived: the still-cue synthetic session
        # derives "nothing"-action/original outcome without review overrides
        assert not s.get("_reviewed")

    def test_confirm_locks_and_marks_reviewed(self, tmp_path):
        import json
        from billiards_trainer.companion.corrections_watcher import scan_once
        from billiards_trainer.vision.actions import classify_and_mark
        from billiards_trainer.vision.analysis_cache import SidecarReader
        from billiards_trainer.vision.outcomes import derive_and_correct
        from billiards_trainer.vision.shots_export import summary_path
        vid = self._session(tmp_path)
        box = tmp_path / "corrections"
        box.mkdir()
        (box / "ok.json").write_text(json.dumps(
            {"session": "session-test.mp4", "start": 8.0, "confirm": True}))
        assert scan_once(tmp_path) == 1
        s = SidecarReader(vid).shots[0]
        assert s["reviewed_ok"] and not s.get("corrected")
        doc = json.loads(summary_path(vid).read_text())
        assert doc["shots"][0]["reviewed"] is True
        # locked: derived passes stand down on both channels
        before = (s["outcome"], s.get("action", "stroke"))
        derive_and_correct(vid)
        classify_and_mark(vid)
        s2 = SidecarReader(vid).shots[0]
        assert (s2["outcome"], s2.get("action", "stroke")) == before


class TestRifeRequests:
    def test_rife_request_renders_and_playlists(self, tmp_path, monkeypatch):
        """A {rife:true} request renders the clip (mocked), appends it to
        the PC-owned Slow-mo playlist, and archives the request."""
        import json
        import os
        import time
        from billiards_trainer.companion import corrections_watcher as cw
        from billiards_trainer.companion import rife_render
        vid = tmp_path / "session-x.mp4"
        vid.write_bytes(b"0")
        old = time.time() - 3600
        os.utime(vid, (old, old))          # not a live recording
        (tmp_path / (vid.name + ".analysis.jsonl")).write_text(
            json.dumps({"type": "meta", "v": 1, "fps": 30}) + "\n",
            encoding="utf-8")
        box = tmp_path / "corrections"
        box.mkdir()
        (box / "r1.json").write_text(json.dumps(
            {"session": vid.name, "start": 10.0, "end": 16.0,
             "rife": True}), encoding="utf-8")
        rendered = {}
        def fake_render(video, start, end):
            out = tmp_path / "slowmo" / rife_render.slowmo_name(video, start)
            out.parent.mkdir(exist_ok=True)
            out.write_bytes(b"clip")
            rendered["ok"] = (video.name, start, end)
            return out
        monkeypatch.setattr(rife_render, "render_slowmo", fake_render)
        n = cw.scan_once(tmp_path)
        assert n == 1 and rendered["ok"][0] == vid.name
        doc = json.loads((tmp_path / "playlists.json").read_text())
        pl = next(q for q in doc["playlists"] if q["name"] == "Slow-mo")
        assert pl["clips"][0]["slowmo"].startswith("slowmo-session-x")
        assert (box / "done" / "r1.json").is_file()

    def test_rife_deferred_while_recording(self, tmp_path):
        """Joe's presence outranks renders: an active recording keeps the
        request queued (not archived)."""
        import json
        from billiards_trainer.companion import corrections_watcher as cw
        vid = tmp_path / "session-x.mp4"
        vid.write_bytes(b"0")     # fresh mtime = active recording
        (tmp_path / (vid.name + ".analysis.jsonl")).write_text(
            json.dumps({"type": "meta", "v": 1, "fps": 30}) + "\n",
            encoding="utf-8")
        box = tmp_path / "corrections"
        box.mkdir()
        (box / "r1.json").write_text(json.dumps(
            {"session": vid.name, "start": 10.0, "end": 16.0,
             "rife": True}), encoding="utf-8")
        assert cw.scan_once(tmp_path) == 0
        assert (box / "r1.json").is_file()      # still queued
