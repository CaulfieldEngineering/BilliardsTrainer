"""The analysis sidecar: write once, play back forever (Joe's architecture:
'by the time we're playing back, it should all be cached data')."""

from billiards_trainer.core.types import BallClass, Track
from billiards_trainer.events.shot_detector import ShotEvent, ShotOutcome
from billiards_trainer.vision.analysis_cache import (
    SidecarReader,
    SidecarWriter,
    sidecar_path,
)


def _track(tid, x, y, num=3):
    return Track(id=tid, x=x, y=y, radius=15.0, number=num,
                 cls=BallClass.SOLID, active=True)


def _write_sample(video):
    # times are RECORDING-relative by contract: the first frame defines t=0
    # (the writer rebases whatever pipeline time it is fed)
    w = SidecarWriter(video, {"fps": 30.0})
    w.add_frame(0.0, [_track(1, 100.0, 200.0), _track(2, 500.0, 600.0, num=5)])
    w.add_frame(1.0, [_track(1, 200.0, 300.0), _track(2, 500.0, 600.0, num=5)])
    w.add_shot(ShotEvent(outcome=ShotOutcome.MAKE, num_pocketed=1,
                         start_t=0.2, end_t=0.9))
    w.close()


class TestSidecar:
    def test_roundtrip_and_interpolation(self, tmp_path):
        video = tmp_path / "session-x.mp4"
        _write_sample(video)
        r = SidecarReader(video)
        assert len(r) == 2
        assert r.shots[0]["outcome"] == "make"
        mid = r.tracks_at(0.5)                       # halfway between states
        t1 = next(t for t in mid if t.id == 1)
        assert abs(t1.x - 150.0) < 0.5 and abs(t1.y - 250.0) < 0.5
        assert t1.number == 3
        still = next(t for t in mid if t.id == 2)
        assert abs(still.x - 500.0) < 0.5

    def test_edges_clamp(self, tmp_path):
        video = tmp_path / "session-x.mp4"
        _write_sample(video)
        r = SidecarReader(video)
        assert r.tracks_at(0.0)[0].x == 100.0        # before first state
        assert r.tracks_at(99.0)[0].x == 200.0       # after last state

    def test_gap_snaps_instead_of_tweening(self, tmp_path):
        video = tmp_path / "session-g.mp4"
        w = SidecarWriter(video, {"fps": 30.0})
        w.add_frame(1.0, [_track(1, 100.0, 100.0)])
        w.add_frame(9.0, [_track(1, 900.0, 900.0)])  # 8s hole (paused/covered)
        w.close()
        r = SidecarReader(video)
        t5 = r.tracks_at(5.0)[0]
        assert t5.x == 100.0, "must not glide a ball across a detection gap"

    def test_exists_helper(self, tmp_path):
        video = tmp_path / "session-x.mp4"
        assert not SidecarReader.exists(video)
        _write_sample(video)
        assert SidecarReader.exists(video)
        assert sidecar_path(video).name == "session-x.mp4.analysis.jsonl"


class TestHandContextV2:
    def test_carried_ids_roundtrip(self, tmp_path):
        video = tmp_path / "s.mp4"
        video.write_bytes(b"x")
        w = SidecarWriter(video, {"fps": 30})
        tr = Track(id=3, x=100, y=100, radius=14, number=5,
                   cls=BallClass.SOLID, active=True)
        w.add_frame(1.0, [tr])                                  # no hands
        w.add_frame(2.0, [tr], carried_ids={3}, foreign_frac=0.04)
        w.add_frame(3.0, [tr])
        w.close()
        r = SidecarReader(video)
        assert r.has_hand_context
        # times are RECORDING-relative now: frames written at 1.0/2.0/3.0
        # land at 0/1/2 (the first frame defines t=0)
        ids, peak = r.hand_context(0.7, 1.3)
        assert ids == {3} and peak >= 0.04
        ids2, _ = r.hand_context(1.8, 2.4)
        assert ids2 == set()

    def test_v1_sidecar_reports_no_hand_context(self, tmp_path):
        video = tmp_path / "s.mp4"
        video.write_bytes(b"x")
        w = SidecarWriter(video, {"fps": 30})
        tr = Track(id=1, x=50, y=50, radius=14, number=2,
                   cls=BallClass.SOLID, active=True)
        w.add_frame(1.0, [tr])
        w.close()
        r = SidecarReader(video)
        assert not r.has_hand_context
        assert r.hand_context(0.0, 2.0) == (set(), 0.0)


class TestRecordingTimebase:
    def test_writer_rebases_to_first_frame(self, tmp_path):
        video = tmp_path / "s.mp4"
        video.write_bytes(b"x")
        w = SidecarWriter(video, {"fps": 30})
        tr = Track(id=1, x=5, y=5, radius=12, number=2,
                   cls=BallClass.SOLID, active=True)
        w.add_frame(1160.5, [tr])
        w.add_frame(1161.5, [tr])
        class Ev:
            start_t, end_t = 1168.8, 1173.2
            class outcome:
                value = "miss"
            num_pocketed = 0
        w.add_shot(Ev)
        w.close()
        r = SidecarReader(video)
        assert r._times[0] == 0.0 and abs(r._times[1] - 1.0) < 1e-6
        assert abs(r.shots[0]["start"] - 8.3) < 0.01

    def test_reader_normalizes_legacy_offset(self, tmp_path):
        video = tmp_path / "s.mp4"
        video.write_bytes(b"x")
        sc = tmp_path / "s.mp4.analysis.jsonl"
        sc.write_text('{"type":"meta","v":1,"fps":30}\n'
                      '{"type":"f","t":1160.5,"tracks":[[1,5,5,12,2,"solid",true]]}\n'
                      '{"type":"f","t":1161.5,"tracks":[[1,5,5,12,2,"solid",true]]}\n'
                      '{"type":"shot","start":1168.8,"end":1173.2,'
                      '"outcome":"miss","pocketed":0}\n')
        r = SidecarReader(video)
        assert r._times[0] == 0.0
        assert abs(r.shots[0]["start"] - 8.3) < 0.01


class TestVerdictCarryover:
    """A --force re-backfill rewrites the sidecar; Joe's phone verdicts
    must ride across (and re-attach by containment when segmentation
    shifted) while legacy machine records must NOT."""

    def _old_sidecar(self, tmp_path):
        import json
        p = tmp_path / "old.analysis.jsonl"
        rows = [
            {"type": "meta", "v": 1, "fps": 30},
            {"type": "shot", "start": 10.0, "end": 14.0, "outcome": "make"},
            {"type": "correction", "start": 10.0, "outcome": "miss",
             "src": "derived"},                       # machine: stays behind
            {"type": "correction", "start": 10.0, "outcome": "miss"},
                                                      # legacy untagged: stays
            {"type": "correction", "start": 10.0, "outcome": "scratch",
             "src": "review"},                        # human: carries
            {"type": "action", "start": 10.0, "action": "rearrange",
             "src": "review"},                        # human: carries
            {"type": "note", "start": 10.0, "text": "my comment"},  # carries
            {"type": "reviewed", "start": 10.0},                    # carries
        ]
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                     encoding="utf-8")
        return p

    def test_only_human_records_carry(self, tmp_path):
        import json

        from billiards_trainer.vision.analysis_cache import carry_review_verdicts
        old = self._old_sidecar(tmp_path)
        vid = tmp_path / "session-x.mp4"
        _write_sample(vid)
        new = tmp_path / "session-x.mp4.analysis.jsonl"
        n = carry_review_verdicts(old, new)
        assert n == 4
        carried = [json.loads(x) for x in
                   new.read_text(encoding="utf-8").splitlines()][-4:]
        assert [c["type"] for c in carried] == ["correction", "action",
                                                "note", "reviewed"]
        assert all(c.get("src") != "derived" for c in carried)

    def test_carried_verdict_reattaches_by_containment(self, tmp_path):
        import json

        from billiards_trainer.vision.analysis_cache import carry_review_verdicts
        vid = tmp_path / "session-x.mp4"
        vid.write_bytes(b"0")
        new = tmp_path / "session-x.mp4.analysis.jsonl"
        # rebuilt sidecar: segmentation shifted — shot now [8.5, 13.0]
        new.write_text(json.dumps({"type": "meta", "v": 1, "fps": 30}) + "\n"
                       + json.dumps({"type": "shot", "start": 8.5,
                                     "end": 13.0, "outcome": "make"}) + "\n",
                       encoding="utf-8")
        carry_review_verdicts(self._old_sidecar(tmp_path), new)
        r = SidecarReader(vid)
        s = r.shots[0]
        assert s["outcome"] == "scratch", "review verdict lost in rebuild"
        assert s["action"] == "rearrange"
        assert s["note"] == "my comment"
        assert s["_reviewed"] and s["_action_reviewed"]
