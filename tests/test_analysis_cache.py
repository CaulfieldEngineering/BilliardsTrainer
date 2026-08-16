"""The analysis sidecar: write once, play back forever (Joe's architecture:
'by the time we're playing back, it should all be cached data')."""

from billiards_trainer.events.shot_detector import ShotEvent, ShotOutcome
from billiards_trainer.vision.analysis_cache import (
    SidecarReader,
    SidecarWriter,
    sidecar_path,
)
from billiards_trainer.vision.types import BallClass, Track


def _track(tid, x, y, num=3):
    return Track(id=tid, x=x, y=y, radius=15.0, number=num,
                 cls=BallClass.SOLID, active=True)


def _write_sample(video):
    w = SidecarWriter(video, {"fps": 30.0})
    w.add_frame(1.0, [_track(1, 100.0, 200.0), _track(2, 500.0, 600.0, num=5)])
    w.add_frame(2.0, [_track(1, 200.0, 300.0), _track(2, 500.0, 600.0, num=5)])
    w.add_shot(ShotEvent(outcome=ShotOutcome.MAKE, num_pocketed=1,
                         start_t=1.2, end_t=1.9))
    w.close()


class TestSidecar:
    def test_roundtrip_and_interpolation(self, tmp_path):
        video = tmp_path / "session-x.mp4"
        _write_sample(video)
        r = SidecarReader(video)
        assert len(r) == 2
        assert r.shots[0]["outcome"] == "make"
        mid = r.tracks_at(1.5)                       # halfway between states
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
