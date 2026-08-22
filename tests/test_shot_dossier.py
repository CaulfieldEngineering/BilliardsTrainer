"""Shot dossier export: the VLM-coach data contract."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from billiards_trainer.core.types import BallClass, Track
from billiards_trainer.vision.analysis_cache import SidecarWriter


def _mk(tid, x, y, num, cls=BallClass.SOLID):
    t = Track(id=tid, x=x, y=y, radius=12, number=num, cls=cls)
    t.active = True
    return t


def _sidecar(tmp_path):
    video = tmp_path / "session-t.mp4"
    video.write_bytes(b"x")
    w = SidecarWriter(video, {"fps": 30})
    # cue rolls right and bends off a rail; the 5 gets bumped; the 9 rests
    for i in range(30):
        t = i * 0.1
        cx = 100 + i * 12 if i < 15 else 280 - (i - 15) * 4
        cy = 300 - i * 2 if i < 15 else 270 + (i - 15) * 6
        tracks = [_mk(1, cx, cy, 0, BallClass.CUE), _mk(3, 500, 500, 9)]
        if i >= 12:
            tracks.append(_mk(2, 300 + (i - 12) * 8, 300, 5))
        w.add_frame(t, tracks)
    class Ev:
        start_t, end_t = 0.3, 2.8
        class outcome:
            value = "make"
        num_pocketed = 1
    w.add_shot(Ev)
    w.close()
    return video


class TestDossier:
    def test_export_shape_and_participants(self, tmp_path):
        from export_shot_dossier import export_shot

        from billiards_trainer.vision.analysis_cache import SidecarReader
        video = _sidecar(tmp_path)
        r = SidecarReader(video)
        d = export_shot(video, r, 1, r.shots[0], tmp_path / "out")
        doc = json.loads((d / "shot.json").read_text())
        assert doc["outcome"] == "make" and doc["duration_s"] == 2.5
        nums = {p["number"] for p in doc["participants"]}
        assert 0 in nums and 5 in nums, f"cue + struck 5 must participate: {nums}"
        assert 9 not in nums, "the resting 9 is not a participant"
        assert doc["cue_stroke"]["peak_speed_px_s"] > 50
        assert doc["cue_stroke"]["direction_changes"], "rail bend must register"
        assert (d / "trajectory.png").exists()
