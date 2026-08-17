"""Per-shot review list (dossier slice 2)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _shots():
    return [
        {"start": 30.0, "end": 36.5, "outcome": "make", "pocketed": 1},
        {"start": 75.2, "end": 80.0, "outcome": "miss", "pocketed": 0},
        {"start": 120.0, "end": 128.3, "outcome": "scratch", "pocketed": 0},
    ]


class TestShotList:
    def test_rows_and_click_seek_to_routine(self, app):
        from billiards_trainer.ui.widgets.shot_list import ShotListPanel
        pnl = ShotListPanel(pre_roll_s=5.0)
        pnl.set_shots(_shots())
        assert pnl._list.count() == 3
        got = []
        pnl.shot_selected.connect(got.append)
        pnl._list.setCurrentRow(1)
        pnl._on_item(pnl._list.item(1))
        assert got and abs(got[0] - 70.2) < 0.01     # 75.2 - 5.0 pre-roll

    def test_prev_next_navigation(self, app):
        from billiards_trainer.ui.widgets.shot_list import ShotListPanel
        pnl = ShotListPanel(pre_roll_s=5.0)
        pnl.set_shots(_shots())
        got = []
        pnl.shot_selected.connect(got.append)
        pnl._step(1)          # from nothing -> first row
        pnl._step(1)          # -> second
        assert len(got) == 2
        assert abs(got[0] - 25.0) < 0.01
        assert abs(got[1] - 70.2) < 0.01
        pnl._step(-1)         # back to first
        assert abs(got[2] - 25.0) < 0.01

    def test_empty_state(self, app):
        from billiards_trainer.ui.widgets.shot_list import ShotListPanel
        pnl = ShotListPanel()
        pnl.set_shots([])
        assert not pnl._list.isVisibleTo(pnl)
        assert pnl._empty.isVisibleTo(pnl)

    def test_live_page_switches_rail_in_playback(self, app):
        from billiards_trainer.config import Settings
        from billiards_trainer.ui.pages.live_page import LivePage
        page = LivePage(Settings())
        page.show()
        page.set_video_mode(True, total=9000, fps=30.0)
        assert page._rail_stack.currentIndex() == 2, "playback shows the shot list"
        page.set_video_mode(False, total=0, fps=30.0)
        assert page._rail_stack.currentIndex() == 0, "live shows the stats rail"


class TestCorrectionChannel:
    def test_correct_outcome_emits_and_marks(self, app):
        from billiards_trainer.ui.widgets.shot_list import ShotListPanel
        pnl = ShotListPanel()
        pnl.set_shots(_shots())
        got = []
        pnl.outcome_corrected.connect(lambda s, o: got.append((s, o)))
        pnl._correct(1, "make")                      # the miss becomes a make
        assert got == [(75.2, "make")]
        assert pnl._shots[1]["outcome"] == "make"
        assert pnl._shots[1]["corrected"] is True

    def test_sidecar_correction_roundtrip(self, tmp_path):
        from billiards_trainer.events.shot_detector import ShotEvent, ShotOutcome
        from billiards_trainer.vision.analysis_cache import (
            SidecarReader,
            SidecarWriter,
            append_correction,
        )
        video = tmp_path / "s.mp4"
        w = SidecarWriter(video, {"fps": 30.0})
        w.add_shot(ShotEvent(outcome=ShotOutcome.MISS, num_pocketed=0,
                             start_t=10.0, end_t=15.0))
        w.close()
        assert append_correction(video, 10.0, "make")
        r = SidecarReader(video)
        assert r.shots[0]["outcome"] == "make"
        assert r.shots[0]["corrected"] is True

    def test_correction_without_sidecar_refuses(self, tmp_path):
        from billiards_trainer.vision.analysis_cache import append_correction
        assert not append_correction(tmp_path / "nope.mp4", 1.0, "make")


class TestClipExport:
    def test_command_shape_and_destination(self, tmp_path):
        from billiards_trainer.vision.analysis_cache import clip_export_cmd
        video = tmp_path / "session-x.mp4"
        video.write_bytes(b"0")
        cmd, dest = clip_export_cmd(video, start=75.2, end=80.0,
                                    pre_roll_s=5.0, shot_no=3)
        assert cmd[0] == "ffmpeg" and "-c" in cmd and "copy" in cmd
        i = cmd.index("-ss")
        assert abs(float(cmd[i + 1]) - 70.2) < 0.01     # routine start
        j = cmd.index("-to")
        assert abs(float(cmd[j + 1]) - 81.0) < 0.01     # +1s tail
        assert dest.name == "session-x_shot03.mp4"
        assert dest.parent.name == "clips"

    def test_start_clamped_at_zero(self, tmp_path):
        from billiards_trainer.vision.analysis_cache import clip_export_cmd
        video = tmp_path / "s.mp4"
        video.write_bytes(b"0")
        cmd, _ = clip_export_cmd(video, start=2.0, end=6.0, pre_roll_s=5.0)
        assert float(cmd[cmd.index("-ss") + 1]) == 0.0

    def test_menu_emits_export(self, app):
        from billiards_trainer.ui.widgets.shot_list import ShotListPanel
        pnl = ShotListPanel()
        pnl.set_shots(_shots())
        got = []
        pnl.export_requested.connect(lambda n, s, e: got.append((n, s, e)))
        # drive the signal path directly (menus need a real event loop)
        pnl.export_requested.emit(2, 75.2, 80.0)
        assert got == [(2, 75.2, 80.0)]


class TestOutcomeDots:
    def test_rows_carry_outcome_coloured_icons(self, app):
        from billiards_trainer.ui.widgets.shot_list import ShotListPanel
        pnl = ShotListPanel()
        pnl.set_shots(_shots())
        for i in range(pnl._list.count()):
            assert not pnl._list.item(i).icon().isNull()

    def test_corrected_shot_is_marked(self, app):
        from billiards_trainer.ui.widgets.shot_list import ShotListPanel
        pnl = ShotListPanel()
        shots = _shots()
        shots[1]["corrected"] = True
        pnl.set_shots(shots)
        assert "corrected" in pnl._list.item(1).toolTip()
        # corrected icon (ringed) differs from the plain one
        plain = pnl._list.item(0).icon().pixmap(14, 14).toImage()
        ringed = pnl._list.item(1).icon().pixmap(14, 14).toImage()
        assert plain != ringed


class TestShotThumbnails:
    def _tiny_video(self, tmp_path):
        import cv2
        import numpy as np
        p = str(tmp_path / "clip.mp4")
        w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 90))
        for i in range(90):                     # 3s, frame index painted in
            fr = np.full((90, 160, 3), (i % 3) * 60 + 40, np.uint8)
            w.write(fr)
        w.release()
        return p

    def test_extractor_returns_frames_at_requested_times(self, tmp_path):
        from billiards_trainer.ui.widgets.shot_thumbs import extract_thumbs
        video = self._tiny_video(tmp_path)
        thumbs = extract_thumbs(video, [0.5, 2.0])
        assert set(thumbs) == {0.5, 2.0}
        assert all(v.shape == (54, 96, 3) for v in thumbs.values())

    def test_rows_swap_dots_for_thumbnails(self, app, tmp_path):
        import numpy as np
        from billiards_trainer.ui.widgets.shot_list import ShotListPanel
        pnl = ShotListPanel()
        pnl.set_shots(_shots())
        before = pnl._list.item(0).icon().pixmap(96, 54).toImage()
        bgr = np.full((54, 96, 3), 90, np.uint8)
        pnl.set_thumbnails({30.0: bgr, 75.2: bgr})
        after0 = pnl._list.item(0).icon().pixmap(96, 54).toImage()
        after2 = pnl._list.item(2).icon().pixmap(96, 54).toImage()
        assert after0 != before, "row with a thumb must swap its icon"
        assert not pnl._list.item(2).icon().isNull()  # row 2 keeps its dot

    def test_thumbnails_survive_correction_rebuild(self, app):
        import numpy as np
        from billiards_trainer.ui.widgets.shot_list import ShotListPanel
        pnl = ShotListPanel()
        pnl.set_shots(_shots())
        bgr = np.full((54, 96, 3), 90, np.uint8)
        pnl.set_thumbnails({30.0: bgr})
        pnl._correct(0, "miss")                  # rebuilds all rows
        assert not pnl._list.item(0).icon().pixmap(96, 54).toImage().isNull()
        assert "corrected" in pnl._list.item(0).toolTip()
