"""Mini view + narrow-window mode (Joe's side-by-side ask).

Headless (offscreen platform): construction, frame painting, geometry
persistence round-trip, and the responsive hide thresholds.
"""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class _Packet:
    def __init__(self, persp=None, bird=None):
        self.perspective = persp
        self.birdseye = bird


class TestMiniView:
    def test_construct_and_paint(self, app):
        from billiards_trainer.ui.widgets.mini_view import MiniView
        m = MiniView()
        m.show()
        frame = np.zeros((120, 160, 3), np.uint8)
        m.on_frame(_Packet(persp=frame))
        m.on_recording(True)
        assert m._rec_btn.isChecked() and "STOP" in m._rec_btn.text()
        m.on_recording(False)
        assert not m._rec_btn.isChecked() and "REC" in m._rec_btn.text()
        m.close()

    def test_hidden_mini_skips_painting(self, app):
        """Painting work must stop when the mini is closed — on_frame is wired
        permanently to the controller signal."""
        from billiards_trainer.ui.widgets.mini_view import MiniView
        m = MiniView()
        # never shown
        m.on_frame(_Packet(persp=np.zeros((10, 10, 3), np.uint8)))
        assert m._video._pixmap is None

    def test_geometry_roundtrip(self, app):
        from billiards_trainer.ui.widgets.mini_view import MiniView
        m = MiniView()
        m.setGeometry(50, 60, 400, 330)
        s = m.geometry_string()
        m2 = MiniView()
        m2.apply_geometry_string(s)
        assert m2.geometry().width() == 400
        assert m2.geometry().height() == 330

    def test_bad_geometry_string_ignored(self, app):
        from billiards_trainer.ui.widgets.mini_view import MiniView
        m = MiniView()
        m.apply_geometry_string("garbage")
        m.apply_geometry_string("")

    def test_aspect_snap_follows_frame(self, app):
        """The window hugs the video: height follows width by frame aspect."""
        from billiards_trainer.ui.widgets.mini_view import MiniView
        m = MiniView()
        m.show()
        m.resize(400, 400)
        # a 2:1 frame (h=200, w=400); two sends — the paint throttle
        # (every 2nd frame) is part of the contract now
        for _ in range(2):
            m.on_frame(_Packet(persp=np.zeros((200, 400, 3), np.uint8)))
        assert abs(m.width() / m.height() - 2.0) < 0.1
        m.close()

    def test_record_button_emits_toggle(self, app):
        from billiards_trainer.ui.widgets.mini_view import MiniView
        m = MiniView()
        got = []
        m.record_toggled.connect(got.append)
        m._rec_btn.click()
        assert got == [True]
        m._rec_btn.click()
        assert got == [True, False]

    def test_double_click_toggles_view(self, app):
        from billiards_trainer.ui.widgets.mini_view import MiniView
        m = MiniView()
        m.show()
        persp = np.zeros((10, 10, 3), np.uint8)
        bird = np.full((10, 10, 3), 200, np.uint8)
        assert m._show_birdseye is False
        m._show_birdseye = True                      # what the double-click flips
        for _ in range(2):                           # paint throttle passes 2nd
            m.on_frame(_Packet(persp=persp, bird=bird))
        # birdseye variant painted: pixmap exists
        assert m._video._pixmap is not None
        m.close()


class TestNarrowMode:
    def test_config_field_persists(self):
        from billiards_trainer.config import Settings
        s = Settings()
        assert hasattr(s.ui, "mini_geometry")

    def test_narrow_mode_keeps_stats_hides_bar_chrome(self, app):
        """Joe: stats must survive narrowing. The bar sheds context-dead
        controls instead (playback transport while live)."""
        from billiards_trainer.config import Settings
        from billiards_trainer.ui.pages.live_page import LivePage
        page = LivePage(Settings())
        page.show()                # hidden widgets defer resize events
        page.resize(700, 600)
        app.processEvents()
        assert page._rail_stack.isVisibleTo(page), "stats rail must survive"
        assert not page._alert_badge.isVisibleTo(page)
        assert not page._seek.isVisibleTo(page), "transport is dead weight live"
        assert page._rec_capsule.isVisibleTo(page), "record must stay while live"
        page.resize(1200, 700)
        app.processEvents()
        assert page._alert_badge.isVisibleTo(page)
        assert page._seek.isVisibleTo(page)

    def test_compact_playback_keeps_transport_drops_record(self, app):
        from billiards_trainer.config import Settings
        from billiards_trainer.ui.pages.live_page import LivePage
        page = LivePage(Settings())
        page.show()
        page.resize(700, 600)
        page.set_video_mode(True, total=100, fps=30.0)
        app.processEvents()
        assert page._seek.isVisibleTo(page), "transport is the point in playback"
        assert not page._rec_capsule.isVisibleTo(page), "cannot record playback"
