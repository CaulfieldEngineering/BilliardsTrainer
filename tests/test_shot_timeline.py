"""DAW shot timeline: mapping, clip hit-tests, pre-roll seeks."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _tl(app):
    from billiards_trainer.ui.widgets.shot_timeline import ShotTimeline
    tl = ShotTimeline(pre_roll_s=5.0)
    tl.resize(600, 34)
    tl.set_duration(600.0)          # ten minutes
    return tl


class TestShotTimeline:
    def test_clip_lookup_includes_pre_roll(self, app):
        tl = _tl(app)
        tl.add_shot(100.0, 106.0, "make", 1)
        assert tl.shot_at(96.0) is not None       # inside the routine window
        assert tl.shot_at(103.0) is not None      # inside the shot
        assert tl.shot_at(90.0) is None           # before the routine

    def test_click_on_clip_seeks_to_routine_start(self, app):
        tl = _tl(app)
        tl.add_shot(100.0, 106.0, "miss")
        got = []
        tl.clicked.connect(got.append)

        class Ev:
            def button(self):
                from PySide6.QtCore import Qt
                return Qt.LeftButton
            def pos(self):
                from PySide6.QtCore import QPoint
                return QPoint(int(600 * 101.0 / 600.0), 17)   # t=101s
        tl.mousePressEvent(Ev())
        assert got and abs(got[0] - 95.0) < 0.6, f"expected routine start, got {got}"

    def test_live_duration_grows_with_shots(self, app):
        from billiards_trainer.ui.widgets.shot_timeline import ShotTimeline
        tl = ShotTimeline()
        tl.add_shot(30.0, 38.0, "make")
        assert tl._duration >= 38.0

    def test_paints_without_error(self, app):
        tl = _tl(app)
        tl.add_shot(10.0, 15.0, "make")
        tl.add_shot(50.0, 58.0, "scratch")
        tl.show()
        tl.repaint()
        tl.close()


class TestHoverCards:
    def test_hover_identifies_the_clip(self, app):
        tl = _tl(app)
        tl.add_shot(100.0, 106.0, "make", 1)
        text = tl.hover_text(101.0)
        assert "Shot 1" in text and "MAKE" in text and "6.0s" in text

    def test_hover_off_clip_gives_hint(self, app):
        tl = _tl(app)
        tl.add_shot(100.0, 106.0, "make", 1)
        assert "Click to seek" in tl.hover_text(300.0)

    def test_hover_marks_corrections(self, app):
        tl = _tl(app)
        tl.add_shot(100.0, 106.0, "make", 1)
        tl._shots[0]["corrected"] = True
        assert "(corrected)" in tl.hover_text(101.0)
