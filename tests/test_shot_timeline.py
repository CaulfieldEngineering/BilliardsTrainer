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
        tl.mouseReleaseEvent(Ev())    # click semantics decide on release
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

    def test_hover_off_clip_is_quiet(self, app):
        # Joe: "I don't need the tooltip to be so prevalent" — bare lane
        # hovers say nothing now; only shot regions speak.
        tl = _tl(app)
        tl.add_shot(100.0, 106.0, "make", 1)
        assert tl.hover_text(300.0) is None

    def test_hover_marks_corrections(self, app):
        tl = _tl(app)
        tl.add_shot(100.0, 106.0, "make", 1)
        tl._shots[0]["corrected"] = True
        assert "(corrected)" in tl.hover_text(101.0)


class TestZoomPan:
    """Editor zoom/pan: wheel zooms around the cursor, middle-drag pans —
    all through pure methods (zoom_at / pan_by / view_range)."""

    def test_zoom_keeps_cursor_time_at_same_pixel(self, app):
        tl = _tl(app)
        tl.zoom_at(300.0, 2.0)               # cursor mid-lane = t=300s
        lo, hi = tl.view_range()
        assert (lo, hi) == (150.0, 450.0)
        assert abs(tl._x(300.0) - 300.0) < 1e-6   # t=300 still under cursor

    def test_zoom_clamps_to_min_span(self, app):
        tl = _tl(app)
        for _ in range(20):
            tl.zoom_at(300.0, 2.0)
        lo, hi = tl.view_range()
        assert hi - lo >= tl.MIN_SPAN_S - 1e-9

    def test_zoom_out_past_full_resets_to_whole_session(self, app):
        tl = _tl(app)
        tl.zoom_at(300.0, 2.0)
        tl.zoom_at(300.0, 0.25)              # zooming way out
        assert tl.view_range() == (0.0, 600.0)
        assert tl._view is None

    def test_pan_clamps_at_session_edges(self, app):
        tl = _tl(app)
        tl.zoom_at(300.0, 2.0)               # (150, 450)
        tl.pan_by(600.0)                      # drag right -> earlier, clamped
        assert tl.view_range() == (0.0, 300.0)
        tl.pan_by(-2400.0)                    # drag far left -> later, clamped
        assert tl.view_range() == (300.0, 600.0)

    def test_pan_is_noop_when_not_zoomed(self, app):
        tl = _tl(app)
        tl.pan_by(200.0)
        assert tl.view_range() == (0.0, 600.0)

    def test_click_seeks_through_zoomed_view(self, app):
        tl = _tl(app)
        tl.zoom_at(300.0, 2.0)               # view (150, 450)
        got = []
        tl.clicked.connect(got.append)

        class Ev:
            def button(self):
                from PySide6.QtCore import Qt
                return Qt.LeftButton
            def pos(self):
                from PySide6.QtCore import QPoint
                return QPoint(0, 17)          # left edge of the zoomed view
        tl.mousePressEvent(Ev())
        tl.mouseReleaseEvent(Ev())    # click semantics decide on release
        assert got and abs(got[0] - 150.0) < 1.0

    def test_live_follow_lane_ignores_zoom(self, app):
        tl = _tl(app)
        tl.follow_window_s = 120.0
        tl.set_live_clock(600.0)
        tl.zoom_at(300.0, 2.0)
        # The live window rolls CONTINUOUSLY now (smooth scroll interpolates
        # between clock syncs), so compare with tolerance, not equality.
        lo, hi = tl.view_range()
        assert abs(lo - 480.0) < 0.5 and abs(hi - 600.0) < 0.5, \
            f"zoom hijacked the live window: {(lo, hi)}"
        assert abs((hi - lo) - 120.0) < 1e-6

    def test_clear_resets_zoom(self, app):
        tl = _tl(app)
        tl.zoom_at(300.0, 2.0)
        tl.clear()
        assert tl._view is None


class TestFilmstripV3:
    def test_strip_times_are_stable_cache_keys(self, app):
        tl = _tl(app)
        a = tl.strip_times(0.0, 600.0)
        b = tl.strip_times(0.0, 600.0)
        assert a == b and len(a) >= 10
        assert all(isinstance(t, int) for t in a)

    def test_full_shot_region_paints_with_thumbs(self, app, tmp_path):
        import cv2
        import numpy as np
        video = tmp_path / "clip.mp4"
        w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 90))
        for _ in range(60):
            w.write(np.full((90, 160, 3), 60, np.uint8))
        w.release()
        tl = _tl(app)
        tl.set_media_source(str(video))
        tl.add_shot(100.0, 112.0, "make", 1)
        tl.show()
        tl.repaint()          # requests thumbs; paints placeholders first
        tl.close()

    def test_live_mode_skips_filmstrip(self, app):
        tl = _tl(app)
        tl.set_media_source("nonexistent.mp4")
        tl.follow_window_s = 120.0
        tl.set_live_clock(300.0)
        tl.show()
        tl.repaint()          # must not request strips in live-follow
        tl.close()
        assert not tl._strip_pending


class TestScrubOverhaul:
    def _drag(self, tl, x0, x1):
        from PySide6.QtCore import QPoint, Qt

        class Press:
            def button(self):
                return Qt.LeftButton
            def pos(self):
                return QPoint(int(x0), 50)

        class Move:
            def buttons(self):
                return Qt.LeftButton
            def pos(self):
                return QPoint(int(x1), 50)

        class Release:
            def button(self):
                return Qt.LeftButton
            def pos(self):
                return QPoint(int(x1), 50)
        tl.mousePressEvent(Press())
        tl.mouseMoveEvent(Move())
        tl.mouseReleaseEvent(Release())

    def test_drag_scrubs_instead_of_clicking(self, app):
        tl = _tl(app)
        tl.add_shot(100.0, 106.0, "make")
        started, scrubs, ended, clicks = [], [], [], []
        tl.scrub_started.connect(lambda: started.append(1))
        tl.scrubbed.connect(scrubs.append)
        tl.scrub_ended.connect(lambda: ended.append(1))
        tl.clicked.connect(clicks.append)
        self._drag(tl, 100, 300)          # 200px drag = scrub
        assert started and ended and scrubs, "drag must emit scrub sequence"
        assert not clicks, "a drag is not a click"
        assert abs(scrubs[-1] - 300.0) < 2.0   # 600px lane / 600s = 1s per px

    def test_click_still_snaps_shot_to_routine(self, app):
        tl = _tl(app)
        tl.add_shot(100.0, 106.0, "make")
        clicks, scrubs = [], []
        tl.clicked.connect(clicks.append)
        tl.scrubbed.connect(scrubs.append)
        self._drag(tl, 101, 103)          # 2px = click, not drag
        assert clicks and abs(clicks[0] - 95.0) < 1.0
        assert not scrubs

    def test_tooltip_quiet_off_shots(self, app):
        tl = _tl(app)
        tl.add_shot(100.0, 106.0, "make")
        assert tl.hover_text(300.0) is None, "no tooltip on bare lane"
        assert "Shot 1" in tl.hover_text(101.0)


class TestCompactLane:
    """Joe (v3.2): 'a tighter rectangle across the top... too much padding
    and negative space' — the lane is compact and the big empty-state help
    text is gone for good."""

    def test_lane_is_compact(self, app):
        from billiards_trainer.ui.widgets.shot_timeline import ShotTimeline
        tl = ShotTimeline()
        assert tl.height() <= 80, f"lane grew back to {tl.height()}px"

    def test_no_placeholder_help_text(self):
        from pathlib import Path
        import billiards_trainer.ui.widgets.shot_timeline as m
        src = Path(m.__file__).read_text(encoding="utf-8")
        assert "clips appear here" not in src, \
            "the empty-state help text Joe removed came back"


class TestMasterContainer:
    """Joe: 'the timeline has no master border. It just appears to be a
    massive negative space.' The lane must paint a visible card — fill and
    border — even when completely empty."""

    def test_empty_lane_paints_a_container(self, app):
        from billiards_trainer.ui.theme import PALETTE
        from billiards_trainer.ui.widgets.shot_timeline import ShotTimeline
        tl = ShotTimeline()
        tl.resize(600, tl.height())
        img = tl.grab().toImage()
        top_centre = img.pixelColor(300, 3).name().lower()
        assert top_centre == PALETTE.bg_elevated.lower(), \
            f"empty lane top edge paints {top_centre}, not the card fill " \
            f"({PALETTE.bg_elevated}) — the master container is missing"

    def test_splitter_handles_are_invisible(self):
        from billiards_trainer.ui.theme import build_stylesheet
        qss = build_stylesheet()
        assert "QSplitter::handle { background: transparent; }" in qss, \
            "splitter slabs came back (Joe: 'rounded cards separated by " \
            "rectangular dividers is bad design')"


class TestFlatPanelLanguage:
    """Joe: 'do away with the round card method altogether' and 'label the
    Shot Timeline window just like the others.'"""

    def test_round_card_language_is_dead(self):
        from billiards_trainer.ui.theme import build_stylesheet
        qss = build_stylesheet()
        assert "border-radius: 14px" not in qss, "rounded cards came back"
        assert "QFrame#StatTile" in qss, "stat tiles regained chrome"

    def test_timeline_panel_is_labeled(self):
        from pathlib import Path
        import billiards_trainer.ui.pages.live_page as m
        src = Path(m.__file__).read_text(encoding="utf-8")
        assert "SHOT TIMELINE" in src, "the timeline lost its caption"
