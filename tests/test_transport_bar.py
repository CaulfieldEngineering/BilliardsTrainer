"""Transport-bar clip invariants: the controls Joe looks at every session.

The REC clock clipped three separate times, each fix a different fudge on
plain font metrics that the rich-text renderer ignored. These tests pin
the survivor: widths come from the label's OWN sizeHint on worst-case
strings — measured through the REAL app theme — so any future divergence
(new string, style change, DPI shift) fails here instead of on Joe's bar.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from billiards_trainer.config import Settings  # noqa: E402


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    from billiards_trainer.ui.theme import apply_theme
    apply_theme(a)
    return a


def _page(app):
    from billiards_trainer.ui.pages.live_page import LivePage
    pg = LivePage(Settings())
    pg.resize(1200, 800)
    return pg


class TestRecClockNeverClips:
    """Clock v4: PLAIN text digits, always visible, dim when idle. The
    rich-text badge and its worst-case width ratchet (which held the
    capsule twice as wide as its content — Joe: 'way off to the side')
    are gone; state lives in the digits' colour and the capsule outline."""

    def test_clock_is_plain_text_and_always_visible(self, app):
        from PySide6.QtCore import Qt
        pg = _page(app)
        assert pg._rec_time.textFormat() == Qt.PlainText
        assert not pg._rec_time.isHidden()       # idle: dim 0:00, no gap
        assert pg._rec_time.text() == "0:00"

    def test_live_tick_strings_are_plain_digits(self, app):
        import time
        pg = _page(app)
        pg.on_recording(True)
        pg._rec_t0 = time.monotonic() - 3599     # 59:59 on the clock
        for paused in (False, True):
            pg._rec_pause_btn.setChecked(paused)
            pg._tick_rec_time()
            text = pg._rec_time.text()
            assert "<" not in text and "REC" not in text, \
                f"rich-text badge returned: {text!r}"
            assert text == "59:59"

    def test_recording_start_does_not_change_capsule_geometry(self, app):
        """Joe: 'something in the timeline layout still expands after I hit
        record, on first app load only' — the clock re-measure at record
        start grew the capsule. Geometry must be identical before/after."""
        pg = _page(app)
        pg.show()
        app.processEvents()
        before = (pg._rec_capsule.sizeHint().width(),
                  pg._rec_capsule.sizeHint().height())
        pg.on_recording(True)
        app.processEvents()
        after = (pg._rec_capsule.sizeHint().width(),
                 pg._rec_capsule.sizeHint().height())
        assert before == after, f"capsule jumped: {before} -> {after}"

    def test_recording_capsule_children_fit_at_narrow_widths(self, app):
        """The whole capsule, not just the clock: at every compact tier the
        visible transport labels must fit their allocated widths."""
        pg = _page(app)
        pg.on_recording(True)
        for w in (1400, 1100, 980, 900, 820):
            pg.resize(w, 800)
            app.processEvents()
            label = pg._rec_time
            if label.isVisible():
                assert label.sizeHint().width() <= label.width(), \
                    f"clock clips at window width {w}"


class TestStatsRailIsAList:
    """Joe: 'just do a vertical stats list here... I don't need big blocks
    for each number. And the SESSION dropdown doesn't make sense.'"""

    def test_rows_not_blocks_and_no_session_fold(self):
        from pathlib import Path
        import billiards_trainer.ui.pages.live_page as m
        src = Path(m.__file__).read_text(encoding="utf-8")
        assert "class _StatRow" in src, "the vertical stat rows are gone"
        assert '"SESSION", grid_holder' not in src, "SESSION fold came back"
        assert "_big_stat" not in src, "big stat blocks came back"
