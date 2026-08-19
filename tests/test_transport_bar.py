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
    def test_every_tick_string_fits_the_fixed_width(self, app):
        pg = _page(app)
        pg.on_recording(True)                    # styling final + re-measure
        label = pg._rec_time
        for worst in pg.REC_CLOCK_WORST:
            label.setText(worst)
            need = label.sizeHint().width()
            assert need <= label.width(), \
                f"clock clips: needs {need}px, has {label.width()}px for {worst!r}"

    def test_live_tick_strings_fit_too(self, app):
        import time
        pg = _page(app)
        pg.on_recording(True)
        pg._rec_t0 = time.monotonic() - 3599     # 59:59 on the clock
        for paused in (False, True):
            pg._rec_pause_btn.setChecked(paused)
            pg._tick_rec_time()
            label = pg._rec_time
            assert label.sizeHint().width() <= label.width(), \
                f"tick string clips (paused={paused}): {label.text()!r}"

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
