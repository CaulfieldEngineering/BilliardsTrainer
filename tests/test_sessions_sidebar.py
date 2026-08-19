"""Explorer-style session table: four sortable headers (Joe's ask, only that)."""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _sidebar(app, tmp_path):
    from billiards_trainer.ui.widgets.sessions_sidebar import SessionsSidebar
    now = time.time()
    for name, age, size in (("session-a.mp4", 3600, 100),
                            ("session-b.mp4", 60, 5000),
                            ("zclip.mp4", 7200, 900)):
        f = tmp_path / name
        f.write_bytes(b"0" * size)
        os.utime(f, (now - age, now - age))
    sb = SessionsSidebar()
    sb.recordings_dir = tmp_path
    sb.refresh()
    return sb


class TestSessionTable:
    def test_columns_and_default_sort_newest_first(self, app, tmp_path):
        sb = _sidebar(app, tmp_path)
        t = sb._list
        assert [t.headerItem().text(i) for i in range(4)] == \
            ["Name", "Date", "Len", "Shots"]
        first = t.topLevelItem(0)
        assert "session-b" in first.data(0, Qt.UserRole)   # newest on top

    def test_sort_by_length_is_numeric(self, app, tmp_path):
        sb = _sidebar(app, tmp_path)
        t = sb._list
        # inject summaries: 9s vs 42m vs 600s — numeric order, not lexical
        for i in range(t.topLevelItemCount()):
            it = t.topLevelItem(i)
            dur = [9.0, 2520.0, 600.0][i]
            sb._apply_summary(it.data(0, Qt.UserRole),
                              {"dur_s": dur, "shots": i})
        t.sortByColumn(2, Qt.DescendingOrder)
        durs = [t.topLevelItem(i).data(2, sb._Row.SORT_ROLE)
                for i in range(t.topLevelItemCount())]
        assert durs == sorted(durs, reverse=True)

    def test_click_emits_session_path(self, app, tmp_path):
        sb = _sidebar(app, tmp_path)
        got = []
        sb.session_selected.connect(got.append)
        sb._on_item(sb._list.topLevelItem(0), 0)
        assert got and got[0].endswith(".mp4")


class TestHeadersFit:
    def test_all_four_headers_fit_at_minimum_width(self, app, tmp_path):
        """Joe: '"Shots" is "Sho" and I can't expand it.' At the sidebar's
        minimum width every header must have at least its text width."""
        from PySide6.QtGui import QFontMetrics
        sb = _sidebar(app, tmp_path)
        sb.resize(sb.minimumWidth(), 600)
        app.processEvents()
        t = sb._list
        fm = QFontMetrics(t.header().font())
        for col in range(4):
            need = fm.horizontalAdvance(t.headerItem().text(col)) + 12
            assert t.header().sectionSize(col) >= need, \
                f"column {col} ({t.headerItem().text(col)!r}) clips: " \
                f"{t.header().sectionSize(col)}px < {need}px"

    def test_sidebar_is_not_fixed_width(self, app, tmp_path):
        sb = _sidebar(app, tmp_path)
        assert sb.maximumWidth() > sb.minimumWidth(), "sidebar must be resizable"
