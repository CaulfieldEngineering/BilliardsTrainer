"""The sessions sidebar — the app's only navigation.

Broadcast-suite model: a persistent left rail listing LIVE plus every recorded
session, newest first. Click LIVE to watch the camera; click a session to play
it back through the analysis. The gear at the bottom opens Settings.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from ...config import EXPORTS_DIR
from ...version import __version__
from ..icons import icon
from ..theme import PALETTE


class SessionsSidebar(QFrame):
    live_selected = Signal()
    session_selected = Signal(str)   # absolute path to a session mp4
    settings_toggled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Sidebar")
        # Resizable via the main-window splitter (a fixed width clipped the
        # "Shots" header to "Sho" with no way to widen). The minimum keeps
        # all four headers legible at the default drag stop.
        self.setMinimumWidth(252)
        self.setMaximumWidth(560)
        # Where recordings are listed from; main_window points this at the
        # configured recordings folder (Settings -> Recording).
        self.recordings_dir: Path = EXPORTS_DIR
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 10, 8, 8)
        lay.setSpacing(8)

        brand = QLabel(f"<span style='color:{PALETTE.accent}'>●</span>  "
                       f"<b>Billiards Trainer</b>")
        brand.setStyleSheet("font-size: 15px;")
        lay.addWidget(brand)

        cap = QLabel("SESSIONS")
        cap.setObjectName("StatLabel")
        lay.addWidget(cap)

        # LIVE is a fixed button — sorting must never shuffle it.
        self._live_btn = QPushButton("▶  LIVE")
        self._live_btn.setObjectName("Ghost")
        self._live_btn.setCursor(Qt.PointingHandCursor)
        self._live_btn.clicked.connect(self.live_selected.emit)
        lay.addWidget(self._live_btn)

        # Explorer-style session table (Joe's ask, and ONLY that): four
        # sortable headers. Click a header to sort; default = Date, newest
        # first. Everything else (click-to-open, right-click menu) unchanged.
        self._list = QTreeWidget()
        self._list.setFrameShape(QFrame.NoFrame)
        self._list.setRootIsDecorated(False)
        self._list.setHeaderLabels(["Name", "Date", "Len", "Shots"])
        self._list.setSortingEnabled(True)
        self._list.sortByColumn(1, Qt.DescendingOrder)
        hdr = self._list.header()
        f = self._list.font()
        f.setPointSizeF(max(7.5, f.pointSizeF() - 0.5))
        self._list.setFont(f)
        # Section floors from MEASURED header text, not constants: constants
        # fit one theme's font and clip under another ("Shots" -> "Sho" was
        # exactly this class of bug). Name gets whatever remains.
        from PySide6.QtGui import QFontMetrics
        fm = QFontMetrics(hdr.font())
        widths = [fm.horizontalAdvance(t) + 14
                  for t in ("Name", "Date", "Len", "Shots")]
        for col in (1, 2, 3):
            hdr.resizeSection(col, widths[col])
        hdr.setMinimumSectionSize(min(widths))
        hdr.resizeSection(0, max(78, widths[0]))
        self.setMinimumWidth(max(252, sum(widths) + max(78, widths[0])
                                 - widths[0] + 24))
        self._list.itemClicked.connect(self._on_item)
        self._list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._context_menu)
        lay.addWidget(self._list, 1)

        gear = QPushButton("  Settings")
        gear.setObjectName("Ghost")
        gear.setIcon(icon("settings", PALETTE.text_dim))
        gear.setCursor(Qt.PointingHandCursor)
        gear.clicked.connect(self.settings_toggled.emit)
        lay.addWidget(gear)

        ver = QLabel(f"v{__version__}")
        ver.setObjectName("Faint")
        lay.addWidget(ver)

        self.refresh()

    # ------------------------------------------------------------------ #
    class _Row(QTreeWidgetItem):
        """Sorts by the hidden numeric role so Length sorts by seconds and
        Date by mtime — not alphabetically ("9m" < "42m" must be true)."""
        SORT_ROLE = Qt.UserRole + 1

        def __lt__(self, other):
            col = self.treeWidget().sortColumn() if self.treeWidget() else 0
            a = self.data(col, self.SORT_ROLE)
            b = other.data(col, self.SORT_ROLE)
            if a is not None and b is not None:
                return a < b
            return self.text(col).lower() < other.text(col).lower()

    def refresh(self) -> None:
        """Rebuild the session table (LIVE is its own button above it)."""
        current = self._current_path()
        self._list.setSortingEnabled(False)
        self._list.clear()
        try:
            clips = sorted(
                (p for ext in ("*.mp4", "*.mov", "*.m4v", "*.avi", "*.mkv")
                 for p in self.recordings_dir.glob(ext)
                 if not p.name.startswith(".")),
                key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            clips = []
        for p in clips[:60]:
            st = p.stat()
            when = datetime.fromtimestamp(st.st_mtime)
            name = when.strftime("%b %d %H:%M") if p.name.startswith("session-")                 else p.stem[:18]
            it = self._Row([name, when.strftime("%m/%d %H:%M"), "…", "…"])
            it.setData(0, Qt.UserRole, str(p))
            it.setData(0, self._Row.SORT_ROLE, name.lower())
            it.setData(1, self._Row.SORT_ROLE, st.st_mtime)
            it.setData(2, self._Row.SORT_ROLE, -1.0)
            it.setData(3, self._Row.SORT_ROLE, -1)
            it.setToolTip(0, f"{p.name}  ·  {st.st_size / 1e6:.0f} MB")
            self._list.addTopLevelItem(it)
        self._list.setSortingEnabled(True)
        self._load_summaries([str(p) for p in clips[:60]])
        if current:
            for i in range(self._list.topLevelItemCount()):
                if self._list.topLevelItem(i).data(0, Qt.UserRole) == current:
                    self._list.setCurrentItem(self._list.topLevelItem(i))
                    break

    def _load_summaries(self, paths: list[str]) -> None:
        """Fill rows with duration + shots, computed off-thread, applied on
        the UI thread over a queued signal (same bridge pattern as the shot
        thumbnails). Cache makes every pass after the first instant."""
        import threading

        from PySide6.QtCore import QObject, Signal

        class _Bridge(QObject):
            ready = Signal(str, dict)     # (path, summary)

        self._sum_bridge = _Bridge(self)
        self._sum_bridge.ready.connect(self._apply_summary)

        def work(bridge=self._sum_bridge, targets=list(paths)):
            from .. import session_summaries as ss
            cache = ss.load_cache()
            dirty = False
            for sp in targets:
                p = Path(sp)
                if not p.exists():
                    continue
                before = len(cache)
                try:
                    s = ss.summarize(p, cache)
                except Exception:  # noqa: BLE001 - one bad file, keep going
                    continue
                dirty = dirty or len(cache) != before
                try:
                    bridge.ready.emit(sp, s)
                except RuntimeError:
                    # The sidebar (and its bridge) died while we were
                    # summarizing — a refresh mid-teardown. Nothing to
                    # deliver to; save what we learned and stop quietly.
                    break
            if dirty:
                ss.save_cache(cache)

        threading.Thread(target=work, daemon=True, name="session-summaries").start()

    def _apply_summary(self, path: str, summary: dict) -> None:
        from .. import session_summaries as ss
        from PySide6.QtGui import QColor
        for i in range(self._list.topLevelItemCount()):
            it = self._list.topLevelItem(i)
            if it.data(0, Qt.UserRole) != path:
                continue
            dur = float(summary.get("dur_s") or 0.0)
            shots = summary.get("shots")
            if dur >= 3600:
                dtext = f"{int(dur // 3600)}h{int(dur % 3600 // 60):02d}"
            elif dur >= 60:
                dtext = f"{int(dur // 60)}m"
            else:
                dtext = f"{int(dur)}s" if dur > 0 else "–"
            it.setText(2, dtext)
            it.setData(2, self._Row.SORT_ROLE, dur)
            it.setText(3, "–" if shots is None else str(shots))
            it.setData(3, self._Row.SORT_ROLE, -1 if shots is None else int(shots))
            try:
                mb = Path(path).stat().st_size / 1e6
            except OSError:
                return
            if ss.is_stub(summary, mb):
                dim = QColor(PALETTE.text_faint)
                for c in range(4):
                    it.setForeground(c, dim)
            return

    def select_live(self) -> None:
        self._list.clearSelection()
        self._list.setCurrentItem(None)

    def _current_path(self) -> str:
        it = self._list.currentItem()
        return it.data(0, Qt.UserRole) if it else ""

    def _on_item(self, item: QTreeWidgetItem, _col: int = 0) -> None:
        path = item.data(0, Qt.UserRole)
        if path:
            self.session_selected.emit(path)

    def mark_live(self) -> None:
        """Programmatic switch back to LIVE (e.g. after an error)."""
        self.select_live()

    def _context_menu(self, pos) -> None:
        item = self._list.itemAt(pos)
        path = item.data(0, Qt.UserRole) if item else ""
        if not path:
            return   # LIVE row / empty area: nothing to reveal
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        from ..reveal import reveal_label
        reveal_act = menu.addAction(reveal_label())
        delete = menu.addAction("Delete recording…")
        chosen = menu.exec(self._list.mapToGlobal(pos))
        if chosen is reveal_act:
            from ..reveal import reveal
            reveal(path)
        elif chosen is delete:
            from PySide6.QtWidgets import QMessageBox
            if QMessageBox.question(
                    self, "Delete recording",
                    f"Delete {Path(path).name}? This can't be undone.",
            ) == QMessageBox.StandardButton.Yes:
                Path(path).unlink(missing_ok=True)
                self.refresh()
