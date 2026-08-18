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
    QListWidget,
    QListWidgetItem,
    QPushButton,
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
        self.setFixedWidth(210)
        # Where recordings are listed from; main_window points this at the
        # configured recordings folder (Settings -> Recording).
        self.recordings_dir: Path = EXPORTS_DIR
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 18, 14, 14)
        lay.setSpacing(8)

        brand = QLabel(f"<span style='color:{PALETTE.accent}'>●</span>  "
                       f"<b>Billiards Trainer</b>")
        brand.setStyleSheet("font-size: 15px;")
        lay.addWidget(brand)

        cap = QLabel("SESSIONS")
        cap.setObjectName("StatLabel")
        lay.addWidget(cap)

        self._list = QListWidget()
        self._list.setFrameShape(QFrame.NoFrame)
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
    def refresh(self) -> None:
        """Rebuild the list: LIVE on top, then session recordings newest-first."""
        current = self._current_path()
        self._list.clear()
        live = QListWidgetItem("▶  LIVE")
        live.setData(Qt.UserRole, "")           # empty payload = live camera
        f = live.font()
        f.setBold(True)
        live.setFont(f)
        self._list.addItem(live)
        # ANY video dropped into the recordings folder is a session — recorded
        # by the app (session-*.mp4) or copied in by hand.
        try:
            # pathlib.glob matches dotfiles, so in-progress/aborted hidden
            # ".session-*.part.mp4" files showed up in the list — skip them.
            clips = sorted(
                (p for ext in ("*.mp4", "*.mov", "*.m4v", "*.avi", "*.mkv")
                 for p in self.recordings_dir.glob(ext)
                 if not p.name.startswith(".")),
                key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            clips = []
        for p in clips[:60]:
            mb = p.stat().st_size / 1e6
            if p.name.startswith("session-"):
                when = datetime.fromtimestamp(p.stat().st_mtime).strftime("%b %d  %H:%M")
                label = f"{when}   ·  {mb:.0f} MB"
            else:  # a hand-copied clip: its filename IS its identity
                label = f"{p.stem[:22]}   ·  {mb:.0f} MB"
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, str(p))
            it.setToolTip(p.name)
            self._list.addItem(it)
        # Rows fill in with duration + shot count off-thread (cached after
        # the first pass); "13 MB" answers nothing Joe actually asks.
        self._load_summaries([str(p) for p in clips[:60]])
        # restore selection (default LIVE)
        idx = 0
        if current:
            for i in range(self._list.count()):
                if self._list.item(i).data(Qt.UserRole) == current:
                    idx = i
                    break
        self._list.setCurrentRow(idx)

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
                bridge.ready.emit(sp, s)
            if dirty:
                ss.save_cache(cache)

        threading.Thread(target=work, daemon=True, name="session-summaries").start()

    def _apply_summary(self, path: str, summary: dict) -> None:
        from .. import session_summaries as ss
        for i in range(1, self._list.count()):
            it = self._list.item(i)
            if it.data(Qt.UserRole) != path:
                continue
            p = Path(path)
            try:
                mb = p.stat().st_size / 1e6
            except OSError:
                return
            if p.name.startswith("session-"):
                when = datetime.fromtimestamp(p.stat().st_mtime).strftime("%b %d  %H:%M")
                it.setText(ss.row_text(when, mb, summary))
            it.setToolTip(f"{p.name}  ·  {mb:.0f} MB")
            if ss.is_stub(summary, mb):
                from PySide6.QtGui import QColor
                it.setForeground(QColor(PALETTE.text_faint))
            return

    def select_live(self) -> None:
        self._list.setCurrentRow(0)

    def _current_path(self) -> str:
        it = self._list.currentItem()
        return it.data(Qt.UserRole) if it else ""

    def _on_item(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        if path:
            self.session_selected.emit(path)
        else:
            self.live_selected.emit()

    def mark_live(self) -> None:
        """Programmatic switch back to LIVE (e.g. after an error)."""
        self.select_live()

    def _context_menu(self, pos) -> None:
        item = self._list.itemAt(pos)
        path = item.data(Qt.UserRole) if item else ""
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
