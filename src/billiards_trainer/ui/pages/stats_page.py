"""Statistics page — historical make/miss performance.

Populated from the DB in Phase 5. Until a session exists it shows an empty
state. ``refresh()`` re-queries the repository.
"""

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...config import EXPORTS_DIR
from ..widgets.common import Card, EmptyState, StatCard, section_header


class StatsPage(QWidget):
    def __init__(self, repository=None, parent=None):
        super().__init__(parent)
        self._repo = repository
        self._stack = QStackedWidget(self)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self._stack)

        self._empty = EmptyState(
            "stats", "No stats yet",
            "Play a session on the Live tab and your make/miss history, streaks, "
            "and pocket distribution will appear here.")
        self._stack.addWidget(self._empty)

        self._content = self._build_content()
        self._stack.addWidget(self._content)
        self.refresh()

    def _build_content(self) -> QWidget:
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)

        header = QHBoxLayout()
        header.addWidget(section_header("Performance"))
        header.addStretch(1)
        self._export_status = QLabel("")
        self._export_status.setObjectName("Faint")
        header.addWidget(self._export_status)
        csv_btn = QPushButton("Export CSV")
        csv_btn.setCursor(Qt.PointingHandCursor)
        csv_btn.clicked.connect(lambda: self._export("csv"))
        json_btn = QPushButton("Export JSON")
        json_btn.setCursor(Qt.PointingHandCursor)
        json_btn.clicked.connect(lambda: self._export("json"))
        header.addWidget(csv_btn)
        header.addWidget(json_btn)
        root.addLayout(header)

        kpis = QHBoxLayout()
        kpis.setSpacing(14)
        self._k_sessions = StatCard("Sessions", "0")
        self._k_shots = StatCard("Total shots", "0")
        self._k_make = StatCard("Make %", "0%", accent=True)
        self._k_streak = StatCard("Best streak", "0")
        for k in (self._k_sessions, self._k_shots, self._k_make, self._k_streak):
            kpis.addWidget(k)
        root.addLayout(kpis)

        body = QGridLayout()
        body.setSpacing(16)
        pocket_card = Card(padding=18)
        pocket_card.add(self._h("By pocket"))
        self._pocket_table = self._table(["Pocket", "Makes", "Attempts", "Make %"])
        pocket_card.add(self._pocket_table)
        body.addWidget(pocket_card, 0, 0)

        sessions_card = Card(padding=18)
        sessions_card.add(self._h("Recent sessions"))
        self._sessions_table = self._table(["Date", "Mode", "Shots", "Makes", "Make %"])
        sessions_card.add(self._sessions_table)
        body.addWidget(sessions_card, 0, 1)
        body.setColumnStretch(0, 1)
        body.setColumnStretch(1, 1)
        root.addLayout(body, 1)
        return w

    @staticmethod
    def _h(text: str):
        from PySide6.QtWidgets import QLabel
        lbl = QLabel(f"<b>{text}</b>")
        return lbl

    @staticmethod
    def _table(headers: list[str]) -> QTableWidget:
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        t.setSelectionMode(QTableWidget.NoSelection)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        t.setShowGrid(False)
        return t

    def _export(self, kind: str) -> None:
        if self._repo is None:
            return
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            if kind == "csv":
                dest = self._repo.export_csv(EXPORTS_DIR / "shots.csv")
            else:
                dest = self._repo.export_json(EXPORTS_DIR / "sessions.json")
        except Exception as exc:  # noqa: BLE001
            self._export_status.setText(f"Export failed: {exc}")
            return
        self._export_status.setText(f"Saved {dest.name}")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(dest.parent)))

    def refresh(self) -> None:
        if self._repo is None:
            self._stack.setCurrentWidget(self._empty)
            return
        try:
            summary = self._repo.global_summary()
        except Exception:
            self._stack.setCurrentWidget(self._empty)
            return
        if not summary or summary.get("sessions", 0) == 0:
            self._stack.setCurrentWidget(self._empty)
            return
        self._stack.setCurrentWidget(self._content)
        self._k_sessions.set_value(str(summary["sessions"]))
        self._k_shots.set_value(str(summary["shots"]))
        self._k_make.set_value(f"{summary['make_pct']:.0f}%")
        self._k_streak.set_value(str(summary["best_streak"]))
        self._fill(self._pocket_table, summary.get("by_pocket", []))
        self._fill(self._sessions_table, summary.get("recent_sessions", []))

    @staticmethod
    def _fill(table: QTableWidget, rows: list[list]) -> None:
        table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                if c > 0:
                    item.setTextAlignment(Qt.AlignCenter)
                table.setItem(r, c, item)
