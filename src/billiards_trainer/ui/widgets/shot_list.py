"""Per-shot review list: the session's shots as navigable rows.

Dossier slice 2 (Joe: "parse shots, per session, so I can review my
gameplay and review the quality of the tracking"). The playback rail shows
every shot as a row — number, outcome, clock position, duration, balls
potted — click one (or use Prev/Next) and playback seeks to the start of
the pre-shot routine. The same data feeds the timeline lane; this is the
list view of it.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..theme import PALETTE

_OUTCOME = {
    "make": ("●", "#3FB950", "MAKE"),
    "miss": ("●", "#7D8590", "MISS"),
    "scratch": ("●", "#E3B341", "SCRATCH"),
}


class ShotListPanel(QWidget):
    """``shot_selected(seconds)`` asks the owner to seek (routine start)."""

    shot_selected = Signal(float)

    def __init__(self, pre_roll_s: float = 5.0, parent=None):
        super().__init__(parent)
        self.pre_roll_s = pre_roll_s
        self._shots: list[dict] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        head = QHBoxLayout()
        cap = QLabel("SHOTS")
        cap.setObjectName("StatLabel")
        head.addWidget(cap)
        self._count = QLabel("")
        self._count.setObjectName("Faint")
        head.addWidget(self._count)
        head.addStretch(1)
        self._prev = QPushButton("‹")
        self._next = QPushButton("›")
        for b, tip in ((self._prev, "Previous shot"), (self._next, "Next shot")):
            b.setFixedSize(24, 24)
            b.setToolTip(tip)
            b.setCursor(Qt.PointingHandCursor)
        self._prev.clicked.connect(lambda: self._step(-1))
        self._next.clicked.connect(lambda: self._step(1))
        head.addWidget(self._prev)
        head.addWidget(self._next)
        root.addLayout(head)

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SingleSelection)
        self._list.itemActivated.connect(self._on_item)
        self._list.itemClicked.connect(self._on_item)
        root.addWidget(self._list, 1)

        self._empty = QLabel("No shots detected in this session yet.")
        self._empty.setObjectName("Faint")
        self._empty.setWordWrap(True)
        root.addWidget(self._empty)
        self._sync_empty()

    # ------------------------------------------------------------------ #
    def set_shots(self, shots: list[dict]) -> None:
        self._shots = list(shots or [])
        self._list.clear()
        for i, s in enumerate(self._shots):
            dot, colour, name = _OUTCOME.get(s.get("outcome", "miss"),
                                             ("●", PALETTE.text_dim, "?"))
            start = float(s.get("start", 0.0))
            dur = max(0.0, float(s.get("end", start)) - start)
            pot = int(s.get("pocketed", 0))
            mm, ss = int(start) // 60, int(start) % 60
            text = f"{i + 1:>3}   {name:<7} {mm}:{ss:02d}   {dur:.1f}s"
            if pot > 1:
                text += f"   ×{pot}"
            item = QListWidgetItem(f"{dot}  {text}")
            item.setForeground(Qt.GlobalColor.white)
            item.setData(Qt.UserRole, start)
            item.setToolTip(f"Shot {i + 1}: {name.lower()}, {dur:.1f}s"
                            + (f", {pot} balls potted" if pot else ""))
            self._list.addItem(item)
            # colour the dot by re-setting rich-ish text is not possible in
            # QListWidgetItem; the outcome name carries the meaning.
        self._count.setText(f"({len(self._shots)})")
        self._sync_empty()

    def add_shot(self, shot: dict) -> None:
        self._shots.append(shot)
        self.set_shots(self._shots)

    def _sync_empty(self) -> None:
        has = bool(self._shots)
        self._list.setVisible(has)
        self._empty.setVisible(not has)
        self._prev.setEnabled(has)
        self._next.setEnabled(has)

    # ------------------------------------------------------------------ #
    def _on_item(self, item: QListWidgetItem) -> None:
        start = float(item.data(Qt.UserRole) or 0.0)
        self.shot_selected.emit(max(0.0, start - self.pre_roll_s))

    def _step(self, delta: int) -> None:
        if not self._shots:
            return
        row = self._list.currentRow()
        row = 0 if row < 0 else max(0, min(len(self._shots) - 1, row + delta))
        self._list.setCurrentRow(row)
        self._on_item(self._list.item(row))
