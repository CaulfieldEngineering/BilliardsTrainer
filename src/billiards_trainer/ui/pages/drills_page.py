"""Drills page — pick a practice drill template to run on the Live tab.

Drill definitions come from ``game.drills``. Selecting one emits ``drill_chosen``
so the main window can switch to Live in Drill mode.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...game.drills import DRILL_TEMPLATES, Drill
from ..widgets.common import Card, section_header


class DrillCard(Card):
    chosen = Signal(object)

    def __init__(self, drill: Drill, parent=None):
        super().__init__(parent, padding=18, spacing=8)
        self._drill = drill
        title = QLabel(f"<b>{drill.name}</b>")
        self.add(title)
        diff = QLabel(drill.difficulty.upper())
        diff.setObjectName("Faint")
        self.add(diff)
        desc = QLabel(drill.description)
        desc.setObjectName("Muted")
        desc.setWordWrap(True)
        self.add(desc)
        goal = QLabel(f"Goal: {drill.target_makes} makes · {drill.balls} balls")
        goal.setObjectName("Faint")
        self.add(goal)
        btn = QPushButton("Start drill")
        btn.setObjectName("Accent")
        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(lambda: self.chosen.emit(self._drill))
        self.add(btn)


class DrillsPage(QWidget):
    drill_chosen = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)
        root.addWidget(section_header("Drill library"))
        sub = QLabel("Structured practice routines. Pick one to run with live tracking.")
        sub.setObjectName("Muted")
        root.addWidget(sub)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        holder = QWidget()
        grid = QHBoxLayout(holder)
        grid.setSpacing(16)
        grid.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(16)
        wrap = QWidget()
        wrap.setLayout(col)

        flow = _FlowColumns(DRILL_TEMPLATES, self.drill_chosen)
        scroll.setWidget(flow)
        root.addWidget(scroll, 1)


class _FlowColumns(QWidget):
    """Simple responsive-ish 3-column grid of drill cards."""

    def __init__(self, drills: list[Drill], chosen_signal: Signal, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import QGridLayout
        grid = QGridLayout(self)
        grid.setSpacing(16)
        grid.setContentsMargins(0, 0, 8, 0)
        cols = 3
        for i, drill in enumerate(drills):
            card = DrillCard(drill)
            card.chosen.connect(chosen_signal.emit)
            grid.addWidget(card, i // cols, i % cols)
        grid.setRowStretch((len(drills) // cols) + 1, 1)
