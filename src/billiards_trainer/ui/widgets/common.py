"""Small composable widgets used across pages: cards, stat tiles, segmented
control, section headers, and an empty-state placeholder."""


from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..icons import icon
from ..theme import PALETTE


class Card(QFrame):
    """An elevated rounded panel."""

    def __init__(self, parent=None, padding: int = 16, spacing: int = 12):
        super().__init__(parent)
        self.setObjectName("Card")
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(padding, padding, padding, padding)
        self._lay.setSpacing(spacing)

    def layout(self) -> QVBoxLayout:  # type: ignore[override]
        return self._lay

    def add(self, w: QWidget) -> QWidget:
        self._lay.addWidget(w)
        return w


class StatCard(Card):
    """A KPI tile: small label, big value, optional sub-caption."""

    def __init__(self, label: str, value: str = "—", accent: bool = False,
                 sub: str = "", parent=None):
        super().__init__(parent, padding=16, spacing=4)
        self.setMinimumWidth(140)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self._label = QLabel(label.upper())
        self._label.setObjectName("StatLabel")
        self._value = QLabel(value)
        self._value.setObjectName("StatValueAccent" if accent else "StatValue")
        self._sub = QLabel(sub)
        self._sub.setObjectName("Faint")
        self._sub.setVisible(bool(sub))

        self._lay.addWidget(self._label)
        self._lay.addWidget(self._value)
        self._lay.addWidget(self._sub)

    def set_value(self, value: str) -> None:
        self._value.setText(value)

    def set_sub(self, sub: str) -> None:
        self._sub.setText(sub)
        self._sub.setVisible(bool(sub))


def section_header(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("H2")
    return lbl


def hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet(f"color: {PALETTE.border_soft};")
    line.setFixedHeight(1)
    return line


class SegmentedControl(QWidget):
    """A pill row of mutually-exclusive options (mode switcher)."""

    changed = Signal(str)

    def __init__(self, options: list[tuple[str, str]], current: str = "", parent=None):
        """options: list of (key, label)."""
        super().__init__(parent)
        self.setObjectName("Segmented")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        self.setStyleSheet(
            f"QWidget#Segmented {{ background:{PALETTE.surface}; border-radius:10px; }}"
        )
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}
        for key, label in options:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setObjectName("SegItem")
            btn.setStyleSheet(self._seg_qss())
            btn.clicked.connect(lambda _=False, k=key: self._on_click(k))
            self._group.addButton(btn)
            self._buttons[key] = btn
            lay.addWidget(btn)
        if current in self._buttons:
            self._buttons[current].setChecked(True)
        elif self._buttons:
            next(iter(self._buttons.values())).setChecked(True)

    @staticmethod
    def _seg_qss() -> str:
        p = PALETTE
        return f"""
        QPushButton#SegItem {{ background: transparent; border: none; border-radius: 8px;
            padding: 7px 16px; color: {p.text_dim}; font-weight: 600; }}
        QPushButton#SegItem:hover {{ color: {p.text}; }}
        QPushButton#SegItem:checked {{ background: {p.surface_hi}; color: {p.text}; }}
        """

    def _on_click(self, key: str) -> None:
        self.changed.emit(key)

    def current(self) -> str:
        for key, btn in self._buttons.items():
            if btn.isChecked():
                return key
        return ""

    def set_current(self, key: str) -> None:
        if key in self._buttons:
            self._buttons[key].setChecked(True)


class EmptyState(QWidget):
    """Centered icon + headline + body + optional action button — used when a
    page has no data yet (good UX > blank screen)."""

    action = Signal()

    def __init__(self, icon_name: str, title: str, body: str = "",
                 action_text: str = "", parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.addStretch(1)

        row = QHBoxLayout()
        row.addStretch(1)
        col = QVBoxLayout()
        col.setSpacing(10)
        col.setAlignment(Qt.AlignCenter)

        ic = QLabel()
        ic.setPixmap(icon(icon_name, PALETTE.text_faint, 56, 1.6).pixmap(56, 56))
        ic.setAlignment(Qt.AlignCenter)
        col.addWidget(ic)

        head = QLabel(title)
        head.setObjectName("H2")
        head.setAlignment(Qt.AlignCenter)
        col.addWidget(head)

        if body:
            sub = QLabel(body)
            sub.setObjectName("Muted")
            sub.setAlignment(Qt.AlignCenter)
            sub.setWordWrap(True)
            sub.setMaximumWidth(420)
            col.addWidget(sub)

        if action_text:
            btn = QPushButton(action_text)
            btn.setObjectName("Accent")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(self.action.emit)
            btn_row = QHBoxLayout()
            btn_row.addStretch(1)
            btn_row.addWidget(btn)
            btn_row.addStretch(1)
            col.addLayout(btn_row)

        row.addLayout(col)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(1)


def nav_button(icon_name: str, text: str) -> QPushButton:
    btn = QPushButton(f"  {text}")
    btn.setObjectName("NavItem")
    btn.setCheckable(True)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setIcon(icon(icon_name, PALETTE.text_dim, 20))
    btn.setIconSize(_icon_size())
    return btn


def _icon_size():
    from PySide6.QtCore import QSize
    return QSize(20, 20)


class Badge(QLabel):
    """A small status pill (e.g. LIVE / IDLE / make / miss)."""

    def __init__(self, text: str, color: str | None = None, parent=None):
        super().__init__(text, parent)
        self._color = color or PALETTE.text_dim
        self._restyle()

    def set_text_color(self, text: str, color: str) -> None:
        self.setText(text)
        self._color = color
        self._restyle()

    def _restyle(self) -> None:
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            f"background: {self._color}22; color: {self._color}; border: 1px solid {self._color}55;"
            f"border-radius: 9px; padding: 3px 10px; font-weight: 700; font-size: 11px;"
        )
