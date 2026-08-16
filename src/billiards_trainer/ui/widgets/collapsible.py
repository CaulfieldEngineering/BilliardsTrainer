"""Collapsible section: a clickable header that folds its body, web-style.

Joe's ask: sections he can collapse when the window is narrow, and an app
that behaves "like a web page" — so the fold is ANIMATED (a quick ease-out
height tween, not a pop) and the collapsed state persists per section key
in ``ui.collapsed_sections`` so the layout reopens the way he left it.
"""

from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    Qt,
)
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from ..theme import PALETTE

#: Qt's "no maximum" sentinel — restored after the expand animation so the
#: body can grow with content again.
_QWIDGETSIZE_MAX = 16777215


class CollapsibleSection(QWidget):
    """Header + foldable body. ``key`` identifies it in persisted settings."""

    def __init__(self, title: str, content: QWidget, key: str,
                 settings=None, parent=None):
        super().__init__(parent)
        self._key = key
        self._settings = settings
        self._content = content

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        self._head = QPushButton()
        self._head.setCursor(Qt.PointingHandCursor)
        self._head.setStyleSheet(
            "QPushButton { background: transparent; border: none; text-align: left;"
            f"color: {PALETTE.text_dim}; font-size: 10px; font-weight: 800;"
            "letter-spacing: 1.2px; padding: 2px 0; }"
            f"QPushButton:hover {{ color: {PALETTE.text}; }}")
        self._title = title
        self._head.clicked.connect(self.toggle)
        root.addWidget(self._head)
        root.addWidget(content)

        self._anim = QPropertyAnimation(content, b"maximumHeight", self)
        self._anim.setDuration(160)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.finished.connect(self._after_anim)

        self._collapsed = False
        if settings is not None and key in getattr(settings.ui, "collapsed_sections", []):
            # restore folded without animating a window that is still building
            self._collapsed = True
            content.setMaximumHeight(0)
        self._set_chevron()

    # ------------------------------------------------------------------ #
    @property
    def collapsed(self) -> bool:
        return self._collapsed

    def toggle(self) -> None:
        self._collapsed = not self._collapsed
        self._set_chevron()
        self._anim.stop()
        if self._collapsed:
            start = self._content.height()
            self._content.setMaximumHeight(start)
            self._anim.setStartValue(start)
            self._anim.setEndValue(0)
        else:
            self._anim.setStartValue(self._content.maximumHeight())
            self._anim.setEndValue(max(1, self._content.sizeHint().height()))
        self._anim.start()
        self._persist()

    def _after_anim(self) -> None:
        if not self._collapsed:
            # free the body to grow with its content again
            self._content.setMaximumHeight(_QWIDGETSIZE_MAX)

    def _set_chevron(self) -> None:
        self._head.setText(("▸  " if self._collapsed else "▾  ") + self._title)

    def _persist(self) -> None:
        if self._settings is None:
            return
        keys = set(getattr(self._settings.ui, "collapsed_sections", []))
        (keys.add if self._collapsed else keys.discard)(self._key)
        self._settings.ui.collapsed_sections = sorted(keys)
        try:
            self._settings.save()
        except OSError:
            pass
