"""Shown when a self-update failed to launch and we rolled back (or a frozen
integrity check tripped). Explains the likely cause (antivirus quarantining the
unsigned download) and offers concrete recovery actions."""

import sys
import webbrowser

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ...version import RELEASES_PAGE_URL


class RecoveryDialog(QDialog):
    def __init__(self, reason: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Update couldn't start")
        self.setMinimumWidth(520)
        self._build(reason)

    def _build(self, reason: str | None) -> None:
        from ...update.recovery import install_dir
        self._dir = str(install_dir())

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 18)
        root.setSpacing(12)

        title = QLabel("The update couldn't start — your previous version was restored")
        title.setObjectName("H2")
        title.setWordWrap(True)
        root.addWidget(title)

        why = reason or ("This almost always means antivirus (e.g. Windows Defender) "
                         "quarantined part of the freshly-downloaded app while it was "
                         "starting. The app is unsigned, so strict AV can be cautious.")
        body = QLabel(
            f"{why}\n\nYou're back on a working version. To update successfully:\n"
            "  1. Add this folder to your antivirus exclusions, then try again, or\n"
            "  2. Download the latest version manually from GitHub and run it.")
        body.setObjectName("Muted")
        body.setWordWrap(True)
        root.addWidget(body)

        folder = QLabel(f"Install folder:  {self._dir}")
        folder.setObjectName("Faint")
        folder.setTextInteractionFlags(Qt.TextSelectableByMouse)
        folder.setWordWrap(True)
        root.addWidget(folder)

        self._copied = QLabel("")
        self._copied.setObjectName("Faint")
        root.addWidget(self._copied)

        buttons = QHBoxLayout()
        open_folder = QPushButton("Open install folder")
        open_folder.setObjectName("Ghost")
        open_folder.clicked.connect(self._open_folder)
        buttons.addWidget(open_folder)

        copy_excl = QPushButton("Copy Defender exclusion command")
        copy_excl.setObjectName("Ghost")
        copy_excl.clicked.connect(self._copy_exclusion)
        buttons.addWidget(copy_excl)

        buttons.addStretch(1)
        releases = QPushButton("Download from GitHub")
        releases.setObjectName("Accent")
        releases.setCursor(Qt.PointingHandCursor)
        releases.clicked.connect(lambda: webbrowser.open(RELEASES_PAGE_URL))
        buttons.addWidget(releases)

        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        buttons.addWidget(close)
        root.addLayout(buttons)

    def _open_folder(self) -> None:
        try:
            if sys.platform == "win32":
                import os
                os.startfile(self._dir)  # type: ignore[attr-defined]
            else:
                import subprocess
                subprocess.Popen(["xdg-open", self._dir])
        except OSError:
            pass

    def _copy_exclusion(self) -> None:
        cmd = f'Add-MpPreference -ExclusionPath "{self._dir}"'
        QGuiApplication.clipboard().setText(cmd)
        self._copied.setText("Copied — paste into an *admin* PowerShell, then re-try the update.")
