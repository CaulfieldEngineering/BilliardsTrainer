"""Update-available dialog: shows release notes, downloads with progress, and
launches the installer (then quits the app so files can be replaced)."""

from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from ...update.updater import (
    DownloadWorker,
    UpdateInfo,
    install_and_relaunch,
    run_in_thread,
)
from ...version import __version__
from ..icons import icon


class UpdateDialog(QDialog):
    def __init__(self, info: UpdateInfo, parent=None):
        super().__init__(parent)
        self._info = info
        self._thread = None
        self._worker = None
        self.setWindowTitle("Update available")
        self.setMinimumWidth(460)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 20)
        root.setSpacing(14)

        title = QLabel(f"Billiards Trainer {self._info.version} is available")
        title.setObjectName("H2")
        root.addWidget(title)

        sub = QLabel(f"You're on {__version__}. Update to get the latest fixes.")
        sub.setObjectName("Muted")
        root.addWidget(sub)

        if self._info.notes:
            notes = QTextEdit()
            notes.setReadOnly(True)
            notes.setPlainText(self._info.notes)
            notes.setMaximumHeight(140)
            root.addWidget(notes)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.hide()
        root.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setObjectName("Faint")
        self._status.hide()
        root.addWidget(self._status)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self._later = QPushButton("Later")
        self._later.setObjectName("Ghost")
        self._later.clicked.connect(self.reject)
        buttons.addWidget(self._later)

        self._update_btn = QPushButton("  Download & install")
        self._update_btn.setObjectName("Accent")
        self._update_btn.setIcon(icon("download", "#0A0E12"))
        self._update_btn.clicked.connect(self._start_download)
        buttons.addWidget(self._update_btn)
        root.addLayout(buttons)

    def _start_download(self) -> None:
        if not self._info.url:
            self._status.setText("No download URL in the release manifest.")
            self._status.show()
            return
        self._update_btn.setEnabled(False)
        self._later.setEnabled(False)
        self._progress.show()
        self._status.setText("Downloading…")
        self._status.show()

        self._worker = DownloadWorker(self._info.url)
        self._worker.progress.connect(self._progress.setValue)
        self._worker.finished.connect(self._on_downloaded)
        self._worker.failed.connect(self._on_failed)
        self._thread = run_in_thread(self._worker)

    def _on_failed(self, msg: str) -> None:
        self._status.setText(f"Download failed: {msg}")
        self._update_btn.setEnabled(True)
        self._later.setEnabled(True)

    def _on_downloaded(self, path: str) -> None:
        if self._thread:
            self._thread.quit()
            self._thread.wait(2000)
        self._status.setText("Installing update…")
        try:
            install_and_relaunch(path)
        except OSError as exc:
            self._on_failed(str(exc))
            return
        QApplication.quit()


def maybe_prompt_update(info: UpdateInfo, parent=None) -> None:
    """Show the update dialog modally if ``info`` is present."""
    if info is None:
        return
    UpdateDialog(info, parent).exec()
