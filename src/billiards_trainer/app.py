"""Application entry point.

Sets up logging, loads settings, applies the dark theme, and shows the main
window. ``main()`` is the console-script / PyInstaller entry.
"""

import logging
import sys

from .config import LOGS_DIR, Settings, ensure_dirs
from .version import APP_NAME, ORG_NAME, __version__


def _setup_logging() -> None:
    ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(LOGS_DIR / "billiards_trainer.log", encoding="utf-8"),
        ],
    )


def main() -> int:
    _setup_logging()
    log = logging.getLogger("app")
    log.info("Starting %s %s", APP_NAME, __version__)

    # Import Qt lazily so logging/paths are ready first and import errors are clear.
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setApplicationVersion(__version__)

    from .config import resource_path

    for cand in ("app.ico", "packaging/app.ico"):
        icon_path = resource_path(cand)
        if icon_path.exists():
            from PySide6.QtGui import QIcon
            app.setWindowIcon(QIcon(str(icon_path)))
            break

    settings = Settings.load()

    from .ui.theme import apply_theme, load_bundled_fonts

    load_bundled_fonts()
    apply_theme(app, settings.ui.accent)

    from .ui.main_window import MainWindow

    window = MainWindow(settings)
    window.show()

    # Center on the primary screen at a comfortable default size.
    screen = QGuiApplication.primaryScreen()
    if screen:
        geo = screen.availableGeometry()
        window.resize(min(1480, int(geo.width() * 0.9)), min(940, int(geo.height() * 0.9)))
        window.move(geo.center() - window.rect().center())

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
