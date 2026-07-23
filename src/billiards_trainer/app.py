"""Application entry point.

Sets up logging, loads settings, applies the dark theme, and shows the main
window. ``main()`` is the console-script / PyInstaller entry.
"""

import logging
import sys

from .config import LOGS_DIR, Settings, ensure_dirs
from .version import APP_NAME, ORG_NAME, __version__

_CRASH_FILE = None  # kept alive for the lifetime of the process


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
    # faulthandler dumps the C-level Python stack of ALL threads on a hard crash
    # (segfault / abort) — the only way to see a native crash that try/except and
    # sys.excepthook can't catch (e.g. a Qt/OpenCV abort). Lands in crash.log.
    global _CRASH_FILE
    try:
        import faulthandler
        _CRASH_FILE = open(LOGS_DIR / "crash.log", "a", buffering=1, encoding="utf-8")
        faulthandler.enable(file=_CRASH_FILE, all_threads=True)
    except Exception:  # noqa: BLE001 - diagnostics must never block startup
        pass


def _install_excepthook() -> None:
    """Log unhandled exceptions instead of letting them crash silently.

    A bare Python exception raised inside a Qt slot can otherwise terminate the
    app with no trace (PySide6 may treat it as fatal). Routing it through the log
    gives us a traceback to act on; we keep the app alive so one bad frame/click
    doesn't kill the session."""
    log = logging.getLogger("excepthook")

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        log.error("UNHANDLED EXCEPTION", exc_info=(exc_type, exc, tb))

    sys.excepthook = hook
    try:  # also catch exceptions raised inside Qt threads (PySide6 6.5+)
        import threading
        threading.excepthook = lambda a: log.error(
            "UNHANDLED THREAD EXCEPTION", exc_info=(a.exc_type, a.exc_value, a.exc_traceback))
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    _setup_logging()
    _install_excepthook()
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

    # We reached a live QApplication => the Python runtime + Qt DLLs loaded.
    # Tell the self-update swap batch the new exe started OK (so it won't roll
    # back). Done as early as possible after the window system is up.
    from .update import recovery

    recovery.mark_launched_ok()

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

    # macOS camera permission: request it on the MAIN thread once the event loop
    # is live (the only context where macOS shows the prompt for a bundled app).
    # When granted, re-scan so the camera list populates without a restart.
    from .capture.macos_permissions import (
        AUTHORIZED,
        DENIED,
        camera_auth_status,
        request_camera_access,
    )

    if camera_auth_status() != AUTHORIZED:
        from PySide6.QtCore import QTimer

        _perm_state = {"granted": None}

        def _fire_prompt() -> None:
            log.info("requesting camera permission on main loop")
            request_camera_access(lambda g: _perm_state.__setitem__("granted", g))

        def _poll_perm() -> None:
            if _perm_state["granted"] is True or camera_auth_status() == AUTHORIZED:
                log.info("camera authorized — re-scanning sources")
                _perm_timer.stop()
                try:
                    window.retry_camera()
                except Exception:  # noqa: BLE001
                    log.exception("camera re-scan after grant failed")
            elif _perm_state["granted"] is False or camera_auth_status() == DENIED:
                _perm_timer.stop()
                log.warning("camera permission denied — enable it in System "
                            "Settings › Privacy & Security › Camera")

        QTimer.singleShot(400, _fire_prompt)
        _perm_timer = QTimer()
        _perm_timer.setInterval(500)
        _perm_timer.timeout.connect(_poll_perm)
        _perm_timer.start()

    # Microphone permission (audio in session recordings). Same main-thread
    # rule as the camera; the ffmpeg capture child inherits this app's grant.
    from .capture.macos_permissions import (
        microphone_auth_status,
        request_microphone_access,
    )

    if settings.recording.audio and microphone_auth_status() != AUTHORIZED:
        from PySide6.QtCore import QTimer
        QTimer.singleShot(900, lambda: request_microphone_access())

    # If a prior self-update was rolled back (or files look incomplete), explain it.
    integrity = recovery.verify_frozen_integrity()
    if recovery.consume_update_failed() or integrity:
        log.warning("Showing post-update recovery dialog (integrity=%s)", integrity)
        from .ui.dialogs.recovery_dialog import RecoveryDialog
        RecoveryDialog(reason=integrity, parent=window).exec()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
