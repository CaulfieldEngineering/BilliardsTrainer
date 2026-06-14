"""UI construction smoke test (offscreen).

Builds the whole window with the Qt 'offscreen' platform so import errors, QSS
problems, and signal/slot signature mismatches are caught in CI without a
display. Skipped gracefully if Qt can't initialise headlessly.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication
    instance = QApplication.instance() or QApplication([])
    yield instance


def test_window_builds(app):
    from billiards_trainer.config import Settings
    from billiards_trainer.ui.main_window import MainWindow
    from billiards_trainer.ui.theme import apply_theme

    settings = Settings()
    settings.updates.auto_check = False
    apply_theme(app, settings.ui.accent)
    win = MainWindow(settings)
    app.processEvents()
    # four nav destinations
    assert win._stack.count() == 4
    win.close()
    app.processEvents()


def test_theme_stylesheet_nonempty(app):
    from billiards_trainer.ui.theme import build_stylesheet
    qss = build_stylesheet("#3DDC97")
    assert "QPushButton" in qss and "#3DDC97" in qss


def test_icons_render(app):
    from billiards_trainer.ui.icons import icon
    ic = icon("play", "#FFFFFF", 24)
    assert not ic.isNull()
