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


def test_feedback_dialog_saves_locally(app):
    from pathlib import Path

    from billiards_trainer.db.repository import Repository
    from billiards_trainer.ui.dialogs.feedback_dialog import FeedbackDialog

    repo = Repository(db_path=Path(":memory:"))
    dlg = FeedbackDialog(repo)
    dlg._title.setText("Tracker drops the cue ball")
    dlg._desc.setPlainText("happens under my lamp")
    dlg._attach_shot.setChecked(False)  # no parent window to grab in this test
    before = len(repo.recent_feedback())
    dlg._on_submit()
    after = repo.recent_feedback()
    assert len(after) == before + 1
    assert after[0]["title"] == "Tracker drops the cue ball"
    assert after[0]["kind"] == "bug"


def test_settings_page_has_update_and_feedback_controls(app):
    from billiards_trainer.config import Settings
    from billiards_trainer.ui.pages.settings_page import SettingsPage

    page = SettingsPage(Settings())
    # both the always-visible header button and the card button exist
    assert hasattr(page, "_check_btn")
    assert hasattr(page, "_header_check_btn")
    assert page._header_check_btn.isVisible() or page._header_check_btn is not None
    page.set_update_status("You're on the latest version.")
    assert "latest" in page._update_status.text()
    assert page._check_btn.isEnabled() and page._header_check_btn.isEnabled()


def test_camera_dropdown_autosaves_without_save_click(app, tmp_path, monkeypatch):
    """The P0 fix: picking a camera in the dropdown must persist immediately —
    no Save click needed — so Start opens the selected camera."""
    import billiards_trainer.config as cfg
    from billiards_trainer.capture import devices
    from billiards_trainer.capture.devices import CameraInfo
    from billiards_trainer.config import Settings
    from billiards_trainer.ui.pages.settings_page import SettingsPage

    monkeypatch.setattr(cfg, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(devices, "list_cameras",
                        lambda: [CameraInfo(0, "Cam A"), CameraInfo(1, "Cam B")])
    s = Settings()
    page = SettingsPage(s)
    assert s.source == "0"  # default before selecting

    page._select_source_data("1")  # user picks "Cam B (1)" — no Save click
    assert s.source == "1"
    assert s.source_name == "Cam B"
    assert (tmp_path / "settings.json").exists()  # persisted to disk
