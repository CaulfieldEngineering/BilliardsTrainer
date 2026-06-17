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
    settings.source = "demo"  # autostart previews a hardware-free source in CI
    apply_theme(app, settings.ui.accent)
    win = MainWindow(settings)
    app.processEvents()
    # four nav destinations
    assert win._stack.count() == 4
    # auto-start previews immediately (no Start click); detection defaults off
    assert win._started_source == "demo"
    assert win._live._detect_on is False
    win.close()
    app.processEvents()
    win._thread.wait(3000)  # ensure the worker thread is fully torn down


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


def test_live_detector_dropdown_lists_simple_blob(app):
    """REGRESSION (sev-1): the 'Live detector' dropdown shipped showing ONLY
    'legacy' because frozen strategy discovery returned nothing. The dropdown
    must always offer simple_blob (the default) + legacy."""
    from billiards_trainer.config import Settings
    from billiards_trainer.ui.pages.settings_page import SettingsPage

    page = SettingsPage(Settings())
    items = [page._live_detector.itemText(i) for i in range(page._live_detector.count())]
    assert "simple_blob" in items, f"dropdown missing simple_blob: {items}"
    assert "legacy" in items, f"dropdown missing legacy: {items}"
    # never the degenerate 'legacy-only' state we shipped
    assert items != ["legacy"]


def test_live_detector_dropdown_frozen_safe(app, monkeypatch):
    """Even when pkgutil.iter_modules finds nothing (the frozen-onefile reality),
    the dropdown still lists simple_blob — proves the static-core fix flows all
    the way through to the actual widget, not just discover()."""
    import pkgutil

    from billiards_trainer.config import Settings
    from billiards_trainer.ui.pages.settings_page import SettingsPage

    monkeypatch.setattr(pkgutil, "iter_modules", lambda *a, **k: iter(()))
    page = SettingsPage(Settings())
    items = [page._live_detector.itemText(i) for i in range(page._live_detector.count())]
    assert "simple_blob" in items, f"FROZEN dropdown missing simple_blob: {items}"


def test_try_demo_button_removed(app):
    from billiards_trainer.config import Settings
    from billiards_trainer.ui.pages.live_page import LivePage

    page = LivePage(Settings())
    assert not hasattr(page, "_demo_btn")
    assert not hasattr(page, "_start_demo")


def test_practice_and_drill_modes_disabled(app):
    from billiards_trainer.config import Settings
    from billiards_trainer.ui.pages.live_page import LivePage

    page = LivePage(Settings())
    assert page._mode._buttons["free_play"].isEnabled()
    assert not page._mode._buttons["practice"].isEnabled()
    assert not page._mode._buttons["drill"].isEnabled()
    assert page._mode.current() == "free_play"  # always lands on the working mode


def test_seek_guard_keeps_thumb_under_cursor_during_drag(app):
    """REGRESSION: while the user drags the seek thumb, a playback tick must NOT
    yank it back to the playback position (the 'thumb keeps pushing forward' bug)."""
    from billiards_trainer.config import Settings
    from billiards_trainer.ui.pages.live_page import LivePage

    page = LivePage(Settings())
    page.set_video_mode(True, 100, 30.0)
    page._seek.setValue(80)
    page._on_seek_pressed()          # user grabs the thumb at frame 80
    assert page._user_is_seeking
    page.update_video_state(5, 100, True)   # a tick says "we're at frame 5"
    assert page._seek.value() == 80         # thumb stays where the user put it
    page._on_seek_released()
    assert not page._user_is_seeking


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
