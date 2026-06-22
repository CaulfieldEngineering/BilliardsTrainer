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
    # auto-start opens the source immediately (no Start click); detection now
    # defaults ON (the trained model is reliable, so we track out of the box).
    assert win._started_source == "demo"
    assert win._controller._detection_enabled is True
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


def test_settings_page_has_no_dev_detector_knobs(app):
    """The detector dropdown / AI-backend / weights-URL dev knobs were removed —
    detection is automatic and tuned from the Sandbox panel, not here."""
    from billiards_trainer.config import Settings
    from billiards_trainer.ui.pages.settings_page import SettingsPage

    page = SettingsPage(Settings())
    for gone in ("_live_detector", "_backend", "_yolo_url", "_param2", "_preset", "_fusion"):
        assert not hasattr(page, gone), f"dev knob {gone} should be removed from Settings"


def test_live_tuning_panel_present(app):
    """The Sandbox page exposes a live tuning panel (cue status + sensitivity +
    display toggles) so settings can be adjusted while watching the clip."""
    from billiards_trainer.config import Settings
    from billiards_trainer.ui.pages.live_page import LivePage

    page = LivePage(Settings())
    assert hasattr(page, "_cue_status")
    assert hasattr(page, "_conf_val")


def test_training_mode_label_correct_add_save(app):
    """Training Mode labelling: the model's guesses become editable on the frame,
    correcting a number + adding a missed ball flows to a save payload. Tested on a
    bare LivePage (no worker threads) so the offscreen suite exits clean."""
    import types

    import numpy as np

    from billiards_trainer.config import Settings
    from billiards_trainer.ui.pages.live_page import LivePage
    from billiards_trainer.vision.types import BallClass, Detection

    lp = LivePage(Settings())
    lp.set_training(True)
    assert lp._training

    frame = np.zeros((480, 640, 3), np.uint8)
    raw = [Detection(100, 100, 12, (200, 200, 200), BallClass.SOLID, 0.9, number=2),
           Detection(300, 200, 12, (245, 245, 245), BallClass.CUE, 0.9, number=0)]
    pkt = types.SimpleNamespace(perspective=frame, birdseye=None, fps=30, n_balls=2,
                                tracks=[], raw_dets=raw, status="tracking", deviated=False,
                                shot_state="settled", clock_enabled=False)
    lp.on_frame(pkt)
    assert len(lp._label_balls) == 2

    lp._on_label_click(100 / 640, 100 / 480)   # select the mis-id'd ball
    lp._assign_label(10)                         # correct 2 -> 10
    lp._on_label_click(0.8, 0.8)                 # empty spot -> add a missed ball
    lp._assign_label(7)
    boxes = []
    lp.save_training_frame_requested.connect(lambda b: boxes.extend(b))
    lp._save_label_frame()
    assert sorted(b[0] for b in boxes) == [0, 7, 10]
    lp.set_training(False)
    assert not lp._training


def test_training_entry_on_paused_frame_no_phantom_ball(app):
    """REGRESSION: entering Training Mode on a paused frame must label THAT frame
    (showing the model's guess per ball) instead of leaving the size unknown — a
    click then selected a phantom ball stuck at image (0,0). Also: a click before
    any frame is loaded must not drop a ball at the origin."""
    import types

    import numpy as np

    from billiards_trainer.config import Settings
    from billiards_trainer.ui.pages.live_page import LivePage
    from billiards_trainer.vision.types import BallClass, Detection

    # A: no frame yet -> click is a no-op, never a phantom ball at (0,0)
    lp = LivePage(Settings())
    lp.set_training(True)
    lp._on_label_click(0.5, 0.5)
    assert lp._label_balls == []

    # B: a normal (non-training) frame arrives — like a paused video — THEN the
    # user switches into Training Mode. The current frame must be ingested.
    lp2 = LivePage(Settings())
    frame = np.zeros((1080, 1920, 3), np.uint8)
    raw = [Detection(800, 500, 14, (200, 200, 200), BallClass.SOLID, 0.9, number=2),
           Detection(1000, 600, 14, (245, 245, 245), BallClass.CUE, 0.9, number=0),
           Detection(600, 400, 14, (50, 50, 50), BallClass.STRIPE, 0.9, number=-1)]
    pkt = types.SimpleNamespace(perspective=frame, birdseye=None, fps=30, n_balls=3,
                                tracks=[], raw_dets=raw, status="tracking", deviated=False,
                                shot_state="settled", clock_enabled=False)
    lp2.on_frame(pkt)                       # arrives while NOT in training mode
    assert lp2._frame_wh == (1920, 1080)    # size tracked even outside Training
    lp2.set_training(True)                  # enter on the paused frame
    assert len(lp2._label_balls) == 3       # the on-screen frame is now labellable
    assert [b[0] for b in lp2._label_balls] == [2, 0, -1]  # model's guesses kept

    # clicking the mis-id'd 2-ball selects IT (not a phantom at the origin)
    lp2._on_label_click(800 / 1920, 500 / 1080)
    assert lp2._label_sel == 0
    assert lp2._label_balls[0][1] == 800 and lp2._label_balls[0][2] == 500


def test_ball_trainer_dialog_and_store(app, tmp_path):
    """The in-app Ball-ID trainer constructs, and its labelled-data store does a
    save/read round-trip in the YOLO format the fine-tune consumes."""
    import numpy as np

    from billiards_trainer.config import Settings
    from billiards_trainer.train.store import LabeledBall, TrainingStore
    from billiards_trainer.ui.dialogs.ball_trainer_dialog import BallTrainerDialog

    s = Settings()
    s.source = ""  # no clip auto-loaded
    dlg = BallTrainerDialog(s)
    assert dlg._store is not None

    store = TrainingStore(tmp_path / "ballid")
    img = np.zeros((480, 640, 3), np.uint8)
    n = store.add_frame(img, [LabeledBall(0, 0.5, 0.5, 0.1, 0.1),   # cue
                              LabeledBall(9, 0.3, 0.3, 0.1, 0.1),   # 9-ball
                              LabeledBall(-1, 0.1, 0.1, 0.1, 0.1)])  # 'not a ball' dropped
    assert n == 2
    assert store.count() == 1
    assert store.class_counts() == {0: 1, 9: 1}
    assert store.write_data_yaml().exists()


def test_training_jump_buttons(app):
    """Training Mode 'jump to a new frame' controls: +30s/+1m/+5m jump ahead by
    fps*seconds (clamped to the clip end) and Random lands in-range. Hidden for
    non-seekable (live camera) sources."""
    from billiards_trainer.config import Settings
    from billiards_trainer.ui.pages.live_page import LivePage

    lp = LivePage(Settings())
    seeks = []
    lp.video_seek.connect(lambda f: seeks.append(f))

    lp.set_video_mode(False, 0, 30.0)
    assert lp._jump_section.isHidden()           # no jumping a live camera

    lp.set_video_mode(True, 1000, 30.0)          # 1000 frames @ 30fps
    assert not lp._jump_section.isHidden()
    lp.set_training(True)

    lp._seek.setValue(100)
    lp._jump_ahead(1)
    assert seeks[-1] == 130                       # 100 + 1*30
    lp._jump_ahead(60)
    assert seeks[-1] == 999                        # 100 + 1800 -> clamped to last
    for _ in range(30):
        lp._jump_random()
        assert 0 <= seeks[-1] <= 999


def test_widget_state_animations_disabled(app):
    """REGRESSION (native crash): widget state-transition animations must be OFF.
    Fusion fades hover/enabled transitions via QStyleAnimation objects that own
    QTimers; rapidly toggling the Training number buttons' enabled-state freed an
    in-flight animation while its timer was still registered, delivering a stale
    QTimerEvent to freed memory (crash in QCoreApplication::notifyInternal2,
    confirmed from a dump). apply_theme wraps Fusion to zero the animation hint."""
    from PySide6.QtWidgets import QPushButton, QStyle

    from billiards_trainer.ui.theme import apply_theme

    apply_theme(app)
    dur = app.style().styleHint(
        QStyle.StyleHint.SH_Widget_Animation_Duration, None, QPushButton())
    assert dur == 0, f"widget animations must be disabled, got {dur}ms"


def test_worker_playback_timer_is_parented(app):
    """REGRESSION (native crash): the controller's worker-thread QTimer must be
    PARENTED to the controller. A parentless QTimer on a worker thread can be freed
    by PySide6 while its OS timer is still registered with the event dispatcher; the
    next tick then delivers a QTimerEvent into freed memory and crashes natively in
    QCoreApplication::notifyInternal2 (confirmed from two crash dumps). Parenting
    ties the timer's lifetime to the controller for the whole session."""
    from pathlib import Path

    from billiards_trainer.config import Settings
    from billiards_trainer.db.repository import Repository
    from billiards_trainer.workers.controller import PipelineController

    ctrl = PipelineController(Settings(), Repository(db_path=Path(":memory:")))
    ctrl.on_started()                       # creates the playback timer
    assert ctrl._timer is not None
    assert ctrl._timer.parent() is ctrl     # owned by the controller, not Python-only


def test_entering_training_mode_autopauses_video(app):
    """Entering Training Mode must auto-pause a video source: labelling is done on
    a still frame, and the streaming repaint racing GPU inference was the suspected
    native-crash trigger. Also stops incoming frames from clearing the selection."""
    import types
    from pathlib import Path

    from billiards_trainer.config import Settings
    from billiards_trainer.db.repository import Repository
    from billiards_trainer.workers.controller import PipelineController

    ctrl = PipelineController(Settings(), Repository(db_path=Path(":memory:")))
    ctrl._source = types.SimpleNamespace(is_video=True)   # seekable video stub
    ctrl._pipeline = None                                 # skip the frame re-emit
    assert ctrl._video_paused is False
    ctrl.set_label_mode(True)
    assert ctrl._video_paused is True      # paused for labelling
    assert ctrl._label_mode is True

    # a live (non-seekable) camera is never auto-paused (nothing to pause)
    ctrl2 = PipelineController(Settings(), Repository(db_path=Path(":memory:")))
    ctrl2._source = types.SimpleNamespace(is_video=False)
    ctrl2.set_label_mode(True)
    assert ctrl2._video_paused is False


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
