"""Unit tests for geometry, the updater version logic, settings, and classify."""

import numpy as np

from billiards_trainer.config import Settings
from billiards_trainer.update.updater import is_newer, parse_version
from billiards_trainer.vision.balls import classify_ball
from billiards_trainer.vision.geometry import TableModel
from billiards_trainer.vision.types import BallClass


# ---- geometry ------------------------------------------------------------- #
def test_table_has_six_pockets():
    t = TableModel.from_rect((488, 895), 40)
    assert len(t.pockets) == 6
    names = {p.name for p in t.pockets}
    assert "top-left" in names and "left-side" in names


def test_pocket_at_detects_capture_zone():
    t = TableModel.from_rect((488, 895), 40)
    br = next(p for p in t.pockets if p.name == "bottom-right")
    assert t.pocket_at(br.x, br.y) is not None
    assert t.pocket_at(t.width / 2, t.height / 2) is None  # centre is open table


def test_on_table_bounds():
    t = TableModel.from_rect((488, 895), 40)
    assert t.on_table(244, 447)
    assert not t.on_table(5, 5)


# ---- updater -------------------------------------------------------------- #
def test_parse_version():
    assert parse_version("0.1.42") == (0, 1, 42)
    assert parse_version("v1.2.3") == (1, 2, 3)
    assert parse_version("") == (0,)


def test_is_newer():
    assert is_newer("0.1.43", "0.1.42")
    assert is_newer("0.2.0", "0.1.99")
    assert not is_newer("0.1.42", "0.1.42")
    assert not is_newer("0.1.41", "0.1.42")
    assert is_newer("1.0", "0.9.9")


# ---- settings ------------------------------------------------------------- #
def test_settings_roundtrip(tmp_path):
    s = Settings()
    s.felt.h_min = 33
    s.shot_clock.seconds = 45
    s.ui.accent = "#FF8800"
    path = tmp_path / "settings.json"
    s.save(path)
    loaded = Settings.load(path)
    assert loaded.felt.h_min == 33
    assert loaded.shot_clock.seconds == 45
    assert loaded.ui.accent == "#FF8800"


def test_settings_forward_compatible(tmp_path):
    # Unknown keys ignored, missing keys defaulted.
    path = tmp_path / "settings.json"
    path.write_text('{"source": "demo", "felt": {"h_min": 10}, "unknown": 7}')
    s = Settings.load(path)
    assert s.source == "demo"
    assert s.felt.h_min == 10
    assert s.felt.h_max == 105  # default preserved


def test_settings_load_missing_returns_defaults(tmp_path):
    s = Settings.load(tmp_path / "nope.json")
    assert s.source == "0"


# ---- classify ------------------------------------------------------------- #
def test_classify_white_is_cue():
    patch = np.full((20, 20, 3), 240, np.uint8)
    cls, _ = classify_ball(patch)
    assert cls == BallClass.CUE


def test_classify_black_is_eight():
    patch = np.full((20, 20, 3), 20, np.uint8)
    cls, _ = classify_ball(patch)
    assert cls == BallClass.EIGHT


def test_classify_colored_is_solid():
    patch = np.zeros((20, 20, 3), np.uint8)
    patch[:, :] = (40, 90, 200)  # red-ish (BGR)
    cls, _ = classify_ball(patch)
    assert cls in (BallClass.SOLID, BallClass.STRIPE)


# ---- felt auto-estimation ------------------------------------------------- #
def test_estimate_felt_settings_keys_on_centre_colour():
    import cv2

    from billiards_trainer.config import FeltSettings
    from billiards_trainer.vision.felt import estimate_felt_settings

    # a frame whose centre is a distinctly blue felt (hue 110) the defaults miss
    hsv = np.full((400, 400, 3), (110, 150, 200), np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    fs = estimate_felt_settings(bgr, FeltSettings())
    assert abs(fs.picked_hsv[0] - 110) <= 2
    assert fs.h_min <= 110 <= fs.h_max
