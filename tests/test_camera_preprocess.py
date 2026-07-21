"""Frame preprocessing (rotation + colour correction) and the overhead vision gates."""

import numpy as np

from billiards_trainer.capture.preprocess import (
    color_correct,
    preprocess_frame,
    rotate_frame,
)
from billiards_trainer.config import CameraSettings, ColorCorrectionSettings, Settings


# ---- rotation ------------------------------------------------------------ #
def test_rotate_90_swaps_dimensions():
    f = np.zeros((720, 1280, 3), np.uint8)
    assert rotate_frame(f, 90).shape == (1280, 720, 3)
    assert rotate_frame(f, 270).shape == (1280, 720, 3)
    assert rotate_frame(f, 180).shape == (720, 1280, 3)
    assert rotate_frame(f, 0).shape == (720, 1280, 3)


def test_rotate_bad_value_is_noop():
    f = np.zeros((10, 20, 3), np.uint8)
    assert rotate_frame(f, 45).shape == f.shape  # unsupported angle -> unchanged


def test_rotate_90_then_270_restores():
    f = (np.arange(6 * 4 * 3).reshape(6, 4, 3) % 256).astype(np.uint8)
    back = rotate_frame(rotate_frame(f, 90), 270)
    assert np.array_equal(back, f)


# ---- colour correction --------------------------------------------------- #
def test_color_correct_off_is_identity():
    f = (np.random.rand(40, 40, 3) * 255).astype(np.uint8)
    out = color_correct(f, ColorCorrectionSettings(mode="off"))
    assert np.array_equal(out, f)


def test_color_correct_auto_neutralizes_a_blue_cast():
    # Grey scene lit with a blue cast (B boosted). White-patch balance should pull
    # the bright pixels back toward neutral, shrinking the channel spread.
    f = np.full((60, 60, 3), 120, np.uint8)
    f[:, :, 0] = 210  # strong blue channel (BGR)
    f[0:5, 0:5] = (230, 200, 200)  # a few bright near-neutral pixels as the reference
    out = color_correct(f, ColorCorrectionSettings(mode="auto"))
    spread_before = float(np.ptp(f.reshape(-1, 3).mean(0)))
    spread_after = float(np.ptp(out.reshape(-1, 3).mean(0)))
    assert spread_after < spread_before


def test_color_correct_manual_gains_scale_channels():
    f = np.full((10, 10, 3), 100, np.uint8)
    cc = ColorCorrectionSettings(mode="manual", gain_b=1.5, gain_g=1.0, gain_r=0.5)
    out = color_correct(f, cc)
    b, g, r = out.reshape(-1, 3).mean(0)
    assert b > g > r  # B up, R down


def test_preprocess_frame_noop_defaults_preserve_shape():
    f = (np.random.rand(48, 64, 3) * 255).astype(np.uint8)
    out = preprocess_frame(f, CameraSettings(rotation=0, color=ColorCorrectionSettings(mode="off")))
    assert np.array_equal(out, f)


# ---- overhead vision gates ----------------------------------------------- #
def test_overhead_disables_parallax_and_far_rail():
    # A default (overhead) Settings must route the pipeline away from the oblique
    # parallax pose estimate — the fronto-parallel decomposition is degenerate.
    import types

    from billiards_trainer.vision.geometry import TableModel
    from billiards_trainer.vision.pipeline import Pipeline

    s = Settings()
    assert s.camera.overhead is True
    calib = types.SimpleNamespace(
        H=np.eye(3), Hinv=np.eye(3),
        table=TableModel.from_rect((567, 1053), 40, nose_inset_frac=0.055))
    pipe = Pipeline(s, source="")
    assert pipe._camera_position(calib, (1080, 1920, 3)) is None
