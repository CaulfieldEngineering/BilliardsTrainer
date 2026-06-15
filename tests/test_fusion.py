"""Multi-modal evidence fusion: background subtraction, optical flow, and the
weighted fused-activity score that suppresses single-signal noise."""

import numpy as np

from billiards_trainer.config import (
    BallSettings,
    DetectionSettings,
    apply_detection_preset,
)
from billiards_trainer.events.shot_detector import ShotDetector
from billiards_trainer.vision.background import BackgroundModel, downscale, flow_activity


def test_background_warmup_then_detects_change():
    bg = BackgroundModel()
    base = np.full((80, 160), 120, np.uint8)
    # warm up on a static frame
    for _ in range(70):
        bg.update(base)
    assert bg.ready
    quiet = bg.update(base)
    moved = base.copy()
    moved[30:50, 60:90] = 240          # a bright object appears
    fg = bg.update(moved)
    assert fg > quiet                  # the change registers as foreground


def test_flow_activity_zero_without_motion():
    a = np.full((80, 160), 100, np.uint8)
    assert flow_activity(a, a) == 0.0   # identical frames -> no flow


def test_downscale_caps_width():
    big = np.zeros((400, 800), np.uint8)
    assert downscale(big, max_w=160).shape[1] == 160


def test_fused_score_dominated_by_foreground():
    d = DetectionSettings()
    det = ShotDetector(d, BallSettings())
    # flicker: high motion + high (spurious) flow but ~no foreground -> below gate
    flicker = det._fuse(1.5, {"motion": 1.5, "flow": 15.0, "fg": 0.02})
    # real ball: real foreground present -> above gate
    ball = det._fuse(0.6, {"motion": 0.6, "flow": 2.0, "fg": 0.8})
    assert flicker < d.fusion_active
    assert ball >= d.fusion_active
    assert ball > flicker


def test_presets_set_gates():
    d = DetectionSettings()
    apply_detection_preset(d, "conservative")
    assert d.preset == "conservative" and d.min_travel_px >= 150
    apply_detection_preset(d, "aggressive")
    assert d.preset == "aggressive" and d.min_travel_px <= 100
