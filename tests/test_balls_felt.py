"""Felt-aware ball classification (the blue-felt 'everything is a 2/10' fix)."""

import cv2
import numpy as np

from billiards_trainer.core.balls import classify_pool_ball
from billiards_trainer.core.types import BallClass

FELT_HSV = (99, 199, 255)   # Joe's blue felt, as calibrated


def _patch(hsv, size=40):
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = hsv
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def _ball_on_felt(ball_hsv, size=40):
    """A ball-coloured disc centred on a felt-coloured square."""
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = FELT_HSV
    bgr = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    ball = cv2.cvtColor(np.full((1, 1, 3), ball_hsv, np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    cv2.circle(bgr, (size // 2, size // 2), int(size * 0.42), ball.tolist(), -1)
    return bgr


def test_pure_felt_patch_is_unknown_not_blue():
    """A phantom detection on bare felt must NOT classify as the blue 2/10."""
    cls, num, _ = classify_pool_ball(_patch(FELT_HSV), felt_hsv=FELT_HSV)
    assert cls == BallClass.UNKNOWN and num == -1


def test_shadowed_felt_is_unknown():
    """Darker felt (shadow of a hand/hat) is still felt, not a blue ball."""
    cls, num, _ = classify_pool_ball(_patch((99, 199, 140)), felt_hsv=FELT_HSV)
    assert cls == BallClass.UNKNOWN and num == -1


def test_red_ball_on_felt_still_reads_red():
    cls, num, _ = classify_pool_ball(_ball_on_felt((3, 220, 200)), felt_hsv=FELT_HSV)
    assert num == 3 and cls == BallClass.SOLID


def test_purple_ball_on_blue_felt_reads_purple_not_blue():
    """Measured on Joe's camera: the purple 4 sits at hue ~118, right on the
    generic blue reference — on blue felt a surviving blue-family vote is
    really purple (true blue is excluded with the cloth)."""
    cls, num, _ = classify_pool_ball(_ball_on_felt((120, 200, 170)), felt_hsv=FELT_HSV)
    assert num == 4 and cls == BallClass.SOLID


def test_without_felt_reference_behaviour_unchanged():
    """No felt_hsv (green-felt tables, old callers): blue stays blue."""
    cls, num, _ = classify_pool_ball(_ball_on_felt((110, 200, 170)))
    assert num == 2 and cls == BallClass.SOLID
