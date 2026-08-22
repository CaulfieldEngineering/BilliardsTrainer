"""Blur recovery: finding a ball the detector lost to motion smear.

Until now this was only exercised end-to-end on real footage, which is how
it shipped with three separate bugs in it. These pin the contract directly.
"""

from collections import deque

import cv2
import numpy as np

from billiards_trainer.core.geometry import TableModel
from billiards_trainer.vision.blur_recovery import BlurRecovery

FELT = (200, 120, 40)          # BGR, a cool blue like the real cloth
PURPLE = (139, 30, 60)         # the solid 4
WHITE = (245, 245, 245)        # the cue ball
SIZE = 400


def _frame(balls=()):
    """A felt-coloured frame with discs painted on it."""
    f = np.full((SIZE, SIZE, 3), FELT, dtype=np.uint8)
    for (x, y, bgr) in balls:
        cv2.circle(f, (int(x), int(y)), 10, bgr, -1)
    return f


class _Track:
    def __init__(self, tid, x, y, bgr, misses=2):
        self.id = tid
        self.x, self.y = float(x), float(y)
        self.vx = self.vy = 0.0
        self.confirmed = True
        self.settled = False        # so the colour veto stays out of the way
        self.misses = misses
        self.committed_number = 4
        self.mbgr_hist = deque([tuple(bgr)] * 10, maxlen=20)
        self.colour_hist = deque(maxlen=40)


class _Tracker:
    max_dist_frac = 0.16
    _ball_r = 10.0

    def __init__(self, tracks):
        self._tracks = list(tracks)


class _Calib:
    H = np.eye(3)
    table = TableModel.from_rect((SIZE, SIZE), pad=0)


def _warm(rec, frame, n=8):
    """Fill the history the median background is built from."""
    for _ in range(n):
        rec._buf.append(frame)


def test_a_still_scene_recovers_nothing():
    """Nothing moved, so there is nothing to find — recovery must not
    invent a detection out of sensor noise or its own subtraction."""
    rec = BlurRecovery()
    still = _frame([(200, 200, PURPLE)])
    _warm(rec, still)
    tracker = _Tracker([_Track(1, 200, 200, PURPLE)])
    assert rec.find(still, _Calib(), tracker, []) == []


def test_a_moved_ball_is_found_and_attributed_to_its_track():
    """The ball left its spot and the finder missed it. Recovery should
    locate it and say WHICH track it belongs to — association refuses to
    re-judge it on distance precisely because it is far away."""
    rec = BlurRecovery()
    _warm(rec, _frame([(200, 200, PURPLE)]))
    moved = _frame([(300, 250, PURPLE)])      # same ball, well outside any gate
    tracker = _Tracker([_Track(1, 200, 200, PURPLE)])
    out = rec.find(moved, _Calib(), tracker, [])
    assert len(out) == 1, "the ball was in plain sight and was not recovered"
    d = out[0]
    assert d.recovered_for == 1
    assert abs(d.x - 300) < 14 and abs(d.y - 250) < 14, f"found at ({d.x},{d.y})"
    assert d.number == -1, "recovery must never NAME a ball"


def test_a_blob_that_looks_like_another_ball_loses_the_contest():
    """The cue ball rolling through the purple 4's search window must not be
    handed to the 4. Blur washes colour out, so the test is relative: the
    blob goes to whichever track it resembles most, and this one is white."""
    rec = BlurRecovery()
    _warm(rec, _frame([(200, 200, PURPLE), (80, 80, WHITE)]))
    # the 4 is gone; only the WHITE ball has moved into its window
    moved = _frame([(300, 250, WHITE)])
    tracker = _Tracker([_Track(1, 200, 200, PURPLE),
                        _Track(2, 80, 80, WHITE, misses=0)])
    out = rec.find(moved, _Calib(), tracker, [])
    assert not [d for d in out if d.recovered_for == 1], \
        "the purple 4's track adopted a white ball"


def test_history_is_required_before_the_median_is_trusted():
    """A cold start has no idea what the static scene looks like, so it must
    not guess — every pixel would read as motion."""
    rec = BlurRecovery()
    rec._buf.append(_frame([(200, 200, PURPLE)]))
    tracker = _Tracker([_Track(1, 200, 200, PURPLE)])
    assert rec.find(_frame([(300, 250, PURPLE)]), _Calib(), tracker, []) == []
