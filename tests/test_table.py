"""Regression tests for the table-detection port (felt + rectify).

These guard the crown-jewel CV pipeline against regressions by checking the
Python output against the labelled C++ reference capture.
"""

import numpy as np

from billiards_trainer.config import FeltSettings
from billiards_trainer.vision.felt import detect_felt
from billiards_trainer.vision.rectify import project_points, rectify_tabletop


def _rmse(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def test_felt_detects_corners(table_frame):
    res = detect_felt(table_frame, FeltSettings())
    assert res.ok
    assert res.has_corners
    assert res.corners.shape == (4, 2)
    # felt should occupy a meaningful but not absurd fraction of the frame
    assert 0.05 < res.area_ratio < 0.6


def test_felt_corner_rmse_under_threshold(table_frame, table_truth):
    res = detect_felt(table_frame, FeltSettings())
    rmse = _rmse(res.corners, table_truth["corners"])
    # Verified ~7.8 px at port time; 25 px leaves headroom for cv2 version drift.
    assert rmse < 25.0, f"corner RMSE {rmse:.1f}px exceeds 25px"


def test_felt_corner_order(table_frame):
    res = detect_felt(table_frame, FeltSettings())
    tl, tr, br, bl = res.corners
    assert tl[0] < tr[0] and bl[0] < br[0]   # left corners left of right corners
    assert tl[1] < bl[1] and tr[1] < br[1]   # top corners above bottom corners


def test_rectify_produces_portrait(table_frame):
    res = detect_felt(table_frame, FeltSettings())
    rect = rectify_tabletop(table_frame, res.mask, res.corners, pad_px=40, aspect=2.0)
    assert rect.ok
    w, h = rect.dst_size
    assert h > w  # portrait
    interior_w = w - 2 * 40
    interior_h = h - 2 * 40
    assert abs((interior_h / interior_w) - 2.0) < 0.05


def test_rectify_homography_invertible(table_frame):
    res = detect_felt(table_frame, FeltSettings())
    rect = rectify_tabletop(table_frame, res.mask, res.corners, pad_px=40)
    assert rect.H is not None and rect.Hinv is not None
    identity = rect.H @ rect.Hinv
    assert np.allclose(identity, np.eye(3), atol=1e-6)


def test_rectify_corners_map_to_dst(table_frame):
    res = detect_felt(table_frame, FeltSettings())
    rect = rectify_tabletop(table_frame, res.mask, res.corners, pad_px=40, refine=False)
    # Without refinement the detected corners must map onto the padded rectangle.
    mapped = project_points(res.corners, rect.H)
    assert _rmse(mapped, rect.dst_quad) < 1.0
