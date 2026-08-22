"""Regression tests for the table-detection port (felt + rectify).

These guard the crown-jewel CV pipeline against regressions by checking the
Python output against the labelled C++ reference capture.
"""

import numpy as np

from billiards_trainer.config import FeltSettings
from billiards_trainer.core.rectify import project_points, rectify_tabletop
from billiards_trainer.vision.felt import detect_felt


def _rmse(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def test_expected_ball_radius_scales_with_bed_size():
    """The known ball radius (2.25in ball) in rectified px scales with table size:
    a smaller bed packs the same ball into proportionally more pixels."""
    from billiards_trainer.core.geometry import TableModel, expected_ball_radius_px

    # short side = play width; build a table whose short side is 400 px
    table = TableModel.from_rect((400 + 40, 800 + 40), pad=20)
    assert abs(table.short_side - 400.0) < 1.0
    r9 = expected_ball_radius_px(table, "9ft")   # 50in bed
    r7 = expected_ball_radius_px(table, "7ft")   # 39in bed -> bigger px ball
    assert abs(r9 - 400.0 * 1.125 / 50.0) < 0.01
    assert r7 > r9
    # unknown size falls back to 9ft, never crashes
    assert expected_ball_radius_px(table, "bogus") == r9


def test_render_schematic_fixed_radius_overrides_track_radius():
    """fixed_radius makes every ball the same drawn size regardless of the
    tracker's per-ball radius (the bird's-eye size-normalization)."""
    from billiards_trainer.core.geometry import TableModel
    from billiards_trainer.core.types import BallClass, Track
    from billiards_trainer.vision.overlay import render_schematic

    table = TableModel.from_rect((300, 560), pad=20)
    cx = table.x0 + 40
    small = Track(id=1, x=cx, y=120, radius=3.0, cls=BallClass.CUE, bgr=(240, 240, 240))
    big = Track(id=2, x=cx, y=300, radius=18.0, cls=BallClass.CUE, bgr=(240, 240, 240))

    def white_span(img, y):
        row = img[int(y)]
        mask = (row[:, 0] > 180) & (row[:, 1] > 180) & (row[:, 2] > 180)
        return int(mask.sum())

    img = render_schematic(table, [small, big], fixed_radius=10.0, show_ids=False)
    # both balls drawn at radius 10 -> similar horizontal white span at their centers
    assert abs(white_span(img, 120) - white_span(img, 300)) <= 4
    # without fixed_radius the spans differ (3 vs 18)
    img2 = render_schematic(table, [small, big], show_ids=False)
    assert white_span(img2, 300) > white_span(img2, 120) + 6


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
