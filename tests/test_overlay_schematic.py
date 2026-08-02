"""Modern overhead schematic renderer: tournament-blue cloth, glossy colour-only
balls (no numbers). Guards shape, determinism, and that every ball class renders."""

import numpy as np

from billiards_trainer.vision.geometry import TableModel
from billiards_trainer.vision.overlay import render_schematic
from billiards_trainer.vision.types import BallClass, Track


def _table():
    return TableModel.from_rect((410, 800), 28, 0.045)


def _mixed_tracks(table):
    x0, y0, x1, y1 = table.x0, table.y0, table.x1, table.y1
    w, h = x1 - x0, y1 - y0
    r = table.short_side * 0.03
    spec = [
        (BallClass.CUE, (245, 245, 245), 0.5, 0.8),
        (BallClass.SOLID, (30, 200, 245), 0.4, 0.3),
        (BallClass.EIGHT, (24, 24, 26), 0.5, 0.45),
        (BallClass.STRIPE, (40, 40, 210), 0.6, 0.6),
        (BallClass.UNKNOWN, (200, 200, 200), 0.3, 0.55),
    ]
    return [Track(id=i, x=x0 + w * fx, y=y0 + h * fy, radius=r, cls=c, bgr=b)
            for i, (c, b, fx, fy) in enumerate(spec)]


def test_schematic_shape_and_all_ball_classes_render():
    table = _table()
    out = render_schematic(table, _mixed_tracks(table), fixed_radius=table.short_side * 0.03)
    # display-only image is exactly the table dimensions (no click-coord coupling)
    assert out.shape == (table.height, table.width, 3)
    assert out.dtype == np.uint8
    assert out.any()


def test_schematic_is_deterministic():
    table = _table()
    tracks = _mixed_tracks(table)
    a = render_schematic(table, tracks, fixed_radius=table.short_side * 0.03)
    b = render_schematic(table, tracks, fixed_radius=table.short_side * 0.03)
    assert np.array_equal(a, b), "same inputs must render identically (seeded texture)"


def test_schematic_moving_ball_draws_trail_without_error():
    table = _table()
    cue = _mixed_tracks(table)[0]
    cue.vx, cue.vy = -6.0, -9.0
    cue.history = [(cue.x - cue.vx * k * 0.5, cue.y - cue.vy * k * 0.5) for k in range(10)]
    out = render_schematic(table, [cue], show_traj=True, fixed_radius=table.short_side * 0.03)
    assert out.shape == (table.height, table.width, 3)
