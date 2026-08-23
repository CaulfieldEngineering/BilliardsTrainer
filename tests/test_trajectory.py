"""Trajectory fit v1: straight legs, named rail reflections, honest residuals."""

import math

from billiards_trainer.vision.trajectory import fit_shot


class _T:
    x0, y0, x1, y1 = 0.0, 0.0, 533.0, 1067.0


def _leg(t0, x0, y0, x1, y1, n=8):
    return [(t0 + i * 0.1, x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n)
            for i in range(n + 1)]


def test_single_straight_roll():
    obs = [(0.0, 200.0, 500.0)] * 3 + _leg(0.3, 200, 500, 120, 900)
    f = fit_shot(obs, _T, 12.0)
    assert f is not None and len(f.segments) == 1 and f.rail is None
    ux, uy = f.departure
    ex, ey = (120 - 200) / math.hypot(80, 400), (900 - 500) / math.hypot(80, 400)
    assert ux * ex + uy * ey > 0.999, "direction off"
    assert f.residual < 1.0


def test_bottom_rail_bounce_is_found_and_named():
    down = _leg(0.0, 200, 500, 150, 1055)          # to the bottom cushion
    back = _leg(0.9, 150, 1055, 120, 700)          # and back up
    f = fit_shot([(0.0, 200.0, 500.0)] + down + back[1:], _T, 12.0)
    assert f is not None and len(f.segments) == 2
    assert f.rail == "bottom"
    assert f.departure[1] > 0, "first leg must head DOWN the table"


def test_gap_does_not_bend_the_fit():
    """A hole in the middle of a straight roll must not invent a rail."""
    a = _leg(0.0, 200, 500, 160, 700, n=4)
    b = _leg(1.5, 120, 900, 100, 1000, n=4)        # same line, later
    f = fit_shot(a + b, _T, 12.0)
    assert f is not None
    assert f.rail is None, "a straight roll with a hole grew a phantom rail"


def test_mid_table_bend_is_a_contact_not_a_rail():
    a = _leg(0.0, 260, 500, 260, 700)
    b = _leg(1.0, 260, 700, 400, 800)              # bends mid-table
    f = fit_shot(a + b[1:], _T, 12.0)
    assert f is not None and len(f.segments) == 2
    assert f.rail is None, "a mid-table bend is not a cushion"


def test_too_little_data_returns_none():
    assert fit_shot([(0.0, 10.0, 10.0)] * 4, _T, 12.0) is None


def test_first_leg_survives_a_rattle_heavy_path():
    """The global 2-leg fit fails on rattle + aftermath; the first leg is
    still perfectly measurable — the backlog's dominant failure mode."""
    from billiards_trainer.vision.trajectory import fit_first_leg
    leg = _leg(0.0, 200, 500, 150, 1000)               # clean departure
    rattle = [(1.1, 155, 1040), (1.2, 148, 1010), (1.3, 158, 1035),
              (1.4, 150, 1005), (1.5, 160, 1030), (1.6, 300, 900),
              (1.7, 310, 700)]                          # chaos afterwards
    f = fit_first_leg([(0.0, 200.0, 500.0)] * 2 + leg + rattle, 12.0)
    assert f is not None
    ux, uy, n, r = f
    import math as m
    ex, ey = -50 / m.hypot(50, 500), 500 / m.hypot(50, 500)
    assert ux * ex + uy * ey > 0.995, f"direction off ({ux:.2f},{uy:.2f})"
    assert n >= 8, "should have used most of the clean leg"
