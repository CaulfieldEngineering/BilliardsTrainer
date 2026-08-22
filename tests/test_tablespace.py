"""True-inch table frame: the measurement basis for miss analytics.

Scale is anchored on the ball (2.25in, always); the bed comes from
calibration, never from the ball-position cloud (measured: the cloud
under-measures whenever play is concentrated)."""

import math

from billiards_trainer.vision.tablespace import (
    BALL_DIAM_IN,
    TableSpace,
    from_calibration,
)


class _Table:
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1


def test_scale_comes_from_the_ball():
    # a 12px-radius ball is 2.25in across -> 10.667 px/in
    ts = from_calibration(_Table(0, 0, 533, 1067), 12.0)
    assert abs(ts.px_per_in - (24.0 / BALL_DIAM_IN)) < 1e-9
    assert abs(ts.bed_short_in - 533 / ts.px_per_in) < 1e-6
    assert abs(ts.dist_in(0, 0, ts.px_per_in * 10, 0) - 10.0) < 1e-6


def test_size_inferred_from_measured_bed():
    ppi = 24.0 / BALL_DIAM_IN                       # 10.667 px/in
    assert from_calibration(_Table(0, 0, 50 * ppi, 100 * ppi), 12.0).size == "9ft"
    assert from_calibration(_Table(0, 0, 46 * ppi, 92 * ppi), 12.0).size == "8ft"
    assert from_calibration(_Table(0, 0, 39 * ppi, 78 * ppi), 12.0).size == "7ft"


def test_pockets_portrait_have_side_pockets_on_long_rails():
    ts = from_calibration(_Table(100, 200, 400, 800), 12.0)   # portrait
    names = {n for n, _x, _y in ts.pockets()}
    assert names == {"top-left", "top-right", "bottom-left", "bottom-right",
                     "left-middle", "right-middle"}
    mid = next((x, y) for n, x, y in ts.pockets() if n == "left-middle")
    assert mid == (100, 500)


def test_inch_conversion_round_trip():
    ts = TableSpace(x0=10, y0=20, x1=520, y1=1040, px_per_in=10.0,
                    ball_r_px=11.25, size="9ft", n_samples=5000)
    ix, iy = ts.to_in(110, 220)
    assert math.isclose(ix, 10.0) and math.isclose(iy, 20.0)
    assert math.isclose(ts.dist_in(10, 20, 10, 1020), 100.0)
