"""The v2 delivery picker: strike minus FORWARD-SWING ONSET.

Library measurement that forced the rewrite: the v1 anchor (rearmost
point) charged the whole back-pause — and sometimes an entire practice
cycle — to the delivery: 15.8% of deliveries read >1500ms, 12.9%
<100ms, and delivery correlated with pause (0.29). v2 walks back from
the last visible pre-strike sample while the tip moves forward, and
abstains instead of reporting garbage.
"""

import numpy as np

from billiards_trainer.vision.stroke_vision import (
    DELIVER_MAX_MS,
    _backstroke,
)

FPS = 30.0
REST = np.array([0.0, 0.0])


def _s(x):
    """A tip sample on the x-axis aim line."""
    return {"tip": np.array([float(x), 0.0]), "dir": np.array([1.0, 0.0]),
            "dist_aim": 0.0}


def _timeline(segments, t_strike):
    """segments: list of (t0, t1, x0, x1) linear tip moves, sampled at 30fps."""
    tl, seen = [], set()
    for t0, t1, x0, x1 in segments:
        for t in np.arange(t0, t1, 1.0 / FPS):
            x = x0 + (x1 - x0) * (t - t0) / max(1e-9, t1 - t0)
            tt = round(float(t), 3)
            if tt not in seen:       # video frames are strictly increasing
                seen.add(tt)
                tl.append((tt, _s(x)))
    return [e for e in tl if e[0] < t_strike]


class TestDeliveryPicker:
    def test_normal_stroke_measures_the_final_swing(self):
        # address, pull back, 1.0s pause, 0.4s swing, strike at 10.0
        tl = _timeline([(5.0, 8.0, -20, -20), (8.0, 8.5, -20, -120),
                        (8.5, 9.5, -120, -120), (9.5, 9.9, -120, -15)], 10.0)
        out = _backstroke(tl, 10.0, REST, FPS)
        assert out["backstroke_conf"] != "none"
        # v1 would say (10.0 - 8.5) = 1500ms; the swing itself is ~500ms
        assert 350 <= out["delivery_ms"] <= 650
        assert out["pause_ms"] >= 700

    def test_pause_length_no_longer_leaks_into_delivery(self):
        short = _timeline([(5.0, 8.7, -20, -20), (8.7, 9.2, -20, -120),
                           (9.2, 9.5, -120, -120), (9.5, 9.9, -120, -15)], 10.0)
        long = _timeline([(5.0, 7.8, -20, -20), (7.8, 8.3, -20, -120),
                          (8.3, 9.5, -120, -120), (9.5, 9.9, -120, -15)], 10.0)
        d_short = _backstroke(short, 10.0, REST, FPS)["delivery_ms"]
        d_long = _backstroke(long, 10.0, REST, FPS)["delivery_ms"]
        # v1: these differ by ~900ms (the pause). v2: same swing, same number.
        assert abs(d_long - d_short) <= 150

    def test_offscreen_delivery_abstains(self):
        # tip lost a full second before the strike: the swing was never seen
        tl = _timeline([(5.0, 8.0, -20, -20), (8.0, 8.5, -20, -120),
                        (8.5, 9.0, -120, -120)], 10.0)
        out = _backstroke(tl, 10.0, REST, FPS)
        assert out["backstroke_conf"] != "none"
        assert "delivery_ms" not in out, "unseen swing must abstain, not guess"
        assert out["back_depth_px"] > 90    # the rest is still measured

    def test_implausibly_long_swing_abstains(self):
        # a 1.6s forward crawl (v1's classic mispick shape) is not a delivery
        tl = _timeline([(5.0, 8.0, -20, -20), (8.0, 8.3, -20, -120),
                        (8.3, 9.9, -120, -10)], 10.0)
        out = _backstroke(tl, 10.0, REST, FPS)
        d = out.get("delivery_ms")
        assert d is None or d <= DELIVER_MAX_MS
        assert "back_depth_px" in out
