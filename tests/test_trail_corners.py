"""A bank must look like a bank.

Joe, 2026-08-30: "we are still not getting enough responsiveness
resolution to accurately reflect the bank shot."

Two causes, both measured on the pinned session's shot 5 (the 3's
140-degree cushion contact at 86.47s), drawn from the shipped samples:

    quadratic smoothing throughout : off by 18.8 units (0.72 ball-widths)
    with hard corners              : off by 10.0 units (0.39 ball-widths)

so the renderer's corner-rounding was about half the error, and the
0.15s sampling the rest - the ball moves up to 210 plane units, eight
ball-widths, between two samples, so a cushion contact can fall entirely
between them.

The exporter now keeps its 0.15s grid (the renderer interpolates the live
tip between consecutive samples, and uneven spacing there is what made
the tail move "in chunks") and ADDS samples where the path turns. The
renderer draws through a sharp vertex instead of past it.
"""

import math
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "companion-cloud" / "public" / "app.js"


@pytest.fixture(scope="module")
def app_js() -> str:
    return APP.read_text(encoding="utf-8")


def _code(js: str) -> str:
    return "\n".join(ln.split("//")[0] for ln in js.splitlines())


class TestRendererDrawsCorners:
    def test_both_builders_share_one_corner_rule(self, app_js):
        code = _code(app_js)
        assert "function segThrough(" in code, "the corner rule vanished"
        # both path builders must go through it, not carry their own copy
        for fn in ("function stablePrefix(", "function smoothPath("):
            i = code.find(fn)
            assert i > 0, f"{fn} vanished"
            body = code[i:i + 800]
            assert "segThrough(" in body, f"{fn} no longer uses the one rule"
            assert "quadraticCurveTo" not in body.split("segThrough")[0][-300:], (
                f"{fn} re-grew its own smoothing")

    def test_a_sharp_vertex_is_drawn_through(self, app_js):
        code = _code(app_js)
        i = code.find("function segThrough(")
        body = code[i:i + 600]
        assert "HARD_TURN_DEG" in body, "the threshold is gone"
        assert "lineTo" in body, (
            "a sharp turn must be drawn THROUGH the sample; a quadratic "
            "uses it only as a control point and never touches it")


class TestExporterSamplesTheCorner:
    """The exporter's densifier, reimplemented here against synthetic
    paths - the rule is what matters, not the file plumbing."""

    @staticmethod
    def _dense(rows, turn_deg=20.0, floor=2.0):
        keep = set()
        for i in range(1, len(rows) - 1):
            ax, ay = rows[i][1] - rows[i - 1][1], rows[i][2] - rows[i - 1][2]
            bx, by = rows[i + 1][1] - rows[i][1], rows[i + 1][2] - rows[i][2]
            na, nb = math.hypot(ax, ay), math.hypot(bx, by)
            if na < floor or nb < floor:
                continue
            cos = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
            if math.degrees(math.acos(cos)) > turn_deg:
                keep.update((i - 1, i, i + 1))
        return keep

    def test_a_bounce_gets_extra_samples(self):
        # straight in, sharp bounce, straight out - 20 units a frame
        rows = [(i / 30.0, 20.0 * i, 100.0) for i in range(10)]
        rows += [(0.33 + i / 30.0, 180.0 - 20.0 * i, 100.0 + 20.0 * i)
                 for i in range(1, 10)]
        keep = self._dense(rows)
        assert keep, "a 135-degree bounce produced no extra samples"
        assert any(8 <= i <= 11 for i in keep), (
            f"the extra samples are not at the corner: {sorted(keep)}")

    def test_a_resting_ball_gets_none(self):
        """THE GUARD THAT COST 10x. Without a travel floor the angle test
        fires on detector noise at rest - the first cut of this took the
        bench export from 920 points to 9,763 (24 KB -> 224 KB)."""
        import random
        random.seed(7)
        rows = [(i / 30.0, 300.0 + random.uniform(-0.6, 0.6),
                 400.0 + random.uniform(-0.6, 0.6)) for i in range(120)]
        assert not self._dense(rows), (
            "jitter at rest is being densified again - every frame of a "
            "still ball reads as a corner without the travel floor")

    def test_the_floor_is_still_in_the_exporter(self):
        src = (ROOT / "src" / "billiards_trainer" / "vision"
               / "shots_export.py").read_text(encoding="utf-8")
        i = src.find("A BANK NEEDS SAMPLES AT THE CUSHION")
        assert i > 0, "the densifier vanished"
        seg = src[i:i + 2600]
        assert re.search(r"na < 2\.0 or nb < 2\.0", seg), (
            "the travel floor is gone; a resting ball will be densified")
        assert "STEP, FINE" in seg, "the uniform grid was dropped"
