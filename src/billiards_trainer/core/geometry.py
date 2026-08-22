"""Table geometry in rectified (bird's-eye) coordinates.

After rectification the table is a padded portrait rectangle: the playing
surface spans ``(pad, pad)..(W-pad, H-pad)`` and the long rails are the vertical
(left/right) edges. This module models the six pockets and answers the queries
the event layer needs: nearest pocket, capture test, and on/off-table tests.
"""

import math
from dataclasses import dataclass

import numpy as np

# Stable pocket identifiers (used as DB keys + stat buckets).
POCKET_TL = "top-left"
POCKET_TR = "top-right"
POCKET_BR = "bottom-right"
POCKET_BL = "bottom-left"
POCKET_LEFT = "left-side"
POCKET_RIGHT = "right-side"
POCKET_ORDER = (POCKET_TL, POCKET_TR, POCKET_RIGHT, POCKET_BR, POCKET_BL, POCKET_LEFT)

# Physical pool dimensions (inches). A regulation ball is 2.25" diameter on every
# table; only the bed size changes. The playing-area SHORT side (width) is half the
# nominal table length. We use these to render every ball at its KNOWN size in the
# bird's-eye instead of whatever wobbly radius the detector returned per-frame.
BALL_DIAMETER_IN = 2.25
_TABLE_WIDTH_IN = {"9ft": 50.0, "8ft": 44.0, "7ft": 39.0, "10ft": 56.0}


def expected_ball_radius_px(table: "TableModel", table_size: str = "9ft") -> float:
    """The ball's known physical radius expressed in rectified pixels, from the
    table's short-side (width) pixel span and the real bed width for ``table_size``.
    Same ratio the detector's radius band is built around (~0.0225·short)."""
    width_in = _TABLE_WIDTH_IN.get(table_size, 50.0)
    return float(table.short_side) * (BALL_DIAMETER_IN / 2.0) / width_in


@dataclass(frozen=True)
class Pocket:
    name: str
    x: float
    y: float
    is_side: bool = False

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)


#: Prose forms for descriptions — the identifiers above are DB keys and
#: must stay stable, but "into the left-side pocket" reads badly, so the
#: sentence layer gets its own words from the SAME single vocabulary.
POCKET_LABEL = {
    POCKET_TL: "top-left", POCKET_TR: "top-right",
    POCKET_BR: "bottom-right", POCKET_BL: "bottom-left",
    POCKET_LEFT: "left-middle", POCKET_RIGHT: "right-middle",
}


def pockets_for_rect(x0: float, y0: float, x1: float, y1: float) -> list:
    """The six pockets of a playing rectangle — THE definition.

    Three modules had each grown their own copy (tablespace, describe,
    outcomes), and they had even drifted apart on naming: two called the
    side pockets "left-middle" while the DB keys said "left-side". One
    definition, one vocabulary. Side pockets sit at the midpoints of the
    LONG rails, which is why the orientation test is here and not in
    each caller."""
    portrait = (y1 - y0) >= (x1 - x0)
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    out = [Pocket(POCKET_TL, x0, y0), Pocket(POCKET_TR, x1, y0),
           Pocket(POCKET_BL, x0, y1), Pocket(POCKET_BR, x1, y1)]
    if portrait:
        out += [Pocket(POCKET_LEFT, x0, my, True),
                Pocket(POCKET_RIGHT, x1, my, True)]
    else:
        out += [Pocket(POCKET_LEFT, mx, y0, True),
                Pocket(POCKET_RIGHT, mx, y1, True)]
    return out


def nearest_pocket_in_rect(x: float, y: float, rect: tuple):
    """Closest pocket of ``rect`` = (x0, y0, x1, y1) to a point."""
    return min(pockets_for_rect(*rect),
               key=lambda p: (p.x - x) ** 2 + (p.y - y) ** 2)


@dataclass
class TableModel:
    """Playing-surface rectangle + pockets in rectified pixels.

    The detected felt spans the cloth — the bed PLUS the rail tops the cloth wraps
    over. The PLAYING area (cushion-nose to cushion-nose) is smaller: ``x0..y1`` is
    that inset playing rectangle, while ``width``/``height`` are the full cloth
    image. ``from_rect`` computes the inset from ``nose_inset_frac``."""

    width: int                 # full rectified image width
    height: int                # full rectified image height
    pad: int                   # padding between image edge and the cloth
    pockets: list[Pocket]
    pocket_radius: float
    x0: float = 0.0            # playing area (cushion-nose) rectangle
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0

    @property
    def play_w(self) -> float:
        return self.x1 - self.x0

    @property
    def play_h(self) -> float:
        return self.y1 - self.y0

    @property
    def short_side(self) -> float:
        return min(self.play_w, self.play_h)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_rect(cls, dst_size: tuple[int, int], pad: int,
                  pocket_radius_frac: float = 0.045,
                  nose_inset_frac: float = 0.0) -> "TableModel":
        w, h = dst_size
        # cloth extent (what felt detection found)
        cx0, cy0 = float(pad), float(pad)
        cx1, cy1 = float(w - pad), float(h - pad)
        # playing area = cloth inset by the rail-top width (cushion noses)
        inset = max(0.0, nose_inset_frac) * min(cx1 - cx0, cy1 - cy0)
        x0, y0 = cx0 + inset, cy0 + inset
        x1, y1 = cx1 - inset, cy1 - inset
        cy = (y0 + y1) / 2.0
        pockets = [
            Pocket(POCKET_TL, x0, y0),
            Pocket(POCKET_TR, x1, y0),
            Pocket(POCKET_BR, x1, y1),
            Pocket(POCKET_BL, x0, y1),
            Pocket(POCKET_LEFT, x0, cy, is_side=True),
            Pocket(POCKET_RIGHT, x1, cy, is_side=True),
        ]
        short = min(x1 - x0, y1 - y0)
        return cls(width=w, height=h, pad=pad, pockets=pockets,
                   pocket_radius=short * pocket_radius_frac,
                   x0=x0, y0=y0, x1=x1, y1=y1)

    # ------------------------------------------------------------------ #
    def nearest_pocket(self, x: float, y: float) -> tuple[Pocket, float]:
        best = min(self.pockets, key=lambda p: math.hypot(p.x - x, p.y - y))
        return best, math.hypot(best.x - x, best.y - y)

    def pocket_at(self, x: float, y: float, scale: float = 1.0) -> Pocket | None:
        """Return the pocket whose capture zone contains (x, y), else None."""
        pocket, dist = self.nearest_pocket(x, y)
        return pocket if dist <= self.pocket_radius * scale else None

    def on_table(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (self.x0 - margin) <= x <= (self.x1 + margin) and \
               (self.y0 - margin) <= y <= (self.y1 + margin)

    def diamonds(self) -> list[tuple[float, float]]:
        """Sight-diamond positions (3 per short rail, 6 per long rail), for the
        UI diagram. Standard regulation layout."""
        pts: list[tuple[float, float]] = []
        # long rails (vertical): 6 gaps -> diamonds at 1/8..7/8 excluding pockets
        for frac in (1, 2, 3, 5, 6, 7):
            y = self.y0 + self.play_h * frac / 8.0
            pts.append((self.x0, y))
            pts.append((self.x1, y))
        # short rails (horizontal): 3 diamonds at 1/4, 1/2, 3/4
        for frac in (1, 2, 3):
            x = self.x0 + self.play_w * frac / 4.0
            pts.append((x, self.y0))
            pts.append((x, self.y1))
        return pts


def velocity_to_speed(vx: float, vy: float) -> float:
    return float(np.hypot(vx, vy))
