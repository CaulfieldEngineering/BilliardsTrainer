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


@dataclass(frozen=True)
class Pocket:
    name: str
    x: float
    y: float
    is_side: bool = False

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class TableModel:
    """Playing-surface rectangle + pockets in rectified pixels."""

    width: int                 # full rectified image width
    height: int                # full rectified image height
    pad: int                   # padding between image edge and playing surface
    pockets: list[Pocket]
    pocket_radius: float

    @property
    def x0(self) -> float:
        return float(self.pad)

    @property
    def y0(self) -> float:
        return float(self.pad)

    @property
    def x1(self) -> float:
        return float(self.width - self.pad)

    @property
    def y1(self) -> float:
        return float(self.height - self.pad)

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
                  pocket_radius_frac: float = 0.045) -> "TableModel":
        w, h = dst_size
        x0, y0 = float(pad), float(pad)
        x1, y1 = float(w - pad), float(h - pad)
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
                   pocket_radius=short * pocket_radius_frac)

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
