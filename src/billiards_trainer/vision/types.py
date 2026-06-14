"""Shared value types for the vision pipeline.

Kept dependency-light (numpy only) so they can be imported by the event layer,
the DB layer, and tests without pulling in the whole pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# Corner ordering used everywhere: top-left, top-right, bottom-right, bottom-left.
CORNER_LABELS = ("TL", "TR", "BR", "BL")


@dataclass
class FeltResult:
    """Output of felt/table-surface detection on the original camera frame."""

    ok: bool = False
    has_corners: bool = False
    mask: np.ndarray | None = None          # CV_8U 0/255
    contour: np.ndarray | None = None        # Nx1x2 int
    corners: np.ndarray | None = None         # (4,2) float32 in TL,TR,BR,BL order
    area_ratio: float = 0.0

    def corners_dict(self) -> dict:
        if self.corners is None:
            return {}
        return {
            lbl: {"x": float(self.corners[i, 0]), "y": float(self.corners[i, 1])}
            for i, lbl in enumerate(CORNER_LABELS)
        }


@dataclass
class RectifyResult:
    """Output of the homography/bird's-eye rectification step."""

    ok: bool = False
    rectified_bgr: np.ndarray | None = None
    rectified_mask: np.ndarray | None = None
    H: np.ndarray | None = None               # 3x3 float64  src -> rectified
    Hinv: np.ndarray | None = None             # 3x3 float64  rectified -> src
    dst_size: tuple[int, int] = (0, 0)            # (w, h)
    src_quad: np.ndarray | None = None         # (4,2) original corners
    dst_quad: np.ndarray | None = None         # (4,2) destination corners


class BallClass(str, Enum):
    CUE = "cue"
    SOLID = "solid"
    STRIPE = "stripe"
    EIGHT = "eight"
    UNKNOWN = "unknown"


@dataclass
class Detection:
    """A single ball-like detection on the rectified (bird's-eye) view."""

    x: float          # center, rectified pixels
    y: float
    radius: float
    bgr: tuple[int, int, int] = (200, 200, 200)
    cls: BallClass = BallClass.UNKNOWN
    score: float = 1.0

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Track:
    """A tracked ball with identity persistence and a short motion history."""

    id: int
    x: float
    y: float
    radius: float
    vx: float = 0.0
    vy: float = 0.0
    cls: BallClass = BallClass.UNKNOWN
    bgr: tuple[int, int, int] = (200, 200, 200)
    age: int = 0               # total frames seen
    hits: int = 0              # frames matched to a detection
    misses: int = 0            # consecutive frames unmatched
    active: bool = True
    history: list[tuple[float, float]] = field(default_factory=list)

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))
