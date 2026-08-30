"""Shared value types for the vision pipeline.

Kept dependency-light (numpy only) so they can be imported by the event layer,
the DB layer, and tests without pulling in the whole pipeline.
"""

import math
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
    number: int = -1   # 0 = cue, 1..15 ball number, -1 = unknown
    # Colour actually MEASURED from this frame's pixels (glare-trimmed crop
    # median / stripe band), set only by code that sampled the frame. bgr
    # above is usually a canonical palette constant — evidence built on it
    # merely echoes the classifier's guess (review finding: the tracker's
    # colour consensus was an echo chamber). None = no measurement made.
    measured_bgr: tuple[int, int, int] | None = None
    #: Did the trained IDENTITY model actually read this ball, or is its
    #: number coming from measured colour alone? Round 81 measured that
    #: the identity model never once reads the cold table's black 8 (0 of
    #: 92) and reads its burgundy 7 twice in 85 - it emits no box at all
    #: at the dark balls, not a weak one. Those balls are still named
    #: ~100% correctly, entirely by the measured-colour path, so colour
    #: is not a backstop there: it is the only thing holding them up.
    #: Nothing recorded that, so it took a bespoke sweep to find. This
    #: carries it to the sidecar the way Track.read carries the vote
    #: evidence (round 68).
    identified: bool = False

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
    number: int = -1           # 0 = cue, 1..15 ball number, -1 = unknown
    bgr: tuple[int, int, int] = (200, 200, 200)
    age: int = 0               # total frames seen
    hits: int = 0              # frames matched to a detection
    misses: int = 0            # consecutive frames unmatched
    active: bool = True
    history: list[tuple[float, float]] = field(default_factory=list)
    coasting: bool = False     # this frame's position is PREDICTED, not seen.
                               # The offline engine has always published this
                               # (round 11) so consumers can tell a sighting
                               # from an estimate; it lives here now because
                               # there is one track type, not two (Joe,
                               # 2026-08-30: "it should all just become one
                               # file, or one module/class").
    id_read: bool = False      # the identity MODEL read this ball this
                               # frame; False means its name comes from
                               # measured colour alone. Round 81: the
                               # model never once reads the cold table's
                               # black 8 (0 of 92), so for the dark balls
                               # colour is not a backstop but the floor.
    read: int = -1             # WHAT THIS TRACK ACTUALLY SAW: the majority of
                               # its retained detection reads, BEFORE
                               # arbitration, the age bar, hysteresis and the
                               # uniqueness belt decide what it may SHOW.
                               # `number` is the verdict; this is the evidence.
                               # Round 68: five gates sit between the two and
                               # none of them left a trace, so a track could
                               # publish 13 on 330 frames while its own reads
                               # backed that on 8 of 366 (round 65) and nothing
                               # downstream could see the disagreement - it
                               # took a bespoke GPU sweep to find. This is a
                               # MEASUREMENT, not a verdict: nothing reads it
                               # to name a ball.

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)
