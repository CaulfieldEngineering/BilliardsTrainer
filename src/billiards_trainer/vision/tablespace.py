"""TRUE-INCH table frame — the shared measurement basis for analytics.

Every miss-forensic in docs/design/MISS_ANALYTICS.md had to rebuild
calibration by hand before it could measure anything, because a stored
transform can be wrong (the recovered file's mapped the "bed" to a
2.98-aspect region that wasn't the table). Root-cause analytics live or
die on inches and degrees, so the frame is derived HERE, from the
session's own data, with the scale anchored on the one invariant in the
picture: a pool ball is 2.25 inches, always.

Angles are scale-free — a bad scale never flips an aim/delivery verdict
— but distances, tolerances and "missed by 1.2 inches" all need this.
"""

import logging
from dataclasses import dataclass, field

log = logging.getLogger("vision.tablespace")

#: Regulation ball diameter (inches). The anchor for every measurement.
BALL_DIAM_IN = 2.25

#: Playing-surface short side by table size (inches).
BED_SHORT_IN = {"9ft": 50.0, "8ft": 46.0, "7ft": 39.0}


@dataclass
class TableSpace:
    """Rect-pixel <-> true-inch mapping for one session."""

    x0: float
    y0: float
    x1: float
    y1: float
    px_per_in: float
    ball_r_px: float
    size: str                    # inferred from geometry, not configured
    n_samples: int
    notes: list = field(default_factory=list)

    @property
    def bed_short_in(self) -> float:
        return min(self.x1 - self.x0, self.y1 - self.y0) / self.px_per_in

    @property
    def bed_long_in(self) -> float:
        return max(self.x1 - self.x0, self.y1 - self.y0) / self.px_per_in

    def to_in(self, x: float, y: float) -> tuple:
        """Rect pixels -> inches from the bed's top-left corner."""
        return ((x - self.x0) / self.px_per_in,
                (y - self.y0) / self.px_per_in)

    def dist_in(self, ax: float, ay: float, bx: float, by: float) -> float:
        return (((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5) / self.px_per_in

    def pockets(self) -> list:
        """(name, x, y) pocket centres in rect pixels — four corners plus
        the two long-rail midpoints, named from the overhead view."""
        mx, my = (self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0
        if (self.y1 - self.y0) >= (self.x1 - self.x0):   # portrait
            side = [("left-middle", self.x0, my), ("right-middle", self.x1, my)]
        else:
            side = [("top-middle", mx, self.y0), ("bottom-middle", mx, self.y1)]
        return [("top-left", self.x0, self.y0), ("top-right", self.x1, self.y0),
                ("bottom-left", self.x0, self.y1),
                ("bottom-right", self.x1, self.y1)] + side


def from_calibration(table, ball_r_px: float, n_samples: int = 0
                     ) -> TableSpace:
    """AUTHORITATIVE frame: the felt-detected cushion-nose rectangle for
    the bed, the ball for the scale.

    Preferred over from_reader: the ball-position CLOUD is only a lower
    bound on the bed (measured — drilling one shot never sends a ball to
    the far rails, and 34/35 archive sessions under-measured the short
    side by 5-20in that way), whereas table detection sees the cushions
    whether or not a ball ever visits them.
    """
    x0, y0 = float(table.x0), float(table.y0)
    x1, y1 = float(table.x1), float(table.y1)
    px_per_in = (2.0 * float(ball_r_px)) / BALL_DIAM_IN
    short_in = min(x1 - x0, y1 - y0) / px_per_in
    size = min(BED_SHORT_IN, key=lambda k: abs(BED_SHORT_IN[k] - short_in))
    return TableSpace(x0=x0, y0=y0, x1=x1, y1=y1, px_per_in=px_per_in,
                      ball_r_px=float(ball_r_px), size=size,
                      n_samples=n_samples, notes=["from calibration"])


def ball_radius_px(reader) -> float | None:
    """Median detected radius of NUMBERED balls across the session."""
    radii = sorted(row[3] for fr in reader._frames for row in fr
                   if row[6] and row[4] >= 0)
    return radii[len(radii) // 2] if len(radii) >= 50 else None


def from_reader(reader) -> TableSpace | None:
    """WEAK frame from the tracking cloud alone (no video decode).

    The envelope of ball positions is a LOWER BOUND on the bed — use
    only when calibration is unavailable, and treat its size inference
    as a hint. from_calibration is the real answer.

    Scale comes from the BALL (2.25in), never from an assumed table
    size — that is what makes the frame self-validating: the bed
    dimensions then FALL OUT in inches and can be checked against the
    known table sizes instead of being assumed correct.
    """
    xs, ys, radii = [], [], []
    for fr in reader._frames:
        for row in fr:
            if not row[6]:
                continue
            xs.append(row[1])
            ys.append(row[2])
            if row[4] >= 0:                      # numbered = a real ball
                radii.append(row[3])
    if len(xs) < 200 or len(radii) < 50:
        return None
    xs.sort(), ys.sort(), radii.sort()
    n = len(xs)
    # 0.5/99.5 percentile envelope: robust to a stray ghost track, and
    # the cloud's edge is one ball RADIUS inside the cushion nose.
    r = radii[len(radii) // 2]
    x0, x1 = xs[int(0.005 * n)] - r, xs[min(n - 1, int(0.995 * n))] + r
    y0, y1 = ys[int(0.005 * n)] - r, ys[min(n - 1, int(0.995 * n))] + r
    px_per_in = (2.0 * r) / BALL_DIAM_IN
    short_in = min(x1 - x0, y1 - y0) / px_per_in
    size = min(BED_SHORT_IN, key=lambda k: abs(BED_SHORT_IN[k] - short_in))
    return TableSpace(x0=x0, y0=y0, x1=x1, y1=y1, px_per_in=px_per_in,
                      ball_r_px=r, size=size, n_samples=n,
                      notes=["from track cloud (lower bound)"])


def space_for_video(video, reader, settings=None) -> TableSpace | None:
    """Best available frame for a session: warm the pipeline on a few
    frames to get the felt-detected table, else fall back to the cloud.
    One decode per session; callers should cache the result."""
    r = ball_radius_px(reader)
    if r is None:
        return from_reader(reader)
    try:
        import cv2

        from ..config import Settings
        from .pipeline import Pipeline
        st = settings or Settings.load()
        pipe = Pipeline(st)
        cap = cv2.VideoCapture(str(video))
        calib = None
        for k in range(40):
            cap.set(cv2.CAP_PROP_POS_MSEC, (20.0 + k * 0.5) * 1000)
            ok, frame = cap.read()
            if not ok:
                break
            pipe.process(frame, 20.0 + k * 0.5, annotate=False, detect=True)
            if pipe.calib.calib is not None:
                calib = pipe.calib.calib
                break
        cap.release()
        if calib is not None:
            return from_calibration(calib.table, r, len(reader._times))
    except Exception:  # noqa: BLE001 - calibration is best-effort
        log.exception("tablespace: calibration warmup failed")
    return from_reader(reader)


def audit(reader, configured_size: str = "", space=None) -> dict:
    """Calibration gates for one session: is this footage measurable?

    Returns {"ok": bool, "gates": [(name, ok, detail)], "space": ts|None}.
    Anything failing means the session's INCH figures are untrustworthy
    (its angles may still be fine — that distinction is the point).
    """
    ts = space if space is not None else from_reader(reader)
    gates = []
    if ts is None:
        return {"ok": False, "space": None,
                "gates": [("samples", False, "too few tracked states")]}

    def gate(name, ok, detail):
        gates.append((name, bool(ok), detail))

    gate("ball_radius", ts.ball_r_px >= 4.0,
         f"median ball radius {ts.ball_r_px:.1f}px")
    aspect = ts.bed_long_in / max(1e-6, ts.bed_short_in)
    gate("bed_aspect", 1.75 <= aspect <= 2.25,
         f"bed aspect {aspect:.2f} (expect ~2.0)")
    gate("bed_size_known", abs(BED_SHORT_IN[ts.size] - ts.bed_short_in) <= 6.0,
         f"short side {ts.bed_short_in:.1f}in -> {ts.size}")
    # SIZE IS A CONFIGURED FACT, NOT AN INFERENCE. Measured short sides
    # ran 42.5-47.8in across sessions (ball-radius detection varies with
    # lighting/parallax), which straddles 8ft and 9ft — the geometry
    # cannot settle it, the owner can. So this is a soft note: it fires
    # only when the measurement is FAR from the configured size, which
    # would mean the configuration is wrong rather than the optics noisy.
    if configured_size and configured_size in BED_SHORT_IN:
        off = abs(BED_SHORT_IN[configured_size] - ts.bed_short_in)
        gate("scale_plausible", off <= 9.0,
             f"measured {ts.bed_short_in:.1f}in vs configured "
             f"{configured_size} ({BED_SHORT_IN[configured_size]:.0f}in), "
             f"off by {off:.1f}in")
    # A session where the balls never spread over the bed cannot pin the
    # envelope — the frame would be too small in one axis.
    gate("coverage", ts.bed_long_in >= 0.7 * 2.0 * BED_SHORT_IN[ts.size],
         f"long side {ts.bed_long_in:.1f}in")
    gate("samples", ts.n_samples >= 2000, f"{ts.n_samples} states")
    ok = all(g[1] for g in gates)
    return {"ok": ok, "space": ts, "gates": gates}


def audit_summary_transform(tf: dict, ts: TableSpace) -> tuple:
    """(ok, detail) for a stored shots.json transform: does it map the
    measured bed to a plausible region of the video?

    This is the gate that catches the recovered file, whose stored
    transform mapped the bed to a 2.98-aspect strip that was not the
    table — every overlay drawn from it lands in the wrong place.
    """
    try:
        import numpy as np
        h = np.asarray(tf["hinv"], dtype=float)
        w, hh = float(tf["w"]), float(tf["h"])
    except Exception:  # noqa: BLE001
        return False, "unreadable transform"
    pts = []
    for (x, y) in ((ts.x0, ts.y0), (ts.x1, ts.y0),
                   (ts.x0, ts.y1), (ts.x1, ts.y1)):
        import numpy as np
        v = h @ np.array([x, y, 1.0])
        pts.append((v[0] / v[2], v[1] / v[2]))
    bw = max(p[0] for p in pts) - min(p[0] for p in pts)
    bh = max(p[1] for p in pts) - min(p[1] for p in pts)
    if bw <= 1 or bh <= 1:
        return False, "degenerate mapping"
    asp = max(bw, bh) / min(bw, bh)
    inside = all(-0.15 * w <= p[0] <= 1.15 * w
                 and -0.15 * hh <= p[1] <= 1.15 * hh for p in pts)
    big = min(bw / w, bh / hh) > 0.25
    ok = bool(inside and big and 1.6 <= asp <= 2.4)
    return ok, (f"video box {bw:.0f}x{bh:.0f} aspect {asp:.2f}"
                f"{'' if inside else ', OUTSIDE frame'}"
                f"{'' if big else ', too small'}")
