"""Multi-object tracker for balls.

A compact ByteTrack-style tracker: greedy distance association with gating, a
constant-velocity motion model so tracks coast through brief occlusions, and a
hit/miss lifecycle for stable IDs. Class labels are majority-voted over a short
history so cue/solid/stripe doesn't flicker frame to frame.

No scipy/lap dependency — greedy nearest-neighbour with a distance gate is plenty
for <=16 near-rigid objects and keeps the installer lean.
"""

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from .types import BallClass, Detection, Track


@dataclass
class _Internal:
    id: int
    x: float
    y: float
    radius: float
    vx: float = 0.0
    vy: float = 0.0
    age: int = 0
    hits: int = 0
    misses: int = 0
    confirmed: bool = False
    bgr: tuple = (200, 200, 200)
    cls_hist: deque = field(default_factory=lambda: deque(maxlen=15))
    pos_hist: deque = field(default_factory=lambda: deque(maxlen=64))

    def predict(self) -> None:
        self.x += self.vx
        self.y += self.vy
        self.age += 1

    @property
    def cls(self) -> BallClass:
        if not self.cls_hist:
            return BallClass.UNKNOWN
        return Counter(self.cls_hist).most_common(1)[0][0]


class BallTracker:
    def __init__(self, max_dist_frac: float = 0.08, max_misses: int = 12,
                 min_hits: int = 2, vel_alpha: float = 0.6,
                 pos_alpha_slow: float = 0.15, pos_alpha_fast: float = 0.85,
                 speed_lo: float = 3.0, speed_hi: float = 8.0):
        self.max_dist_frac = max_dist_frac
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.vel_alpha = vel_alpha
        # Adaptive position smoothing: heavy (slow alpha) when a ball is ~still so
        # the bird's-eye doesn't jitter while Joe tunes, light (fast alpha) when it
        # moves so a struck cue ball is FOLLOWED, not trailed by a foot. Alpha is
        # interpolated between slow/fast over [speed_lo, speed_hi] px/frame.
        self.pos_alpha_slow = pos_alpha_slow
        self.pos_alpha_fast = pos_alpha_fast
        self.speed_lo = speed_lo
        self.speed_hi = speed_hi
        self._tracks: list[_Internal] = []
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    # ------------------------------------------------------------------ #
    def update(self, detections: list[Detection], short_side: float) -> list[Track]:
        gate = max(8.0, self.max_dist_frac * short_side)

        for t in self._tracks:
            t.predict()

        unmatched_dets = set(range(len(detections)))
        # Build all (track, det) pairs within the gate, then assign greedily.
        pairs = []
        for ti, t in enumerate(self._tracks):
            for di in unmatched_dets:
                d = detections[di]
                dist = np.hypot(t.x - d.x, t.y - d.y)
                if dist <= gate:
                    pairs.append((dist, ti, di))
        pairs.sort(key=lambda p: p[0])

        matched_tracks = set()
        matched_dets = set()
        for _dist, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            self._apply_match(self._tracks[ti], detections[di])

        # Unmatched tracks -> coast / age out
        for ti, t in enumerate(self._tracks):
            if ti in matched_tracks:
                continue
            t.misses += 1
            # damp velocity while coasting so it doesn't drift forever
            t.vx *= 0.6
            t.vy *= 0.6

        # Unmatched detections -> new tentative tracks
        for di, d in enumerate(detections):
            if di in matched_dets:
                continue
            self._spawn(d)

        self._tracks = [t for t in self._tracks if t.misses <= self.max_misses]
        return self._public()

    def _apply_match(self, t: _Internal, d: Detection) -> None:
        meas_vx = d.x - t.x
        meas_vy = d.y - t.y
        spd = (meas_vx * meas_vx + meas_vy * meas_vy) ** 0.5
        t.vx = self.vel_alpha * meas_vx + (1 - self.vel_alpha) * t.vx
        t.vy = self.vel_alpha * meas_vy + (1 - self.vel_alpha) * t.vy
        if spd < self.speed_lo:
            # a ~still ball shouldn't carry velocity, or predict() injects jitter
            t.vx *= 0.25
            t.vy *= 0.25
        # adaptive position alpha: follow fast motion, smooth slow motion
        frac = (spd - self.speed_lo) / max(1e-6, self.speed_hi - self.speed_lo)
        frac = max(0.0, min(1.0, frac))
        pos_a = self.pos_alpha_slow + (self.pos_alpha_fast - self.pos_alpha_slow) * frac
        t.x = pos_a * d.x + (1 - pos_a) * t.x
        t.y = pos_a * d.y + (1 - pos_a) * t.y
        # Radius: once a track is established, reject wild size jumps (sensor-noise
        # outliers) and smooth slowly, so a held ball stops "pumping" in size.
        if t.confirmed and t.radius > 0 and abs(d.radius - t.radius) > 0.35 * t.radius:
            pass  # keep the stable estimate; this frame's size is an outlier
        else:
            a = 0.2 if t.confirmed else 0.5
            t.radius = a * d.radius + (1 - a) * t.radius
        t.bgr = d.bgr
        t.cls_hist.append(d.cls)
        t.pos_hist.append((t.x, t.y))
        t.hits += 1
        t.misses = 0
        if t.hits >= self.min_hits:
            t.confirmed = True

    def _spawn(self, d: Detection) -> None:
        t = _Internal(id=self._next_id, x=d.x, y=d.y, radius=d.radius, bgr=d.bgr)
        t.cls_hist.append(d.cls)
        t.pos_hist.append((d.x, d.y))
        t.hits = 1
        self._next_id += 1
        self._tracks.append(t)

    def _public(self) -> list[Track]:
        out = []
        for t in self._tracks:
            if not t.confirmed:
                continue
            tr = Track(
                id=t.id, x=t.x, y=t.y, radius=t.radius, vx=t.vx, vy=t.vy,
                cls=t.cls, bgr=t.bgr, age=t.age, hits=t.hits, misses=t.misses,
                active=(t.misses == 0), history=list(t.pos_hist),
            )
            out.append(tr)
        return out

    @property
    def tracks(self) -> list[Track]:
        return self._public()
