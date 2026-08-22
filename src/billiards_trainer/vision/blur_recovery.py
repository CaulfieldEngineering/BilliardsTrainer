"""Find balls the ball-finder could not see because they were MOVING.

Joe: "a remedy is mandatory, we can't move forward if we can't get accuracy
on this 4. This shot will be very common." He is right — it is a medium
8ft/s cut, not a break.

WHY THE DETECTOR FAILS. Measured on session-20260820-005048 @233: at the
~1/75s shutter that footage was shot at, the ball smears ~17px past its own
28px diameter. The smear is not even the main problem — CONTRAST collapses,
because each pixel only holds the ball for part of the exposure and averages
the rest to felt. The model is trained on solid discs, so it returned
nothing at all for half a second.

The pixels still hold the ball. Two things make it recoverable:

MEDIAN BACKGROUND, not frame differencing. A frame difference lights up
everything that moved, and the thing that moves most beside a struck cue
ball is Joe's cue stick withdrawing through the same space — an earlier
attempt adopted the stick and drifted the track the wrong way. A median over
the last ~1.5s is the static scene (felt, rails, chalk), so subtracting it
isolates what is there NOW without ranking the stick above the ball.

RELATIVE colour, not a threshold. Blur washes a ball toward felt, so "within
X of purple" fails exactly when it is needed. But the ball still resembles
ITS OWN measured colour more than any other ball's, and every track carries
that reference. So the test is a contest, not a bar. Measured on the
decisive frame: the ball scored 51, felt blobs 234 and 254, and the stick
was rejected on size.

Nothing here NAMES a ball. A recovered blob is handed over as an unnumbered
detection carrying only the id of the track it was found for, so identity
still comes from track continuity — this can keep a ball attached, never
rename one.
"""

import math
import os
from collections import deque

import cv2
import numpy as np

from ..core.types import BallClass, Detection

#: frames of history behind the median background (~1.5s at the detect cadence)
_BUF = 15
#: how many frames of history before the median is trustworthy
_MIN_BUF = 6
#: a track is a candidate while its detection has been missing this long. A
#: struck ball can be gone the better part of a second (measured: 0.5s), so
#: the window stays open — the colour contest is what keeps that safe.
_MAX_MISSES = 12
#: per-channel BGR difference that counts as "this pixel changed"
_MOVED = 18
#: the biggest a smeared BALL gets, in expected radii. A cue stick or a
#: forearm sweeping through is far larger and is rejected here.
_MAX_BLOB_R = 3.2


def track_colour(t) -> np.ndarray:
    """A track's own measured colour: the per-channel median of its recent
    sampled readings, so one glare frame cannot move it."""
    n = len(t.mbgr_hist)
    return np.array([sorted(c[i] for c in t.mbgr_hist)[n // 2]
                     for i in range(3)], dtype=float)


class BlurRecovery:
    """Holds the rolling frame history the median background is built from."""

    def __init__(self) -> None:
        self._buf: deque = deque(maxlen=_BUF)

    def find(self, frame, calib, tracker, detections) -> list:
        """Detections for tracks whose ball the finder lost to blur.

        Returns RAW-frame detections; the caller projects them. Each carries
        ``recovered_for`` — the track it belongs to — because association
        must not re-judge it on distance: a struck ball is far outside any
        gate by the time it is found again, which is the whole reason it
        needed recovering.
        """
        dbg = os.environ.get("RECOV_DEBUG")

        def say(*a):
            if dbg:
                print("   [recov]", *a, flush=True)

        from .tracking import BallTracker

        tracks = getattr(tracker, "_tracks", [])
        # Gate on "its detection just VANISHED". A struck ball was at REST one
        # frame earlier, so anything keyed on move_streak excludes the only
        # case this exists for (measured: it never once fired for the cue).
        lost = [t for t in tracks if t.confirmed and 1 <= t.misses <= _MAX_MISSES
                and len(t.mbgr_hist) >= 5]
        self._buf.append(frame)
        say(f"called: buf={len(self._buf)} lost="
            f"{[(t.id, t.misses, t.committed_number) for t in lost]}")
        if len(self._buf) < _MIN_BUF or not lost:
            return []
        try:
            hinv = np.linalg.inv(np.asarray(calib.H, dtype=float))
        except Exception:  # noqa: BLE001 - recovery is best-effort, never fatal
            return []

        h, w = frame.shape[:2]
        r_raw = max(4.0, 0.5 * (h + w) / 90.0)
        short = calib.table.short_side
        gate = max(8.0, tracker.max_dist_frac * short)
        # every OTHER live track's colour, for the contest
        rivals = {o.id: track_colour(o) for o in tracks if len(o.mbgr_hist) >= 5}

        out = []
        for t in lost:
            speed = float(np.hypot(t.vx, t.vy))
            px, py = t.x + t.vx, t.y + t.vy
            # Coverage means a detection this track could actually MATCH. One
            # its own colour veto has refused is NOT coverage — that bug made
            # recovery skip the search at exactly the moment the ball was
            # closest and most findable, because the (rejected) cue ball
            # happened to be 26px away.
            covered = [(d.x, d.y) for d in detections
                       if not BallTracker._colour_contradicts(t, d)]
            if any((px - cx) ** 2 + (py - cy) ** 2 <= max(gate, 1.2 * speed) ** 2
                   for cx, cy in covered):
                say(f"id{t.id}: skipped, detector already covers it")
                continue
            v = hinv @ np.array([px, py, 1.0])
            if abs(v[2]) < 1e-9:
                continue
            sx, sy = float(v[0] / v[2]), float(v[1] / v[2])
            # A ball that vanished FROM REST has no velocity to scale a window
            # by, and its prediction stays parked where it left — so the gap
            # between prediction and ball GROWS every frame it stays missing.
            # The window has to grow with it. (+2 because misses counts frames
            # since the track went unmatched, which lags the moment the ball
            # actually left: when misses first read 1 the ball was already
            # 292px away and a 190px window missed it.)
            per_frame = max(gate, 1.3 * speed, 0.30 * short)
            reach = per_frame * (max(1, t.misses) + 2)
            scale = r_raw / max(6.0, getattr(tracker, "_ball_r", 0) or 12.0)
            win = int(np.clip(reach * scale, 3 * r_raw, 520))
            x0, y0 = int(max(0, sx - win)), int(max(0, sy - win))
            x1, y1 = int(min(w, sx + win)), int(min(h, sy + win))
            say(f"id{t.id} misses={t.misses} seed=({sx:.0f},{sy:.0f}) win={win}")
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue

            stack = np.stack([f[y0:y1, x0:x1] for f in self._buf]).astype(np.float32)
            bg = np.median(stack, axis=0)
            cur = frame[y0:y1, x0:x1]
            moved = np.linalg.norm(cur.astype(np.float32) - bg, axis=2)
            mask = (moved > _MOVED).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            mine = track_colour(t)
            # What the table looks like here. The window is mostly cloth, so
            # its background median IS the felt — and a ball, by definition,
            # is not felt-coloured. This is what rejects the holes: a ball
            # that leaves registers as motion at the spot it vacated as well
            # as where it went, and those vacancies are bare table.
            felt = np.median(bg.reshape(-1, 3), axis=0)
            best, best_d = None, 1e18
            for c in cnts:
                a = cv2.contourArea(c)
                if a < 0.25 * math.pi * r_raw * r_raw:
                    continue
                (_bx, _by), enc_r = cv2.minEnclosingCircle(c)
                if enc_r > _MAX_BLOB_R * r_raw:
                    say(f"   blob r={enc_r:.0f} too big — stick or arm")
                    continue
                m = cv2.moments(c)
                if m["m00"] <= 0:
                    continue
                cm = np.zeros(mask.shape, np.uint8)
                cv2.drawContours(cm, [c], -1, 255, -1)
                mean = np.array(cv2.mean(cur, mask=cm)[:3], dtype=float)
                d_mine = float(np.linalg.norm(mean - mine))
                # NOT THE HOLE IT LEFT. A departed ball registers as motion
                # twice: once where it now is, and once at the spot it
                # vacated — where the background still remembers it and the
                # current pixels are bare felt. Both blobs can win a colour
                # contest against the other balls, and the vacancy sits
                # exactly under the parked track, so adopting it pins the
                # track to its own absence. Tell them apart physically: at
                # the ball, the CURRENT pixels look like the ball; at the
                # hole, the BACKGROUND does.
                was = np.array(cv2.mean(bg.astype(np.uint8), mask=cm)[:3],
                               dtype=float)
                if float(np.linalg.norm(was - mine)) < d_mine:
                    say(f"   blob a={a:.0f} is the hole THIS ball left")
                    continue
                if float(np.linalg.norm(mean - felt)) < d_mine:
                    say(f"   blob a={a:.0f} is bare table, not a ball")
                    continue
                # THE CONTEST: it must look more like this track's ball than
                # like anybody else's. The stick and the cue ball lose here.
                rival = min((float(np.linalg.norm(mean - r))
                             for k, r in rivals.items() if k != t.id),
                            default=1e9)
                say(f"   blob a={a:.0f} col={tuple(int(q) for q in mean)} "
                    f"mine={d_mine:.0f} rival={rival:.0f} "
                    f"{'LOSES' if rival < d_mine else 'WINS'}")
                if rival < d_mine:
                    continue
                if d_mine < best_d:
                    best, best_d = (m["m10"] / m["m00"] + x0,
                                    m["m01"] / m["m00"] + y0, mean), d_mine
            say(f"id{t.id}: " + (f"RECOVERED at ({best[0]:.0f},{best[1]:.0f})"
                                 if best else "nothing accepted"))
            if best is not None:
                rec = Detection(x=float(best[0]), y=float(best[1]),
                                radius=float(r_raw),
                                bgr=tuple(int(q) for q in best[2]),
                                cls=BallClass.UNKNOWN, score=0.30, number=-1)
                rec.recovered_for = t.id
                out.append(rec)
        return out
