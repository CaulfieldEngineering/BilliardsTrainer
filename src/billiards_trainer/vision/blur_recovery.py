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
#: frames a detection must have been missing before the search is worth its
#: cost. A BLINK is not a loss: a resting ball whose detection drops for a
#: single frame is back on the same spot next frame, and coasting held its
#: position exactly in the meantime, so hunting it can only burn time —
#: measured, a resting "4" on the bench flickered into the candidate set 150
#: times in 900 frames and was never once actually missing. A ball that was
#: truly struck and lost stays gone the better part of a second (~15 frames),
#: so one frame of patience costs it nothing.
#:
#: NOT a velocity gate: that was tried first and is WRONG. A ball struck from
#: rest and lost on the very first frame of motion has no measured velocity
#: yet, and that is the exact case this module was built for (its own
#: docstring says so, and tests/test_blur_recovery.py pins it).
_MIN_MISSES = 2



def _missed(t) -> int:
    """Consecutive frames with no detection for this track.

    The surviving tracker keeps that as `miss_frames`, because its
    `misses` counts SECONDS (coasting is time-based). Trackers that
    count frames in `misses` still work - this reads whichever the
    tracker offers rather than forcing one spelling on both."""
    v = getattr(t, "miss_frames", None)
    return int(v if v is not None else getattr(t, "misses", 0))
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

    def sweep(self, frame, calib, tracker, detections) -> list:
        """PRESENCE channel: ball-sized MOVING blobs anywhere on the table
        that the model produced no detection for. The fusion design's
        always-on subtraction pass (FUSION.md item 3), promoted from
        lost-track-only recovery at Joe's direction: "rerun all of the
        footage via subtraction". Emits UNNUMBERED low-score detections —
        identity still comes from tracking continuity and the colour
        contest, never from here.
        """
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // 2, h // 2),
                           interpolation=cv2.INTER_AREA)
        buf = getattr(self, "_small", None)
        if buf is None:
            buf = self._small = deque(maxlen=_BUF)
        buf.append(small)
        if len(buf) < _MIN_BUF:
            return []
        self._sweep_n = getattr(self, "_sweep_n", 0) + 1
        if self._sweep_n % 4 == 1 or getattr(self, "_bg_small", None) is None:
            self._bg_small = np.median(
                np.stack(buf).astype(np.float32), axis=0)
        moved = np.linalg.norm(small.astype(np.float32) - self._bg_small,
                               axis=2)
        mask = (moved > _MOVED).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        r_raw = max(4.0, 0.5 * (h + w) / 90.0)
        r_s = r_raw / 2.0                       # half-res radius
        felt = np.median(self._bg_small.reshape(-1, 3), axis=0)
        out = []
        for c in cnts:
            a = cv2.contourArea(c)
            if not (0.3 * math.pi * r_s * r_s <= a <= 12 * math.pi * r_s * r_s):
                continue
            (_bx, _by), enc_r = cv2.minEnclosingCircle(c)
            if enc_r > _MAX_BLOB_R * r_s:
                continue                        # stick / forearm streak
            m = cv2.moments(c)
            if m["m00"] <= 0:
                continue
            x, y = m["m10"] / m["m00"] * 2, m["m01"] / m["m00"] * 2
            if any((x - d.x) ** 2 + (y - d.y) ** 2 <= (2.0 * r_raw) ** 2
                   for d in detections):
                continue                        # the model already has it
            cm = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(cm, [c], -1, 255, -1)
            mean = np.array(cv2.mean(small, mask=cm)[:3], dtype=float)
            was = np.array(cv2.mean(self._bg_small.astype(np.uint8),
                                    mask=cm)[:3], dtype=float)
            # a VACANCY looks like felt now and like a ball before; a BALL
            # is the reverse — same physics as _locate's hole guard
            if (np.linalg.norm(mean - felt)
                    < np.linalg.norm(was - felt)):
                continue
            rec = Detection(x=float(x), y=float(y), radius=float(r_raw),
                            bgr=tuple(int(q) for q in mean),
                            cls=BallClass.UNKNOWN, score=0.25, number=-1)
            out.append(rec)
            if len(out) >= 4:
                break                           # phantom-churn bound
        return out

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


        tracks = getattr(tracker, "_tracks", [])
        if isinstance(tracks, dict):
            tracks = list(tracks.values())
        # This reads the tracker's own per-track state. The guard makes
        # an unfamiliar tracker stand down rather than raise once per
        # frame behind the pipeline's catch-all.
        if not all(hasattr(t, "confirmed") and hasattr(t, "mbgr_hist")
                   for t in tracks):
            return []
        # Gate on "its detection just VANISHED". A struck ball was at REST one
        # frame earlier, so anything keyed on move_streak excludes the only
        # case this exists for (measured: it never once fired for the cue).
        #
        # AND ON "IT IS A BALL WE CAN NAME" (round 48). This search is
        # expensive - measured at 287ms per candidate - and the whole
        # point of it is IDENTITY: keeping a known ball attached to its
        # track through a smear. A track with no number has no identity
        # to preserve, so recovering it attaches nothing. When measured
        # colour finally reached the offline tracker and this code could
        # run at all, it took 66% of total engine wall time and 87% of
        # its work was hunting NAMELESS blobs: over 900 bench frames,
        # 271 of 311 searches were for unnamed tracks (one phantom alone
        # accounted for 244), against 14 for a real named ball - and it
        # recovered nothing on the whole clip. Chasing phantoms is not
        # free; it was tripling the cost of every clip in the library.
        lost = [t for t in tracks if t.confirmed
                and t.committed_number >= 0
                and _MIN_MISSES <= _missed(t) <= _MAX_MISSES
                and len(t.mbgr_hist) >= 5]
        self._buf.append(frame)
        say(f"called: buf={len(self._buf)} lost="
            f"{[(t.id, _missed(t), t.committed_number) for t in lost]}")
        if len(self._buf) < _MIN_BUF or not lost:
            return []
        try:
            hinv = np.linalg.inv(np.asarray(calib.H, dtype=float))
        except Exception:  # noqa: BLE001 - recovery is best-effort, never fatal
            return []

        h, w = frame.shape[:2]
        r_raw = max(4.0, 0.5 * (h + w) / 90.0)
        short = calib.table.short_side
        # Search radius for "is this blob plausibly that track's ball". The
        # 0.16 came from BallTracker's 2026-07-02 autotune sweep and is
        # kept as the default now that the surviving tracker gates by ball
        # radii instead of a table fraction - this is a SEARCH window, not
        # an association rule, so it does not have to match the tracker's.
        gate = max(8.0, getattr(tracker, "max_dist_frac", 0.16) * short)
        ball_r = getattr(tracker, "_ball_r", 0)
        # every OTHER live track's colour, for the contest
        rivals = {o.id: track_colour(o) for o in tracks if len(o.mbgr_hist) >= 5}

        out = []
        for t in lost:
            got = self._locate(t, frame, detections, hinv, (w, h), r_raw,
                               short, gate, ball_r, rivals, say)
            if got is not None:
                out.append(got)
        return out

    def _locate(self, t, frame, detections, hinv, dims, r_raw, short, gate,
                ball_r, rivals, say):
        """Hunt one lost track's ball in this frame. Returns a RAW-frame
        Detection stamped with the track's id, or None."""
        from .tracking import BallTracker

        w, h = dims
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
            return None
        v = hinv @ np.array([px, py, 1.0])
        if abs(v[2]) < 1e-9:
            return None
        sx, sy = float(v[0] / v[2]), float(v[1] / v[2])
        # A ball that vanished FROM REST has no velocity to scale a window
        # by, and its prediction stays parked where it left — so the gap
        # between prediction and ball GROWS every frame it stays missing.
        # The window has to grow with it. (+2 because misses counts frames
        # since the track went unmatched, which lags the moment the ball
        # actually left: when misses first read 1 the ball was already
        # 292px away and a 190px window missed it.)
        per_frame = max(gate, 1.3 * speed, 0.30 * short)
        reach = per_frame * (max(1, _missed(t)) + 2)
        scale = r_raw / max(6.0, ball_r or 12.0)
        win = int(np.clip(reach * scale, 3 * r_raw, 520))
        x0, y0 = int(max(0, sx - win)), int(max(0, sy - win))
        x1, y1 = int(min(w, sx + win)), int(min(h, sy + win))
        say(f"id{t.id} misses={_missed(t)} seed=({sx:.0f},{sy:.0f}) win={win}")
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None

        # CACHED BACKGROUND (round 48). `win` clips to 520, so this crop
        # is routinely ~1040x1040 and the median ran over 15 of them -
        # ~195MB of float32 - from scratch for every candidate on every
        # frame. Measured at 287ms per search, which was 66% of total
        # engine wall time once recovery could finally run at all. The
        # felt does not change in a tenth of a second, and sweep() has
        # always refreshed its own background every 4th call for exactly
        # this reason; _locate simply never did. Same window, same
        # buffer depth -> reuse for 4 calls.
        key = (x0, y0, x1, y1)
        self._bg_n = getattr(self, "_bg_n", 0) + 1
        cached = getattr(self, "_bg_cache", None)
        if cached is not None and cached[0] == key and self._bg_n - cached[1] < 4:
            bg = cached[2]
        else:
            stack = np.stack([f[y0:y1, x0:x1]
                              for f in self._buf]).astype(np.float32)
            bg = np.median(stack, axis=0)
            self._bg_cache = (key, self._bg_n, bg)
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
        if best is None:
            return None
        rec = Detection(x=float(best[0]), y=float(best[1]),
                        radius=float(r_raw),
                        bgr=tuple(int(q) for q in best[2]),
                        cls=BallClass.UNKNOWN, score=0.30, number=-1)
        rec.recovered_for = t.id
        return rec
