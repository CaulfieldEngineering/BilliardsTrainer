"""Live streaming cue-ball impact detector + per-shot stroke metrics.

Port of the reference app's live-detector flow (gui._det_feed_accel /
_det_eval / _compute_live_stats), which was verified to produce identical
output to the offline ``analysis.detect_impacts`` path: candidates are |a|
peaks above a floor; each is evaluated with the video-validated
``analysis.evaluate_impact`` signature once its ±0.4 s window has streamed in.

Kinematic metrics need the follow-through, so the caller computes them
``METRICS_DELAY_S`` after a confirmed hit via :meth:`compute_metrics` — a
local-window fusion (~9 s of samples, cheap pure Python).

Timestamps are whatever the caller feeds (the worker feeds wall-clock epoch
seconds so hits can be joined with the vision pipeline's shot events); only
time DIFFERENCES matter to the math.

Pure Python + stdlib: testable without Qt, bleak, or a device.
"""

import bisect
import math
from collections import deque

from . import analysis

DET_WIN = 0.4          # ± evaluation window around a candidate (s)
DET_REFRACTORY = 1.2   # min spacing between hits (s)
METRICS_DELAY_S = 2.6  # kinematics need the follow-through to have streamed in
HISTORY_S = 12.0       # rolling sample history (metrics read [hit-6.5, hit+2.6])
CUE_AXIS = (0.0, 0.0, 1.0)  # butt→tip = +Z in the sensor body frame (mounting truth)


class LiveImpactDetector:
    """Feed decoded samples; get confirmed impacts (and, later, full metrics)."""

    def __init__(self, impact_g: float = 1.6):
        self.impact_g = float(impact_g)
        self.accel: deque = deque()   # {"t","x","y","z"} history, ~HISTORY_S
        self.gyro: deque = deque()
        self._gmags: deque = deque()  # (t, |gyro|) for the impact gate
        self._pend: tuple | None = None   # (t_peak, mag) awaiting its window
        self._last_hit = -1e9

    def reset(self) -> None:
        self.accel.clear()
        self.gyro.clear()
        self._gmags.clear()
        self._pend = None
        self._last_hit = -1e9

    def feed(self, kind: str, d: dict, t: float) -> dict | None:
        """Feed one decoded sample. Returns a confirmed-impact dict
        ({t0, t1, dur, hit_t, peak_g, peak_rate, rot_deg}) or None."""
        rec = {"t": t, "x": d["x"], "y": d["y"], "z": d["z"]}
        if kind == "gyro":
            self.gyro.append(rec)
            self._gmags.append((t, math.sqrt(d["x"] ** 2 + d["y"] ** 2 + d["z"] ** 2)))
            self._trim(t)
            return None
        self.accel.append(rec)
        self._trim(t)
        mag = math.sqrt(d["x"] ** 2 + d["y"] ** 2 + d["z"] ** 2)
        if self._pend is None:
            if mag >= self.impact_g and t - self._last_hit > DET_REFRACTORY:
                self._pend = (t, mag)
            return None
        pt, pm = self._pend
        if mag > pm and t - pt < 0.25:
            self._pend = (t, mag)          # track the true local peak
        elif t >= pt + DET_WIN + 0.05:
            return self._evaluate()
        return None

    def flush(self) -> dict | None:
        """Evaluate a still-pending candidate (e.g. the stream is stopping)."""
        return self._evaluate() if self._pend is not None else None

    def _trim(self, t: float) -> None:
        for dq in (self.accel, self.gyro):
            while dq and dq[0]["t"] < t - HISTORY_S:
                dq.popleft()
        while self._gmags and self._gmags[0][0] < t - HISTORY_S:
            self._gmags.popleft()

    def _evaluate(self) -> dict | None:
        (pt, pm), self._pend = self._pend, None
        aw = list(self.accel)
        if len(aw) < 8:
            return None
        i = min(range(len(aw)), key=lambda j: abs(aw[j]["t"] - pt))
        ok, feats = analysis.evaluate_impact(aw, list(self._gmags), i,
                                             mag_floor=self.impact_g, win=DET_WIN)
        if not ok:
            return None
        near = [s for s in aw if abs(s["t"] - pt) < 0.25
                and math.sqrt(s["x"] ** 2 + s["y"] ** 2 + s["z"] ** 2) > self.impact_g]
        t0 = near[0]["t"] if near else pt
        t1 = near[-1]["t"] if near else pt
        rot_deg, peak_rate, last = 0.0, 0.0, None
        for g in self.gyro:
            if g["t"] < t0 or g["t"] > t1:
                continue
            rate = math.sqrt(g["x"] ** 2 + g["y"] ** 2 + g["z"] ** 2)
            peak_rate = max(peak_rate, rate)
            if last is not None:
                rot_deg += rate * (g["t"] - last)
            last = g["t"]
        self._last_hit = pt
        return {"t0": t0, "t1": t1, "dur": t1 - t0, "hit_t": pt, "peak_g": pm,
                "peak_rate": peak_rate, "rot_deg": rot_deg, "feats": feats}

    def compute_metrics(self, stroke: dict, prev_hit_t: float | None = None) -> dict | None:
        """Full per-shot metrics for a confirmed impact, from a local window
        around it. Call ~METRICS_DELAY_S after ``stroke['hit_t']`` so the
        follow-through has streamed in; earlier calls just yield None fields.

        Returns the analysis.shot_metrics dict ({v_impact, stroke_len,
        back_depth, pause, steer_side, steer_deg, yaw_swing, steer_ratio,
        finish, interval}) or None if the reconstruction failed.
        """
        hit = stroke["hit_t"]
        ta = [a["t"] for a in self.accel]
        tg = [g["t"] for g in self.gyro]
        aw = list(self.accel)[bisect.bisect_left(ta, hit - 6.5):]
        gw = list(self.gyro)[bisect.bisect_left(tg, hit - 6.5):]
        try:
            ot = analysis.fuse_track(aw, gw)
            kin, info = analysis.shot_kinematics(aw, ot, CUE_AXIS, [stroke],
                                                 pre=4.0, post=2.0)
            _, fg = analysis.stroke_frame(aw, gw, ot, CUE_AXIS)
            ang = analysis.shot_angles(fg, [stroke], pre=4.0, post=2.0)
            st = analysis.shot_metrics(aw, [stroke], kin, info, ang=ang)
            m = st[0] if st else None
        except Exception:  # noqa: BLE001 - metrics are best-effort, never fatal
            m = None
        if m is not None and prev_hit_t is not None:
            m["interval"] = hit - prev_hit_t
        return m
