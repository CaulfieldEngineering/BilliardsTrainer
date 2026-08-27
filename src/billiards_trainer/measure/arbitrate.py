"""Video-truth arbitration for dense-vs-sparse trail disagreements.

The bootstrap agreement gate trusted sparse as the auditor — and the
autopsy proved it backwards at the worst divergence (shot@362: the ball
rests IN the pocket jaw, dense ends on it, sparse stops mid-felt). The
honest judge is the video: at trail-end the ball is at rest somewhere
visible (or pocketed). Whichever trail ends where reality is, wins.

Verdicts: "dense" | "sparse" | "unknown" (ambiguous keeps sparse — the
conservative default never regresses a replay).
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("measure.arbitrate")

BALL_TOL = 0.024      # normalized frame-widths ~ 1 ball radius
POCKET_TOL = 0.05


def endpoint_verdict(dense_end, sparse_end, ball_spots, pocket_spots,
                     was_pocketed: bool) -> str:
    """Pure verdict. ball_spots: normalized (x, y) of balls DETECTED on
    the trail-end frame; pocket_spots: normalized pocket mouths."""
    def near(p, spots, tol):
        return any(((p[0] - sx) ** 2 + (p[1] - sy) ** 2) ** 0.5 <= tol
                   for (sx, sy) in spots)
    d_ball = near(dense_end, ball_spots, BALL_TOL)
    s_ball = near(sparse_end, ball_spots, BALL_TOL)
    if was_pocketed:
        # a pocketed ball is at no ball spot; the true trail ENDS at a pocket
        d_pock = near(dense_end, pocket_spots, POCKET_TOL)
        s_pock = near(sparse_end, pocket_spots, POCKET_TOL)
        if d_pock and not s_pock:
            return "dense"
        if s_pock and not d_pock:
            return "sparse"
        return "unknown"
    if d_ball and not s_ball:
        return "dense"
    if s_ball and not d_ball:
        return "sparse"
    return "unknown"


class VideoArbiter:
    """Detects balls on trail-end frames (one finder pass per contested
    frame) and renders verdicts in normalized video space."""

    def __init__(self, video, calib, hinv, w, h):
        import cv2

        from ..detector_strategies import discover
        self._cap = cv2.VideoCapture(str(video))
        self._calib = calib
        self._hinv = np.asarray(hinv, dtype=float)
        self._w, self._h = float(w), float(h)
        strat = discover()["ensemble_findid"]
        strat.inference_provider = "dml"
        self._finder = strat._finder
        self._pipe = None           # set by caller for prepare_detections
        # pocket mouths in normalized video space
        self._pockets = []
        try:
            for p in calib.table.pockets:
                self._pockets.append(self._norm(p.x, p.y))
        except Exception:  # noqa: BLE001 - pockets optional for arbitration
            pass

    def _norm(self, x, y):
        v = self._hinv @ np.array([x, y, 1.0])
        return (v[0] / v[2] / self._w, v[1] / v[2] / self._h)

    def ball_spots(self, t_video: float) -> list:
        """Normalized positions of balls detected at t_video."""
        self._cap.set(0x0, 0)  # noop guard for stubs
        import cv2
        self._cap.set(cv2.CAP_PROP_POS_MSEC, t_video * 1000)
        ok, frame = self._cap.read()
        if not ok:
            return []
        found = self._finder.detect(frame, self._calib) or []
        if self._pipe is not None:
            found = self._pipe.prepare_detections(found, self._calib,
                                                  frame.shape, frame=frame)
        return [self._norm(d.x, d.y) for d in found]

    def verdict(self, t_end_video: float, dense_end, sparse_end,
                was_pocketed: bool) -> str:
        spots = self.ball_spots(t_end_video + 0.4)
        return endpoint_verdict(dense_end, sparse_end, spots,
                                self._pockets, was_pocketed)

    def close(self):
        try:
            self._cap.release()
        except Exception:  # noqa: BLE001
            pass
