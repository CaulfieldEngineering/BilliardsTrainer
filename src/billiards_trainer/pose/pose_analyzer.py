"""MediaPipe-backed pose/body-fundamentals analysis.

Real when the optional ``[pose]`` extra (mediapipe) is installed; otherwise a
clean no-op that reports unavailability. Computes a few coaching-relevant
fundamentals from the player's pose:

  * head_down      — is the head low over the cue (chin near the line)?
  * stance_width   — feet/shoulder spread, normalised
  * stillness      — body motion between frames (lower is steadier)

This is a stretch feature and is not yet wired into the live UI (it wants a
second, player-height camera). The structure is here so it can be enabled later.
"""

import importlib.util
from dataclasses import dataclass, field

import numpy as np


def is_available() -> bool:
    return importlib.util.find_spec("mediapipe") is not None


@dataclass
class PoseResult:
    ok: bool = False
    landmarks: np.ndarray | None = None       # (33, 3) normalised x,y,visibility
    head_down: bool | None = None
    stance_width: float = 0.0
    stillness: float = 0.0
    notes: list[str] = field(default_factory=list)


class PoseAnalyzer:
    def __init__(self):
        if not is_available():
            raise RuntimeError(
                "MediaPipe not installed. Install the optional extra: pip install -e \".[pose]\""
            )
        import mediapipe as mp  # noqa: F401

        self._mp = mp
        self._pose = mp.solutions.pose.Pose(
            model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self._prev: np.ndarray | None = None

    def analyze(self, frame_bgr: np.ndarray) -> PoseResult:
        import cv2

        res = PoseResult()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out = self._pose.process(rgb)
        if not out.pose_landmarks:
            return res

        lm = np.array([[p.x, p.y, p.visibility] for p in out.pose_landmarks.landmark],
                      dtype=np.float32)
        res.landmarks = lm
        res.ok = True

        # MediaPipe Pose landmark indices
        NOSE, L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK = 0, 11, 12, 23, 24, 27, 28
        shoulders_y = (lm[L_SH, 1] + lm[R_SH, 1]) / 2
        hips_y = (lm[L_HIP, 1] + lm[R_HIP, 1]) / 2
        # head "down" if the nose has dropped toward shoulder/hip line (cue address)
        res.head_down = bool(lm[NOSE, 1] > shoulders_y - 0.02)
        # stance: ankle spread normalised by shoulder width
        sh_w = abs(lm[L_SH, 0] - lm[R_SH, 0]) + 1e-6
        ank_w = abs(lm[L_ANK, 0] - lm[R_ANK, 0])
        res.stance_width = float(ank_w / sh_w)

        if self._prev is not None:
            res.stillness = float(np.mean(np.abs(lm[:, :2] - self._prev[:, :2])))
        self._prev = lm

        if res.head_down is False:
            res.notes.append("Get your head down over the cue.")
        if res.stance_width < 0.8:
            res.notes.append("Widen your stance for a more stable base.")
        if res.stillness > 0.02:
            res.notes.append("Stay still through the shot.")
        if hips_y < shoulders_y:  # sanity / posture
            res.notes.append("Check posture — hips above shoulders.")
        return res

    def close(self) -> None:
        try:
            self._pose.close()
        except Exception:  # noqa: BLE001
            pass
