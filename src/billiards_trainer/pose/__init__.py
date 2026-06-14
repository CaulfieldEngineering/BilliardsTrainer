"""Optional body-fundamentals analysis (Phase 8, stretch) via MediaPipe Pose.

Lazy and optional: importing this package never requires mediapipe. Use
``is_available()`` to check, and ``PoseAnalyzer`` only when the ``[pose]`` extra
is installed. Best used with a second camera at player height — see docs/BLOCKERS.
"""

from .pose_analyzer import PoseAnalyzer, PoseResult, is_available

__all__ = ["PoseAnalyzer", "PoseResult", "is_available"]
