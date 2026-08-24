"""One-line stroke summary shared by every per-shot surface.

Reads the camera-measured stroke record from either the live key
(``stroke``, set by on_stroke_measured / shots.json) or the sidecar
attach key (``_stroke``). Returns None when there is nothing to show.
"""


def stroke_text(shot: dict) -> str | None:
    sv = shot.get("stroke") or shot.get("_stroke") or {}
    if not sv or sv.get("confidence") == "none":
        return None
    bits = []
    if sv.get("stay_down_s") is not None:
        bits.append(f"stay-down {float(sv['stay_down_s']):.1f}s")
    if sv.get("popped_early"):
        bits.append("POPPED UP EARLY")
    if sv.get("pause_ms") is not None:
        bits.append(f"pause {int(sv['pause_ms'])}ms")
    if sv.get("back_depth_px") is not None:
        bits.append(f"back {round(float(sv['back_depth_px']))}px")
    if sv.get("practice_strokes") is not None:
        bits.append(f"{int(sv['practice_strokes'])} practice")
    return " · ".join(bits) if bits else None
