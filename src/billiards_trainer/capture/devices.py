"""Camera enumeration + stable resolution.

Listing cameras by *friendly name* (so the user picks "Logitech BRIO", not "1")
is the whole point. On Windows we use DirectShow via pygrabber, whose device
order matches OpenCV's ``CAP_DSHOW`` indices. Everywhere else (and if pygrabber
is unavailable in a frozen build) we fall back to probing indices with OpenCV.

Windows reorders cameras when you plug/unplug USB devices, so we also persist the
friendly name and ``resolve_camera`` re-finds the device by name even if its
index shifted.
"""

import logging
import sys
from dataclasses import dataclass

import cv2

log = logging.getLogger("capture.devices")


@dataclass(frozen=True)
class CameraInfo:
    index: int
    name: str

    def label(self) -> str:
        return f"{self.name} ({self.index})"


def _windows_cameras() -> list[CameraInfo]:
    from pygrabber.dshow_graph import FilterGraph

    names = FilterGraph().get_input_devices()
    return [CameraInfo(i, n or f"Camera {i}") for i, n in enumerate(names)]


def _probe_cameras(max_index: int = 5) -> list[CameraInfo]:
    """Fallback: open indices and keep the ones that deliver a frame. Slow and
    can conflict with a camera already in use, so it's bounded + last-resort."""
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    found: list[CameraInfo] = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, backend)
        try:
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    found.append(CameraInfo(i, f"Camera {i}"))
        finally:
            cap.release()
    return found


def _macos_cameras() -> list[CameraInfo]:
    """AVFoundation enumeration: real device names WITHOUT opening anything.

    Probing (cv2.VideoCapture per index) actively starts each device — which
    wakes a paired iPhone via Continuity Camera every launch. Enumeration is
    passive. The index mirrors AVFoundation's device order, which is what
    OpenCV's AVFoundation backend uses. Continuity (iPhone/iPad) cameras are
    excluded outright: the overhead rig must never grab the phone."""
    import AVFoundation as AV

    devs = list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo))
    cams: list[CameraInfo] = []
    for i, d in enumerate(devs):
        try:
            name = str(d.localizedName())
            dtype = str(d.deviceType())
        except Exception:  # noqa: BLE001
            name, dtype = f"Camera {i}", ""
        if "ContinuityCamera" in dtype or "iPhone" in name or "iPad" in name:
            continue
        cams.append(CameraInfo(i, name or f"Camera {i}"))
    return cams


def list_cameras() -> list[CameraInfo]:
    """Enumerate connected cameras with friendly names where possible."""
    cams: list[CameraInfo] = []
    if sys.platform == "win32":
        try:
            cams = _windows_cameras()
        except Exception as exc:  # noqa: BLE001 - pygrabber/comtypes can fail; probe instead
            log.warning("pygrabber enumeration failed (%s); probing instead", exc)
    elif sys.platform == "darwin":
        try:
            cams = _macos_cameras()
        except Exception as exc:  # noqa: BLE001 - pyobjc binding absent
            log.warning("AVFoundation enumeration failed (%s); probing instead", exc)
    if not cams:
        cams = _probe_cameras()
    log.info("Cameras detected: %s", [c.label() for c in cams] or "none")
    return cams


def resolve_camera(index: int, name: str = "") -> tuple[int, str, str]:
    """Pick the right camera index, surviving index reshuffles.

    Returns ``(index, name, warning)``. Matches the saved friendly name first
    (so a re-ordered USB cam is still found), then the saved index, then falls
    back to the first available camera.
    """
    cams = list_cameras()
    if not cams:
        return index, name, "No cameras detected. Plug one in and press Refresh."
    if name:
        for c in cams:
            if c.name == name:
                if c.index != index:
                    return c.index, c.name, f"'{name}' is now camera {c.index} (was {index})."
                return c.index, c.name, ""
    # name didn't match (or none saved) — fall back to the saved index if valid
    for c in cams:
        if c.index == index:
            if name and c.name != name:
                return c.index, c.name, f"'{name}' not found — using camera {index} ({c.name})."
            return c.index, c.name, ""
    first = cams[0]
    return first.index, first.name, f"Camera {index} not found — using {first.name}."
