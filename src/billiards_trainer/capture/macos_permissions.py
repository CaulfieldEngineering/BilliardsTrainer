"""macOS camera-permission (TCC) handling.

OpenCV's AVFoundation backend requests camera access from whatever thread first
opens a device — here the capture GRAB thread. macOS only shows the permission
prompt from the main thread while an app event loop is running, so from a worker
thread (or before the GUI loop starts) the request is silently denied and the
camera enumerates as "none" with no prompt. The fix: on the main thread, once the
Qt event loop is live, ask AVFoundation for access (which shows the real dialog),
then re-scan when it's granted.

No-op on non-macOS / when the AVFoundation binding is absent (graceful degrade)."""

from __future__ import annotations

import logging
import sys

log = logging.getLogger("capture.permissions")

# AVAuthorizationStatus
NOT_DETERMINED = 0
RESTRICTED = 1
DENIED = 2
AUTHORIZED = 3


def _av():
    import AVFoundation
    return AVFoundation


def camera_auth_status() -> int:
    """Current camera authorization status (AUTHORIZED on non-mac/no binding)."""
    if sys.platform != "darwin":
        return AUTHORIZED
    try:
        av = _av()
        return int(av.AVCaptureDevice.authorizationStatusForMediaType_(
            av.AVMediaTypeVideo))
    except Exception:  # noqa: BLE001 - binding missing
        return AUTHORIZED


def microphone_auth_status() -> int:
    """Current mic authorization status (AUTHORIZED on non-mac/no binding)."""
    if sys.platform != "darwin":
        return AUTHORIZED
    try:
        av = _av()
        return int(av.AVCaptureDevice.authorizationStatusForMediaType_(
            av.AVMediaTypeAudio))
    except Exception:  # noqa: BLE001 - binding missing
        return AUTHORIZED


def request_microphone_access(on_result=None) -> None:
    """Fire the macOS microphone prompt (MAIN THREAD, event loop running).
    Needed for audio in session recordings — the ffmpeg child inherits this
    app's TCC grant."""
    if sys.platform != "darwin":
        if on_result:
            on_result(True)
        return
    try:
        av = _av()
    except Exception:  # noqa: BLE001
        if on_result:
            on_result(True)
        return

    def _handler(granted) -> None:
        log.info("microphone permission %s", "granted" if granted else "refused")
        if on_result:
            try:
                on_result(bool(granted))
            except Exception:  # noqa: BLE001
                pass

    av.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
        av.AVMediaTypeAudio, _handler)


def request_camera_access(on_result=None) -> None:
    """Fire the macOS camera prompt (MAIN THREAD, event loop must be running).

    ``on_result(granted: bool)`` is invoked from an arbitrary thread when the
    user answers — keep it thread-safe (e.g. flip a flag a QTimer polls)."""
    if sys.platform != "darwin":
        if on_result:
            on_result(True)
        return
    try:
        av = _av()
    except Exception:  # noqa: BLE001
        if on_result:
            on_result(True)
        return

    def _handler(granted) -> None:
        log.info("camera permission %s", "granted" if granted else "refused")
        if on_result:
            try:
                on_result(bool(granted))
            except Exception:  # noqa: BLE001
                pass

    av.AVCaptureDevice.requestAccessForMediaType_completionHandler_(
        av.AVMediaTypeVideo, _handler)
