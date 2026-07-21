"""Tethered Canon DSLR capture over USB — native PTP, no external binaries.

A Canon EOS 600D (T3i) has no UVC/webcam mode, so OpenCV cannot open it. This
module streams its liveview over USB using our own PTP driver (``eos_ptp``,
pyusb/libusb) — the same approach commercial tether apps use, because on macOS
the ``gphoto2`` CLI loses an unwinnable fight against Apple's ``ptpcamerad``
daemon (validated at length on real hardware; see eos_ptp's module docstring).

Fully optional and graceful: no pyusb/libusb or no camera attached just means
``opened == False`` and the app falls back to the normal camera path — the same
degrade contract as the Bluetooth cue sensor.

Liveview on the T3i is ~960x640 regardless of the sensor's 18MP still size.
Focus policy for a fixed overhead rig: drive AF once at connect, then lock.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time

import cv2
import numpy as np

from ..config import TetherSettings
from .camera import FrameSource

log = logging.getLogger("capture.tether")


def gphoto2_available() -> bool:
    """Whether the tether transport is usable (name kept for callers; the
    implementation is now native PTP — no gphoto2 binary involved)."""
    from .eos_ptp import usb_available
    return usb_available()


def list_tethered_cameras() -> list[str]:
    """Product names of connected Canon bodies (empty if none/no libusb)."""
    from .eos_ptp import list_eos_cameras
    return list_eos_cameras()


# --------------------------------------------------------------------------- #
# The app-lifetime camera session.
#
# Live-validated firmware law (T3i): the body accepts exactly ONE PTP session
# per power-on — even a cleanly closed session doesn't earn a second one. So
# the app connects once and HOLDS that session for its entire run; every
# control action flows through it. Also validated live: the HDMI feed keeps
# streaming (~25fps out the capture dongle) while this session is held, even
# during autofocus — so holding it costs nothing.
# --------------------------------------------------------------------------- #
_session_lock = threading.Lock()
_session_cam = None
_session_atexit_registered = False


def _get_session():
    """Return the connected app-lifetime EOSCamera, establishing it if needed."""
    global _session_cam, _session_atexit_registered
    from .eos_ptp import EOSCamera
    with _session_lock:
        if _session_cam is not None and _session_cam.connected:
            return _session_cam
        cam = EOSCamera()
        cam.connect()  # tight-claim; raises with the power-cycle message if deaf
        _session_cam = cam
        if not _session_atexit_registered:
            import atexit
            # Leave liveview running at exit so the dongle feed survives the
            # app; the camera needs a power cycle for the next session anyway.
            atexit.register(lambda: _close_session_quietly())
            _session_atexit_registered = True
        return cam


def _close_session_quietly() -> None:
    global _session_cam
    cam, _session_cam = _session_cam, None
    if cam is not None:
        try:
            cam.close(leave_liveview=True)
        except Exception:  # noqa: BLE001
            pass


_keeper_started = threading.Event()


def start_session_keeper(tether: TetherSettings) -> None:
    """Own the camera for the app's lifetime — the zero-restart guarantee.

    On a dedicated rig the app runs continuously, so instead of connecting on
    demand (and pushing recovery work onto the user when the one
    session-per-power-on is already spent), a resident keeper thread:

    * keeps macOS's camera daemons suppressed (they are the session-burners),
    * the INSTANT a camera appears on USB — app launch, power blip, smart-plug
      schedule — tight-claims it, opens THE session, turns liveview on for the
      HDMI dongle, applies saved exposure config, and (per focus policy) drives
      one AF pass,
    * holds that session until the camera disappears, then waits for it to
      return and does it all again.

    The user never restarts anything as part of normal play: power events
    still happen, but recovering from them is the keeper's job. Idempotent —
    call as often as convenient."""
    if _keeper_started.is_set():
        return
    _keeper_started.set()

    def _power_cycle_via_plug() -> bool:
        """Restart the camera through its smart plug — the app's job, never the
        user's. Returns True if the cycle visibly happened on USB."""
        from .eos_ptp import eos_camera_present
        if not (tether.plug_off_cmd.strip() and tether.plug_on_cmd.strip()):
            return False
        log.info("session keeper: power-cycling the camera via smart plug…")
        try:
            subprocess.run(tether.plug_off_cmd, shell=True, capture_output=True,
                           timeout=30)
        except subprocess.SubprocessError:
            return False
        t0 = time.time()
        while eos_camera_present():
            if time.time() - t0 > 20:
                log.warning("session keeper: camera did not power OFF via plug")
                return False
            time.sleep(0.3)
        time.sleep(1.5)  # let the supply actually drop
        try:
            subprocess.run(tether.plug_on_cmd, shell=True, capture_output=True,
                           timeout=30)
        except subprocess.SubprocessError:
            return False
        t0 = time.time()
        while not eos_camera_present():
            if time.time() - t0 > 30:
                log.warning("session keeper: camera did not return after plug ON")
                return False
            time.sleep(0.2)
        return True

    def _keep() -> None:
        from .eos_ptp import ensure_daemon_suppression, eos_camera_present
        ensure_daemon_suppression()
        was_present: bool | None = None
        synced_this_poweron = False
        last_cycle = 0.0
        while True:
            present = eos_camera_present()
            if present and not was_present:
                log.info("session keeper: camera appeared on USB")
                if was_present is not None:
                    synced_this_poweron = False  # fresh power-on
            elif not present and was_present:
                log.info("session keeper: camera left USB — watching for return")
                _close_session_quietly()
                synced_this_poweron = False
            if present and not synced_this_poweron:
                # One transient sync per power-on: exposure config + AF, then
                # let go so movie-mode native liveview owns the HDMI feed.
                msg = remote_camera_sync(tether, autofocus=True)
                if not msg:
                    synced_this_poweron = True
                    log.info("session keeper: camera synced (transient)")
                elif time.time() - last_cycle > 120 and _power_cycle_via_plug():
                    last_cycle = time.time()
                    msg2 = remote_camera_sync(tether, autofocus=True)
                    synced_this_poweron = not msg2
                    log.info("session keeper: auto power-cycle -> %s",
                             msg2 or "synced")
                else:
                    time.sleep(4.0)  # back off — don't hammer a refusing camera
            was_present = present
            time.sleep(0.3)

    threading.Thread(target=_keep, name="tether-keeper", daemon=True).start()
    log.info("camera session keeper started (zero-restart SOP)")


def remote_camera_sync(tether: TetherSettings, autofocus: bool = True) -> str:
    """Touchless control for the ceiling-mounted camera over the held session.

    Ensures the app-lifetime session is up, turns liveview on with output to
    the camera screen (HDMI mirrors it -> the capture dongle lights up),
    applies exposure config, disables auto-power-off, and optionally drives one
    AF pass. Never disconnects. Returns '' on success or a short status; a
    spent-session failure is the session keeper's cue to power-cycle the camera
    via its smart plug and retry — the user does nothing.
    Blocking (~2-6s) — call from the worker thread, not the UI thread.
    """
    from .eos_ptp import usb_available
    if not usb_available():
        return "Camera control unavailable: pyusb/libusb not installed."
    if not list_tethered_cameras():
        return "No Canon camera detected on USB."
    try:
        cam = _get_session()
        try:
            # With the mode dial on Movie the camera runs liveview natively and
            # refuses PTP EVF control (DeviceBusy) — that's fine, skip it.
            cam.ensure_hdmi_liveview()
        except Exception:  # noqa: BLE001
            log.info("EVF control refused (movie-dial rig) — native liveview rules")
        cam.drain_events()
        cam.disable_auto_poweroff()
        cam.apply_exposure(tether.iso, tether.whitebalance)
        if autofocus and (tether.focus_mode or "") != "manual":
            cam.autofocus_once(poll_evf=False)
    except Exception as exc:  # noqa: BLE001 - surface, never raise into the app
        _close_session_quietly()
        return str(exc)
    finally:
        # Transient by design: a HELD remote-mode session blocks movie-mode
        # liveview on the body (observed live: "press REC" prompt instead of
        # picture). Clean closes are safe to repeat — abandoned sessions are
        # what poison a power-on — so control connects briefly and lets go.
        _close_session_quietly()
    log.info("remote camera sync OK (transient session, autofocus=%s)", autofocus)
    return ""


class GphotoSource(FrameSource):
    """Live EVF feed from a tethered EOS, with the same take-semantics
    ``read()`` and dedicated grab thread as :class:`ThreadedCameraSource`.

    The grab thread owns ALL USB I/O (PTP is strictly single-conversation);
    outside callers communicate via flags (``refocus()``)."""

    is_live = True

    def __init__(self, tether: TetherSettings):
        self.name = "Tethered camera"
        self._tether = tether
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._stop = threading.Event()
        self._refocus = threading.Event()
        self._thread: threading.Thread | None = None
        self._cam = None
        self._opened = False

        from .eos_ptp import usb_available
        if not usb_available():
            log.warning("tether: pyusb/libusb unavailable — pip install pyusb "
                        "(and `brew install libusb` on macOS)")
            return
        try:
            cam = _get_session()  # app-lifetime session (one per camera power-on)
            cam.disable_auto_poweroff()
            cam.apply_exposure(tether.iso, tether.whitebalance)
            cam.start_liveview()  # redirect EVF to USB; HDMI dark while we stream
            # First frames need a beat; confirm liveview actually flows.
            frame = self._await_first_frame(cam, timeout_s=6.0)
            if frame is None:
                raise RuntimeError("liveview produced no frames")
            if (tether.focus_mode or "auto_once_lock") in ("auto_once_lock", "continuous"):
                cam.autofocus_once()
        except Exception as exc:  # noqa: BLE001 - degrade, never raise into the app
            log.warning("tether: connect failed: %s", exc)
            return

        self._cam = cam
        with self._lock:
            self._latest = frame
        self._opened = True
        self._thread = threading.Thread(target=self._grab_loop,
                                        name="tether-grab", daemon=True)
        self._thread.start()
        log.info("tether: EOS liveview streaming (%dx%d)",
                 frame.shape[1], frame.shape[0])

    @staticmethod
    def _await_first_frame(cam, timeout_s: float) -> np.ndarray | None:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            jpeg = cam.liveview_frame()
            if jpeg:
                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
            time.sleep(0.1)
        return None

    # -- grab thread (owns the USB bus) ---------------------------------- #
    def _grab_loop(self) -> None:
        from .eos_ptp import PTPError
        cam = self._cam
        last_keepalive = last_drain = time.time()
        try:
            while not self._stop.is_set():
                if self._refocus.is_set():
                    self._refocus.clear()
                    cam.autofocus_once()
                try:
                    jpeg = cam.liveview_frame()
                except PTPError as e:
                    log.debug("EVF frame error: %s", e)
                    jpeg = None
                if jpeg:
                    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8),
                                         cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self._lock:
                            self._latest = frame
                else:
                    time.sleep(0.03)
                now = time.time()
                if now - last_keepalive > 20:
                    cam.keep_alive()
                    last_keepalive = now
                if now - last_drain > 2:
                    cam.drain_events()
                    last_drain = now
        except Exception as exc:  # noqa: BLE001 - unplugged / bus error
            log.warning("tether: stream lost: %s", exc)
            cam.connected = False  # force a reconnect attempt on next use
        finally:
            self._opened = False
            # Keep the app-lifetime session; just hand the picture back to the
            # camera screen so the HDMI/dongle feed resumes.
            if cam.connected:
                try:
                    cam.ensure_hdmi_liveview()
                except Exception:  # noqa: BLE001
                    pass

    # -- FrameSource API -------------------------------------------------- #
    @property
    def opened(self) -> bool:
        return self._opened

    def read(self) -> np.ndarray | None:
        with self._lock:
            frame, self._latest = self._latest, None
        return frame

    def refocus(self) -> None:
        """Ask the grab thread to re-run AF once (safe from any thread)."""
        if self._tether.focus_mode != "manual":
            self._refocus.set()

    def release(self) -> None:
        """Stop streaming. Deliberately does NOT close the PTP session — the
        T3i grants one session per power-on, so the app-lifetime session must
        survive source stop/start (the grab thread hands the picture back to
        the camera screen on its way out)."""
        self._stop.set()
        t, self._thread = self._thread, None
        if t is not None:
            t.join(timeout=4.0)
        self._opened = False

    @property
    def fps(self) -> float:
        return 24.0  # EVF over USB2 lands ~15-25 fps; pipeline adapts anyway
