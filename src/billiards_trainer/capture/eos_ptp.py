"""Native PTP driver for Canon EOS bodies over USB (pyusb/libusb).

Why this exists: on macOS, Apple's ``ptpcamerad`` claims every PTP camera and
SIP forbids disabling it. Killing it frees the USB claim but leaves the camera
with a half-open PTP session that renders its bulk endpoints silent — which is
why the ``gphoto2`` CLI approach fails on a Mac even though the same libusb
works fine (commercial tether apps ship their own PTP stack for this reason).
The winning sequence, proven on the real T3i: suppress the daemons, force a USB
re-enumeration so the camera comes back with a virgin PTP state, and claim it
before anything else can speak to it. Once we hold the exclusive claim the
daemons are locked out and plain PTP flows perfectly.

Opcodes/properties verified against libgphoto2's camlibs/ptp2/ptp.h (the
authoritative community source for Canon's unpublished EOS extensions).

Threading contract: ALL USB I/O after ``connect()`` happens on the grab thread.
External callers use flags/queues (``request_refocus``) — never the bus.
"""

from __future__ import annotations

import atexit
import logging
import struct
import subprocess
import sys
import threading
import time

log = logging.getLogger("capture.eos_ptp")

CANON_VID = 0x04A9

# -- PTP standard --------------------------------------------------------- #
PTP_OC_GetDeviceInfo = 0x1001
PTP_OC_OpenSession = 0x1002
PTP_OC_CloseSession = 0x1003
PTP_RC_OK = 0x2001
PTP_RC_CANON_NOT_READY = 0xA102

# -- Canon EOS extensions (libgphoto2 camlibs/ptp2/ptp.h) ------------------ #
EOS_SetDevicePropValueEx = 0x9110
EOS_SetRemoteMode = 0x9114
EOS_SetEventMode = 0x9115
EOS_GetEvent = 0x9116
EOS_KeepDeviceOn = 0x911D
EOS_GetViewFinderData = 0x9153
EOS_DoAf = 0x9154
EOS_AfCancel = 0x9160

DPC_Aperture = 0xD101
DPC_ShutterSpeed = 0xD102
DPC_ISOSpeed = 0xD103
DPC_WhiteBalance = 0xD109
DPC_AutoPowerOff = 0xD114
DPC_EVFOutputDevice = 0xD1B0
DPC_EVFMode = 0xD1B1

# Canon EOS coded values (subset that matters for a fixed overhead rig).
ISO_CODES = {
    "auto": 0x00, "50": 0x40, "100": 0x48, "125": 0x4B, "160": 0x4D,
    "200": 0x50, "250": 0x53, "320": 0x55, "400": 0x58, "500": 0x5B,
    "640": 0x5D, "800": 0x60, "1000": 0x63, "1250": 0x65, "1600": 0x68,
    "2000": 0x6B, "2500": 0x6D, "3200": 0x70, "4000": 0x73, "5000": 0x75,
    "6400": 0x78, "12800": 0x80,
}
WB_CODES = {
    "auto": 0, "daylight": 1, "cloudy": 2, "tungsten": 3,
    "fluorescent": 4, "flash": 5, "manual": 6, "shade": 8,
}

EP_OUT = 0x02
EP_IN = 0x81


class PTPError(RuntimeError):
    def __init__(self, code: int, op: str):
        super().__init__(f"{op} -> PTP rc {hex(code)}")
        self.code = code


def _usb():
    """Import pyusb lazily so the app runs fine without it installed."""
    import usb.core
    import usb.util
    return usb.core, usb.util


def usb_available() -> bool:
    try:
        _usb()
        return True
    except Exception:  # noqa: BLE001 - missing module OR missing libusb dylib
        return False


def eos_camera_present() -> bool:
    """Cheap USB-enumeration check (no claim, no session)."""
    if not usb_available():
        return False
    core, _ = _usb()
    try:
        return core.find(idVendor=CANON_VID) is not None
    except Exception:  # noqa: BLE001
        return False


def list_eos_cameras() -> list[str]:
    """Product strings of connected Canon cameras (control-transfer only, so it
    works even while another app holds the claim)."""
    if not usb_available():
        return []
    core, util = _usb()
    names: list[str] = []
    try:
        for dev in core.find(find_all=True, idVendor=CANON_VID) or []:
            try:
                names.append(util.get_string(dev, dev.iProduct) or "Canon camera")
            except Exception:  # noqa: BLE001
                names.append("Canon camera")
    except Exception:  # noqa: BLE001 - no libusb backend
        return []
    return names


_suppress_lock = threading.Lock()
_suppress_stop: threading.Event | None = None


def ensure_daemon_suppression() -> None:
    """Start (once, process-wide) continuous suppression of macOS's PTP daemons.

    Live-hardware findings that force this design (T3i, macOS 15):
    * The daemons open a PTP session the moment they see a camera and hold it.
    * Canon's firmware goes PERMANENTLY deaf on the bulk endpoints when any
      session-holder vanishes without CloseSession — and nothing software-side
      (USB re-enumeration, PTP class Device-Reset) revives it; only a physical
      power cycle does.
    * Therefore the daemons must NEVER touch the camera while this app wants
      USB control. Suppression runs for the app's whole life once camera
      control is first used: daemons are killed the instant launchd respawns
      them, before they can reach the camera. After ONE power cycle under
      suppression the camera stays healthy indefinitely, because our own
      sessions always close cleanly.

    Harmless side effect: Image Capture/Photos can't see PTP cameras while the
    app runs — acceptable on a dedicated rig."""
    global _suppress_stop
    if sys.platform != "darwin":
        return
    with _suppress_lock:
        if _suppress_stop is not None:
            return
        stop = threading.Event()

        def _loop() -> None:
            while not stop.is_set():
                subprocess.run(["killall", "-9", "ptpcamerad", "mscamerad-xpc"],
                               capture_output=True)
                stop.wait(0.2)

        threading.Thread(target=_loop, name="ptp-daemon-suppress",
                         daemon=True).start()
        _suppress_stop = stop
        atexit.register(stop.set)
        log.info("macOS camera-daemon suppression active (app lifetime)")


class EOSCamera:
    """One exclusive PTP session with an EOS body. Not thread-safe by design —
    the owner (grab thread) is the only caller after connect()."""

    def __init__(self):
        self._dev = None
        self._buf = bytearray()
        self._tid = 0
        self._evf_started = False
        self.connected = False

    # -- connection ----------------------------------------------------- #
    def connect(self) -> None:
        """Claim the camera and bring up a working EOS session.

        Flow: start app-lifetime daemon suppression -> exclusive claim ->
        OpenSession. A camera the daemons ever touched carries an un-closed
        session that leaves it deaf (see ensure_daemon_suppression); the only
        cure is one power cycle done under suppression, after which control
        stays reliable for the whole app run — the error message says exactly
        that so the UI can show it."""
        core, util = _usb()
        self._util = util
        ensure_daemon_suppression()
        # Tight acquisition: claim the INSTANT the device is claimable. Any
        # settle-sleep before the claim is a window in which a respawned daemon
        # can open its session first (a race we measurably lost once) — the
        # exclusive claim is the only real lock, so take it immediately and
        # retry rapidly while enumeration completes.
        dev = None
        claimed = False
        t0 = time.time()
        while time.time() - t0 < 5.0:
            dev = core.find(idVendor=CANON_VID)
            if dev is not None:
                try:
                    util.claim_interface(dev, 0)  # exclusive: daemons locked out
                    claimed = True
                    break
                except Exception:  # noqa: BLE001 - enumeration still settling
                    try:
                        util.dispose_resources(dev)
                    except Exception:  # noqa: BLE001
                        pass
            time.sleep(0.05)
        if dev is None:
            raise RuntimeError("no Canon camera on USB")
        if not claimed:
            raise RuntimeError("could not claim the camera (another app holds it?)")
        self._dev = dev
        time.sleep(0.3)  # brief settle AFTER the claim, before first PTP traffic
        try:
            self._open_session(timeout=2500)
        except Exception:  # noqa: BLE001 - stale session or wedge
            log.info("tether: camera deaf on first try — class reset, one retry")
            self._class_reset()
            try:
                self._open_session(timeout=2500)
            except Exception as e:
                raise RuntimeError(
                    "camera unresponsive (macOS's camera daemon left a stale "
                    "session) — power-cycle the camera once (off/on), then retry; "
                    "it stays reliable while the app is open"
                ) from e
        self.connected = True

    def _class_reset(self) -> None:
        """USB Still-Image class Device Reset (0x66): clears the camera's PTP
        state machine — including a zombie session — without re-enumeration."""
        try:
            self._dev.ctrl_transfer(0x21, 0x66, 0, 0, None, timeout=5000)
        except Exception as e:  # noqa: BLE001 - some stacks stall this; not fatal
            log.debug("class reset not accepted: %s", e)
        time.sleep(0.8)  # camera needs a beat to rebuild its PTP state
        for ep in (EP_IN, EP_OUT):
            try:
                self._dev.clear_halt(ep)
            except Exception:  # noqa: BLE001
                pass
        self._buf.clear()

    def _open_session(self, timeout: int = 5000) -> None:
        self.check(PTP_OC_OpenSession, 1, timeout=timeout, op="OpenSession")
        self.check(EOS_SetRemoteMode, 1, timeout=timeout, op="SetRemoteMode")
        self.check(EOS_SetEventMode, 1, timeout=timeout, op="SetEventMode")
        self.drain_events()

    def close(self, leave_liveview: bool = False) -> None:
        """End the session. ``leave_liveview=True`` keeps the camera's own
        (TFT→HDMI) liveview running so the capture-dongle feed survives the app
        — the right choice for the permanent rig, where the camera needs a
        power cycle before the next session anyway (one-session-per-power-on)."""
        if self._dev is None:
            return
        steps = []
        if self._evf_started and not leave_liveview:
            steps += [lambda: self.set_prop(DPC_EVFOutputDevice, 0),
                      lambda: self.set_prop(DPC_EVFMode, 0)]
        elif self._evf_started:
            # We were consuming EVF over USB: hand the picture back to the
            # camera screen (HDMI mirrors it) instead of shutting liveview off.
            steps += [lambda: self.set_prop(DPC_EVFOutputDevice, 1)]
        steps += [lambda: self.command(EOS_SetRemoteMode, 0),
                  lambda: self.command(PTP_OC_CloseSession)]
        for step in steps:
            try:
                step()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
        try:
            self._util.release_interface(self._dev, 0)
            self._util.dispose_resources(self._dev)
        except Exception:  # noqa: BLE001
            pass
        self._dev = None
        self.connected = False

    # -- transport ------------------------------------------------------ #
    def _read_exact(self, n: int, timeout: int) -> bytes:
        while len(self._buf) < n:
            chunk = self._dev.read(EP_IN, 65536, timeout=timeout)
            self._buf.extend(chunk)
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out

    def _read_container(self, timeout: int) -> tuple[int, int, bytes]:
        hdr = self._read_exact(12, timeout)
        length, ctype, code, _tid = struct.unpack("<IHHI", hdr)
        payload = self._read_exact(length - 12, timeout) if length > 12 else b""
        return ctype, code, payload

    def command(self, code: int, *params: int, data: bytes | None = None,
                timeout: int = 5000) -> tuple[int, bytes]:
        """Run one PTP transaction. Returns (response_code, data_payload)."""
        self._tid += 1
        body = b"".join(struct.pack("<I", p) for p in params)
        self._dev.write(EP_OUT, struct.pack("<IHHI", 12 + len(body), 1, code,
                                            self._tid) + body, timeout=timeout)
        if data is not None:
            self._dev.write(EP_OUT, struct.pack("<IHHI", 12 + len(data), 2, code,
                                                self._tid) + data, timeout=timeout)
        payload = b""
        while True:
            ctype, rcode, cdata = self._read_container(timeout)
            if ctype == 2:
                payload = cdata
            elif ctype == 3:
                return rcode, payload

    def check(self, code: int, *params: int, data: bytes | None = None,
              timeout: int = 5000, op: str = "") -> bytes:
        rc, payload = self.command(code, *params, data=data, timeout=timeout)
        if rc != PTP_RC_OK:
            raise PTPError(rc, op or hex(code))
        return payload

    # -- EOS session ----------------------------------------------------- #
    def drain_events(self) -> None:
        try:
            self.command(EOS_GetEvent, timeout=3000)
        except Exception:  # noqa: BLE001 - event drain is best-effort
            pass

    def keep_alive(self) -> None:
        try:
            self.command(EOS_KeepDeviceOn, timeout=2000)
        except Exception:  # noqa: BLE001
            pass

    def set_prop(self, prop: int, value: int) -> None:
        payload = struct.pack("<III", 12, prop, value)
        self.check(EOS_SetDevicePropValueEx, data=payload, op=f"SetProp {hex(prop)}")
        self.drain_events()

    # -- liveview -------------------------------------------------------- #
    def start_liveview(self) -> None:
        """Redirect liveview to USB (we consume the frames; HDMI goes dark)."""
        self.set_prop(DPC_EVFMode, 1)
        self.set_prop(DPC_EVFOutputDevice, 2)  # 2 = PC output
        self._evf_started = True

    def ensure_hdmi_liveview(self) -> None:
        """Turn liveview ON with output on the camera screen — HDMI mirrors it.
        This is what lights up the capture dongle after a camera power-on
        (validated live: ~25fps out the dongle while this session stays held,
        including during DoAf).

        DeviceBusy (0x2019) here usually means the camera's MENU is open on the
        body — Canon refuses remote liveview from inside the menu — or it's
        still settling right after boot. Drain events and retry a few times;
        a persistent refusal is reported to the caller."""
        last: Exception | None = None
        for attempt in range(4):
            try:
                self.set_prop(DPC_EVFMode, 1)
                self.set_prop(DPC_EVFOutputDevice, 1)  # 1 = TFT -> HDMI mirror
                self._evf_started = True
                return
            except PTPError as e:
                last = e
                if e.code != 0x2019:  # only DeviceBusy is worth retrying
                    raise
                self.drain_events()
                time.sleep(0.8 + 0.4 * attempt)
        raise PTPError(0x2019, "EVF enable (camera menu open on the body?)") from last

    def liveview_frame(self, timeout: int = 2000) -> bytes | None:
        """One EVF JPEG, or None while the camera is still warming up."""
        rc, data = self.command(EOS_GetViewFinderData, 0x00200000, 0, 0,
                                timeout=timeout)
        if rc == PTP_RC_CANON_NOT_READY:
            return None
        if rc != PTP_RC_OK:
            raise PTPError(rc, "GetViewFinderData")
        start = data.find(b"\xff\xd8\xff")
        end = data.rfind(b"\xff\xd9")
        if start < 0 or end <= start:
            return None
        return data[start:end + 2]

    # -- focus / exposure ------------------------------------------------ #
    def autofocus_once(self, settle_s: float = 2.0, poll_evf: bool = True) -> None:
        """Drive contrast-detect AF, then cancel so the lens never hunts again —
        the right behaviour for a fixed overhead rig.

        ``poll_evf`` keeps pulling EVF frames while AF runs (needed when WE are
        the liveview consumer). A transient control session with the video going
        out over HDMI instead passes False and just waits."""
        try:
            self.check(EOS_DoAf, op="DoAf")
        except PTPError as e:
            log.info("DoAf not accepted (%s) — lens on MF?", e)
            return
        t0 = time.time()
        while time.time() - t0 < settle_s:
            if poll_evf:
                try:
                    self.liveview_frame()  # keep EVF flowing while AF runs
                except PTPError:
                    pass
            time.sleep(0.05)
        try:
            self.command(EOS_AfCancel)
        except Exception:  # noqa: BLE001
            pass

    def apply_exposure(self, iso: str, whitebalance: str) -> None:
        code = ISO_CODES.get((iso or "auto").strip().lower())
        if code is not None and code != 0:
            try:
                self.set_prop(DPC_ISOSpeed, code)
            except PTPError as e:
                log.info("ISO not set: %s", e)
        wb = WB_CODES.get((whitebalance or "auto").strip().lower())
        if wb is not None and wb != 0:
            try:
                self.set_prop(DPC_WhiteBalance, wb)
            except PTPError as e:
                log.info("WB not set: %s", e)

    def disable_auto_poweroff(self) -> None:
        try:
            self.set_prop(DPC_AutoPowerOff, 0)
            log.info("camera auto-power-off disabled")
        except PTPError as e:
            log.info("AutoPowerOff not settable: %s", e)
