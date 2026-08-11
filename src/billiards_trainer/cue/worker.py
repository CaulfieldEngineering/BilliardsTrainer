"""Threaded BLE cue-sensor worker.

A plain QObject owning a background asyncio thread (the pattern proven in the
reference analyzer app): ALL Bluetooth work happens on that daemon thread and
results surface as Qt signals, which auto-queue across threads. Nothing here
may ever raise on a Qt thread — no sensor, no Bluetooth radio, and no bleak
install each land in a ``status_changed`` state and a quiet retry loop, so the
rest of the app behaves identically with or without the hardware.

Live flow per the reference implementation: decoded samples feed the
LiveImpactDetector as they stream; a confirmed cue-ball impact emits
``impact`` immediately (peak g is known at the strike), and the full stroke
metrics emit ~2.6 s later once the follow-through has streamed in.

READ-ONLY: this worker never writes to the sensor (see cue/protocol.py).
"""

import asyncio
import logging
import threading
import time

from PySide6.QtCore import QObject, Signal

from . import protocol
from .detector import METRICS_DELAY_S, LiveImpactDetector

log = logging.getLogger("cue.worker")

_SCAN_SECONDS = 8.0        # one BLE discovery pass
_RESCAN_WAIT = 15.0        # idle between failed scans (sensor off / out of range)
_RECONNECT_WAIT = 1.5      # after a connection drop
_RADIO_WAIT = 20.0         # after a radio-level error (Bluetooth off)


class CueSensorWorker(QObject):
    # (state, detail) — states: disabled | unavailable | bluetooth_off | scanning
    #                 | no_sensor | connecting | connected | reconnecting | error
    status_changed = Signal(str, object)
    sample = Signal(str, object, float)   # kind ('accel'|'gyro'), {x,y,z}, epoch t
    impact = Signal(object)               # confirmed hit (immediate): peak_g etc.
    metrics = Signal(object)              # full stroke metrics, ~2.6 s after the hit
    address_resolved = Signal(str)        # sensor MAC discovered (persist for UX)
    # Every device a discovery pass saw, for the pairing UI: a list of
    # {address, name, rssi, is_sensor}. Emitted only when a scan was REQUESTED,
    # so the normal connect loop stays quiet.
    devices_found = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._det = LiveImpactDetector()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._enabled = False
        self._closing = False
        self._enabled_evt = asyncio.Event()
        self._wake = asyncio.Event()
        self._impact_floor = 1.6
        self._saved_address = ""
        self._accel_lsb = protocol.ACCEL_LSB_PER_G
        self._gyro_lsb = protocol.GYRO_LSB_PER_DPS
        self._buffer = bytearray()
        self._prev_hit_t: float | None = None
        self._state = "disabled"
        self._scan_req = False       # one-shot: emit devices_found next pass

    # ------------------------------------------------------------------ #
    # Public API (call from any thread)
    # ------------------------------------------------------------------ #
    def apply_settings(self, cue) -> None:
        """Adopt CueSettings: enable/disable the loop, floor, saved address."""
        self._impact_floor = max(1.0, float(getattr(cue, "impact_g", 1.6)))
        self._saved_address = getattr(cue, "address", "") or ""
        want = bool(getattr(cue, "enabled", False))
        if want and self._thread is None:
            self._start_thread()
        if self._loop is not None:
            def _apply():
                self._det.impact_g = self._impact_floor
                self._enabled = want
                if want:
                    self._enabled_evt.set()
                else:
                    self._enabled_evt.clear()
                    self._emit_status("disabled", None)
                self._wake.set()
            try:
                self._loop.call_soon_threadsafe(_apply)
            except RuntimeError:
                pass  # loop already closed (shutdown race)
        elif not want:
            self._emit_status("disabled", None)

    def request_scan(self) -> None:
        """Ask for one discovery pass and emit everything it sees.

        Runs on the worker's OWN BLE thread rather than opening a second radio
        session from the UI — two concurrent scans is exactly the kind of thing
        that works on one Windows Bluetooth stack and fails on another. Works
        while DISABLED too, so a sensor can be found and chosen before the
        feature is switched on: the idle wait is released for the pass and
        re-entered afterwards.
        """
        if self._thread is None:
            self._start_thread()
        if self._loop is None:
            return

        def _req():
            self._scan_req = True
            self._enabled_evt.set()   # break the idle wait for one pass
            self._wake.set()          # cut short any backoff sleep
        try:
            self._loop.call_soon_threadsafe(_req)
        except RuntimeError:
            pass

    def select_device(self, address: str) -> None:
        """Pin the sensor to ``address`` and (re)connect to it now."""
        self._saved_address = address or ""
        if self._loop is None:
            return

        def _sel():
            self._wake.set()          # abandon any wait; reconnect immediately
        try:
            self._loop.call_soon_threadsafe(_sel)
        except RuntimeError:
            pass

    @property
    def state(self) -> str:
        return self._state

    def shutdown(self) -> None:
        loop, thread = self._loop, self._thread
        if loop is None:
            return
        def _stop():
            self._closing = True
            self._enabled = False
            self._enabled_evt.set()   # release the idle wait so _main can exit
            self._wake.set()
        try:
            loop.call_soon_threadsafe(_stop)
        except RuntimeError:
            return
        if thread is not None:
            thread.join(timeout=3.0)

    # ------------------------------------------------------------------ #
    def _start_thread(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="cue-ble", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception:  # noqa: BLE001 - the loop must die quietly, never crash Qt
            log.exception("cue BLE loop crashed")
        finally:
            try:
                self._loop.close()
            except Exception:  # noqa: BLE001
                pass

    def _emit_status(self, state: str, detail) -> None:
        self._state = state
        log.info("cue sensor: %s %s", state, detail if detail is not None else "")
        self.status_changed.emit(state, detail)

    async def _sleep(self, seconds: float) -> None:
        """Interruptible wait — enable/disable/shutdown wake it immediately."""
        self._wake.clear()
        try:
            await asyncio.wait_for(self._wake.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    # ------------------------------------------------------------------ #
    async def _main(self) -> None:
        while not self._closing:
            # A requested scan runs even while disabled, so the sensor can be
            # found and picked before the feature is switched on.
            if not self._enabled and not self._scan_req:
                self._enabled_evt.clear()
                await self._enabled_evt.wait()
                continue
            try:
                import bleak  # noqa: F401 - deferred so the app runs without it
                from bleak import BleakClient, BleakScanner
            except Exception as exc:  # noqa: BLE001 - ImportError or backend init
                self._emit_status("unavailable", f"bleak not available: {exc}")
                self._enabled_evt.clear()
                self._enabled = False
                continue

            self._emit_status("scanning", None)
            try:
                found = await BleakScanner.discover(timeout=_SCAN_SECONDS,
                                                    return_adv=True)
            except Exception as exc:  # noqa: BLE001 - radio off/absent, WinRT errors
                self._emit_status("bluetooth_off", f"{type(exc).__name__}: {exc}")
                if self._scan_req:
                    self._scan_req = False
                    self.devices_found.emit([])
                await self._sleep(_RADIO_WAIT)
                continue
            if self._scan_req:
                self._scan_req = False
                self.devices_found.emit(self._describe(found))
                if not self._enabled:
                    continue          # scan-only pass: back to idle
            device, adv = self._pick(found)
            if device is None:
                self._emit_status("no_sensor", None)
                await self._sleep(_RESCAN_WAIT)
                continue

            # Scale from the passive broadcast (carries the configured ranges).
            battery = None
            bc = protocol.broadcast_from_adv(adv) if adv is not None else None
            if bc:
                if bc.get("accel_range_g"):
                    self._accel_lsb = protocol.FULL_SCALE / bc["accel_range_g"]
                if bc.get("gyro_range_dps"):
                    self._gyro_lsb = protocol.FULL_SCALE / bc["gyro_range_dps"]
                battery = bc.get("battery")

            self._emit_status("connecting", device.address)
            try:
                async with BleakClient(device.address, timeout=20) as client:
                    try:
                        v = bytes(await client.read_gatt_char(protocol.BATTERY_CHAR))
                        battery = v[0] if v else battery
                    except Exception:  # noqa: BLE001 - battery is nice-to-have
                        pass
                    self._buffer.clear()
                    self._det.reset()
                    self._det.impact_g = self._impact_floor
                    await client.start_notify(protocol.DATA_CHAR, self._on_notify)
                    name = getattr(device, "name", "") or ""
                    self._emit_status("connected", {"address": device.address,
                                                    "name": name, "battery": battery})
                    self.address_resolved.emit(device.address)
                    while (client.is_connected and self._enabled
                           and not self._closing):
                        await asyncio.sleep(0.3)
                    try:
                        await client.stop_notify(protocol.DATA_CHAR)
                    except Exception:  # noqa: BLE001 - already disconnected
                        pass
            except Exception as exc:  # noqa: BLE001 - connect/stream failure
                self._emit_status("error", f"{type(exc).__name__}: {exc}")
            if self._enabled and not self._closing:
                self._emit_status("reconnecting", None)
                await self._sleep(_RECONNECT_WAIT)

    @staticmethod
    def _describe(found: dict) -> list:
        """Flatten a discovery into rows for the pairing UI, strongest first.

        Everything is listed, not just recognised sensors: this exists precisely
        for the case where the device ISN'T recognised — a sensor that has never
        been paired, advertises an unhelpful hex name, and would otherwise be
        invisible to an auto-picker.
        """
        rows = []
        for device, adv in found.values():
            name = (getattr(adv, "local_name", None)
                    or getattr(device, "name", "") or "")
            uuids = [u.lower() for u in (adv.service_uuids or [])]
            rows.append({
                "address": device.address,
                "name": name,
                "rssi": getattr(adv, "rssi", None),
                "is_sensor": protocol.looks_like_sensor(name, uuids),
                "uuids": uuids,
            })
        rows.sort(key=lambda r: (not r["is_sensor"],
                                 -(r["rssi"] if r["rssi"] is not None else -999)))
        return rows

    def _pick(self, found: dict):
        """Choose the sensor from a scan: saved address first, else identity."""
        best = None
        for device, adv in found.values():
            if self._saved_address and device.address == self._saved_address:
                return device, adv
            name = getattr(adv, "local_name", None) or getattr(device, "name", "") or ""
            if best is None and protocol.looks_like_sensor(name, adv.service_uuids):
                best = (device, adv)
        return best if best else (None, None)

    # ------------------------------------------------------------------ #
    def _on_notify(self, _char, data: bytearray) -> None:
        """Runs on the asyncio thread for every BLE notification."""
        t = time.time()
        self._buffer.extend(data)
        for ftype, payload in protocol.parse_frames(self._buffer):
            if ftype == protocol.FRAME_ACCEL:
                kind, d = "accel", protocol.decode_accel(payload, self._accel_lsb)
            elif ftype == protocol.FRAME_GYRO:
                kind, d = "gyro", protocol.decode_gyro(payload, self._gyro_lsb)
            else:
                continue
            if d is None:
                continue
            self.sample.emit(kind, d, t)
            hit = self._det.feed(kind, d, t)
            if hit is not None:
                self._on_hit(hit)

    def _on_hit(self, stroke: dict) -> None:
        stroke["hit_epoch"] = stroke["hit_t"]  # detector time IS wall-clock epoch
        stroke.pop("feats", None)              # diagnostics-only detail
        log.info("cue impact: %.1f g at %.3f", stroke["peak_g"], stroke["hit_t"])
        self.impact.emit(dict(stroke))
        prev = self._prev_hit_t
        self._prev_hit_t = stroke["hit_t"]
        # kinematic metrics need the follow-through — compute once it has streamed
        self._loop.call_later(METRICS_DELAY_S, self._compute_metrics, stroke, prev)

    def _compute_metrics(self, stroke: dict, prev_hit_t: float | None) -> None:
        m = self._det.compute_metrics(stroke, prev_hit_t=prev_hit_t)
        out = {"hit_epoch": stroke["hit_t"], "peak_g": stroke["peak_g"],
               "peak_rate": stroke["peak_rate"], "rot_deg": stroke["rot_deg"]}
        if m:
            out.update(m)
        self.metrics.emit(out)
