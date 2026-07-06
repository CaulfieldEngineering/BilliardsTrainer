"""JINOU JO-BEC12-2 sensor protocol: frame parsing, decoding, scaling.

Decoding follows the official datasheet (archived in the pool-stroke-analyzer
project): frames and broadcast are ``55 AA <type> <len> <payload>``; sensor
values are little-endian int16 scaled as  value = range * raw / 32768.

Pure stdlib — no bleak import here, so the app can always load this module
(tests, frozen builds, machines without Bluetooth). The BLE connection itself
lives in worker.py.

SAFETY: the app NEVER writes to the device. The command characteristic (B3A2)
is deliberately absent from this module — a malformed write bricked sensor
unit #1. All configuration goes through the standalone analyzer's configure.py,
run deliberately by a human.
"""

import struct

# Vendor GATT identifiers
SERVICE_UUID = "0000b3a0-0000-1000-8000-00805f9b34fb"
DATA_CHAR = "0000b3a1-0000-1000-8000-00805f9b34fb"       # notify: sensor stream
BATTERY_CHAR = "00002a19-0000-1000-8000-00805f9b34fb"    # read: battery %
# NOTE: B3A2 (command channel) is intentionally not here.

SYNC = b"\x55\xaa"
FRAME_ACCEL = 0x00
FRAME_GYRO = 0x04

# Range code -> full-scale, and the resulting LSB/unit (value = raw / lsb).
ACCEL_RANGE_G = {0x01: 2.0, 0x02: 4.0, 0x04: 8.0, 0x08: 16.0}
GYRO_RANGE_DPS = {0x01: 250.0, 0x02: 500.0, 0x04: 1000.0, 0x08: 2000.0}
FULL_SCALE = 32768.0

# Defaults (factory: accel ±2g). Streams override these from the broadcast.
ACCEL_LSB_PER_G = FULL_SCALE / 2.0        # 16384 (±2g)
GYRO_LSB_PER_DPS = FULL_SCALE / 500.0     # 65.536 (500 dps, the documented default)

NAME_HINTS = ("joe", "jinou", "jo-", "jo_", "bec")


def accel_lsb_per_g(range_code: int) -> float:
    return FULL_SCALE / ACCEL_RANGE_G.get(range_code, 2.0)


def gyro_lsb_per_dps(range_code: int) -> float:
    return FULL_SCALE / GYRO_RANGE_DPS.get(range_code, 500.0)


def parse_frames(buffer: bytearray):
    """Yield (frame_type, payload_bytes) for each complete frame; keep partial tail."""
    while True:
        start = buffer.find(SYNC)
        if start < 0:
            # A trailing 0x55 may be the first half of the next sync word —
            # keep it (fix over the reference port, which dropped a frame when
            # a notification fragment ended exactly on the sync boundary).
            if buffer and buffer[-1] == SYNC[0]:
                del buffer[:-1]
            else:
                buffer.clear()
            return
        if start:
            del buffer[:start]
        if len(buffer) < 4:
            return
        length = buffer[3]
        if len(buffer) < 4 + length:
            return
        ftype = buffer[2]
        payload = bytes(buffer[4:4 + length])
        del buffer[:4 + length]
        yield ftype, payload


def decode_accel(payload: bytes, lsb_per_g: float = ACCEL_LSB_PER_G):
    """First 3 int16 (LE) are X,Y,Z. Firmware may append extra bytes (ignored)."""
    if len(payload) < 6:
        return None
    x, y, z = struct.unpack_from("<3h", payload, 0)
    return {"x": x / lsb_per_g, "y": y / lsb_per_g, "z": z / lsb_per_g}


def decode_gyro(payload: bytes, lsb_per_dps: float = GYRO_LSB_PER_DPS):
    if len(payload) < 6:
        return None
    x, y, z = struct.unpack_from("<3h", payload, 0)
    return {"x": x / lsb_per_dps, "y": y / lsb_per_dps, "z": z / lsb_per_dps}


def parse_broadcast(company_id: int, payload: bytes):
    """Decode the manufacturer-data broadcast (fully passive — no connection).

    Layout (datasheet): BTaddr(6, big-endian) + accel_range(1) + Xacc,Yacc,Zacc
    (2 each, LE) + battery(1) [+ gyro_range(1) + Xgyro,Ygyro,Zgyro (2 each, LE)].
    The BLE 'company id' is really the first two address bytes, so prepend them.
    """
    full = struct.pack("<H", company_id) + bytes(payload)
    if len(full) < 14:
        return None
    ar = full[6]
    a_lsb = accel_lsb_per_g(ar)
    ax, ay, az = struct.unpack_from("<3h", full, 7)
    out = {
        "bt_address": ":".join(f"{b:02X}" for b in full[0:6]),
        "accel_range_g": ACCEL_RANGE_G.get(ar),
        "accel": {"x": ax / a_lsb, "y": ay / a_lsb, "z": az / a_lsb},
        "battery": full[13],
        "gyro": None,
        "gyro_range_dps": None,
    }
    if len(full) >= 21:
        gr = full[14]
        g_lsb = gyro_lsb_per_dps(gr)
        gx, gy, gz = struct.unpack_from("<3h", full, 15)
        out["gyro_range_dps"] = GYRO_RANGE_DPS.get(gr)
        out["gyro"] = {"x": gx / g_lsb, "y": gy / g_lsb, "z": gz / g_lsb}
    return out


def broadcast_from_adv(adv) -> dict | None:
    """Extract the sensor broadcast from a bleak AdvertisementData, if present."""
    for cid, payload in (getattr(adv, "manufacturer_data", None) or {}).items():
        info = parse_broadcast(cid, payload)
        if info:
            return info
    return None


def looks_like_sensor(name: str, service_uuids) -> bool:
    """Identify the sensor in a scan by service UUID (authoritative) or name hint."""
    uuids = [u.lower() for u in (service_uuids or [])]
    if SERVICE_UUID in uuids:
        return True
    n = (name or "").lower()
    return any(h in n for h in NAME_HINTS) and "whoop" not in n
