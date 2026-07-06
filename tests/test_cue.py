"""Cue-stroke sensor integration tests.

Guards the contracts of the port from pool-stroke-analyzer:
  * protocol framing/decoding matches the datasheet (split notifications,
    garbage prefixes, range scaling, passive broadcast),
  * the LIVE impact detector fires on the video-validated drive→dip signature
    and rejects the labelled handling classes (tilt, radial clip, spin),
    agreeing with the offline detect_impacts path,
  * the app degrades gracefully with no bleak / no sensor (Joe's hard
    requirement: everything must work when no Bluetooth device is connected),
  * stroke metrics ride along on recorded shots (DB column + one-shot join).
"""

import math
import sys
import time
from pathlib import Path

import pytest

from billiards_trainer.cue import analysis, protocol
from billiards_trainer.cue.detector import LiveImpactDetector

# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #


def _frame(ftype: int, payload: bytes) -> bytes:
    return bytes([0x55, 0xAA, ftype, len(payload)]) + payload


def _accel_payload(x: int, y: int, z: int) -> bytes:
    import struct
    return struct.pack("<3h", x, y, z)


def test_parse_frames_reassembles_split_stream_with_garbage():
    buf = bytearray()
    stream = (b"\x01\x02" + _frame(protocol.FRAME_ACCEL, _accel_payload(1, 2, 3))
              + _frame(protocol.FRAME_GYRO, _accel_payload(4, 5, 6)))
    got = []
    # deliver one byte at a time — worst-case BLE fragmentation
    for b in stream:
        buf.append(b)
        got.extend(parse for parse in protocol.parse_frames(buf))
    assert [f[0] for f in got] == [protocol.FRAME_ACCEL, protocol.FRAME_GYRO]
    d = protocol.decode_accel(got[0][1], protocol.ACCEL_LSB_PER_G)
    assert d["x"] == pytest.approx(1 / 16384)


def test_decode_scaling_follows_range_codes():
    # ±8 g (code 0x04): raw 4096 == 1.0 g. ±2000 dps (0x08): raw 16384 == 1000 dps
    assert protocol.accel_lsb_per_g(0x04) == pytest.approx(4096.0)
    d = protocol.decode_accel(_accel_payload(4096, 0, 0), protocol.accel_lsb_per_g(0x04))
    assert d["x"] == pytest.approx(1.0)
    g = protocol.decode_gyro(_accel_payload(0, 16384, 0), protocol.gyro_lsb_per_dps(0x08))
    assert g["y"] == pytest.approx(1000.0)


def test_parse_broadcast_datasheet_layout():
    import struct
    addr = bytes.fromhex("A1B2C3D4E5F6")
    body = addr + bytes([0x04]) + struct.pack("<3h", 0, 4096, 0) + bytes([87]) \
        + bytes([0x08]) + struct.pack("<3h", 0, 0, 16384)
    cid = struct.unpack("<H", body[:2])[0]
    info = protocol.parse_broadcast(cid, body[2:])
    assert info["bt_address"] == "A1:B2:C3:D4:E5:F6"
    assert info["accel_range_g"] == 8.0
    assert info["accel"]["y"] == pytest.approx(1.0)
    assert info["battery"] == 87
    assert info["gyro_range_dps"] == 2000.0
    assert info["gyro"]["z"] == pytest.approx(1000.0)


def test_sensor_identity_matching():
    assert protocol.looks_like_sensor("JINOU-BEC12", [])
    assert protocol.looks_like_sensor("whatever", [protocol.SERVICE_UUID])
    assert not protocol.looks_like_sensor("WHOOP 4.0", [])
    assert not protocol.looks_like_sensor("Sony WH-1000XM4", [])


# --------------------------------------------------------------------------- #
# Live impact detector — synthetic traces built on the validated signature
# --------------------------------------------------------------------------- #

RATE = 27.0
DT = 1.0 / RATE


def _trace(hits=(2.0,), rest=None, dip_z=-1.5, drive_z=1.2, clip_x=None,
           spin_dps=None, dur=None):
    """Synthesize (accel, gyro) sample lists: level cue at rest (gravity on Y),
    a forward DRIVE then the impact DIP at each hit time — the PATH A shape."""
    rest = rest or {"x": 0.0, "y": 1.0, "z": 0.0}
    dur = dur or (max(hits) + 1.5 if hits else 4.0)
    accel, gyro = [], []
    n = int(dur / DT)
    for i in range(n):
        t = i * DT
        a = dict(rest)
        g = {"x": 2.0, "y": 2.0, "z": 2.0}
        for h in hits:
            if h - 0.30 <= t <= h - 0.06:
                a = {**rest, "z": rest["z"] + drive_z}       # forward drive
            elif abs(t - h) < DT / 2:
                a = {**rest, "z": rest["z"] + dip_z}         # the strike dip
                if clip_x is not None:
                    a["x"] = clip_x
            if spin_dps is not None and abs(t - h) < 0.2:
                g = {"x": spin_dps, "y": 0.0, "z": 0.0}
        accel.append({"t": t, **a})
        gyro.append({"t": t + DT / 2, **g})
    return accel, gyro


def _run_live(accel, gyro):
    det = LiveImpactDetector(impact_g=1.6)
    merged = sorted([("accel", a) for a in accel] + [("gyro", g) for g in gyro],
                    key=lambda kv: kv[1]["t"])
    hits = []
    for kind, s in merged:
        r = det.feed(kind, {k: s[k] for k in ("x", "y", "z")}, s["t"])
        if r:
            hits.append(r)
    return det, hits


def test_live_detector_fires_on_driven_stroke():
    accel, gyro = _trace(hits=(2.0,))
    det, hits = _run_live(accel, gyro)
    assert len(hits) == 1
    h = hits[0]
    assert h["hit_t"] == pytest.approx(2.0, abs=DT)
    assert h["peak_g"] == pytest.approx(math.sqrt(1.0 + 1.5 ** 2), abs=0.05)


def test_live_detector_matches_offline_path():
    accel, gyro = _trace(hits=(2.0, 5.0))
    _det, live_hits = _run_live(accel, gyro)
    offline = analysis.detect_impacts(accel, gyro, mag_thresh=1.6)
    assert len(live_hits) == len(offline) == 2
    for lh, oh in zip(live_hits, offline, strict=True):
        assert lh["hit_t"] == pytest.approx(oh["hit_t"], abs=DT)


def test_live_detector_rejects_tilted_cue():
    # gravity on Z == cue pointing down: a table jab, never a shot
    accel, gyro = _trace(hits=(2.0,), rest={"x": 0.0, "y": 0.0, "z": 1.0})
    _det, hits = _run_live(accel, gyro)
    assert hits == []


def test_live_detector_rejects_radial_clip():
    # a ±8 g X rail-out is a knock — the clip is used as a feature
    accel, gyro = _trace(hits=(2.0,), clip_x=8.0)
    _det, hits = _run_live(accel, gyro)
    assert hits == []


def test_live_detector_rejects_fast_spin():
    # flourish/spin: >450 dps near the peak kills the candidate
    accel, gyro = _trace(hits=(2.0,), spin_dps=600.0)
    _det, hits = _run_live(accel, gyro)
    assert hits == []


def test_live_detector_metrics_never_raise():
    accel, gyro = _trace(hits=(4.0,), dur=8.0)
    det, hits = _run_live(accel, gyro)
    assert len(hits) == 1
    m = det.compute_metrics(hits[0], prev_hit_t=1.0)
    # metrics are best-effort (None fields allowed) but must be a well-formed
    # dict for the UI/DB on any input that produced a confirmed impact
    assert m is not None
    assert m["interval"] == pytest.approx(hits[0]["hit_t"] - 1.0)
    for key in ("v_impact", "stroke_len", "pause", "yaw_swing",
                "steer_ratio", "finish"):
        assert key in m


# --------------------------------------------------------------------------- #
# Worker degradation — the app must work with no BLE at all
# --------------------------------------------------------------------------- #


class _Cue:
    def __init__(self, enabled, impact_g=1.6, address=""):
        self.enabled = enabled
        self.impact_g = impact_g
        self.address = address


def test_worker_disabled_starts_nothing():
    from billiards_trainer.cue.worker import CueSensorWorker
    w = CueSensorWorker()
    w.apply_settings(_Cue(enabled=False))
    assert w.state == "disabled"
    assert w._thread is None       # no BLE thread ever spun up
    w.shutdown()                    # no-op, must not raise


def test_worker_without_bleak_reports_unavailable(monkeypatch):
    from billiards_trainer.cue.worker import CueSensorWorker
    monkeypatch.setitem(sys.modules, "bleak", None)   # import bleak -> ImportError
    w = CueSensorWorker()
    w.apply_settings(_Cue(enabled=True))
    deadline = time.time() + 5.0
    while w.state != "unavailable" and time.time() < deadline:
        time.sleep(0.02)
    assert w.state == "unavailable"
    w.shutdown()


# --------------------------------------------------------------------------- #
# DB: stroke metrics ride along on shots
# --------------------------------------------------------------------------- #


def test_shot_stroke_json_roundtrip(tmp_path):
    import json

    from billiards_trainer.db.repository import Repository
    repo = Repository(db_path=Path(":memory:"))
    sid = repo.start_session()
    stroke = {"hit_epoch": 123.0, "peak_g": 3.2, "v_impact": 1.8}
    repo.record_shot(sid, outcome="make", stroke_json=json.dumps(stroke))
    repo.record_shot(sid, outcome="miss")               # no sensor data
    out = tmp_path / "export.json"
    repo.export_json(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    shots = data[-1]["shots"]
    assert shots[0]["stroke"]["peak_g"] == 3.2
    assert shots[1]["stroke"] is None


def test_old_db_gains_stroke_column(tmp_path):
    """A pre-cue-sensor database opens cleanly: the additive migration adds
    stroke_json (and reads old rows as empty)."""
    import sqlite3

    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE sessions (id INTEGER PRIMARY KEY, started_at DATETIME,
            ended_at DATETIME, mode VARCHAR(32), drill_key VARCHAR(64),
            drill_target INTEGER, table_size VARCHAR(16), notes VARCHAR(512));
        CREATE TABLE shots (id INTEGER PRIMARY KEY, session_id INTEGER,
            created_at DATETIME, outcome VARCHAR(16), num_pocketed INTEGER,
            target_pocket VARCHAR(24), cue_scratch BOOLEAN, duration_s FLOAT,
            shot_seconds FLOAT, streak_index INTEGER);
        INSERT INTO sessions (id, mode, drill_target, table_size, notes)
            VALUES (1, 'free_play', 0, '9ft', '');
        INSERT INTO shots (session_id, outcome, num_pocketed, cue_scratch,
            duration_s, shot_seconds, streak_index)
            VALUES (1, 'make', 1, 0, 2.0, 5.0, 1);
    """)
    con.commit()
    con.close()

    from billiards_trainer.db.repository import Repository
    repo = Repository(db_path=db)
    import json
    repo.record_shot(1, outcome="miss", stroke_json=json.dumps({"peak_g": 2.0}))
    summary = repo.session_summary(1)
    assert summary["shots"] == 2   # old row + new row coexist


# --------------------------------------------------------------------------- #
# Controller join: one stroke annotates exactly one shot, only while fresh
# --------------------------------------------------------------------------- #


def test_controller_stroke_join_is_fresh_and_one_shot():
    from billiards_trainer.config import Settings
    from billiards_trainer.db.repository import Repository
    from billiards_trainer.workers.controller import PipelineController

    ctrl = PipelineController(Settings(), Repository(db_path=Path(":memory:")))
    ctrl.on_stroke_metrics({"hit_epoch": time.time(), "peak_g": 3.0})
    first = ctrl._consume_stroke()
    assert '"peak_g": 3.0' in first
    assert ctrl._consume_stroke() == ""      # consumed — can't annotate twice

    ctrl.on_stroke_metrics({"hit_epoch": time.time() - 120.0, "peak_g": 9.9})
    assert ctrl._consume_stroke() == ""      # stale strokes never attach
