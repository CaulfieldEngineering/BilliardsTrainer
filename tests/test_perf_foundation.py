"""Tests for the performance-foundation work: threaded camera grabber,
far-rail rescan gating, cached table mask, and the cached schematic base.

These guard the *behavioural contracts* of the optimizations — semantics that a
future refactor could silently break while everything still "works on my video".
"""

import time

import numpy as np
import pytest

from billiards_trainer.core.geometry import TableModel
from billiards_trainer.core.types import BallClass, Track
from billiards_trainer.detector_strategies import table_polygon_mask
from billiards_trainer.vision.overlay import render_schematic


# --------------------------------------------------------------------------- #
# ThreadedCameraSource
# --------------------------------------------------------------------------- #
class _FakeCap:
    """Stands in for cv2.VideoCapture: delivers a new frame every ~5 ms."""

    def __init__(self, *args):
        self.n = 0
        self.released = False

    def isOpened(self):
        return True

    def set(self, *a):
        return True

    def get(self, prop):
        return 30.0

    def read(self):
        time.sleep(0.005)
        self.n += 1
        return True, np.full((4, 4, 3), self.n % 251, np.uint8)

    def release(self):
        self.released = True


def test_threaded_camera_take_semantics(monkeypatch):
    from billiards_trainer.capture import camera as cam

    monkeypatch.setattr(cam.cv2, "VideoCapture", lambda *a, **k: _FakeCap())
    src = cam.ThreadedCameraSource(0)
    try:
        assert src.fps == 30.0
        # first frame arrives asynchronously
        deadline = time.time() + 2.0
        f = None
        while f is None and time.time() < deadline:
            f = src.read()
            time.sleep(0.002)
        assert f is not None, "grab thread never delivered a frame"
        # take-semantics: a frame is handed out exactly once — an immediate
        # re-read yields None or a NEWER frame, never the same object
        g = src.read()
        assert g is None or g is not f
    finally:
        src.release()
    assert src._thread is not None and not src._thread.is_alive()


def test_threaded_camera_release_without_frames(monkeypatch):
    """release() must not hang even if no frame was ever consumed."""
    from billiards_trainer.capture import camera as cam

    monkeypatch.setattr(cam.cv2, "VideoCapture", lambda *a, **k: _FakeCap())
    src = cam.ThreadedCameraSource(0)
    t0 = time.time()
    src.release()
    assert time.time() - t0 < 3.0


def test_open_source_uses_threaded_camera(monkeypatch):
    """The OpenCV fallback must still be the THREADED source, never a blocking
    one. _ffmpeg_camera is stubbed out so this never touches real hardware —
    open_source now prefers an ffmpeg-owned device where one is present."""
    from billiards_trainer.capture import camera as cam

    monkeypatch.setattr(cam, "_ffmpeg_camera", lambda *a, **k: None)
    monkeypatch.setattr(cam.cv2, "VideoCapture", lambda *a, **k: _FakeCap())
    src = cam.open_source("0")
    try:
        assert isinstance(src, cam.ThreadedCameraSource)
        assert src.is_live
    finally:
        src.release()


def test_open_source_prefers_ffmpeg_owned_device(monkeypatch):
    """When a device can be owned by ffmpeg, the live camera goes through it —
    that is what keeps the RECORDING out of Python's timing entirely."""
    from billiards_trainer.capture import camera as cam

    sentinel = object()
    monkeypatch.setattr(cam, "_ffmpeg_camera", lambda *a, **k: sentinel)
    assert cam.open_source("0") is sentinel


# --------------------------------------------------------------------------- #
# Far-rail rescan gating
# --------------------------------------------------------------------------- #
def _stub_onnx_detector(monkeypatch, calls: list):
    """An OnnxModelStrategy whose _infer records calls and returns one box."""
    from billiards_trainer.detector_strategies.onnx_model import OnnxModelStrategy

    det = OnnxModelStrategy.__new__(OnnxModelStrategy)
    det._conf, det._iou = 0.25, 0.45
    det._nc = 1

    def fake_infer(image, ox=0, oy=0):
        calls.append(image.shape)
        return [(320.0 + ox, 300.0 + oy, 12.0, 0.9, 0)]

    monkeypatch.setattr(det, "_infer", fake_infer)
    return det


def test_far_rail_rescan_toggle(monkeypatch):
    calls: list = []
    det = _stub_onnx_detector(monkeypatch, calls)
    frame = np.full((480, 640, 3), (70, 120, 60), np.uint8)

    det.far_rail_rescan = True
    det.detect(frame, None)
    assert len(calls) == 2, "rescan ON must run a second (far-rail) inference"
    assert calls[1][0] == int(480 * 0.60)

    calls.clear()
    det.far_rail_rescan = False
    det.detect(frame, None)
    assert len(calls) == 1, "rescan OFF must run exactly one inference"

    # per-call override (used by the offline labeller): does NOT mutate the
    # instance and wins over the instance setting
    calls.clear()
    det.detect(frame, None, rescan=True)
    assert len(calls) == 2, "rescan=True override must force the second pass"
    assert det.far_rail_rescan is False, "per-call override must not stick"


def test_pipeline_wires_far_rail_rescan_setting():
    from billiards_trainer.config import Settings
    from billiards_trainer.vision.pipeline import Pipeline

    s = Settings()
    s.balls.far_rail_rescan = False
    pipe = Pipeline(s, source="test")
    strat = pipe._strategy
    if strat is None or not hasattr(strat, "far_rail_rescan"):
        pytest.skip("no ONNX strategy available in this environment")
    try:
        assert strat.far_rail_rescan is False
    finally:
        # the strategy is a process-wide singleton — restore the default so
        # later tests (and dev sessions) aren't left with the rescan disabled
        strat.far_rail_rescan = True


# --------------------------------------------------------------------------- #
# table_polygon_mask cache
# --------------------------------------------------------------------------- #
class _CalibStub:
    def __init__(self, corners):
        self.corners = np.asarray(corners, np.float32)


def test_table_polygon_mask_cached_and_invalidated():
    c1 = _CalibStub([[10, 10], [90, 10], [90, 90], [10, 90]])
    m1 = table_polygon_mask((100, 100, 3), c1)
    m2 = table_polygon_mask((100, 100, 3), c1)
    assert m1 is m2, "same corners + size must return the cached mask"
    assert m1[50, 50] == 255 and m1[5, 5] == 0

    c2 = _CalibStub([[20, 20], [80, 20], [80, 80], [20, 80]])
    m3 = table_polygon_mask((100, 100, 3), c2)
    assert m3 is not m1, "changed corners must rebuild the mask"
    assert m3[15, 15] == 0

    # no calibration -> full-frame mask, never cached to a stale polygon
    m4 = table_polygon_mask((50, 50, 3), None)
    assert m4.min() == 255


# --------------------------------------------------------------------------- #
# Schematic base cache
# --------------------------------------------------------------------------- #
def test_schematic_base_cache_isolated():
    table = TableModel.from_rect((400, 760), 40, 0.045)
    tracks = [Track(id=1, x=200.0, y=300.0, radius=9.0, cls=BallClass.CUE, number=0)]
    a = render_schematic(table, tracks)
    b = render_schematic(table, [])
    # mutating one returned frame must not leak into the next render
    a[:] = 0
    c = render_schematic(table, [])
    assert c.any(), "cached base was corrupted by a caller mutation"
    assert np.array_equal(b, c), "identical inputs must render identically"
    assert a.shape == (table.height, table.width, 3)


def test_settings_survive_a_utf8_bom(tmp_path):
    """A BOM must not wipe the user's configuration.

    Windows editors and PowerShell's `Set-Content -Encoding utf8` both prepend
    one. Plain utf-8 decoding rejects it, and the loader used to answer that by
    silently returning defaults — losing the recording directory, camera
    rotation and encoder quality while the app looked perfectly healthy.
    """
    from billiards_trainer.config import Settings

    p = tmp_path / "settings.json"
    s = Settings()
    s.camera.rotation = 270
    s.save(p)
    p.write_bytes(b"\xef\xbb\xbf" + p.read_bytes())      # prepend a BOM

    assert Settings.load(p).camera.rotation == 270


def test_settings_report_a_corrupt_file(tmp_path, caplog):
    """Falling back to defaults is acceptable; doing it SILENTLY is not."""
    import logging

    from billiards_trainer.config import Settings

    p = tmp_path / "settings.json"
    p.write_text("{ this is not json", encoding="utf-8")
    with caplog.at_level(logging.ERROR):
        Settings.load(p)
    assert any("DEFAULTS" in r.message or "not valid JSON" in r.message
               for r in caplog.records), "a corrupt settings file must be reported"
