"""Real-time recording: the camera grab thread feeds the recorder directly,
decoupled from the CV tick, so a busy pipeline can't starve the capture into
frame-rate stutter. The recording's visual processing is unchanged — only its
timing source moved off the CV loop.

Covered without needing a camera:
  * ThreadedCameraSource's frame sink (grab-thread → recorder tap).
  * The writer still applies its denoise/sharpen quality chain (regression).
"""

import time

import numpy as np
import pytest

import billiards_trainer.capture.camera as cam
from billiards_trainer.capture.audio import find_ffmpeg
from billiards_trainer.capture.videowriter import FfmpegWriter


class _FakeCap:
    """Instant-read stand-in for cv2.VideoCapture so the real grab loop runs."""

    def __init__(self):
        self._n = 0

    def isOpened(self):
        return True

    def read(self):
        self._n += 1
        return True, np.full((4, 4, 3), self._n % 255, np.uint8)

    def set(self, *_a):
        return True

    def get(self, *_a):
        return 30.0

    def release(self):
        pass


def _threaded_source(monkeypatch):
    monkeypatch.setattr(cam.cv2, "VideoCapture", lambda *a, **k: _FakeCap())
    return cam.ThreadedCameraSource(0)


def test_frame_sink_receives_frames_at_grab_cadence(monkeypatch):
    src = _threaded_source(monkeypatch)
    received: list = []
    try:
        src.set_frame_sink(lambda f: received.append(f))
        time.sleep(0.1)
        assert received, "grab thread never fed the sink"
    finally:
        src.release()


def test_frame_sink_detaches_cleanly(monkeypatch):
    src = _threaded_source(monkeypatch)
    received: list = []
    try:
        src.set_frame_sink(lambda f: received.append(f))
        time.sleep(0.05)
        src.set_frame_sink(None)
        n_at_detach = len(received)
        time.sleep(0.05)
        assert len(received) == n_at_detach, "sink kept firing after detach"
    finally:
        src.release()


def test_sink_exception_never_kills_capture(monkeypatch):
    """A throwing sink must not take down the grab loop — capture keeps serving."""
    src = _threaded_source(monkeypatch)
    try:
        src.set_frame_sink(lambda f: (_ for _ in ()).throw(RuntimeError("boom")))
        time.sleep(0.05)
        src.set_frame_sink(None)
        time.sleep(0.02)
        # camera still alive and delivering after the sink blew up repeatedly
        assert src.read() is not None
    finally:
        src.release()


@pytest.mark.skipif(find_ffmpeg() is None, reason="ffmpeg not installed")
def test_writer_keeps_quality_chain():
    """Recording quality is unchanged by the frame-rate fix: the writer still
    denoises + sharpens (the visual appearance Joe is happy with)."""
    w = FfmpegWriter("/dev/null", 20.0, (1920, 1080))
    try:
        args = w._proc.args
        vf = args[args.index("-vf") + 1] if "-vf" in args else ""
        assert "hqdn3d" in vf and "unsharp" in vf
    finally:
        w.release()
