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
from billiards_trainer.capture.videowriter import FfmpegWriter, _CfrRetimer


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


# --------------------------------------------------------------------------- #
# CFR retiming — the actual frame-rate-jitter fix. A rigid wall-clock pacer
# beats against the camera clock and periodically dups/drops a frame (visible
# micro-stutter); slot assignment from capture timestamps must not.
# --------------------------------------------------------------------------- #

def test_retimer_steady_source_with_jitter_never_dups_or_drops():
    """30fps arrivals with ±40% frame-period jitter (a loaded USB dongle) must
    map 1:1 — once the phase servo has centred (first ~2s), every frame is
    emitted exactly once: no periodic beat, no micro-stutter."""
    fps = 30.0
    rt = _CfrRetimer(fps)
    rng = np.random.default_rng(3)
    anomalies_after_settle = 0
    for k in range(3000):   # 100 seconds
        ts = k / fps + float(rng.uniform(-0.4, 0.4)) / fps
        n = rt.advance(ts)
        if k >= 100 and n != 1:
            anomalies_after_settle += 1
    assert anomalies_after_settle == 0
    assert abs(rt.emitted - 3000) <= 3   # servo convergence may slip a frame or two


def test_retimer_clock_skew_does_not_beat():
    """A camera clock 0.1% fast vs the declared fps (typical crystal drift)
    causes at most the ~3 genuinely surplus frames over 100s — not a periodic
    stutter. (The old pacer turned any skew into regular dup/drop pairs.)"""
    fps = 30.0
    rt = _CfrRetimer(fps)
    drops = sum(1 for k in range(3000) if rt.advance(k / (fps * 1.001)) == 0)
    assert drops <= 4  # only the true rate surplus, nothing periodic


def test_retimer_bridges_real_gap_with_duplicates():
    fps = 30.0
    rt = _CfrRetimer(fps)
    for k in range(30):
        assert rt.advance(k / fps) == 1
    # capture stalls 0.5s, then resumes: the gap is filled so audio stays synced
    n = rt.advance(30 / fps + 0.5)
    assert 14 <= n <= 17


def test_retimer_caps_runaway_gap_fill():
    """A gap longer than MAX_GAP_S resyncs the timeline instead of flooding the
    encoder with seconds of frozen duplicates."""
    fps = 30.0
    rt = _CfrRetimer(fps)
    rt.advance(0.0)
    n = rt.advance(10.0)   # 10s dead air
    assert n <= int(_CfrRetimer.MAX_GAP_S * fps)


def test_retimer_drops_same_slot_burst():
    fps = 30.0
    rt = _CfrRetimer(fps)
    assert rt.advance(0.0) == 1
    assert rt.advance(0.001) == 0     # second frame in the same slot: dropped
    assert rt.advance(1 / fps) == 1   # next slot proceeds normally


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
