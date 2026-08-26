"""Regression tests for the v0.1.5 reliability work: calibration persistence,
shot-clock non-interference, felt picking, and tracker robustness."""

import cv2
import numpy as np

from billiards_trainer.capture.camera import DemoSource
from billiards_trainer.config import FeltSettings, Settings, ShotClockSettings
from billiards_trainer.core.types import BallClass, Detection
from billiards_trainer.game.shot_clock import ShotClock
from billiards_trainer.vision.calibration import CalibrationManager
from billiards_trainer.vision.felt import felt_from_point
from billiards_trainer.vision.tracking import BallTracker


# ---- calibration persistence --------------------------------------------- #
def _lock(mgr: CalibrationManager, frame, settings) -> bool:
    """Feed the same frame until the consensus lock lands (or give up)."""
    for _ in range(settings.table.calib_consensus_frames + 2):
        if mgr.calibrate(frame, settings):
            return True
    return False


def test_calibration_save_and_restore(tmp_path):
    frame = DemoSource().read()
    settings = Settings()
    mgr = CalibrationManager()
    assert _lock(mgr, frame, settings)
    path = tmp_path / "calibration.json"
    mgr.save(path, "demo", frame.shape, settings)
    assert path.exists()

    restored = CalibrationManager()
    assert restored.try_load(path, "demo", frame.shape)
    assert restored.is_calibrated
    assert np.allclose(restored.calib.corners, mgr.calib.corners, atol=0.5)
    assert restored.calib.dst_size == mgr.calib.dst_size


def test_calibration_rejects_mismatched_source(tmp_path):
    frame = DemoSource().read()
    settings = Settings()
    mgr = CalibrationManager()
    assert _lock(mgr, frame, settings)
    path = tmp_path / "calibration.json"
    mgr.save(path, "demo", frame.shape, settings)

    other = CalibrationManager()
    assert not other.try_load(path, "0", frame.shape)            # different source
    assert not other.try_load(path, "demo", (10, 10, 3))         # different resolution


def test_calibration_per_source_no_clobber(tmp_path):
    """Locking a DIFFERENT source (e.g. the laptop camera) must not overwrite
    the pool table's saved lock — the exact bug that rendered a wall-quad."""
    frame = DemoSource().read()
    settings = Settings()
    path = tmp_path / "calibration.json"
    mgr = CalibrationManager()
    assert _lock(mgr, frame, settings)
    mgr.save(path, "table.mp4", frame.shape, settings)
    table_corners = mgr.calib.corners.copy()

    # a second source saves its own (different) lock into the same file
    mgr.save(path, "0", frame.shape, settings)

    restored = CalibrationManager()
    assert restored.try_load(path, "table.mp4", frame.shape), \
        "the table's lock must survive another source saving"
    assert np.allclose(restored.calib.corners, table_corners, atol=0.5)
    assert CalibrationManager().try_load(path, "0", frame.shape)


def test_calibration_legacy_single_entry_still_loads(tmp_path):
    """A v1 (single-entry) calibration.json from an older build must restore."""
    import json
    frame = DemoSource().read()
    settings = Settings()
    mgr = CalibrationManager()
    assert _lock(mgr, frame, settings)
    path = tmp_path / "calibration.json"
    mgr.save(path, "demo", frame.shape, settings)
    # rewrite as the legacy flat format
    doc = json.loads(path.read_text())["entries"]["demo"]
    path.write_text(json.dumps(doc))
    assert CalibrationManager().try_load(path, "demo", frame.shape)


def test_calibration_consensus_rejects_occluded_frame():
    """One frame with a corner pulled far inward (someone leaning over the
    table) must not shift the consensus lock."""
    src = DemoSource()
    frame = src.read()
    settings = Settings()
    n = settings.table.calib_consensus_frames
    assert n >= 3, "consensus must be on for this test"

    # clean lock for reference
    ref = CalibrationManager()
    assert _lock(ref, frame, settings)

    # same frames, but one heavily occluded frame in the middle
    occluded = frame.copy()
    h, w = occluded.shape[:2]
    occluded[:, : int(w * 0.45)] = (28, 24, 20)  # dark figure over the left half
    mgr = CalibrationManager()
    locked = False
    frames = [frame, frame, occluded] + [frame] * (n + 2)
    for f in frames:
        if mgr.calibrate(f, settings):
            locked = True
            break
    assert locked
    rmse = float(np.sqrt(np.mean(np.sum(
        (mgr.calib.corners - ref.calib.corners) ** 2, axis=1))))
    assert rmse < 10.0, f"occluded frame shifted the lock by {rmse:.1f}px"


def test_validate_against_rejects_stale_lock():
    """A restored lock that doesn't match the live frame must be rejected."""
    src = DemoSource()
    frame = src.read()
    settings = Settings()
    mgr = CalibrationManager()
    assert _lock(mgr, frame, settings)
    assert mgr.validate_against(frame, settings), "a fresh lock must validate"
    # shift the lock far away — simulating a lock saved for a different scene
    mgr.calib.corners = mgr.calib.corners + 200.0
    assert not mgr.validate_against(frame, settings)


# ---- shot clock must not interfere with sandbox -------------------------- #
def test_shot_clock_disabled_is_noop():
    clock = ShotClock(ShotClockSettings(enabled=False, seconds=30))
    clock.start(0.0)
    assert not clock.running
    assert clock.remaining(100.0) == 30.0      # never counts down
    assert clock.poll(100.0) == ""             # never fires warn/expire
    assert not clock.is_warning(100.0)


def test_shot_clock_enabled_counts_down():
    clock = ShotClock(ShotClockSettings(enabled=True, seconds=10, warn_seconds=3))
    clock.start(0.0)
    assert clock.running
    assert clock.remaining(4.0) == 6.0
    assert clock.poll(0.0) == "start"      # countdown announced (Joe's cue)
    assert clock.poll(8.0) == "warn"
    assert clock.poll(11.0) == "expired"


def test_shot_clock_beep_cadence():
    """Joe's audio spec: single warn beep at 10 s left, one tick each at
    3-2-1, buzz at 0 — each edge fires exactly once however often we poll."""
    clock = ShotClock(ShotClockSettings(enabled=True, seconds=30, warn_seconds=10))
    clock.start(0.0)
    assert clock.poll(0.0) == "start"      # countdown announced (Joe's cue)
    assert clock.poll(19.5) == ""          # 10.5 s left — nothing yet
    assert clock.poll(20.0) == "warn"      # 10 s left — the heads-up beep
    assert clock.poll(20.5) == ""          # warn is one-shot
    assert clock.poll(26.5) == ""          # 3.5 s — cadence not started
    assert clock.poll(27.0) == "tick"      # 3
    assert clock.poll(27.5) == ""          # same second, no re-beep
    assert clock.poll(28.1) == "tick"      # 2
    assert clock.poll(29.05) == "tick"     # 1
    assert clock.poll(29.6) == ""
    assert clock.poll(30.0) == "expired"   # the buzz
    assert not clock.running
    assert clock.poll(31.0) == ""          # expired is one-shot too


def test_shot_clock_warn_inside_cadence_no_double_beep():
    """A warn threshold at/below 3 s must not stack a tick on the same second."""
    clock = ShotClock(ShotClockSettings(enabled=True, seconds=10, warn_seconds=3))
    clock.start(0.0)
    assert clock.poll(0.0) == "start"      # countdown announced (Joe's cue)
    assert clock.poll(7.2) == "warn"       # 2.8 s left
    assert clock.poll(7.3) == ""           # tick for '3' suppressed
    assert clock.poll(8.1) == "tick"       # 1.9 s -> the '2' tick fires
    assert clock.poll(9.05) == "tick"      # 1
    assert clock.poll(10.0) == "expired"


def test_shot_clock_sound_cues_defined(monkeypatch):
    """play() spawns one beep thread per known edge and ignores unknown edges
    (patched Thread — the suite must stay silent)."""
    from billiards_trainer.ui import sounds

    started = []

    class _FakeThread:
        def __init__(self, target=None, args=(), **_kw):
            started.append((target, args))

        def start(self):
            pass

    monkeypatch.setattr(sounds.threading, "Thread", _FakeThread)
    for edge in ("warn", "tick", "expired"):
        sounds.play(edge)
    assert len(started) == 3
    sounds.play("bogus")
    sounds.play("")
    assert len(started) == 3               # unknown edges are silent no-ops


def test_shot_clock_wav_render_for_macos():
    """The macOS playback path renders each cue to a valid mono 16-bit WAV
    (played via afplay); rendering is platform-independent and cached."""
    import wave

    from billiards_trainer.ui import sounds

    path = sounds._render_wav(sounds._CUES["expired"])
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        dur_ms = 1000 * w.getnframes() / w.getframerate()
    assert abs(dur_ms - (260 + 520)) < 5           # the two buzz tones
    assert sounds._render_wav(sounds._CUES["expired"]) == path  # cached


# ---- ball-height parallax correction ------------------------------------- #
def _synthetic_plane_camera(cam_center, img_w=1920, img_h=1080, f=1400.0):
    """Ground-truth rect-plane -> image homography for a pinhole camera at
    ``cam_center`` (rect coords, z above the cloth), aimed at the table centre."""
    fwd = np.array([487 / 2, 1053 / 2, 0], float) - np.asarray(cam_center, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, (0, 0, 1))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd])
    t = -R @ np.asarray(cam_center, float)
    K = np.array([[f, 0, img_w / 2], [0, f, img_h / 2], [0, 0, 1.0]])
    G = K @ np.column_stack([R[:, 0], R[:, 1], t])
    return G / G[2, 2]


def test_camera_position_recovery_from_homography():
    from billiards_trainer.core.rectify import estimate_camera_position
    for true_c in [(283.0, 1900.0, 700.0), (100.0, 1600.0, 500.0),
                   (500.0, 2400.0, 420.0)]:
        G = _synthetic_plane_camera(true_c)
        c = estimate_camera_position(G, (1920, 1080))
        assert c is not None
        assert np.allclose(c, true_c, atol=1.0), (true_c, c)


def test_parallax_correction_pulls_projected_balls_toward_camera():
    """The homography maps the CLOTH plane; ball centres sit one radius above
    it, so projections land radially outward from the camera (rail balls were
    rendered IN the rail). The pipeline must slide each point back along the
    camera ray — and leave points untouched when the flag is off."""
    import types

    from billiards_trainer.core.geometry import TableModel, expected_ball_radius_px
    from billiards_trainer.core.rectify import project_points
    from billiards_trainer.core.types import Detection
    from billiards_trainer.vision.pipeline import Pipeline

    cam = np.array([283.0, 1900.0, 480.0])       # like the real rig: end-on, ~55" up
    G = _synthetic_plane_camera(cam)             # rect -> image
    calib = types.SimpleNamespace(
        H=np.linalg.inv(G), Hinv=G,
        table=TableModel.from_rect((567, 1053), 40, nose_inset_frac=0.055))
    shape = (1080, 1920, 3)

    p_far = np.array([283.0, 80.0])              # far-rail ball, worst case
    img_pt = project_points(p_far[None, :], G)[0]
    det = Detection(float(img_pt[0]), float(img_pt[1]), 8.0, (200, 200, 200))

    s = Settings()
    s.camera.overhead = False  # this test exercises the OBLIQUE parallax correction
    pipe = Pipeline(s, source="")
    out = pipe._project_raw_to_rect([det], calib, shape)[0]
    r = expected_ball_radius_px(calib.table, s.table.size)
    d_before = np.linalg.norm(p_far - cam[:2])
    d_after = np.linalg.norm(np.array([out.x, out.y]) - cam[:2])
    # pulled toward the camera ground point by radius/height of the distance
    assert d_after < d_before
    assert np.isclose(d_after / d_before, 1.0 - r / cam[2], atol=1e-3)
    shift = np.hypot(out.x - p_far[0], out.y - p_far[1])
    assert shift > 2 * r                          # the far-rail bias is REAL money

    # escape hatch: flag off = untouched projection
    s.balls.parallax_correction = False
    pipe2 = Pipeline(s, source="")
    out2 = pipe2._project_raw_to_rect([det], calib, shape)[0]
    assert np.isclose(out2.x, p_far[0], atol=1e-6)
    assert np.isclose(out2.y, p_far[1], atol=1e-6)


# ---- click-to-pick felt -------------------------------------------------- #
def test_felt_from_point_keys_on_clicked_pixel():
    hsv = np.full((300, 300, 3), (100, 150, 200), np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    fs = felt_from_point(bgr, 150, 150, sensitivity=82)
    assert isinstance(fs, FeltSettings)
    assert abs(fs.picked_hsv[0] - 100) <= 2
    assert fs.h_min <= 100 <= fs.h_max


# ---- tracker robustness -------------------------------------------------- #
def test_tracker_handles_off_frame_detection():
    tr = BallTracker(min_hits=2)
    # negative / out-of-frame coordinates must not crash association
    tr.update([Detection(-40, -40, 10, cls=BallClass.SOLID)], 400)
    out = tr.update([Detection(-40, -40, 10, cls=BallClass.SOLID)], 400)
    assert isinstance(out, list)
    # then it vanishes — tracker ages it out without error once the keep-alive
    # budget (max_misses, raised to survive ~1s occlusion) is exceeded.
    for _ in range(tr.max_misses + 2):
        tr.update([], 400)
    assert tr.tracks == []
