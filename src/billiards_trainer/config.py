"""Application paths and persisted settings.

Settings live as a single JSON document in the per-user app data directory so
they survive reinstalls/updates. The schema is a set of nested dataclasses; load
is forward/backward compatible (unknown keys ignored, missing keys defaulted),
so adding a field never breaks an old settings file.
"""

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger("config")


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def _user_data_dir() -> Path:
    """Per-user, writable application data directory.

    Windows:  %LOCALAPPDATA%\\BilliardsTrainer
    macOS:    ~/Library/Application Support/BilliardsTrainer
    Linux:    $XDG_DATA_HOME or ~/.local/share/BilliardsTrainer
    """
    folder = "BilliardsTrainer"
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base) / folder


APP_DIR = _user_data_dir()
SETTINGS_PATH = APP_DIR / "settings.json"
DB_PATH = APP_DIR / "billiards.db"
MODELS_DIR = APP_DIR / "models"
LOGS_DIR = APP_DIR / "logs"
EXPORTS_DIR = APP_DIR / "exports"
CALIBRATION_PATH = APP_DIR / "calibration.json"
SHOTLOG_PATH = LOGS_DIR / "shots.jsonl"
SUPABASE_CONFIG_PATH = APP_DIR / "supabase.json"


def load_supabase_config() -> dict | None:
    """Return Supabase credentials if configured, else None (sync stays a no-op).

    Looks first at env vars (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY), then at
    ``supabase.json`` in the app data dir. The app picks these up on launch.
    """
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not (url and key):
        try:
            data = json.loads(SUPABASE_CONFIG_PATH.read_text(encoding="utf-8"))
            url = url or data.get("url", "")
            key = key or data.get("service_role_key", "")
        except (OSError, json.JSONDecodeError):
            pass
    if url and key:
        return {"url": url.rstrip("/"), "key": key}
    return None


def ensure_dirs() -> None:
    for d in (APP_DIR, MODELS_DIR, LOGS_DIR, EXPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def resource_path(*parts: str) -> Path:
    """Resolve a bundled resource, working both from source and a PyInstaller exe.

    PyInstaller unpacks data files into ``sys._MEIPASS`` at runtime.
    """
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent.parent))
    return base.joinpath(*parts)


# --------------------------------------------------------------------------- #
# Settings schema
# --------------------------------------------------------------------------- #
@dataclass
class FeltSettings:
    # HSV range for the playing-surface colour. Defaults span green (~60) through
    # blue-green (~100) felt and were validated to recover the reference capture's
    # corners to <12 px RMSE (OpenCV H is 0..180). Re-tune via the colour picker.
    h_min: int = 55
    h_max: int = 105
    s_min: int = 40
    s_max: int = 255
    v_min: int = 50
    v_max: int = 255
    sensitivity: int = 82  # 0 (strict) .. 100 (loose) — expands the picked range
    picked_hsv: list[int] = field(default_factory=lambda: [76, 130, 200])


@dataclass
class RectifySettings:
    pad_px: int = 40
    margin_scale: float = 1.0
    # Forced playing-surface aspect (long:short). 2.0 == regulation pool table.
    aspect: float = 2.0


# Regulation playing-surface short side (cushion nose to nose), inches.
_BED_SHORT_IN = {"9ft": 50.0, "8ft": 46.0, "7ft": 39.0}


@dataclass
class TableSettings:
    size: str = "9ft"           # 9ft | 8ft | 7ft (display/scale only)
    pocket_radius_frac: float = 0.045  # pocket capture radius as frac of short side
    # Felt detection finds the edge of the BLUE CLOTH — the outer edge of the
    # cushion top. Balls rebound off the cushion NOSE, which sits inboard of
    # that edge by the cushion width. A physical measurement (Joe's table: 2"),
    # converted to a fraction of the detected felt at calibration time.
    cushion_inset_in: float = 2.0
    # Legacy fractional inset — superseded by cushion_inset_in; kept so old
    # settings files load cleanly.
    nose_inset_frac: float = 0.055

    def computed_nose_inset_frac(self) -> float:
        """Cushion inset as a fraction of the detected felt SHORT side.

        felt_short = bed_short + 2×inset, so frac = inset / (bed + 2×inset)."""
        inset = max(0.0, float(self.cushion_inset_in))
        bed = _BED_SHORT_IN.get(self.size, 50.0)
        return inset / (bed + 2.0 * inset) if inset > 0 else 0.0
    auto_relock: bool = True           # re-detect automatically if the table shifts
    persist_calibration: bool = True   # save/reuse the locked table across launches
    # The lock is the per-corner MEDIAN over this many consecutive successful
    # felt detections (outlier frames — someone leaning over the table — are
    # rejected), so a single occluded frame can never become the session's lock.
    calib_consensus_frames: int = 5


@dataclass
class BallSettings:
    # auto = YOLO when weights are present, else classical. 'classical' forces
    # Hough+colour; 'yolo' forces the net (falling back if deps/weights missing).
    backend: str = "auto"  # auto | classical | yolo
    # Live detector. 'auto' (default) picks the best available: a trained YOLO
    # .onnx model in the models dir if present (the reliable path), else the
    # cue-ball heuristic. Runs on the RAW frame; results project into the
    # bird's-eye for display. Specific names (onnx_*, cue_ball_white, …) force one.
    live_strategy: str = "auto"
    # Temporal median preprocessing (median of the last 3 frames). OFF: a fast
    # ball occupies 3 different spots across those frames, so the median ERASES
    # it — measured on Joe's table as balls "not moving" for the first 6-12
    # inches of travel. The trained model doesn't need the noise suppression;
    # the tracker's settled-lock handles resting-ball shimmer.
    temporal_median: bool = False
    temporal_median_frames: int = 3
    # ONNX detector: run a SECOND inference pass over the foreshortened far-rail
    # region (top of the frame) and merge. Recovers tiny far balls at the cost of
    # doubling per-frame GPU work — turn off to halve detector latency when the
    # camera already resolves the far rail well.
    far_rail_rescan: bool = True
    # Ball-height parallax correction: the homography maps the cloth plane, but a
    # ball's centre sits one radius above it, so an oblique camera projects centres
    # radially outward (rail balls rendered IN the rail). With the camera position
    # recovered from the homography, each point slides back along the camera ray.
    # Escape hatch only — no UI knob.
    parallax_correction: bool = True
    # One-time migration marker. Bumped by Settings.load() after it moves users off
    # 'legacy' — which used to be the ONLY selectable detector in shipped builds
    # (frozen strategy discovery was broken). 0 = not yet migrated.
    detector_migration: int = 0
    # Ball radius bounds as a fraction of the rectified short side (the playing
    # WIDTH). Regulation: a 2.25" ball on a 50"-wide bed => radius ≈ 0.0225·W.
    # The band is kept TIGHT around that real ratio so noise/shadows of the wrong
    # size are rejected outright — real pool balls are all one diameter, so a
    # detector finding "balls" of wildly varying sizes is finding artefacts.
    min_radius_frac: float = 0.016  # ~0.71× regulation — smallest plausible ball
    max_radius_frac: float = 0.034  # ~1.5× regulation — rejects oversized blobs
    # Physical-size prior tolerance (fraction): a rectified detection whose radius
    # differs from the known ball radius by more than this is rejected (pocket
    # shadows too big, speckle too small). 0 disables. Default is generous (0.55)
    # because the blob detector systematically UNDER-reports radius vs the geometric
    # ideal (~7.5px measured vs ~11px geometric on a 9ft bed), so a tight band would
    # kill real balls. Tighten in the tuning sandbox once detector radius is
    # calibrated. (Rendering uses the true geometric size regardless.)
    size_prior_tol: float = 0.55
    detect_param2: int = 18         # Hough accumulator threshold (lower => more circles)
    cue_speed_strike: float = 14.0  # px/frame on rectified view that counts as a strike
    stop_speed: float = 1.2         # px/frame below which a ball is "stopped"
    # Optional YOLO backend: weights are auto-fetched from this URL into the
    # models dir on first use (no turnkey public pool-ball model exists, so this
    # is blank by default — see docs/BLOCKERS.md for how to point it at one).
    yolo_weights_url: str = ""
    yolo_conf: float = 0.25


@dataclass
class DetectionSettings:
    """Hard-evidence gates for shot detection. Defaults are deliberately strict to
    kill false positives from lighting flicker / compression noise; all are
    tunable from Settings -> Detection for a specific table/lighting."""

    # Master switch for AI ball/shot detection (Settings window; no top-bar toggle).
    enabled: bool = True
    warmup_seconds: float = 6.0       # ignore shots right after Start (stabilise)
    cooldown_seconds: float = 4.0     # min gap between counted shots
    # Shots are gated on MOTION ENERGY (mean frame-to-frame change in the playing
    # area) — far more robust than fragile per-ball velocity during fast motion.
    # motion energy = % of playing-area pixels that changed significantly between
    # frames (a moving ball is ~0.5-1.0; an idle table ~0.1).
    motion_active: float = 0.4        # above this counts as "table active"
    motion_quiet: float = 0.2         # below this counts as "settled"
    strike_frames: int = 6            # consecutive active frames to start a shot
    min_travel_px: float = 120.0      # a ball must travel this far for a shot to count
    pocket_frames: int = 12           # consecutive frames in a pocket to count a pot
    require_cue: bool = True          # no cue ball identified => no shot detection
    confidence_floor: float = 0.45    # drop ball detections below this score
    # Render/track floor: when auto-detection is ON, a detection must clear THIS
    # (much stricter) score to be drawn or tracked. The rule is "show nothing
    # rather than something wrong" — a wrong-coloured, wrong-sized phantom is
    # worse than a blank table. For YOLO this is a calibrated probability; for
    # classical it culls all but the cleanest non-felt blobs.
    render_floor: float = 0.85
    manual_confirm: bool = False      # auto-detect SUGGESTS; user commits make/miss
    # Feature flag: normally the Detection toggle is locked off until a pool model
    # is present (no fake detections). Turn this on to experiment with the
    # classical detector on a real feed without a model. Behaviour change, no
    # rebuild — flip it in Settings and it applies live.
    allow_without_model: bool = False
    # Multi-modal evidence fusion: combine motion energy + optical-flow activity +
    # background-subtraction foreground into one weighted "activity" score, so a
    # single noisy signal (e.g. a flickering highlight) can't trip a shot on its
    # own. Weights/threshold are tunable; presets set sensible bundles.
    # DEFERRED: off by default — shot detection (M5+) is gated until cue-ball
    # tracking (M2) is solid, and the Farneback flow + MOG2 it needs are the
    # heaviest ops in the per-frame loop. Off = the real-time budget goes to the
    # detector. Re-enable once shot detection is back on the roadmap.
    use_fusion: bool = False
    # Measured on real ball motion vs a flickering specular highlight:
    #   fg: 0.74 vs 0.013  (50x separation — the strong discriminator)
    #   flow: 1.9 vs 10.5  (HIGHER for flicker — misleading, so weighted ~0)
    #   motion: 0.6 vs 0.8 (overlaps — weak on its own)
    # So bgsub foreground dominates, motion corroborates, flow barely counts.
    w_motion: float = 0.30
    w_flow: float = 0.05
    w_fg: float = 0.65
    fusion_active: float = 0.45       # fused activity (0..1) needed to be "active"
    preset: str = "balanced"          # conservative | balanced | aggressive


@dataclass
class ShotClockSettings:
    enabled: bool = False
    seconds: int = 30
    warn_seconds: int = 10
    audio: bool = True
    auto_reset_on_shot: bool = True


@dataclass
class UiSettings:
    # Rule-of-thirds alignment grid over the live camera view (Settings toggle)
    # — for squaring the physical camera mount to the table.
    alignment_grid: bool = False
    # Small feed-stats chip in the corner of the camera view: container
    # resolution, ACTIVE picture resolution (the HD-vs-480p truth), fps.
    feed_stats: bool = True
    theme: str = "dark"
    accent: str = "#3DDC97"      # mint/green — nods to the felt
    show_trajectories: bool = True
    show_ball_ids: bool = True
    show_overlays: bool = True   # master toggle for all detection overlays
    debug_overlay: bool = False  # draw raw detections + shot-state diagnostics
    schematic_birdseye: bool = True  # clean rendered overhead vs warped camera
    mirror_preview: bool = False
    # Render balls in their MEASURED mean colour (a blue ball looks blue), with a
    # neutral grey "?" when the class is uncertain — instead of a fixed per-class
    # palette that made every solid look yellow and every dark blob look black.
    measured_ball_colors: bool = True
    # Draw every ball on the bird's-eye at its KNOWN physical radius (2.25" ball on
    # the configured bed) instead of the detector's per-frame radius — kills the
    # "balls are all different sizes on the overhead" wobble. The detector still
    # provides position; rendering owns the size.
    normalize_ball_size: bool = True
    # Debug: show the detector's RAW radius instead of the normalized one.
    show_raw_detection_size: bool = False


@dataclass
class UpdateSettings:
    auto_check: bool = True
    last_check_iso: str = ""
    skip_version: str = ""


@dataclass
class PoseSettings:
    enabled: bool = False


@dataclass
class CueSettings:
    """Bluetooth cue-stroke sensor (JINOU JO-BEC12-2 IMU on the cue butt).

    Fully optional: when disabled (default) or when no sensor/radio/bleak is
    present, the app behaves exactly as before. The impact floor is the ONLY
    tuning knob deliberately exposed — the video-validated signature gates in
    cue/analysis.py do the real work of separating hits from handling."""

    enabled: bool = False
    address: str = ""       # last-connected sensor MAC (preferred in scans)
    impact_g: float = 1.6   # impact candidate floor in g (soft pokes ≈ 1.7 g)


@dataclass
class ColorCorrectionSettings:
    """Automatic-or-manual colour correction of the raw frame before detection.

    ``auto`` is tuned for a pool table: it white-balances off the brightest pixels
    (the white cue ball / table lights are a reliable neutral reference), which is
    robust on a scene dominated by green felt — where a naive grey-world balance
    would wrongly try to neutralise the felt itself. ``manual`` exposes per-channel
    gains + saturation; ``off`` passes the frame through untouched."""

    # Default OFF — show the true camera image. The auto white-patch balance can
    # over-warm/‑cool real feeds, so it's opt-in via the COLOUR control on the
    # live view rather than a silent default.
    mode: str = "off"           # off | auto | manual
    auto_strength: float = 1.0  # 0..1 blend of the auto white-balance toward neutral
    # Manual per-channel gains (multipliers, 1.0 = unchanged) and saturation.
    gain_r: float = 1.0
    gain_g: float = 1.0
    gain_b: float = 1.0
    saturation: float = 1.0


@dataclass
class TetherSettings:
    """Canon DSLR (or any libgphoto2 camera) driven over USB via the ``gphoto2``
    CLI — the only way to get live video + focus/exposure control out of a body
    like the EOS 600D (T3i), which has no UVC/webcam mode. Fully optional and
    lazily probed: with no ``gphoto2`` binary or no camera attached the app falls
    back to the normal OpenCV path and this just reports unavailable."""

    enabled: bool = False
    # Fixed overhead rig: focus once at calibration then never re-drive AF (so it
    # can't hunt mid-shot). 'continuous' re-drives every reconnect; 'manual' never.
    focus_mode: str = "auto_once_lock"  # auto_once_lock | continuous | manual
    # Exposure values applied over USB; "auto" or "" leaves the body's own
    # setting untouched. ISO/WB accept the camera's own values (e.g. "400",
    # "daylight"); shutter/aperture are currently set on the body.
    iso: str = "auto"
    shutterspeed: str = "auto"
    aperture: str = "auto"
    whitebalance: str = "auto"
    # Smart-plug power-cycle commands (shell). When set, the session keeper
    # RESTARTS THE CAMERA ITSELF whenever its one-session-per-power-on is spent
    # — the user never power-cycles anything. Examples:
    #   shortcuts run "Camera Off"            (any HomeKit plug via Shortcuts)
    #   kasa --host 10.0.0.5 off              (TP-Link Kasa, python-kasa)
    #   curl -s http://10.0.0.6/relay/0?turn=off   (Shelly)
    plug_off_cmd: str = ""
    plug_on_cmd: str = ""


@dataclass
class CameraSettings:
    """Capture-side settings shared across every frame source: frame rotation,
    colour correction, UVC webcam controls, and the tethered-DSLR path."""

    # Frame rotation applied at ingest (whole pipeline sees it, so detection,
    # felt-picking, display and recording all share one coordinate space).
    # Degrees clockwise: 0/180 = landscape, 90/270 = portrait.
    rotation: int = 0
    # Mirror the frame (applied after rotation). flip_h mirrors left/right,
    # flip_v mirrors top/bottom — needed when the camera is mounted mirrored or
    # the ceiling bracket faces the "wrong" way.
    flip_h: bool = False
    flip_v: bool = False
    # Camera lens height above the table BED in inches (overhead mount). Drives
    # the ball-height parallax correction: a ball's centre sits ~1.1" above the
    # cloth, so even a directly-overhead camera sees rail balls displaced
    # outward by ~r*d/H — enough to draw a cushion-resting ball inside the rail.
    height_in: float = 66.0
    # True once the camera is mounted directly overhead. Disables the oblique-only
    # parallax + far-rail-rescan corrections and tightens the ball-size band, since
    # a top-down view has no foreshortening and balls are one constant size.
    overhead: bool = True
    # UVC webcam controls (OpenCV CAP_PROP_*). Apply only to index-based cameras;
    # the tethered T3i uses TetherSettings instead. -1 / auto flags = driver default.
    auto_focus: bool = True
    focus: float = -1.0            # 0..255 manual focus when auto_focus is off
    auto_exposure: bool = True
    exposure: float = -1.0
    auto_wb: bool = True
    wb_temperature: float = -1.0   # Kelvin when auto_wb is off (driver-dependent)
    gain: float = -1.0
    # Full 1080p by default — an HDMI capture dongle delivers it, and the extra
    # resolution sharpens ball detection + identification (balls are small on an
    # overhead 9-ft table). Detection still runs at 640 internally, so the cost
    # is modest.
    width: int = 1920
    height: int = 1080
    color: ColorCorrectionSettings = field(default_factory=ColorCorrectionSettings)
    tether: TetherSettings = field(default_factory=TetherSettings)


@dataclass
class AutoLabelSettings:
    """AI auto-labelling of a recorded training session via a vision-language
    model. A VLM reads a sheet of enlarged ball crops and returns each ball's
    number — the labelling the user would otherwise do by hand. Config-gated and
    optional (backend 'off'); the user still labels manually without it."""

    backend: str = "off"           # off | openrouter
    api_key: str = ""
    # An OpenRouter vision-capable model id.
    model: str = "anthropic/claude-3.7-sonnet"
    endpoint: str = "https://openrouter.ai/api/v1/chat/completions"
    max_layouts: int = 12          # distinct settled arrangements to label per session


@dataclass
class RecordingSettings:
    """Where session recordings go and whether they carry audio."""

    # Absolute path where session-*.mp4 clips are written and listed from.
    # Empty = the app's own exports folder. Point it at a synced folder
    # (Dropbox/iCloud) to back sessions up automatically.
    directory: str = ""
    # Capture audio alongside video and mux it into the session mp4. Needs
    # ffmpeg and an audio input device; silently records video-only without
    # them. NOTE: the Canon T3i does NOT send live audio over HDMI, so with no
    # separate mic attached the track will be silence.
    audio: bool = True
    # Capture device name: a DirectShow device on Windows (see
    # capture.audio.list_audio_devices), an avfoundation one on macOS.
    # "default" = first device found / the system default input.
    audio_device: str = "default"
    # Encoder quality (QP/CRF): LOWER is better quality and a bigger file.
    # Measured on a real 60s segment at 924x1630@30, against the old 14Mbps
    # constant-bitrate setting which cost 7.31 GB/hour:
    #     16 -> 3.09 GB/hr  SSIM 0.990 (visually transparent)
    #     20 -> 0.97 GB/hr  SSIM 0.986 (excellent)      <- default
    #     24 -> 0.35 GB/hr  SSIM 0.982 (very good)
    # A pool table is motionless most of the time, so a fixed bitrate spent the
    # same bits on a still table as on a break; targeting quality instead is
    # what buys the ~7x saving at no visible cost.
    video_qp: int = 20

    def resolved_dir(self) -> Path:
        d = self.directory.strip()
        return Path(os.path.expanduser(d)) if d else EXPORTS_DIR


@dataclass
class Settings:
    source: str = "0"  # camera index (as str) | path to video | path to image | "demo" | "tether"
    source_name: str = ""  # friendly camera name, to survive index reshuffles
    mode: str = "free_play"  # free_play | practice | drill
    felt: FeltSettings = field(default_factory=FeltSettings)
    rectify: RectifySettings = field(default_factory=RectifySettings)
    table: TableSettings = field(default_factory=TableSettings)
    balls: BallSettings = field(default_factory=BallSettings)
    detection: DetectionSettings = field(default_factory=DetectionSettings)
    shot_clock: ShotClockSettings = field(default_factory=ShotClockSettings)
    ui: UiSettings = field(default_factory=UiSettings)
    updates: UpdateSettings = field(default_factory=UpdateSettings)
    pose: PoseSettings = field(default_factory=PoseSettings)
    cue: CueSettings = field(default_factory=CueSettings)
    camera: CameraSettings = field(default_factory=CameraSettings)
    autolabel: AutoLabelSettings = field(default_factory=AutoLabelSettings)
    recording: RecordingSettings = field(default_factory=RecordingSettings)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        return _build(cls, data or {})

    def save(self, path: Path | None = None) -> None:
        # Guard for tests/headless probes: constructing UI with default Settings
        # must NEVER clobber the real settings file (it did, twice). Only the
        # DEFAULT path is guarded — tests saving to explicit tmp paths still work.
        if path is None and os.environ.get("BILLIARDS_TRAINER_NO_SAVE"):
            return
        path = path or SETTINGS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(path)  # atomic on Windows + POSIX

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or SETTINGS_PATH
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        s = cls.from_dict(data)
        if _migrate_settings(s):
            try:
                s.save(path)
            except OSError:
                pass
        return s


DETECTION_PRESETS = {
    "conservative": dict(motion_active=0.6, strike_frames=8, min_travel_px=160.0,
                         pocket_frames=15, warmup_seconds=8.0, cooldown_seconds=5.0,
                         require_cue=True, confidence_floor=0.55, fusion_active=0.55),
    "balanced": dict(motion_active=0.4, strike_frames=6, min_travel_px=120.0,
                     pocket_frames=12, warmup_seconds=6.0, cooldown_seconds=4.0,
                     require_cue=True, confidence_floor=0.45, fusion_active=0.45),
    "aggressive": dict(motion_active=0.28, strike_frames=4, min_travel_px=80.0,
                       pocket_frames=8, warmup_seconds=4.0, cooldown_seconds=2.5,
                       require_cue=False, confidence_floor=0.35, fusion_active=0.35),
}


def apply_detection_preset(det: "DetectionSettings", name: str) -> None:
    """Bulk-set the detection gates from a named preset (one-click tuning)."""
    p = DETECTION_PRESETS.get(name)
    if not p:
        return
    for k, v in p.items():
        setattr(det, k, v)
    det.preset = name


def _migrate_settings(s: "Settings") -> bool:
    """Apply one-time forward migrations to a freshly-loaded settings object.

    Returns True if anything changed (so the caller re-saves). Currently: move
    users off ``live_strategy == 'legacy'``. Legacy was the ONLY selectable
    detector in shipped builds while frozen strategy discovery was broken, so a
    saved 'legacy' is almost certainly that bug, not a deliberate choice. We flip
    it to the intended default (simple_blob) exactly once and bump the marker, so
    a later *deliberate* choice of legacy sticks.
    """
    changed = False
    if s.balls.detector_migration < 1:
        if s.balls.live_strategy == "legacy":
            log.info("settings migration: live_strategy 'legacy' -> 'simple_blob'")
            s.balls.live_strategy = "simple_blob"
            changed = True
        s.balls.detector_migration = 1
        changed = True
    # Migration 2: the classical blob detectors are demoted in favour of the
    # trained model. Move anyone still on an old blob default onto 'auto', which
    # resolves to the YOLO model when present. A deliberate non-default choice
    # (e.g. an explicit onnx_* name) is left alone.
    if s.balls.detector_migration < 2:
        if s.balls.live_strategy in ("simple_blob", "felt_mask_hough", "classical"):
            log.info("settings migration: live_strategy '%s' -> 'auto'", s.balls.live_strategy)
            s.balls.live_strategy = "auto"
        s.balls.detector_migration = 2
        changed = True
    return changed


def _build(dc_type: type, data: dict[str, Any]) -> Any:
    """Recursively build a (possibly nested) dataclass, defaulting missing keys
    and ignoring unknown ones."""
    kwargs: dict[str, Any] = {}
    for f in fields(dc_type):
        if f.name not in data:
            continue
        value = data[f.name]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[f.name] = _build(f.type, value)
        else:
            kwargs[f.name] = value
    return dc_type(**kwargs)
