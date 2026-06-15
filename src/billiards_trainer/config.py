"""Application paths and persisted settings.

Settings live as a single JSON document in the per-user app data directory so
they survive reinstalls/updates. The schema is a set of nested dataclasses; load
is forward/backward compatible (unknown keys ignored, missing keys defaulted),
so adding a field never breaks an old settings file.
"""

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


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


@dataclass
class TableSettings:
    size: str = "9ft"           # 9ft | 8ft | 7ft (display/scale only)
    pocket_radius_frac: float = 0.045  # pocket capture radius as frac of short side
    nose_inset_frac: float = 0.0       # playable-area inset from felt edge
    auto_relock: bool = True           # re-detect automatically if the table shifts
    persist_calibration: bool = True   # save/reuse the locked table across launches


@dataclass
class BallSettings:
    backend: str = "classical"  # classical | yolo
    min_radius_frac: float = 0.018  # ball radius bounds as frac of rectified short side
    max_radius_frac: float = 0.060
    detect_param2: int = 18         # Hough accumulator threshold (lower => more circles)
    cue_speed_strike: float = 14.0  # px/frame on rectified view that counts as a strike
    stop_speed: float = 1.2         # px/frame below which a ball is "stopped"
    # Optional YOLO backend: weights are auto-fetched from this URL into the
    # models dir on first use (no turnkey public pool-ball model exists, so this
    # is blank by default — see docs/BLOCKERS.md for how to point it at one).
    yolo_weights_url: str = ""
    yolo_conf: float = 0.25


@dataclass
class ShotClockSettings:
    enabled: bool = False
    seconds: int = 30
    warn_seconds: int = 10
    audio: bool = True
    auto_reset_on_shot: bool = True


@dataclass
class UiSettings:
    theme: str = "dark"
    accent: str = "#3DDC97"      # mint/green — nods to the felt
    show_trajectories: bool = True
    show_ball_ids: bool = True
    show_overlays: bool = True   # master toggle for all detection overlays
    mirror_preview: bool = False


@dataclass
class UpdateSettings:
    auto_check: bool = True
    last_check_iso: str = ""
    skip_version: str = ""


@dataclass
class PoseSettings:
    enabled: bool = False


@dataclass
class Settings:
    source: str = "0"  # camera index (as str) | path to video | path to image | "demo"
    source_name: str = ""  # friendly camera name, to survive index reshuffles
    mode: str = "free_play"  # free_play | practice | drill
    felt: FeltSettings = field(default_factory=FeltSettings)
    rectify: RectifySettings = field(default_factory=RectifySettings)
    table: TableSettings = field(default_factory=TableSettings)
    balls: BallSettings = field(default_factory=BallSettings)
    shot_clock: ShotClockSettings = field(default_factory=ShotClockSettings)
    ui: UiSettings = field(default_factory=UiSettings)
    updates: UpdateSettings = field(default_factory=UpdateSettings)
    pose: PoseSettings = field(default_factory=PoseSettings)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        return _build(cls, data or {})

    def save(self, path: Path | None = None) -> None:
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
        return cls.from_dict(data)


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
