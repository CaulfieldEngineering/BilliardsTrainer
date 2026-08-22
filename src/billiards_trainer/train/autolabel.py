"""Auto-labelling core: turn a recorded session into labelled training frames.

Shared by the CLI (`tools/auto_label_session.py`) and the in-app Training tab.
The heavy lifting: load a session (capture zip / frames dir / video), find the
SETTLED frames (balls at rest, between shots), dedupe them into a few distinct
layouts, run the single-class detector to box every ball, and render an enlarged
CROP SHEET per layout. A vision model then reads each sheet and returns the ball
numbers; `apply_labels` writes them into the TrainingStore.

No torch here — detection is the ONNX runtime path; training stays separate.
"""

from __future__ import annotations

import glob
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..config import MODELS_DIR

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
_VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


@dataclass
class Detection:
    index: int
    x: float
    y: float
    r: float
    heur_num: int          # the live colour-heuristic's guess (for the accuracy report)


@dataclass
class Layout:
    frame: np.ndarray                       # the raw training image (BGR)
    dets: list[Detection] = field(default_factory=list)

    def crop_sheet(self, cell: int = 150, cols: int = 6) -> np.ndarray:
        """Grid of enlarged ball crops captioned with detection index — this is
        what a vision model reads to identify colour + solid/stripe."""
        n = len(self.dets)
        if n == 0:
            return np.zeros((cell, cell, 3), np.uint8)
        rows = (n + cols - 1) // cols
        pad = 26
        sheet = np.full((rows * (cell + pad), cols * cell, 3), 30, np.uint8)
        for d in self.dets:
            r = int(max(d.r * 1.6, 10))
            x0, y0 = max(0, int(d.x - r)), max(0, int(d.y - r))
            crop = self.frame[y0:int(d.y + r), x0:int(d.x + r)]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (cell, cell), interpolation=cv2.INTER_CUBIC)
            gy = (d.index // cols) * (cell + pad)
            gx = (d.index % cols) * cell
            sheet[gy + pad:gy + pad + cell, gx:gx + cell] = crop
            cv2.putText(sheet, f"#{d.index}", (gx + 4, gy + 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return sheet


# --------------------------------------------------------------------------- #
def load_frames(src: Path, stride: int = 2) -> list[np.ndarray]:
    """Load session frames from a capture zip, an image folder, or a video."""
    tmp: str | None = None
    try:
        if src.suffix.lower() == ".zip":
            tmp = tempfile.mkdtemp(prefix="autolabel_")
            with zipfile.ZipFile(src) as z:
                z.extractall(tmp)
            src = Path(tmp)
        if src.is_dir():
            paths = [p for p in sorted(src.rglob("*")) if p.suffix.lower() in _IMG_EXT]
            frames = [cv2.imread(str(p)) for p in paths[::stride]]
        elif src.suffix.lower() in _VID_EXT:
            cap = cv2.VideoCapture(str(src))
            frames, i = [], 0
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                if i % stride == 0:
                    frames.append(f)
                i += 1
            cap.release()
        else:
            raise ValueError(f"unsupported session source: {src}")
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    return [f for f in frames if f is not None]


def _motion(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(cv2.resize(a, (160, 90)), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.resize(b, (160, 90)), cv2.COLOR_BGR2GRAY)
    return float(cv2.absdiff(ga, gb).mean())


def _settled(frames: list[np.ndarray], quiet: float = 2.0) -> list[int]:
    return [i for i in range(len(frames) - 1) if _motion(frames[i], frames[i + 1]) < quiet]


def _dedupe(frames: list[np.ndarray], idxs: list[int], min_change: float = 6.0,
            max_layouts: int = 12) -> list[int]:
    kept: list[int] = []
    for i in idxs:
        if not kept or _motion(frames[kept[-1]], frames[i]) > min_change:
            kept.append(i)
        if len(kept) >= max_layouts:
            break
    return kept


def find_latest_session(recordings_dir: Path | None = None) -> str:
    """Newest recorded session (mp4 preferred; legacy capture zips too), or ''.

    Session mp4s live in the configurable recordings folder; capture zips are
    internal training artifacts and stay in the app's exports folder."""
    from ..config import EXPORTS_DIR
    rec_dir = recordings_dir or EXPORTS_DIR
    try:
        cands = sorted(list(rec_dir.glob("session-*.mp4"))
                       + list(EXPORTS_DIR.glob("capture-*.zip")),
                       key=lambda p: p.stat().st_mtime)
    except OSError:
        return ""
    return str(cands[-1]) if cands else ""


def load_detector():
    """The single-class ball-finder that boxes balls (bootstraps the labels)."""
    from ..detector_strategies.onnx_model import OnnxModelStrategy
    root = Path(__file__).resolve().parents[3]
    models = glob.glob(str(MODELS_DIR / "*.onnx")) + glob.glob(str(root / "_eval" / "models" / "*.onnx"))
    if not models:
        raise FileNotFoundError("no .onnx detector found to bootstrap ball boxes")
    det = OnnxModelStrategy(models[0])
    det.far_rail_rescan = True
    return det


def build_layouts(session: Path, max_layouts: int = 12, stride: int = 2,
                  progress=None) -> list[Layout]:
    """Session -> distinct settled layouts, each with detected ball boxes."""
    from ..core.balls import classify_pool_ball
    frames = load_frames(session, stride)
    if not frames:
        return []
    if progress:
        progress(f"{len(frames)} frames loaded; finding settled layouts…")
    settled = _settled(frames) or list(range(0, len(frames), max(1, len(frames) // 10)))
    picks = _dedupe(frames, settled, max_layouts=max_layouts)
    det = load_detector()
    layouts: list[Layout] = []
    for k, fi in enumerate(picks):
        frame = frames[fi]
        raw = det.detect(frame, None)
        lay = Layout(frame=frame)
        for di, d in enumerate(raw):
            crop = frame[max(0, int(d.y - d.radius)):int(d.y + d.radius),
                         max(0, int(d.x - d.radius)):int(d.x + d.radius)]
            _, heur_num, _ = classify_pool_ball(crop)
            lay.dets.append(Detection(di, float(d.x), float(d.y), float(d.radius), int(heur_num)))
        layouts.append(lay)
        if progress:
            progress(f"layout {k + 1}/{len(picks)}: {len(raw)} balls")
    return layouts


def apply_labels(store, layout: Layout, labels: dict[int, int]) -> int:
    """Write a labelled layout into the TrainingStore. ``labels`` maps detection
    index -> ball number (0=cue, 1..15, <0 = drop). Returns balls saved."""
    from .store import LabeledBall
    h, w = layout.frame.shape[:2]
    balls = []
    for d in layout.dets:
        num = labels.get(d.index, labels.get(str(d.index), -1))
        if num is None or int(num) < 0:
            continue
        balls.append(LabeledBall(int(num), d.x / w, d.y / h, 2 * d.r / w, 2 * d.r / h))
    if not balls:
        return 0
    store.add_frame(layout.frame, balls)
    return len(balls)


def autolabel_session(session: Path, settings, store, progress=None) -> dict:
    """End-to-end: session -> layouts -> VLM labels -> TrainingStore. Returns a
    report {layouts, balls_saved, heuristic_agreement_pct}. Requires a configured
    VLM backend (settings.autolabel); raises if unavailable."""
    from . import vlm
    al = settings.autolabel
    if not vlm.available(al):
        raise RuntimeError("AI labelling is not configured — set an OpenRouter API "
                           "key in Settings, or label manually.")
    layouts = build_layouts(session, max_layouts=al.max_layouts, progress=progress)
    if not layouts:
        raise RuntimeError("No usable frames found in that session.")
    saved = agree = total = 0
    for k, lay in enumerate(layouts):
        if progress:
            progress(f"AI labelling layout {k + 1}/{len(layouts)}…")
        labels = vlm.label_crop_sheet(lay.crop_sheet(), al)
        if not labels:
            continue
        saved += apply_labels(store, lay, labels)
        a, t = accuracy(lay, labels)
        agree += a
        total += t
    return {"layouts": len(layouts), "balls_saved": saved,
            "heuristic_agreement_pct": round(100.0 * agree / total, 1) if total else 0.0}


def accuracy(layout: Layout, labels: dict[int, int]) -> tuple[int, int]:
    """(agreements, total) between the colour heuristic and the vision labels —
    the 'grade the real-time analysis' metric."""
    agree = total = 0
    for d in layout.dets:
        num = labels.get(d.index, labels.get(str(d.index), None))
        if num is None:
            continue
        total += 1
        if int(num) == d.heur_num:
            agree += 1
    return agree, total
