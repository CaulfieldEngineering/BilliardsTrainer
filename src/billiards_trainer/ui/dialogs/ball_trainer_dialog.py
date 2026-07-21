"""In-app Ball-ID Trainer — the active-learning loop as a window.

Flow: load a recorded clip of THIS table -> the detector finds balls and guesses
each number -> you click a ball and tap the right number to correct it -> Save
appends the labelled frame to the persistent store -> Train fine-tunes the model
on everything collected so far and switches the app to it. Redo whenever the
camera/angle/lighting changes; the corrections accumulate as the foundation.

Labelling is fully in-app (no torch). Training shells out to a torch env (the
shipped app is torch-free) — auto-detected, with the exact CLI shown if absent.
"""
import subprocess
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ...config import APP_DIR, MODELS_DIR, Settings
from ...train import TrainingStore
from ...train.store import LabeledBall

_DISPLAY_W = 880
_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


class _TrainWorker(QThread):
    done = Signal(bool, str)

    def __init__(self, py: str, data: str, out: str):
        super().__init__()
        self._py, self._data, self._out = py, data, out

    def run(self) -> None:
        root = Path(__file__).resolve().parents[3]
        try:
            p = subprocess.run(
                [self._py, str(root / "tools" / "finetune_ballid.py"),
                 "--data", self._data, "--out", self._out],
                capture_output=True, text=True, timeout=3 * 3600)
            ok = p.returncode == 0 and "FINETUNE_OK" in (p.stdout or "")
            self.done.emit(ok, (p.stdout or "")[-400:] + (p.stderr or "")[-400:])
        except Exception as exc:  # noqa: BLE001
            self.done.emit(False, str(exc))


class BallTrainerDialog(QDialog):
    strategy_retrained = Signal(str)   # the strategy name to switch to after training

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ball ID Trainer — teach the table's balls")
        self.resize(1040, 760)
        self._s = settings
        self._store = TrainingStore(APP_DIR / "training" / "ballid")
        self._cap: cv2.VideoCapture | None = None
        self._img_paths: list[Path] = []   # image-sequence mode (capture zip/folder)
        self._img_i = 0
        self._tmpdir: tempfile.TemporaryDirectory | None = None
        self._frame: np.ndarray | None = None
        self._dets: list = []           # current detections (LabeledBall-like, mutable number)
        self._prev_labeled: list = []   # (cx, cy, number) carried from the last frame
        self._sel = -1                  # selected ball index
        self._scale = 1.0
        self._worker: _TrainWorker | None = None
        self._strategy = self._load_strategy()
        self._build()
        self._open_source(settings.source if Path(str(settings.source)).exists() else "")

    # ------------------------------------------------------------------ #
    def _load_strategy(self):
        try:
            from ...detector_strategies import discover
            from ...vision.pipeline import Pipeline
            return Pipeline._resolve_auto(discover())
        except Exception:  # noqa: BLE001
            return None

    def _build(self) -> None:
        root = QVBoxLayout(self)
        hint = QLabel("Open a recorded clip. Label every ball once (click a ball, tap "
                      "its number) — then Save and step forward: your labels follow "
                      "the balls automatically, so you only fix what moved. Collect "
                      "20–40 varied frames, then Train. Redo if the camera moves.")
        hint.setWordWrap(True)
        hint.setObjectName("Faint")
        root.addWidget(hint)

        self._view = QLabel("Open a clip to begin")
        self._view.setAlignment(Qt.AlignCenter)
        self._view.setMinimumSize(_DISPLAY_W, 500)
        self._view.setStyleSheet("background:#11151a;")
        self._view.mousePressEvent = self._on_click  # type: ignore[method-assign]
        root.addWidget(self._view, 1)

        # number pad: cue, 1..15, and "not a ball"
        pad = QGridLayout()
        self._num_btns = {}
        labels = [("C", 0)] + [(str(i), i) for i in range(1, 16)] + [("✕", -1)]
        for idx, (txt, num) in enumerate(labels):
            b = QPushButton(txt)
            b.setEnabled(False)
            b.clicked.connect(lambda _=False, n=num: self._assign(n))
            self._num_btns[num] = b
            pad.addWidget(b, idx // 9, idx % 9)
        root.addLayout(pad)

        bar = QHBoxLayout()
        for txt, fn in (("Open clip…", self._choose_clip), ("◀ Prev", lambda: self._step(-15)),
                        ("Next ▶", lambda: self._step(15)), ("＋ Save frame", self._save)):
            btn = QPushButton(txt)
            btn.clicked.connect(fn)
            bar.addWidget(btn)
        bar.addStretch(1)
        self._stats = QLabel("")
        self._stats.setObjectName("Muted")
        bar.addWidget(self._stats)
        self._train_btn = QPushButton("Train model on collected data")
        self._train_btn.setObjectName("Accent")
        self._train_btn.clicked.connect(self._train)
        bar.addWidget(self._train_btn)
        root.addLayout(bar)

        self._status = QLabel("")
        self._status.setObjectName("Faint")
        self._status.setWordWrap(True)
        root.addWidget(self._status)
        self._refresh_stats()

    # ------------------------------------------------------------------ #
    def _choose_clip(self) -> None:
        # Accept a video clip OR a "Capture 60s for analysis" bundle (its zip or
        # the unpacked frames folder) — the latter is RAW frames, the right
        # training data (the Record button bakes in overlays).
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a clip or capture of your table",
            filter="Clip or capture (*.mp4 *.avi *.mov *.mkv *.zip);;All files (*)")
        if path:
            self._open_source(path)

    def _open_source(self, path: str) -> None:
        if not path:
            return
        self._release_source()
        self._prev_labeled = []   # don't carry labels across different sources
        p = Path(path)
        if p.suffix.lower() == ".zip" or p.is_dir():
            self._open_images(p)
        else:
            self._cap = cv2.VideoCapture(str(p))
        self._step(0)

    def _open_images(self, p: Path) -> None:
        """Load a capture zip (frames/*.jpg + meta.json) or an image folder."""
        root = p
        if p.suffix.lower() == ".zip":
            self._tmpdir = tempfile.TemporaryDirectory(prefix="ballid_")
            with zipfile.ZipFile(p) as z:
                z.extractall(self._tmpdir.name)
            root = Path(self._tmpdir.name)
        imgs = [q for q in sorted(root.rglob("*")) if q.suffix.lower() in _IMG_EXT]
        self._img_paths = imgs
        self._img_i = 0
        if not imgs:
            self._status.setText("No images found in that capture.")

    def _release_source(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._img_paths = []
        self._img_i = 0
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def _step(self, delta: int) -> None:
        # Snapshot the labels the user just set so they carry to the next frame.
        if self._dets:
            self._prev_labeled = [(d.cx, d.cy, d.number) for d in self._dets]
        frame = None
        if self._img_paths:
            # Image mode: frames are already strided, so a ±15 video step maps to
            # a smaller image step for sensible traversal of distinct layouts.
            stepn = int(np.sign(delta)) * max(1, abs(delta) // 5) if delta else 0
            self._img_i = max(0, min(len(self._img_paths) - 1, self._img_i + stepn))
            frame = cv2.imread(str(self._img_paths[self._img_i]))
        elif self._cap is not None:
            pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos + delta))
            ok, frame = self._cap.read()
            if not ok:
                return
        if frame is None:
            return
        self._frame = frame
        self._detect()
        self._propagate_labels()
        self._render()

    def _propagate_labels(self) -> None:
        """Carry the previous frame's labels onto this frame's detections by
        nearest match — so labelling one settled frame propagates to the rest and
        you only correct what actually moved. Detector guesses are only overridden
        where a confident nearby prior label exists."""
        if not self._prev_labeled or not self._dets:
            return
        r = self._default_radius()
        thresh2 = (2.2 * max(r, 6.0)) ** 2   # within ~2 ball-widths = same ball
        used = [False] * len(self._prev_labeled)
        for d in self._dets:
            best_j, best_d2 = -1, thresh2
            for j, (px, py, pn) in enumerate(self._prev_labeled):
                if used[j] or pn < 0:
                    continue
                dd = (d.cx - px) ** 2 + (d.cy - py) ** 2
                if dd < best_d2:
                    best_d2, best_j = dd, j
            if best_j >= 0:
                d.number = self._prev_labeled[best_j][2]
                used[best_j] = True

    def _detect(self) -> None:
        self._sel = -1
        self._dets = []
        if self._frame is None or self._strategy is None:
            return
        try:
            # Labelling is offline and wants MAX recall — force the far-rail
            # rescan for this call regardless of the live pipeline's perf
            # setting (the strategy object is shared process-wide).
            if hasattr(self._strategy, "far_rail_rescan"):
                raw = self._strategy.detect(self._frame, None, rescan=True)
            else:
                raw = self._strategy.detect(self._frame, None)   # no calib needed to label
        except Exception:  # noqa: BLE001
            raw = []
        for d in raw:
            self._dets.append(LabeledBall(number=d.number, cx=d.x, cy=d.y,
                                          w=2 * d.radius, h=2 * d.radius))

    # ------------------------------------------------------------------ #
    def _render(self) -> None:
        if self._frame is None:
            return
        img = self._frame.copy()
        for i, d in enumerate(self._dets):
            c = (int(d.cx), int(d.cy))
            r = int(max(d.w, d.h) / 2)
            sel = i == self._sel
            col = (0, 255, 255) if sel else (60, 220, 60)
            cv2.circle(img, c, r, col, 3 if sel else 2)
            lbl = "C" if d.number == 0 else (str(d.number) if d.number > 0 else "?")
            cv2.putText(img, lbl, (c[0] - 8, c[1] - r - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, col, 2)
        h, w = img.shape[:2]
        self._scale = _DISPLAY_W / w
        disp = cv2.resize(img, (_DISPLAY_W, int(h * self._scale)))
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        qi = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        self._view.setPixmap(QPixmap.fromImage(qi))
        for b in self._num_btns.values():
            b.setEnabled(self._sel >= 0)

    def _default_radius(self) -> float:
        rs = [max(d.w, d.h) / 2.0 for d in self._dets if d.w > 0]
        if rs:
            return float(np.median(rs))
        w = self._frame.shape[1] if self._frame is not None else 640
        return max(8.0, w * 0.012)

    def _on_click(self, ev) -> None:
        if self._frame is None or self._scale <= 0:
            return
        x = ev.position().x() / self._scale
        y = ev.position().y() / self._scale
        best, bd = -1, 1e18
        for i, d in enumerate(self._dets):
            dd = (d.cx - x) ** 2 + (d.cy - y) ** 2
            if dd < bd:
                bd, best = dd, i
        if best >= 0 and bd < (60 / max(self._scale, 1e-6)) ** 2:
            self._sel = best                      # clicked a ball -> select it
        else:
            # clicked empty felt -> ADD a missed ball (false negative). It starts
            # unlabelled ('?'); tap a number to set it. This is how you teach the
            # model the balls it didn't find.
            r = self._default_radius()
            self._dets.append(LabeledBall(number=-1, cx=float(x), cy=float(y),
                                          w=2 * r, h=2 * r))
            self._sel = len(self._dets) - 1
        self._render()

    def _assign(self, num: int) -> None:
        if 0 <= self._sel < len(self._dets):
            self._dets[self._sel].number = num
            self._sel = -1
            self._render()

    def _save(self) -> None:
        if self._frame is None or not self._dets:
            return
        h, w = self._frame.shape[:2]
        norm = [LabeledBall(d.number, d.cx / w, d.cy / h, d.w / w, d.h / h)
                for d in self._dets]
        n = self._store.add_frame(self._frame, norm)
        self._refresh_stats()
        self._status.setText(f"Saved frame with {n} labelled balls. Keep going, then Train.")
        self._step(15)

    def _refresh_stats(self) -> None:
        cc = self._store.class_counts()
        covered = sorted(cc)
        self._stats.setText(f"Collected: {self._store.count()} frames · "
                            f"numbers seen: {covered if covered else '—'}")

    # ------------------------------------------------------------------ #
    def _find_train_python(self) -> str:
        """A python with torch+ultralytics (the shipped app is torch-free)."""
        root = Path(__file__).resolve().parents[3]
        cands = [root / "_refs" / "pool_coach" / ".venv" / "Scripts" / "python.exe",
                 root / ".trainvenv" / "Scripts" / "python.exe",
                 root / "_refs" / "pool_coach" / ".venv" / "bin" / "python",
                 root / ".trainvenv" / "bin" / "python"]
        for c in cands:
            if c.exists():
                return str(c)
        return ""

    def _train(self) -> None:
        if self._store.count() < 5:
            self._status.setText("Collect at least ~5 frames (more is better) before training.")
            return
        data = str(self._store.write_data_yaml())
        out = str(MODELS_DIR / "pool_ballid.onnx")
        py = self._find_train_python()
        if not py:
            self._status.setText(
                "No training environment found. Run this once in a python that has "
                "torch+ultralytics:\n"
                f"  python tools/finetune_ballid.py --data \"{data}\" --out \"{out}\"\n"
                "then reopen — the app will use the trained model.")
            return
        self._train_btn.setEnabled(False)
        self._status.setText("Training… this runs in the background and can take a few minutes.")
        self._worker = _TrainWorker(py, data, out)
        self._worker.done.connect(self._on_trained)
        self._worker.start()

    def _on_trained(self, ok: bool, log: str) -> None:
        self._train_btn.setEnabled(True)
        if ok:
            self._s.balls.live_strategy = "onnx_pool_ballid"
            self._s.save()
            self.strategy_retrained.emit("onnx_pool_ballid")
            self._status.setText("Trained ✓ — the app is now using your table's model. "
                                 "Close this window to see it.")
        else:
            self._status.setText(f"Training failed. {log[-300:]}")

    def closeEvent(self, ev) -> None:  # noqa: N802
        # NEVER let the dialog die while the training QThread runs — destroying
        # a running QThread aborts the whole process natively (the "crashed in
        # the middle of training" class). Block the close instead.
        if self._worker is not None and self._worker.isRunning():
            self._status.setText("Training is still running — leave this window "
                                 "open until it finishes (a few minutes). It will "
                                 "say 'Trained ✓' when done.")
            ev.ignore()
            return
        self._release_source()
        super().closeEvent(ev)
