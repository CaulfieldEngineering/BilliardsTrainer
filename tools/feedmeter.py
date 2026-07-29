"""Live video-feed quality meter — numbers instead of "hmm, looks better".

Measures what a capture chain actually delivers, per the recipe in
docs/STATE-2026-07-24.md (same definitions, so results stay comparable with
the 2026-07-24 baseline: T3i -> Cam Link 4K measured ~390 effective lines,
laplacian ~35; a real 1080p feed should read 700+ lines and 150+ sharpness):

  container   what the capture device negotiated (WxH @ fps)
  active      bounding box of the actual picture (gray > 12) — HDMI feeds
              often letterbox a small image inside a big container
  eff. lines  spectral cutoff (fraction of Nyquist, vertical) x active height
              = how many rows of REAL information exist, regardless of what
              the container claims. eff. cols is the same horizontally.
  sharpness   variance of the Laplacian over the active crop (a standard
              focus/detail proxy; higher = crisper)
  unique fps  frames that actually CHANGED per second — catches a device
              duplicating 25 fps content up to 60
  comb        interlace-artifact energy (adjacent-row alternation vs 2-row
              blur); > ~1.3 during motion suggests fields are being weaved

Usage (Cam Link plugged into this machine):

    python tools/feedmeter.py --list                # find the device index
    python tools/feedmeter.py --source 1 --label "NTSC + clear overlays"
    python tools/feedmeter.py --video path.mp4      # measure a file instead

Live window keys:  s = log a snapshot row to docs/feedmeter-log.csv (+PNG)
                   l = edit the attempt label
                   q = quit
(The window is PySide6 — the project venv ships opencv-headless, no cv2 GUI.)

Workflow: change ONE thing in the chain (camera video system, ML overlay
mode, splitter in/out, scaler), press s, read the row. ~390 lines = no
change; 700+ = breakthrough.
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOG_CSV = ROOT / "docs" / "feedmeter-log.csv"
SNAP_DIR = ROOT / "docs" / "feedmeter-snaps"

# --------------------------------------------------------------------------- #
# Metrics (definitions match docs/STATE-2026-07-24.md — keep comparable!)
# --------------------------------------------------------------------------- #


def active_bbox(gray: np.ndarray, thresh: int = 12) -> tuple[int, int, int, int]:
    """Bounding box (x, y, w, h) of the real picture inside the container."""
    mask = gray > thresh
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return 0, 0, gray.shape[1], gray.shape[0]
    return int(cols[0]), int(rows[0]), int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1)


def spectral_cutoff(gray_crop: np.ndarray, axis: int) -> float:
    """Fraction of Nyquist (0..1) below which 98% of the AC spectral energy
    lives — a robust "where does the real information end" estimate.

    Criterion chosen by ground-truth experiment (see feedmeter tests): on
    known content upscaled into a 1080 container, the 98%-cumulative-energy
    point recovers 720-line content exactly and tracks 480/390-line content
    within ~6% of Nyquist, across linear/cubic/lanczos interpolation — while
    noise-floor thresholds get fooled by faint interpolation harmonics.
    A crisp native feed reads ~0.9-1.0; the 2026-07-24 T3i baseline ~0.5."""
    g = gray_crop.astype(np.float32)
    g = g - g.mean(axis=axis, keepdims=True)          # kill DC per line
    spec = np.abs(np.fft.rfft(g, axis=axis))
    power = (spec ** 2).mean(axis=1 - axis)
    n = power.size
    if n < 16:
        return 0.0
    power = cv2.GaussianBlur(power.reshape(-1, 1), (1, 9), 0).ravel()
    c = np.cumsum(power)
    if c[-1] <= 0:
        return 0.0
    return float(np.searchsorted(c, c[-1] * 0.98)) / float(n - 1)


def measure_frame(frame: np.ndarray) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x, y, w, h = active_bbox(gray)
    # Degenerate feed (camera off/booting, dongle showing nothing, LV not
    # started): report it as BLANK instead of computing meaningless spectra on
    # a handful of noise pixels.
    if w < 64 or h < 64 or int(gray.max()) < 16:
        return {"active_w": w, "active_h": h, "cut_v": 0.0, "cut_h": 0.0,
                "eff_lines": 0.0, "eff_cols": 0.0, "sharpness": 0.0,
                "comb": 0.0, "luma_lo": int(gray.min()), "luma_hi": int(gray.max()),
                "blank": 1.0}
    crop = gray[y:y + h, x:x + w]
    # centre crop (75%) avoids letterbox edges polluting the spectrum
    cy, cx = crop.shape[0] // 8, crop.shape[1] // 8
    core = crop[cy:crop.shape[0] - cy, cx:crop.shape[1] - cx]
    if core.size < 1024:
        core = crop
    cut_v = spectral_cutoff(core, axis=0)
    cut_h = spectral_cutoff(core, axis=1)
    lap = float(cv2.Laplacian(core, cv2.CV_64F).var())
    # interlace combing: adjacent-row alternation energy vs vertically-blurred
    a = core.astype(np.float32)
    comb = float(np.abs(a[1:-1] - (a[:-2] + a[2:]) * 0.5).mean()
                 / max(np.abs(cv2.blur(a, (1, 3))[1:-1] - a[1:-1]).mean(), 1e-6))
    return {
        "active_w": w, "active_h": h,
        "cut_v": cut_v, "cut_h": cut_h,
        "eff_lines": cut_v * h, "eff_cols": cut_h * w,
        "sharpness": lap, "comb": comb,
        "luma_lo": int(gray.min()), "luma_hi": int(gray.max()),
        "blank": 0.0,
    }


class RollingMedian:
    def __init__(self, n: int = 15):
        self.n = n
        self.hist: list[dict] = []

    def push(self, m: dict) -> dict:
        self.hist.append(m)
        self.hist = self.hist[-self.n:]
        keys = [k for k, v in m.items() if isinstance(v, (int, float))]
        return {k: float(np.median([h[k] for h in self.hist])) for k in keys}


# --------------------------------------------------------------------------- #
def list_devices() -> None:
    try:
        from pygrabber.dshow_graph import FilterGraph
        names = FilterGraph().get_input_devices()
        for i, name in enumerate(names):
            print(f"  {i}: {name}")
        return
    except Exception:
        pass
    print("  (no pygrabber — probing indices)")
    for i in range(8):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        if cap.isOpened():
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"  {i}: opened ({w:.0f}x{h:.0f})")
        cap.release()


def open_capture(source: str, width: int, height: int, fps: float):
    if source.isdigit():
        cap = cv2.VideoCapture(int(source),
                               cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        # ASK for the full mode — capture devices default to tiny formats
        # (a Cam Link left unasked hands OpenCV 640x480 even with 1080p arriving)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    else:
        cap = cv2.VideoCapture(source)
    return cap


def log_row(label: str, med: dict, container: str, ufps: float, frame) -> None:
    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    new = not LOG_CSV.exists()
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        if new:
            wcsv.writerow(["stamp", "label", "container", "unique_fps",
                           "active", "eff_lines", "eff_cols", "cut_v", "cut_h",
                           "sharpness", "comb", "luma"])
        wcsv.writerow([stamp, label, container, f"{ufps:.1f}",
                       f"{med['active_w']:.0f}x{med['active_h']:.0f}",
                       f"{med['eff_lines']:.0f}", f"{med['eff_cols']:.0f}",
                       f"{med['cut_v']:.3f}", f"{med['cut_h']:.3f}",
                       f"{med['sharpness']:.0f}", f"{med['comb']:.2f}",
                       f"{med['luma_lo']:.0f}-{med['luma_hi']:.0f}"])
    png = SNAP_DIR / f"{stamp}-{''.join(c if c.isalnum() else '_' for c in label)[:40]}.png"
    cv2.imwrite(str(png), frame)
    if med.get("blank"):
        print(f"\nLOGGED [{label}] container={container} — BLANK FEED "
              f"(no picture: camera off/booting, liveview not started, or "
              f"dongle idle). luma {med['luma_lo']:.0f}-{med['luma_hi']:.0f}"
              f"\n -> {png.name}")
        return
    print(f"\nLOGGED [{label}] container={container} unique_fps={ufps:.1f} "
          f"active={med['active_w']:.0f}x{med['active_h']:.0f} "
          f"eff_lines={med['eff_lines']:.0f} eff_cols={med['eff_cols']:.0f} "
          f"sharp={med['sharpness']:.0f} comb={med['comb']:.2f}"
          f"  [{_wire_verdict(med['active_w'], med['active_h'])}]"
          f"\n -> {png.name}")


def _wire_verdict(aw: float, ah: float) -> str:
    """HD vs degraded from the ACTIVE BOX ASPECT, not its width.

    Width alone MISLABELS the 480p wire (1730x757) as HD. Measured on this rig
    (docs/feedmeter-log.csv): healthy 1.50-1.78, degraded 1.95 (failed mode)
    and 2.29 (480p). Nothing healthy exceeds the 16:9 container's own aspect.
    """
    if aw < 8 or ah < 8:
        return "no picture"
    a = aw / ah
    return "HD wire" if a <= 1.85 else f"SD/degraded wire (aspect {a:.2f})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="", help="capture device index")
    ap.add_argument("--video", default="", help="measure a video file instead")
    ap.add_argument("--list", action="store_true", help="list capture devices")
    ap.add_argument("--label", default="attempt", help="label for logged rows")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--headless", type=float, default=0.0, metavar="SECONDS",
                    help="no window: measure for N seconds, log one row, exit")
    args = ap.parse_args()

    if args.list:
        list_devices()
        return 0
    src = args.video or args.source
    if not src:
        ap.error("--source N or --video PATH required (or --list)")
    cap = open_capture(src, args.width, args.height, args.fps)
    if not cap.isOpened():
        print(f"could not open {src}")
        return 1
    cw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ch = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cfps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    container = f"{cw:.0f}x{ch:.0f}@{cfps:.0f}"
    print(f"container: {container}   (s=snapshot, l=relabel, q=quit)")

    state = {"label": args.label, "med": {}, "frame": None, "ufps": 0.0,
             "prev_small": None, "uniq": []}
    roll = RollingMedian()

    def step() -> bool:
        """Grab + measure one frame; returns False when the source ends."""
        ok, frm = cap.read()
        if not ok:
            return args.video == ""          # a live camera may just be slow
        state["frame"] = frm
        state["med"] = roll.push(measure_frame(frm))
        small = cv2.resize(frm, (160, 90), interpolation=cv2.INTER_AREA)
        if state["prev_small"] is None or \
                cv2.absdiff(small, state["prev_small"]).mean() > 0.35:
            state["uniq"].append(time.time())
        state["prev_small"] = small
        state["uniq"] = [t for t in state["uniq"] if t > time.time() - 2.0]
        state["ufps"] = len(state["uniq"]) / 2.0
        return True

    if args.headless:
        t_end = time.time() + args.headless
        while time.time() < t_end:
            step()
        if state["med"]:
            log_row(state["label"], state["med"], container, state["ufps"], state["frame"])
        cap.release()
        return 0

    if args.video:                # file: measure up to ~300 frames, log once
        n = 0
        while step() and n < 300:
            n += 1
        if state["med"]:
            log_row(state["label"], state["med"], container, 0.0, state["frame"])
        cap.release()
        return 0

    # Live viewer — PySide6 (the project venv has opencv-HEADLESS: no cv2 GUI)
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import QApplication, QInputDialog, QLabel

    app = QApplication.instance() or QApplication([])

    class Viewer(QLabel):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("feedmeter — s: snapshot · l: label · q: quit")
            self.setMinimumSize(960, 540)
            self.setAlignment(Qt.AlignCenter)

        def keyPressEvent(self, ev):  # noqa: N802 - Qt override
            k = ev.text().lower()
            if k == "q":
                self.close()
            elif k == "s" and state["med"]:
                log_row(state["label"], state["med"], container,
                        state["ufps"], state["frame"])
            elif k == "l":
                txt, ok = QInputDialog.getText(self, "Attempt label",
                                               "label:", text=state["label"])
                if ok and txt.strip():
                    state["label"] = txt.strip()

    view = Viewer()
    view.show()

    def tick():
        if not step() or state["frame"] is None:
            return
        med = state["med"]
        disp = state["frame"].copy()
        h = disp.shape[0]
        lines = [
            f"container {container}   unique fps {state['ufps']:.1f}",
            f"active {med['active_w']:.0f}x{med['active_h']:.0f}",
            f"EFF LINES {med['eff_lines']:.0f}   eff cols {med['eff_cols']:.0f}"
            f"   (cutoff {med['cut_v']:.2f}/{med['cut_h']:.2f} Nyq)",
            f"sharpness {med['sharpness']:.0f}   comb {med['comb']:.2f}"
            f"   luma {med['luma_lo']:.0f}-{med['luma_hi']:.0f}",
            f"label: {state['label']}",
        ]
        for i, txt in enumerate(lines):
            org = (12, h - 12 - 30 * (len(lines) - 1 - i))
            cv2.putText(disp, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(disp, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 120), 2, cv2.LINE_AA)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                      rgb.strides[0], QImage.Format_RGB888)
        view.setPixmap(QPixmap.fromImage(qimg).scaled(
            view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(33)
    app.exec()
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
