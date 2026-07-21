"""Auto-label a free-shooting session into a numbered-ball training set.

The workflow the user actually wants: lay out the balls, shoot until the table
clears, and let a VISION MODEL do the labelling — no mouse clicking.

Pipeline (this file is stages 1 & 3; the vision model is stage 2):

  stage 1  `montages`   — take a session (capture zip / frames dir / video),
            find SETTLED frames (balls still, between shots), dedupe into a few
            distinct layouts, run the single-class detector for ball boxes, and
            emit one montage image per layout with every detection numbered by
            INDEX. Writes work/manifest.json + work/montage_*.png.

  stage 2  (the vision model — Claude now, an API VLM later) VIEWS each montage
            and writes work/labels.json: {layout_index: {det_index: ball_number}}
            where ball_number is 0=cue, 1..15, or -1 = not-a-ball (false positive).

  stage 3  `build`      — apply those labels, PROPAGATE each layout's labels to
            the nearby settled frames of the same layout (tracking by nearest
            box), write a YOLO dataset (class == ball number, nc=16), and a
            data.yaml ready for tools/finetune_ballid.py. Also writes an accuracy
            report comparing the live colour-heuristic guess to the model labels.

Run stage 1 & 3 with the torch-free app venv; stage-2 labelling is external.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
_VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


# --------------------------------------------------------------------------- #
# Session frame loading (capture zip / frames dir / video)
# --------------------------------------------------------------------------- #
def load_frames(src: Path, stride: int, tmp: list) -> list[np.ndarray]:
    if src.suffix.lower() == ".zip":
        d = tempfile.mkdtemp(prefix="autolabel_")
        tmp.append(d)
        with zipfile.ZipFile(src) as z:
            z.extractall(d)
        src = Path(d)
    if src.is_dir():
        paths = [p for p in sorted(src.rglob("*")) if p.suffix.lower() in _IMG_EXT]
        return [cv2.imread(str(p)) for p in paths[::stride]]
    if src.suffix.lower() in _VID_EXT:
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
        return frames
    raise SystemExit(f"unsupported session source: {src}")


def motion(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(cv2.resize(a, (160, 90)), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.resize(b, (160, 90)), cv2.COLOR_BGR2GRAY)
    return float(cv2.absdiff(ga, gb).mean())


def settled_frames(frames: list[np.ndarray], quiet: float = 2.0) -> list[int]:
    """Indices of frames that are ~identical to the next one (balls at rest)."""
    out = []
    for i in range(len(frames) - 1):
        if motion(frames[i], frames[i + 1]) < quiet:
            out.append(i)
    return out


def dedupe_layouts(frames: list[np.ndarray], idxs: list[int],
                   min_change: float = 6.0, max_layouts: int = 12) -> list[int]:
    """Keep settled frames that differ enough from the last kept one — distinct
    ball arrangements, not 900 copies of the same rack."""
    kept: list[int] = []
    for i in idxs:
        if not kept or motion(frames[kept[-1]], frames[i]) > min_change:
            kept.append(i)
        if len(kept) >= max_layouts:
            break
    return kept


# --------------------------------------------------------------------------- #
def _detector():
    import glob

    from billiards_trainer.config import MODELS_DIR
    from billiards_trainer.detector_strategies.onnx_model import OnnxModelStrategy
    models = glob.glob(str(MODELS_DIR / "*.onnx")) + glob.glob(str(ROOT / "_eval" / "models" / "*.onnx"))
    if not models:
        raise SystemExit("no .onnx detector found to bootstrap ball boxes")
    det = OnnxModelStrategy(models[0])
    det.far_rail_rescan = True
    return det


def _write_crop_sheet(path: Path, frame: np.ndarray, dets, cell: int = 150,
                      cols: int = 6) -> None:
    """Grid of enlarged ball crops, each captioned with its detection index —
    this is what the vision model reads to identify colour + solid/stripe."""
    n = len(dets)
    if n == 0:
        return
    rows = (n + cols - 1) // cols
    pad = 26
    sheet = np.full((rows * (cell + pad), cols * cell, 3), 30, np.uint8)
    for i, d in enumerate(dets):
        r = int(max(d.radius * 1.6, 10))  # a bit of surround for context
        x0, y0 = max(0, int(d.x - r)), max(0, int(d.y - r))
        crop = frame[y0:int(d.y + r), x0:int(d.x + r)]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (cell, cell), interpolation=cv2.INTER_CUBIC)
        gy, gx = (i // cols) * (cell + pad), (i % cols) * cell
        sheet[gy + pad:gy + pad + cell, gx:gx + cell] = crop
        cv2.putText(sheet, f"#{i}", (gx + 4, gy + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(str(path), sheet)


def stage_montages(args) -> int:
    from billiards_trainer.vision.balls import classify_pool_ball
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    tmp: list = []
    try:
        frames = [f for f in load_frames(Path(args.session), args.stride, tmp) if f is not None]
    finally:
        for d in tmp:
            shutil.rmtree(d, ignore_errors=True)
    if not frames:
        raise SystemExit("no frames loaded from session")
    print(f"loaded {len(frames)} frames")
    settled = settled_frames(frames)
    layouts = dedupe_layouts(frames, settled or list(range(0, len(frames), max(1, len(frames)//10))),
                             max_layouts=args.max_layouts)
    print(f"{len(settled)} settled, {len(layouts)} distinct layouts")

    det = _detector()
    manifest = {"session": str(args.session), "layouts": []}
    for li, fi in enumerate(layouts):
        frame = frames[fi]
        raw = det.detect(frame, None)
        entry = {"frame_index": fi, "shape": list(frame.shape[:2]), "dets": []}
        montage = frame.copy()
        for di, d in enumerate(raw):
            heur_cls, heur_num, _ = classify_pool_ball(
                frame[max(0, int(d.y-d.radius)):int(d.y+d.radius),
                      max(0, int(d.x-d.radius)):int(d.x+d.radius)])
            entry["dets"].append({"index": di, "x": float(d.x), "y": float(d.y),
                                  "r": float(d.radius), "heur_num": int(heur_num)})
            c = (int(d.x), int(d.y))
            r = max(8, int(d.radius))
            cv2.circle(montage, c, r, (0, 255, 0), 2)
            cv2.putText(montage, str(di), (c[0]-8, c[1]-r-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        manifest["layouts"].append(entry)
        # Save montage (spatial context) + a CROP SHEET (enlarged per-ball crops
        # so the vision model can read colour + solid/stripe) + the raw source
        # frame (the actual training image).
        cv2.imwrite(str(work / f"montage_{li:02d}.png"), montage)
        cv2.imwrite(str(work / f"frame_{li:02d}.png"), frame)
        _write_crop_sheet(work / f"crops_{li:02d}.png", frame, raw)
        print(f"layout {li}: frame {fi}, {len(raw)} detections")
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(layouts)} montages to {work}")
    print("NEXT: a vision model views montage_*.png and writes labels.json "
          "{layout_index: {det_index: ball_number}} (0=cue,1..15,-1=not-a-ball)")
    return 0


def stage_build(args) -> int:
    work = Path(args.work)
    manifest = json.loads((work / "manifest.json").read_text())
    labels = json.loads((work / "labels.json").read_text())
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    # Optional propagation: spread each labelled layout's numbers to nearby
    # settled frames of the SAME arrangement (balls at rest -> same positions),
    # multiplying labelled images ~Nx with zero extra labelling effort.
    prop_frames: list = []
    if args.propagate and manifest.get("session"):
        tmp: list = []
        try:
            prop_frames = [f for f in load_frames(Path(manifest["session"]), args.stride, tmp)
                           if f is not None]
        finally:
            for d in tmp:
                shutil.rmtree(d, ignore_errors=True)
    det = _detector() if prop_frames else None

    def write_sample(name: str, frame, lines: list[str]) -> None:
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n")

    n_imgs = 0
    agree = total = 0
    for li, entry in enumerate(manifest["layouts"]):
        lab = labels.get(str(li)) or labels.get(li)
        if not lab:
            continue
        h, w = entry["shape"]
        frame_png = work / f"frame_{li:02d}.png"
        lines = []
        labeled_pts = []   # (x, y, number) for propagation matching
        for d in entry["dets"]:
            num = lab.get(str(d["index"]), lab.get(d["index"], -1))
            total += 1
            if int(num) == int(d["heur_num"]):
                agree += 1
            if num is None or int(num) < 0:
                continue
            cx, cy = d["x"] / w, d["y"] / h
            bw, bh = 2 * d["r"] / w, 2 * d["r"] / h
            lines.append(f"{int(num)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            labeled_pts.append((d["x"], d["y"], int(num), d["r"]))
        if not lines:
            continue
        shutil.copy2(frame_png, out / "images" / f"L{li:02d}.png")
        (out / "labels" / f"L{li:02d}.txt").write_text("\n".join(lines) + "\n")
        n_imgs += 1

        # Propagate to neighbouring settled frames of this layout.
        if prop_frames and labeled_pts:
            fi = entry["frame_index"]
            base = prop_frames[fi]
            added = 0
            for off in range(-30, 31, 5):
                j = fi + off
                if off == 0 or not (0 <= j < len(prop_frames)):
                    continue
                cand = prop_frames[j]
                if motion(base, cand) > 2.0:      # same arrangement only
                    continue
                raw = det.detect(cand, None)
                plines = []
                used = set()
                thresh2 = (1.2 * np.median([r for *_z, r in labeled_pts])) ** 2
                for dd in raw:
                    best, bd = -1, thresh2
                    for k, (px, py, _pn, _r) in enumerate(labeled_pts):
                        if k in used:
                            continue
                        dist2 = (dd.x - px) ** 2 + (dd.y - py) ** 2
                        if dist2 < bd:
                            bd, best = dist2, k
                    if best >= 0:
                        used.add(best)
                        pn = labeled_pts[best][2]
                        plines.append(f"{pn} {dd.x / w:.6f} {dd.y / h:.6f} "
                                      f"{2 * dd.radius / w:.6f} {2 * dd.radius / h:.6f}")
                if len(plines) >= max(3, len(labeled_pts) // 2):
                    write_sample(f"L{li:02d}_p{j:05d}", cand, plines)
                    n_imgs += 1
                    added += 1
            print(f"layout {li}: +{added} propagated frames")

    names = ["cue"] + [str(i) for i in range(1, 16)]
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images\nval: images\nnc: 16\n"
        f"names: [{', '.join(names)}]\n")
    acc = (100.0 * agree / total) if total else 0.0
    report = {"labeled_images": n_imgs, "boxes": total,
              "heuristic_agreement_pct": round(acc, 1)}
    (work / "accuracy_report.json").write_text(json.dumps(report, indent=2))
    print(f"dataset: {n_imgs} images -> {out}")
    print(f"real-time heuristic agreed with the vision labels on "
          f"{agree}/{total} balls ({acc:.1f}%)")
    print(f"data.yaml: {out / 'data.yaml'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Auto-label a session into a numbered-ball dataset")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("montages", help="stage 1: session -> montages for a vision model")
    m.add_argument("--session", required=True, help="capture zip / frames dir / video")
    m.add_argument("--work", default=str(ROOT / "_train" / "autolabel"))
    m.add_argument("--stride", type=int, default=2)
    m.add_argument("--max-layouts", type=int, default=12)
    m.set_defaults(fn=stage_montages)
    b = sub.add_parser("build", help="stage 3: labels.json -> YOLO dataset + accuracy")
    b.add_argument("--work", default=str(ROOT / "_train" / "autolabel"))
    b.add_argument("--out", default=str(ROOT / "_train" / "autolabel" / "dataset"))
    b.add_argument("--propagate", action="store_true",
                   help="spread each layout's labels to nearby settled frames")
    b.add_argument("--stride", type=int, default=2)
    b.set_defaults(fn=stage_build)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
