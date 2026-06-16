"""Detection eval harness — measure the CURRENT pipeline against ground truth.

Phase 0 philosophy: *measure before you change.* This harness runs the existing
detection pipeline (unmodified) over a set of video clips, compares its per-frame
ball detections to hand-labelled ground truth, and reports honest numbers +
a browsable HTML visual diff. It changes NO detection logic.

Inputs
------
- ``--clips DIR``   directory of ``*.mp4`` clips (default ``_eval/clips``)
- ``--labels DIR``  directory of ``<clip-stem>.json`` ground-truth files
                    (default ``_eval/labels``; clips without a label still get
                    shot-FP + calibration metrics)
- ``--out DIR``     output dir (default ``docs/eval/<timestamp>``)
- ``--subset N``    only the first N clips (CI uses a tiny subset)
- ``--stride K``    process every Kth frame (default 1; CI may downsample)
- ``--max-seconds S`` cap per-clip processing length

Ground-truth label schema (coords are RAW camera-frame pixels)
--------------------------------------------------------------
    {
      "clip": "idle_01.mp4",
      "static": true,              # if true, the single keyframe applies to ALL frames
      "empty": false,              # if true, NO balls expected (any detection = FP)
      "expected_shots": 0,         # makes+misses the clip should produce (FP-shot test)
      "tolerance_px": 45,          # match radius; default = 2.5% of frame width
      "keyframes": {               # frame_index -> list of balls
        "0": [ {"x": 980, "y": 540, "cls": "cue"}, {"x": 1240, "y": 610, "cls": "solid"} ]
      },
      "notes": "..."
    }

Metrics (per clip + aggregate), reported honestly with their definitions:
- precision / recall / F1 over labelled frames (a detection matches a GT ball if
  its centre is within ``tolerance_px`` — greedy nearest, one-to-one)
- AP@dist: average precision of detections (score-ranked, distance-matched). This
  is a CENTRE-DISTANCE AP, *not* IoU mAP — we have no tight GT boxes, our detector
  emits centre+radius. Labelled honestly as such.
- ball-ID accuracy: of correctly-localised detections, fraction with the right class
- false-positive shots / minute: shot events emitted vs ``expected_shots``
- calibration rate: fraction of processed frames where the table locked

Usage
-----
    python tools/eval_harness.py                          # full set -> docs/eval/<ts>
    python tools/eval_harness.py --subset 2 --stride 3    # quick / CI
"""

import argparse
import html
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:  # Windows consoles default to cp1252 — make our output UTF-8 safe
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from billiards_trainer.config import Settings  # noqa: E402
from billiards_trainer.vision.pipeline import Pipeline  # noqa: E402
from billiards_trainer.vision.rectify import project_points  # noqa: E402


# --------------------------------------------------------------------------- #
# Ground-truth loading
# --------------------------------------------------------------------------- #
@dataclass
class ClipLabels:
    static: bool = False
    empty: bool = False
    expected_shots: int | None = None
    tolerance_px: float | None = None
    keyframes: dict[int, list[dict]] = field(default_factory=dict)
    notes: str = ""
    labelled: bool = False

    def balls_at(self, frame_idx: int) -> list[dict] | None:
        """Return the GT balls for a frame, or None if this frame is unlabelled.
        Static clips return their single keyframe for every frame. Otherwise we
        return the nearest labelled keyframe within a small window (no risky
        interpolation of identities — we only score frames we actually labelled,
        plus their immediate neighbourhood for static scenes)."""
        if not self.labelled:
            return None
        if self.empty:
            return []
        if self.static and self.keyframes:
            return next(iter(self.keyframes.values()))
        return self.keyframes.get(frame_idx)


def load_labels(path: Path | None) -> ClipLabels:
    if path is None or not path.exists():
        return ClipLabels(labelled=False)
    data = json.loads(path.read_text(encoding="utf-8"))
    kf = {int(k): v for k, v in data.get("keyframes", {}).items()}
    return ClipLabels(
        static=bool(data.get("static", False)),
        empty=bool(data.get("empty", False)),
        expected_shots=data.get("expected_shots"),
        tolerance_px=data.get("tolerance_px"),
        keyframes=kf, notes=data.get("notes", ""),
        labelled=True,
    )


# --------------------------------------------------------------------------- #
# Matching + metrics
# --------------------------------------------------------------------------- #
def match(dets_xy: list[tuple], gt_xy: list[tuple], tol: float) -> list[tuple[int, int]]:
    """Greedy one-to-one nearest matching within ``tol`` pixels.
    Returns list of (det_index, gt_index)."""
    pairs = []
    for di, (dx, dy) in enumerate(dets_xy):
        for gi, (gx, gy) in enumerate(gt_xy):
            d = float(np.hypot(dx - gx, dy - gy))
            if d <= tol:
                pairs.append((d, di, gi))
    pairs.sort()
    used_d, used_g, out = set(), set(), []
    for _d, di, gi in pairs:
        if di in used_d or gi in used_g:
            continue
        used_d.add(di)
        used_g.add(gi)
        out.append((di, gi))
    return out


@dataclass
class ClipResult:
    name: str
    frames_processed: int = 0
    frames_calibrated: int = 0
    frames_scored: int = 0          # frames with GT available
    total_dets: int = 0             # detections across ALL frames (behaviour signal)
    tp: int = 0
    fp: int = 0
    fn: int = 0
    cls_correct: int = 0
    cls_total: int = 0
    shots: int = 0
    expected_shots: int | None = None
    duration_s: float = 0.0
    # for AP@dist: (score, is_tp) across all scored detections
    score_hits: list[tuple[float, int]] = field(default_factory=list)
    keyframe_imgs: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else float("nan")

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else float("nan")

    @property
    def f1(self):
        p, r = self.precision, self.recall
        if not (p == p and r == r) or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)

    @property
    def id_acc(self):
        return self.cls_correct / self.cls_total if self.cls_total else float("nan")

    @property
    def fp_shots_per_min(self):
        if self.duration_s <= 0:
            return float("nan")
        extra = self.shots - (self.expected_shots or 0)
        return max(0, extra) / (self.duration_s / 60.0)

    @property
    def calib_rate(self):
        return self.frames_calibrated / self.frames_processed if self.frames_processed else 0.0


def average_precision(score_hits: list[tuple[float, int]], n_gt: int) -> float:
    """AP@dist: rank detections by score, sweep precision/recall, integrate."""
    if n_gt == 0 or not score_hits:
        return float("nan")
    s = sorted(score_hits, key=lambda x: -x[0])
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    for _score, hit in s:
        if hit:
            tp += 1
        else:
            fp += 1
        recall = tp / n_gt
        precision = tp / (tp + fp)
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap


# --------------------------------------------------------------------------- #
# Per-clip evaluation
# --------------------------------------------------------------------------- #
_CLS_DRAW = {"cue": (255, 255, 255), "solid": (60, 200, 255), "stripe": (80, 255, 180),
             "eight": (30, 30, 30), "unknown": (200, 200, 200)}


def _raw_detections(pipe: Pipeline, res) -> list[dict]:
    """Project the pipeline's rectified detections back into raw-frame pixels."""
    if not pipe.calib.is_calibrated or not res.detections:
        return []
    calib = pipe.calib.calib
    pts = np.array([[d.x, d.y] for d in res.detections], np.float64)
    raw = project_points(pts, calib.Hinv)
    # radius: project a +r offset point and take the distance
    off = np.array([[d.x + max(d.radius, 1.0), d.y] for d in res.detections], np.float64)
    raw_off = project_points(off, calib.Hinv)
    out = []
    for d, (rx, ry), (ox, oy) in zip(res.detections, raw, raw_off, strict=False):
        rr = float(np.hypot(ox - rx, oy - ry))
        out.append({"x": float(rx), "y": float(ry), "r": rr,
                    "cls": d.cls.value, "score": float(d.score)})
    return out


def eval_clip(clip_path: Path, labels: ClipLabels, out_imgs: Path,
              stride: int, max_seconds: float, n_keyframes: int = 12) -> ClipResult:
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    max_frames = total if max_seconds <= 0 else min(total, int(max_seconds * fps))
    tol = labels.tolerance_px or (0.025 * w)

    r = ClipResult(name=clip_path.name, expected_shots=labels.expected_shots)
    settings = Settings()  # CURRENT defaults — the detector exactly as shipped
    pipe = Pipeline(settings, source=str(clip_path))

    keyframe_every = max(1, max_frames // max(1, n_keyframes))
    n_gt_total = 0
    fidx = 0
    while fidx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if stride > 1 and (fidx % stride) != 0:
            fidx += 1
            continue
        res = pipe.process(frame, t=fidx / fps)
        r.frames_processed += 1
        if pipe.calib.is_calibrated:
            r.frames_calibrated += 1
        if res.shot_event is not None:
            r.shots += 1

        dets = _raw_detections(pipe, res)
        r.total_dets += len(dets)
        gt = labels.balls_at(fidx)
        if gt is not None:
            r.frames_scored += 1
            n_gt_total += len(gt)
            det_xy = [(d["x"], d["y"]) for d in dets]
            gt_xy = [(g["x"], g["y"]) for g in gt]
            pairs = match(det_xy, gt_xy, tol)
            matched_d = {di for di, _ in pairs}
            matched_g = {gi for _, gi in pairs}
            r.tp += len(pairs)
            r.fp += len(dets) - len(matched_d)
            r.fn += len(gt) - len(matched_g)
            for di, gi in pairs:
                if "cls" in gt[gi]:
                    r.cls_total += 1
                    if dets[di]["cls"] == gt[gi]["cls"]:
                        r.cls_correct += 1
            for di, d in enumerate(dets):
                r.score_hits.append((d["score"], 1 if di in matched_d else 0))

        # keyframe visual diff (3-up: raw | detector | ground truth)
        if (fidx % keyframe_every) == 0 and len(r.keyframe_imgs) < n_keyframes:
            img = _composite(frame, dets, gt, res, pipe)
            name = f"{clip_path.stem}_f{fidx:06d}.jpg"
            cv2.imwrite(str(out_imgs / name), img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            r.keyframe_imgs.append(name)
        fidx += 1

    cap.release()
    r.duration_s = r.frames_processed * stride / fps
    r._n_gt = n_gt_total  # type: ignore[attr-defined]
    return r


def _panel_label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _composite(frame, dets, gt, res, pipe):
    h, w = frame.shape[:2]
    scale = 480 / w
    small = cv2.resize(frame, (480, int(h * scale)))
    raw = small.copy()
    _panel_label(raw, "INPUT")
    det_img = small.copy()
    for d in dets:
        c = (int(d["x"] * scale), int(d["y"] * scale))
        col = _CLS_DRAW.get(d["cls"], (200, 200, 200))
        cv2.circle(det_img, c, max(3, int(d["r"] * scale)), col, 2, cv2.LINE_AA)
    # draw the locked table outline if calibrated
    if pipe.calib.is_calibrated and res.corners is not None:
        pts = (res.corners * scale).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(det_img, [pts], True, (60, 220, 160), 1, cv2.LINE_AA)
    _panel_label(det_img, f"DETECTOR ({len(dets)})  calib={pipe.calib.is_calibrated}")
    gt_img = small.copy()
    if gt is not None:
        for g in gt:
            c = (int(g["x"] * scale), int(g["y"] * scale))
            col = _CLS_DRAW.get(g.get("cls", "unknown"), (200, 200, 200))
            cv2.circle(gt_img, c, 8, col, 2, cv2.LINE_AA)
            cv2.drawMarker(gt_img, c, col, cv2.MARKER_CROSS, 10, 1)
        _panel_label(gt_img, f"GROUND TRUTH ({len(gt)})")
    else:
        _panel_label(gt_img, "GROUND TRUTH (unlabelled)")
    return np.hstack([raw, det_img, gt_img])


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
def _fmt(x, pct=False):
    if x != x:  # NaN
        return "—"
    return f"{x*100:.1f}%" if pct else f"{x:.2f}"


def write_report(results: list[ClipResult], out: Path, meta: dict) -> Path:
    agg = _aggregate(results)
    rows = []
    for r in results:
        rows.append(
            f"<tr><td>{html.escape(r.name)}</td>"
            f"<td>{r.frames_processed}</td><td>{_fmt(r.calib_rate, True)}</td>"
            f"<td>{r.frames_scored}</td>"
            f"<td>{_fmt(r.precision, True)}</td><td>{_fmt(r.recall, True)}</td>"
            f"<td>{_fmt(r.f1, True)}</td><td>{_fmt(r.id_acc, True)}</td>"
            f"<td>{r.shots}{'' if r.expected_shots is None else '/' + str(r.expected_shots)}</td>"
            f"<td>{_fmt(r.fp_shots_per_min)}</td></tr>")
    galleries = []
    for r in results:
        imgs = "".join(
            f'<figure><img src="images/{html.escape(i)}" loading="lazy">'
            f'<figcaption>{html.escape(i)}</figcaption></figure>' for i in r.keyframe_imgs)
        galleries.append(f"<h3>{html.escape(r.name)} <small>{html.escape(r.note)}</small></h3>"
                         f'<div class="gallery">{imgs}</div>')
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Detection eval — {meta.get('timestamp','')}</title>
<style>
 body{{font:14px/1.5 -apple-system,Segoe UI,sans-serif;margin:24px;color:#1b1f24;background:#fff}}
 h1{{margin:0 0 4px}} .sub{{color:#667}}
 table{{border-collapse:collapse;margin:16px 0;width:100%}}
 th,td{{border:1px solid #d0d7de;padding:6px 10px;text-align:right}}
 th:first-child,td:first-child{{text-align:left}}
 thead{{background:#f6f8fa}} tr.agg{{font-weight:700;background:#fff8e1}}
 .gallery{{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:24px}}
 figure{{margin:0}} img{{width:480px;border:1px solid #ccc;border-radius:4px}}
 figcaption{{font-size:11px;color:#889}}
 .key{{background:#f6f8fa;padding:10px 14px;border-radius:6px;font-size:13px}}
</style></head><body>
<h1>Detection eval — baseline</h1>
<div class="sub">{html.escape(json.dumps(meta))}</div>
<div class="key"><b>Panels:</b> INPUT (raw frame) · DETECTOR (current pipeline, projected to raw; green = locked table) · GROUND TRUTH (hand-labelled centres).
<b>AP@dist</b> is centre-distance average precision, not IoU mAP (our detector emits centre+radius, GT has no tight boxes).</div>
<table><thead><tr><th>clip</th><th>frames</th><th>calib%</th><th>scored</th>
<th>precision</th><th>recall</th><th>F1</th><th>ID acc</th><th>shots(d/exp)</th><th>FP shots/min</th></tr></thead>
<tbody>
<tr class="agg"><td>AGGREGATE</td><td>{agg['frames']}</td><td>{_fmt(agg['calib_rate'],True)}</td>
<td>{agg['scored']}</td><td>{_fmt(agg['precision'],True)}</td><td>{_fmt(agg['recall'],True)}</td>
<td>{_fmt(agg['f1'],True)}</td><td>{_fmt(agg['id_acc'],True)}</td><td>{agg['shots']}</td>
<td>{_fmt(agg['fp_shots_per_min'])}</td></tr>
{''.join(rows)}
</tbody></table>
<p><b>AP@dist (aggregate):</b> {_fmt(agg['ap'])} &nbsp;·&nbsp; <b>total GT balls:</b> {agg['n_gt']}</p>
<h2>Per-clip visual diffs</h2>
{''.join(galleries)}
</body></html>"""
    report = out / "report.html"
    report.write_text(doc, encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({"meta": meta, "aggregate": agg,
        "clips": [_clip_json(r) for r in results]}, indent=2), encoding="utf-8")
    return report


def _clip_json(r: ClipResult) -> dict:
    return {"clip": r.name, "frames": r.frames_processed, "calib_rate": round(r.calib_rate, 4),
            "total_dets": r.total_dets,
            "scored_frames": r.frames_scored, "tp": r.tp, "fp": r.fp, "fn": r.fn,
            "precision": _num(r.precision), "recall": _num(r.recall), "f1": _num(r.f1),
            "id_acc": _num(r.id_acc), "shots": r.shots, "expected_shots": r.expected_shots,
            "fp_shots_per_min": _num(r.fp_shots_per_min)}


def _num(x):
    return None if x != x else round(float(x), 4)


def _aggregate(results: list[ClipResult]) -> dict:
    tp = sum(r.tp for r in results)
    fp = sum(r.fp for r in results)
    fn = sum(r.fn for r in results)
    cc = sum(r.cls_correct for r in results)
    ct = sum(r.cls_total for r in results)
    frames = sum(r.frames_processed for r in results)
    calib = sum(r.frames_calibrated for r in results)
    n_gt = sum(getattr(r, "_n_gt", 0) for r in results)
    score_hits = [sh for r in results for sh in r.score_hits]
    dur = sum(r.duration_s for r in results)
    shots = sum(r.shots for r in results)
    exp = sum((r.expected_shots or 0) for r in results)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if (prec == prec and rec == rec and (prec + rec)) else float("nan")
    return {"frames": frames, "calib_rate": calib / frames if frames else 0.0,
            "total_dets": sum(r.total_dets for r in results),
            "scored": sum(r.frames_scored for r in results),
            "precision": prec, "recall": rec, "f1": f1,
            "id_acc": cc / ct if ct else float("nan"),
            "ap": average_precision(score_hits, n_gt), "n_gt": n_gt,
            "shots": shots, "expected_shots": exp,
            "fp_shots_per_min": max(0, shots - exp) / (dur / 60.0) if dur else float("nan")}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Eval the current detection pipeline.")
    ap.add_argument("--clips", default=str(ROOT / "_eval" / "clips"))
    ap.add_argument("--labels", default=str(ROOT / "_eval" / "labels"))
    ap.add_argument("--out", default="")
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--timestamp", default="", help="override output subdir name")
    args = ap.parse_args()

    clips_dir = Path(args.clips)
    labels_dir = Path(args.labels)
    # dedupe case-insensitively (Windows globs *.mp4 and *.MP4 to the same file)
    seen, clips = set(), []
    for p in sorted(clips_dir.glob("*.mp4")) + sorted(clips_dir.glob("*.MP4")):
        key = str(p.resolve()).lower()
        if key not in seen:
            seen.add(key)
            clips.append(p)
    if args.subset:
        clips = clips[: args.subset]
    if not clips:
        print(f"No clips in {clips_dir}. Sample some with tools/sample_clips.py first.")
        return 1

    ts = args.timestamp or "run"
    out = Path(args.out) if args.out else (ROOT / "docs" / "eval" / ts)
    out_imgs = out / "images"
    out_imgs.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating {len(clips)} clip(s) -> {out}")
    results = []
    for c in clips:
        lbl = load_labels(labels_dir / f"{c.stem}.json")
        print(f"  {c.name}  (labelled={lbl.labelled}, static={lbl.static}, empty={lbl.empty})")
        r = eval_clip(c, lbl, out_imgs, args.stride, args.max_seconds)
        r.note = lbl.notes
        print(f"    frames={r.frames_processed} calib={_fmt(r.calib_rate,True)} "
              f"P={_fmt(r.precision,True)} R={_fmt(r.recall,True)} F1={_fmt(r.f1,True)} "
              f"shots={r.shots} FPshots/min={_fmt(r.fp_shots_per_min)}")
        results.append(r)

    meta = {"timestamp": ts, "n_clips": len(clips), "stride": args.stride,
            "detector": "v0.2.16 classical (current default)"}
    report = write_report(results, out, meta)
    agg = _aggregate(results)
    print("\n=== AGGREGATE ===")
    print(f"calibration rate: {_fmt(agg['calib_rate'],True)}")
    print(f"precision={_fmt(agg['precision'],True)} recall={_fmt(agg['recall'],True)} "
          f"F1={_fmt(agg['f1'],True)} AP@dist={_fmt(agg['ap'])} ID-acc={_fmt(agg['id_acc'],True)}")
    print(f"FP shots/min={_fmt(agg['fp_shots_per_min'])} (total GT balls={agg['n_gt']})")
    print(f"report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
