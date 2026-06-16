"""Multi-variant detector experiment harness (Phase 1) — self-healing.

Runs every registered detector STRATEGY against the eval clips on the RAW frame,
scores each vs the hand-labelled ground truth (reusing eval_harness scoring), and
produces a ranked comparison + side-by-side visual diff.

SELF-HEALING: each variant runs in its OWN subprocess with a timeout. A variant
that crashes, hangs, or OOMs is marked FAILED/TIMEOUT in the report — it can never
block the others.

SELF-EVALUATION vs the Phase-0 baseline (P 0.264 / R 0.026 / F1 0.047):
  - beats baseline F1            -> CANDIDATE
  - beats baseline F1 by >50%    -> RECOMMENDED
  - >50% recall on idle_02       -> flagged as leading candidate

    python tools/eval_experiments.py                       # run ALL variants
    python tools/eval_experiments.py --only felt_mask_hough,simple_blob
    python tools/eval_experiments.py --variant <name> ...  # internal worker mode
"""

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

import eval_harness as EH  # noqa: E402

BASELINE = {"precision": 0.264, "recall": 0.026, "f1": 0.047}
KEYFRAMES_PER_CLIP = 4
VARIANT_TIMEOUT_S = 1800


# --------------------------------------------------------------------------- #
# Worker: run ONE strategy over the eval set
# --------------------------------------------------------------------------- #
def _calibrate(clip_path, settings):
    """Calibrate a clip once (first frame that locks); shared across all frames."""
    from billiards_trainer.vision.calibration import CalibrationManager
    cap = cv2.VideoCapture(str(clip_path))
    mgr = CalibrationManager()
    tries = 0
    while tries < 30:
        ok, frame = cap.read()
        if not ok:
            break
        tries += 1
        try:
            if mgr.calibrate(frame, settings):
                cap.release()
                return mgr.calib
        except Exception:  # noqa: BLE001
            pass
    cap.release()
    return None


def run_worker(name, args) -> int:
    from billiards_trainer.config import Settings
    from billiards_trainer.detector_strategies import discover
    strategies = discover()
    strat = strategies.get(name)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"variant": name, "status": "success", "error": "", "fps": 0.0,
           "ms_per_frame": 0.0, "keyframes": [], "per_clip": {}}
    if strat is None:
        res["status"] = "failed"
        res["error"] = f"strategy '{name}' not found (discovery: {list(strategies)})"
        (out / "result.json").write_text(json.dumps(res, indent=2))
        return 0
    res["description"] = strat.description

    settings = Settings()
    clips = _list_clips(Path(args.clips))
    labels_dir = Path(args.labels)

    # Probe: try one detect on the first frame so an un-runnable variant (e.g.
    # missing torch) is marked FAILED with a clear error, not silently zero.
    if clips:
        cap0 = cv2.VideoCapture(str(clips[0]))
        ok0, f0 = cap0.read()
        cap0.release()
        if ok0:
            try:
                strat.detect(f0, _calibrate(clips[0], settings))
            except Exception:  # noqa: BLE001
                res["status"] = "failed"
                res["error"] = traceback.format_exc()[-1500:]
                (out / "result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
                print(f"[{name}] FAILED at probe: {res['error'].splitlines()[-1]}")
                return 0

    img_dir = out / "img"
    img_dir.mkdir(exist_ok=True)
    agg = {"tp": 0, "fp": 0, "fn": 0, "cc": 0, "ct": 0, "dets": 0, "n_gt": 0,
           "frames": 0, "calib_frames": 0, "scored": 0}
    score_hits = []
    total_ms, n_calls = 0.0, 0

    for clip in clips:
        lbl = EH.load_labels(labels_dir / f"{clip.stem}.json")
        calib = _calibrate(clip, settings)
        cap = cv2.VideoCapture(str(clip))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        tol = lbl.tolerance_px or 0.025 * w
        maxf = total if args.max_seconds <= 0 else min(total, int(args.max_seconds * fps))
        kf_every = max(1, maxf // KEYFRAMES_PER_CLIP)
        cstat = {"tp": 0, "fp": 0, "fn": 0, "dets": 0, "frames": 0, "calib": calib is not None}
        fidx = 0
        while fidx < maxf:
            ok, frame = cap.read()
            if not ok:
                break
            if args.stride > 1 and fidx % args.stride:
                fidx += 1
                continue
            agg["frames"] += 1
            cstat["frames"] += 1
            if calib is not None:
                agg["calib_frames"] += 1
            try:
                t0 = time.perf_counter()
                dets = strat.detect(frame, calib)
                total_ms += (time.perf_counter() - t0) * 1000.0
                n_calls += 1
            except Exception:  # noqa: BLE001 - one bad frame must not abort the variant
                dets = []
            agg["dets"] += len(dets)
            cstat["dets"] += len(dets)
            gt = lbl.balls_at(fidx)
            if gt is not None:
                agg["scored"] += 1
                agg["n_gt"] += len(gt)
                det_xy = [(d.x, d.y) for d in dets]
                pairs = EH.match(det_xy, [(g["x"], g["y"]) for g in gt], tol)
                md = {di for di, _ in pairs}
                mg = {gi for _, gi in pairs}
                ntp, nfp, nfn = len(pairs), len(dets) - len(md), len(gt) - len(mg)
                agg["tp"] += ntp
                cstat["tp"] += ntp
                agg["fp"] += nfp
                cstat["fp"] += nfp
                agg["fn"] += nfn
                cstat["fn"] += nfn
                for di, gi in pairs:
                    if "cls" in gt[gi]:
                        agg["ct"] += 1
                        if dets[di].cls.value == gt[gi]["cls"]:
                            agg["cc"] += 1
                for di, d in enumerate(dets):
                    score_hits.append((d.score, 1 if di in md else 0))
            if fidx % kf_every == 0 and len([k for k in res["keyframes"] if clip.stem in k]) < KEYFRAMES_PER_CLIP:
                fn = f"{clip.stem}_f{fidx:06d}.jpg"
                cv2.imwrite(str(img_dir / fn), _overlay(frame, dets, gt), [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                res["keyframes"].append(fn)
            fidx += 1
        cap.release()
        res["per_clip"][clip.name] = _rates(cstat)

    res["metrics"] = _rates(agg) | {
        "ap": EH.average_precision(score_hits, agg["n_gt"]),
        "id_acc": agg["cc"] / agg["ct"] if agg["ct"] else float("nan"),
        "calib_rate": agg["calib_frames"] / agg["frames"] if agg["frames"] else 0.0,
        "total_dets": agg["dets"], "n_gt": agg["n_gt"], "scored": agg["scored"],
        "frames": agg["frames"],
    }
    res["ms_per_frame"] = total_ms / n_calls if n_calls else 0.0
    res["fps"] = 1000.0 / res["ms_per_frame"] if res["ms_per_frame"] > 0.01 else 0.0
    (out / "result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[{name}] P={res['metrics']['precision']} R={res['metrics']['recall']} "
          f"F1={res['metrics']['f1']} fps={res['fps']:.1f}")
    return 0


def _rates(d):
    tp, fp, fn = d["tp"], d["fp"], d["fn"]
    p = tp / (tp + fp) if (tp + fp) else float("nan")
    r = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * p * r / (p + r) if (p == p and r == r and (p + r)) else float("nan")
    return {"precision": _n(p), "recall": _n(r), "f1": _n(f1),
            "tp": tp, "fp": fp, "fn": fn}


def _n(x):
    return None if (x is None or x != x) else round(float(x), 4)


def _overlay(frame, dets, gt):
    img = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
    sx = 640 / frame.shape[1]
    for d in dets:
        cv2.circle(img, (int(d.x * sx), int(d.y * sx)), max(3, int(d.radius * sx)),
                   (0, 215, 255), 2, cv2.LINE_AA)
    if gt:
        for g in gt:
            cv2.drawMarker(img, (int(g["x"] * sx), int(g["y"] * sx)), (60, 255, 60),
                           cv2.MARKER_CROSS, 12, 2)
    return img


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def _list_clips(d: Path):
    seen, out = set(), []
    for p in sorted(d.glob("*.mp4")) + sorted(d.glob("*.MP4")):
        k = str(p.resolve()).lower()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def run_orchestrator(args) -> int:
    from billiards_trainer.detector_strategies import discover
    names = list(discover())
    if args.only:
        wanted = set(args.only.split(","))
        names = [n for n in names if n in wanted]
    if not names:
        print("No strategies discovered.")
        return 1
    out = Path(args.out) if args.out else ROOT / "docs" / "eval" / "experiments" / (args.timestamp or "run")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(names)} variants -> {out}\n  variants: {names}")

    results = {}
    for name in names:
        vout = out / "variants" / name
        vout.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(ROOT / "tools" / "eval_experiments.py"),
               "--variant", name, "--clips", args.clips, "--labels", args.labels,
               "--out", str(vout), "--stride", str(args.stride),
               "--max-seconds", str(args.max_seconds)]
        status, err = "success", ""
        t0 = time.perf_counter()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=VARIANT_TIMEOUT_S, cwd=str(ROOT))
            if p.returncode != 0:
                status, err = "failed", (p.stderr or "")[-1500:]
        except subprocess.TimeoutExpired:
            status, err = "timeout", f"exceeded {VARIANT_TIMEOUT_S}s"
        except Exception as exc:  # noqa: BLE001
            status, err = "failed", f"{exc}\n{traceback.format_exc()[-1200:]}"
        wall = time.perf_counter() - t0
        rj = vout / "result.json"
        if rj.exists():
            r = json.loads(rj.read_text(encoding="utf-8"))
            if status != "success" and r.get("status") == "success":
                r["status"] = status
                r["error"] = err
        else:
            r = {"variant": name, "status": status or "failed", "error": err,
                 "metrics": {}, "fps": 0.0, "keyframes": []}
        r["wall_s"] = round(wall, 1)
        results[name] = r
        print(f"  {name}: {r.get('status')} ({wall:.0f}s)")

    _write_comparison(results, out)
    _write_visual(results, out)
    print(f"\ncomparison: {out / 'comparison.md'}")
    return 0


def _verdict(m):
    f1 = m.get("f1")
    if f1 is None:
        return ""
    if f1 > BASELINE["f1"] * 1.5:
        return "RECOMMENDED"
    if f1 > BASELINE["f1"]:
        return "CANDIDATE"
    return ""


def _write_comparison(results, out: Path):
    rows = []
    for name, r in results.items():
        m = r.get("metrics", {}) or {}
        rows.append((name, r, m, _verdict(m) if r.get("status") == "success" else ""))
    rows.sort(key=lambda x: (x[1].get("status") == "success",
                             (x[2].get("f1") or -1)), reverse=True)

    def fmt(x, pct=False):
        if x is None or (isinstance(x, float) and x != x):
            return "—"
        return f"{x*100:.1f}%" if pct else f"{x:.2f}"

    lines = ["# Detector experiments — comparison", "",
             f"Baseline (Phase 0, classical): P {BASELINE['precision']*100:.0f}% · "
             f"R {BASELINE['recall']*100:.1f}% · F1 {BASELINE['f1']*100:.1f}%. "
             "All variants detect on the **raw frame** and are scored in raw coords "
             "vs hand-labelled GT. **CANDIDATE** = beats baseline F1; **RECOMMENDED** "
             "= beats it by >50%.", "",
             "| variant | status | precision | recall | F1 | ID acc | calib | dets | FPS | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r, m, v in rows:
        st = r.get("status", "?")
        badge = {"success": "✅", "failed": "❌", "timeout": "⏱️"}.get(st, st)
        lines.append(
            f"| `{name}` | {badge} {st} | {fmt(m.get('precision'),1)} | "
            f"{fmt(m.get('recall'),1)} | {fmt(m.get('f1'),1)} | {fmt(m.get('id_acc'),1)} | "
            f"{fmt(m.get('calib_rate'),1)} | {m.get('total_dets','—')} | "
            f"{fmt(r.get('fps'))} | **{v}** |")
    # idle_02 spotlight
    lines += ["", "## idle_02 recall (cleanest 7-ball clip)"]
    for name, r, _m, _v in rows:
        pc = (r.get("per_clip") or {}).get("idle_02_0574s.mp4")
        if pc and pc.get("recall") is not None:
            flag = "  ⭐ **>50% recall — leading candidate**" if pc["recall"] > 0.5 else ""
            lines.append(f"- `{name}`: recall {pc['recall']*100:.1f}%{flag}")
    # failures
    fails = [(n, r) for n, r, _m, _v in rows if r.get("status") != "success"]
    if fails:
        lines += ["", "## Failed / timed-out variants (self-healed — did not block others)"]
        for n, r in fails:
            lines.append(f"- `{n}` ({r.get('status')}): {(r.get('error') or '')[:300]}")
    lines += ["", "_Per-frame visual diffs: `visual.html`. Raw metrics: "
              "`variants/<name>/result.json`._"]
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def _write_visual(results, out: Path):
    # group keyframes by (clip, frame) across variants
    frames = {}
    for name, r in results.items():
        for kf in r.get("keyframes", []):
            frames.setdefault(kf, {})[name] = f"variants/{name}/img/{kf}"
    blocks = []
    for kf in sorted(frames):
        cells = "".join(
            f'<figure><img src="{src}" loading="lazy"><figcaption>{n}</figcaption></figure>'
            for n, src in sorted(frames[kf].items()))
        blocks.append(f"<h3>{kf}</h3><div class=row>{cells}</div>")
    doc = ("<!doctype html><meta charset=utf-8><title>experiments visual diff</title>"
           "<style>body{font:13px sans-serif;margin:18px}.row{display:flex;flex-wrap:wrap;gap:8px}"
           "figure{margin:0}img{width:380px;border:1px solid #ccc}"
           "figcaption{font-size:11px}</style>"
           "<h1>Detector experiments — visual diff</h1>"
           "<p>Yellow circles = that variant's detections. Green crosses = ground truth.</p>"
           + "".join(blocks))
    (out / "visual.html").write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=str(ROOT / "_eval" / "clips"))
    ap.add_argument("--labels", default=str(ROOT / "_eval" / "labels"))
    ap.add_argument("--out", default="")
    ap.add_argument("--only", default="", help="comma-separated variant names")
    ap.add_argument("--variant", default="", help="internal worker mode")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--timestamp", default="run")
    args = ap.parse_args()
    if args.variant:
        return run_worker(args.variant, args)
    return run_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
