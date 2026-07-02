"""End-to-end pipeline benchmark on a recorded video.

Answers "how fast is the tracker, and where does the time go" with numbers
instead of vibes. Runs the REAL Pipeline (same code the app runs) over N frames
of a video and reports per-stage medians + effective FPS.

    python tools/bench_pipeline.py                     # testVideo.MP4, 300 frames
    python tools/bench_pipeline.py --video clip.mp4 --frames 500 --start-s 60
    python tools/bench_pipeline.py --strategy onnx_pool_ballid

Stage timings come from ``PipelineResult.diag['stages']`` when the pipeline
provides them; the total is always measured here.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billiards_trainer.config import Settings  # noqa: E402
from billiards_trainer.vision.pipeline import Pipeline  # noqa: E402


def _fmt_ms(vals: list[float]) -> str:
    if not vals:
        return "     -"
    med = statistics.median(vals)
    p95 = sorted(vals)[int(0.95 * (len(vals) - 1))]
    return f"{med:6.1f} / {p95:6.1f}"


def _steady(vals: list[float]) -> list[float]:
    """Drop warm-up frames (calibration + model load) from long runs."""
    return vals[10:] if len(vals) > 30 else vals


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default=str(ROOT / "testVideo.MP4"))
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--start-s", type=float, default=120.0,
                    help="seek offset in seconds (past intro/empty frames)")
    ap.add_argument("--strategy", default="",
                    help="force a detector strategy (default: auto)")
    ap.add_argument("--no-rescan", action="store_true",
                    help="disable the far-rail second inference pass")
    ap.add_argument("--detect-every", type=int, default=1, metavar="N",
                    help="run detection every Nth frame (display-only between)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"could not open {args.video}")
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.start_s * fps))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video: {Path(args.video).name}  {w}x{h} @ {fps:.0f}fps, "
          f"benching {args.frames} frames from t={args.start_s:.0f}s")

    settings = Settings()
    if args.strategy:
        settings.balls.live_strategy = args.strategy
    if args.no_rescan:
        settings.balls.far_rail_rescan = False
    pipe = Pipeline(settings, source=args.video)
    print(f"strategy: {type(pipe._strategy).__name__} "
          f"{getattr(pipe._strategy, 'name', '(none)')}")

    totals: list[float] = []
    stages: dict[str, list[float]] = {}
    n_dets: list[int] = []
    t_bench0 = time.perf_counter()
    for i in range(args.frames):
        ok, frame = cap.read()
        if not ok:
            break
        detect = (i % max(1, args.detect_every)) == 0
        t0 = time.perf_counter()
        res = pipe.process(frame, i / fps, detect=detect)
        totals.append((time.perf_counter() - t0) * 1000.0)
        if detect:
            n_dets.append(len(res.detections))
        for k, v in (res.diag.get("stages") or {}).items():
            stages.setdefault(k, []).append(v)
    wall = time.perf_counter() - t_bench0
    cap.release()

    if not totals:
        print("no frames processed")
        return 2
    # first frames include calibration + model load; report steady state
    steady = _steady(totals)
    print(f"\nframes: {len(totals)}   status: {res.status}   "
          f"balls: {statistics.median(n_dets) if n_dets else 0:.0f}")
    print(f"{'stage':<14} {'med / p95 ms':>15}")
    for k, vals in stages.items():
        print(f"{k:<14} {_fmt_ms(_steady(vals)):>15}")
    print(f"{'TOTAL':<14} {_fmt_ms(steady):>15}")
    med = statistics.median(steady)
    print(f"\npipeline-only FPS (median): {1000.0 / med:.1f}")
    print(f"wall (incl. video decode):  {len(totals) / wall:.1f} fps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
