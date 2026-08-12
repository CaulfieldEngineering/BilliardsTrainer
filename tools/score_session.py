"""Score the real vision pipeline against the laws of pool — no labels needed.

Runs the exact Pipeline the app runs over recorded session footage, feeds its
tracks to the invariant scorer, and reports violations of physical law
(duplicate balls, teleports, identity flicker, ...) per 1000 ball-frames.

This is the honest-baseline tool: run it before and after any vision change,
on any session in the Dropbox folder, unattended.

    python tools/score_session.py --video <session.mp4>
    python tools/score_session.py --video <session.mp4> --start 300 --duration 120
    python tools/score_session.py --video <session.mp4> --stride 3 --game nine
    python tools/score_session.py --latest            # newest session in the recording dir
    python tools/score_session.py --latest --json out.json --dump-worst 8

``--dump-worst`` saves annotated full frames for the most frequent violation
kinds so failures can be *looked at*, not just counted. ``--json`` writes the
full report for the promotion gate / trend tracking.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billiards_trainer.config import Settings  # noqa: E402
from billiards_trainer.eval.invariants import (  # noqa: E402
    SequenceScorer,
    bed_and_pockets,
)
from billiards_trainer.vision.pipeline import Pipeline  # noqa: E402

_GAME_MAX = {"nine": 9, "ten": 10, "eight": 15, "any": None}


def latest_session(settings: Settings) -> Path | None:
    d = Path(settings.recording.resolved_dir())
    vids = sorted(d.glob("session-*.mp4"), key=lambda p: p.stat().st_mtime)
    return vids[-1] if vids else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default="")
    ap.add_argument("--latest", action="store_true",
                    help="score the newest session in the recording folder")
    ap.add_argument("--start", type=float, default=0.0, help="seek offset, seconds")
    ap.add_argument("--duration", type=float, default=0.0, help="0 = to the end")
    ap.add_argument("--stride", type=int, default=1,
                    help="process every Nth frame (temporal checks stay valid: "
                    "speed limits are per *processed* step and scale up)")
    ap.add_argument("--game", choices=sorted(_GAME_MAX), default="any",
                    help="restricts legal ball numbers (nine -> 1..9)")
    ap.add_argument("--json", default="", help="write full report as JSON")
    ap.add_argument("--dump-worst", type=int, default=0, metavar="N",
                    help="save N annotated frames of the worst violations")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    settings = Settings.load()
    video = Path(args.video) if args.video else (latest_session(settings) or Path())
    if not video.is_file():
        print(f"no video: {video}", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"could not open {video}", file=sys.stderr)
        return 2
    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)

    pipeline = Pipeline(settings)
    scorer: SequenceScorer | None = None
    max_number = _GAME_MAX[args.game]

    # Keep raw frames for --dump-worst without holding the whole video: remember
    # only frame indices we may want, second pass extracts them.
    end_t = args.start + args.duration if args.duration > 0 else float("inf")
    n_read = n_proc = 0
    t0 = time.perf_counter()
    frame_of_violation: dict[int, int] = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t > end_t:
            break
        n_read += 1
        if (n_read - 1) % max(1, args.stride):
            continue
        res = pipeline.process(frame, t, annotate=False)
        n_proc += 1

        if scorer is None and res.rect_bgr is not None:
            h, w = res.rect_bgr.shape[:2]
            bed, pockets = bed_and_pockets(float(w), float(h))
            scorer = SequenceScorer(bed=bed, max_number=max_number, pockets=pockets)
        if scorer is not None:
            before = len(scorer.report.violations)
            scorer.add(res.tracks, n_read - 1, tracking=(res.status == "tracking"))
            for v in scorer.report.violations[before:]:
                frame_of_violation.setdefault(v.frame, n_read - 1)
        if not args.quiet and n_proc % 300 == 0:
            el = time.perf_counter() - t0
            print(f"  {n_proc} frames scored ({n_proc / el:5.1f} fps)...")

    cap.release()
    if scorer is None:
        print("pipeline never produced a rectified view — table was never "
              "locked. That itself is the headline failure for this clip.")
        return 1

    rep = scorer.report
    print(f"\n=== {video.name}  ({n_proc} frames scored, stride {args.stride}) ===")
    print(rep.summary())

    if args.json:
        out = {
            "video": video.name,
            "frames": rep.frames,
            "tracking_frames": rep.tracking_frames,
            "ball_frames": rep.ball_frames,
            "coverage": rep.coverage,
            "count_stability": rep.count_stability,
            "degenerate": rep.degenerate,
            "impossible": rep.impossible,
            "implausible": rep.implausible,
            "rate_impossible": rep.rate(severity="impossible"),
            "counts": rep.counts,
            "violations": [
                {"kind": v.kind, "frame": v.frame, "severity": v.severity,
                 "detail": v.detail}
                for v in rep.violations[:2000]     # cap: files stay reviewable
            ],
        }
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"json -> {args.json}")

    if args.dump_worst and rep.violations:
        _dump_worst(video, rep, frame_of_violation, args.dump_worst)
    return 0


def _dump_worst(video: Path, rep, frame_of, n: int) -> None:
    """Second pass: pull the actual frames where the worst violations happened."""
    out_dir = ROOT / "_eval" / "violations" / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = rep.worst(n)
    wanted = {}
    for v in picks:
        if v.frame in frame_of:
            wanted.setdefault(frame_of[v.frame], v)
    cap = cv2.VideoCapture(str(video))
    for idx, v in sorted(wanted.items()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.putText(frame, str(v)[:110], (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        path = out_dir / f"f{idx:07d}_{v.kind}.png"
        cv2.imwrite(str(path), frame)
        print(f"  saved {path.relative_to(ROOT)}")
    cap.release()


if __name__ == "__main__":
    raise SystemExit(main())
