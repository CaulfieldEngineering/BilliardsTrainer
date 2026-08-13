"""Score every recorded session against the laws of pool, unattended.

The corpus IS the test set: 26 GB of real footage under real lighting with
real drills. Runs tools/score_session.py over each session (bounded per-clip
so one marathon video can't eat the night), writes one JSON per session plus
an aggregate, and prints a league table. Run it before and after a vision
change and diff the aggregates — that is the whole quality process.

    python tools/score_corpus.py                       # default: recording dir
    python tools/score_corpus.py --max-seconds 300 --stride 5
    python tools/score_corpus.py --out _eval/corpus/my_experiment

Skips sessions already scored into --out (resumable), so a crashed or
interrupted run continues where it left off.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billiards_trainer.config import Settings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="", help="session folder (default: recording dir)")
    ap.add_argument("--out", default="", help="output folder (default: _eval/corpus/<utc-date>)")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-seconds", type=float, default=300.0,
                    help="cap per session; keeps a 20-video night bounded")
    ap.add_argument("--start", type=float, default=60.0,
                    help="skip the first N seconds of each session (setup, empty table)")
    ap.add_argument("--game", default="nine")
    ap.add_argument("--min-mb", type=float, default=20.0,
                    help="ignore stub recordings smaller than this")
    args = ap.parse_args()

    src = Path(args.dir) if args.dir else Path(Settings.load().recording.resolved_dir())
    vids = sorted(p for p in src.glob("session-*.mp4")
                  if p.stat().st_size >= args.min_mb * 1e6)
    if not vids:
        print(f"no sessions >= {args.min_mb} MB in {src}", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_dir = Path(args.out) if args.out else ROOT / "_eval" / "corpus" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"scoring {len(vids)} sessions -> {out_dir}")
    t0 = time.time()
    for i, v in enumerate(vids, 1):
        dest = out_dir / f"{v.stem}.json"
        if dest.exists():
            print(f"[{i}/{len(vids)}] {v.name}: already scored, skipping")
            continue
        print(f"[{i}/{len(vids)}] {v.name} ...", flush=True)
        cmd = [sys.executable, str(ROOT / "tools" / "score_session.py"),
               "--video", str(v), "--start", str(args.start),
               "--duration", str(args.max_seconds), "--stride", str(args.stride),
               "--game", args.game, "--json", str(dest), "--quiet"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0 or not dest.exists():
            # Score-the-failure principle: a clip the pipeline can't lock on is
            # DATA (the worst kind of failure), not a gap in the report.
            dest.write_text(json.dumps({
                "video": v.name, "failed": True, "returncode": r.returncode,
                "stderr": r.stderr[-2000:], "stdout": r.stdout[-2000:],
            }, indent=2), encoding="utf-8")
            print(f"    FAILED (rc={r.returncode}) — recorded as a failure")

    rows = []
    for f in sorted(out_dir.glob("session-*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        rows.append(d)

    agg = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "sessions": len(rows),
        "failed": sum(1 for d in rows if d.get("failed")),
        "degenerate": sum(1 for d in rows if d.get("degenerate")),
        "total_ball_frames": sum(d.get("ball_frames", 0) for d in rows),
        "total_impossible": sum(d.get("impossible", 0) for d in rows),
        "sessions_detail": [
            {**{k: d.get(k) for k in ("video", "failed", "degenerate", "coverage",
                                      "count_stability", "impossible",
                                      "rate_impossible", "ball_frames")},
             # identifier-quality signal for the challenger gate: stripe
             # misreads show up here, not in the impossible rate
             "out_of_game": (d.get("counts") or {}).get("out_of_game_number", 0)}
            for d in rows],
    }
    bf = agg["total_ball_frames"]
    agg["rate_impossible"] = (1000.0 * agg["total_impossible"] / bf) if bf else None
    (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

    print(f"\n{'session':42s} {'cover':>6s} {'stab':>6s} {'imp/1k':>7s}")
    for d in agg["sessions_detail"]:
        if d.get("failed"):
            print(f"{d['video']:42s} {'FAILED':>6s}")
            continue
        flag = " DEGEN" if d.get("degenerate") else ""
        print(f"{d['video']:42s} {d['coverage']:6.2f} "
              f"{d['count_stability']:6.1%} {d['rate_impossible']:7.2f}{flag}")
    print(f"\naggregate: {agg['sessions']} sessions, {bf} ball-frames, "
          f"impossible {agg['rate_impossible'] and round(agg['rate_impossible'], 2)} /1k, "
          f"{agg['failed']} failed, {agg['degenerate']} degenerate "
          f"({(time.time() - t0) / 60:.0f} min)")
    print(f"-> {out_dir / 'aggregate.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
