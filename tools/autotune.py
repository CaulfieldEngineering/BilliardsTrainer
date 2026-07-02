"""Autonomous parameter tuning for detection + tracking — no human in the loop.

Random-search over the tracker/detector knob space, scoring each candidate with
the self-supervised physics metrics from tools/eval_tracking.py (settled balls
don't move, nothing teleports, numbers are unique, tracks don't churn). Writes a
ranked report + the best configuration to ``docs/autotune/`` so a reviewer (or a
later Claude session) can promote winners into the shipped defaults.

    python tools/autotune.py --trials 40                 # ~45 s per trial
    python tools/autotune.py --trials 6 --segments 100:12 480:12   # smoke run

Designed to run unattended (e.g. nightly via Task Scheduler): deterministic
seeding, per-trial exception containment, atomic report writes.
"""

import argparse
import json
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval_tracking import run_segment, score  # noqa: E402

from billiards_trainer.config import Settings  # noqa: E402

# Search space. Tracker keys go to BallTracker(**kwargs); dotted keys are
# Settings fields. Ranges bracket the shipped defaults (the baseline trial).
SPACE = {
    "tracker.max_dist_frac":   ("f", 0.05, 0.12),
    "tracker.max_misses":      ("i", 12, 45),
    "tracker.min_hits":        ("i", 2, 5),
    "tracker.vel_alpha":       ("f", 0.40, 0.80),
    "tracker.pos_alpha_slow":  ("f", 0.08, 0.30),
    "tracker.pos_alpha_fast":  ("f", 0.70, 0.98),
    "tracker.still_speed_frac": ("f", 0.005, 0.015),
    "tracker.still_frames":    ("i", 4, 10),
    "tracker.lock_dist_frac":  ("f", 0.008, 0.020),
    "detection.confidence_floor": ("f", 0.35, 0.65),
    "balls.far_rail_rescan":   ("c", [True, False]),
}


def sample(rng: random.Random) -> dict:
    out = {}
    for k, spec in SPACE.items():
        kind = spec[0]
        if kind == "f":
            out[k] = round(rng.uniform(spec[1], spec[2]), 4)
        elif kind == "i":
            out[k] = rng.randint(spec[1], spec[2])
        else:
            out[k] = rng.choice(spec[1])
    return out


def evaluate(params: dict | None, video: str, segments: list[str]) -> dict:
    settings = Settings()
    tracker_kwargs = {}
    if params:
        for k, v in params.items():
            if k.startswith("tracker."):
                tracker_kwargs[k.split(".", 1)[1]] = v
            else:
                obj = settings
                *parents, leaf = k.split(".")
                for p in parents:
                    obj = getattr(obj, p)
                setattr(obj, leaf, v)
    per = []
    seg_scores = []
    for seg in segments:
        parts = [float(x) for x in seg.split(":")]
        start, dur = parts[0], parts[1]
        expected = parts[2] if len(parts) > 2 else None  # human-verified count
        m = run_segment(video, start, dur, settings, tracker_kwargs or None)
        per.append(m)
        seg_scores.append(score(m, expected))
    agg = {k: round(statistics.mean(s[k] for s in per), 3)
           for k in per[0]
           if k not in ("segment", "frames", "tracks_total", "median_tracks")}
    agg["score"] = round(statistics.mean(seg_scores), 3)
    agg["median_tracks"] = [s["median_tracks"] for s in per]
    return agg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default=str(ROOT / "testVideo.MP4"))
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--segments", nargs="*", default=["100:20", "480:20", "1140:20"])
    ap.add_argument("--seed", type=int, default=0, help="0 = derive from date")
    ap.add_argument("--out", default=str(ROOT / "docs" / "autotune"))
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    seed = args.seed or int(stamp[:8])
    rng = random.Random(seed)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"autotune: {args.trials} trials, seed {seed}, segments {args.segments}")
    results = []
    t_all = time.perf_counter()
    baseline = evaluate(None, args.video, args.segments)
    print(f"[baseline] score={baseline['score']}  {baseline}")
    results.append({"trial": "baseline", "params": {}, "metrics": baseline})

    for i in range(args.trials):
        params = sample(rng)
        try:
            m = evaluate(params, args.video, args.segments)
        except Exception as exc:  # noqa: BLE001 - a bad combo must not kill the run
            print(f"[{i:03d}] FAILED: {exc}")
            continue
        results.append({"trial": i, "params": params, "metrics": m})
        marker = "  << improves baseline" if m["score"] < baseline["score"] else ""
        print(f"[{i:03d}] score={m['score']}{marker}")

    ranked = sorted(results, key=lambda r: r["metrics"]["score"])
    best = ranked[0]
    wall_min = (time.perf_counter() - t_all) / 60.0

    report = outdir / f"report-{stamp}.md"
    lines = [f"# Autotune report {stamp}",
             f"{args.trials} trials over {args.segments} "
             f"({wall_min:.0f} min, seed {seed})", "",
             f"Baseline score: **{baseline['score']}**  |  "
             f"Best: **{best['metrics']['score']}** "
             f"({'baseline holds' if best['trial'] == 'baseline' else 'trial ' + str(best['trial'])})",
             "", "| rank | trial | score | jitter | phantoms/min | id flips | fps |",
             "|---|---|---|---|---|---|---|"]
    for rank, r in enumerate(ranked[:10], 1):
        m = r["metrics"]
        lines.append(f"| {rank} | {r['trial']} | {m['score']} | {m['jitter_px']} "
                     f"| {m['phantom_rate_per_min']} | {m['id_flips_per_track_min']} "
                     f"| {m['fps']} |")
    lines += ["", "## Best parameters", "```json",
              json.dumps(best["params"], indent=1), "```",
              "", "Promote by changing the corresponding defaults in "
              "`BallTracker.__init__` / `config.py` and re-running "
              "`tools/eval_tracking.py` as a regression check."]
    report.write_text("\n".join(lines), encoding="utf-8")
    (outdir / f"results-{stamp}.json").write_text(json.dumps(results, indent=1))
    (outdir / "best.json").write_text(json.dumps(
        {"stamp": stamp, "baseline": baseline, "best": best}, indent=1))
    print(f"\nbest score {best['metrics']['score']} (baseline {baseline['score']})")
    print(f"report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
