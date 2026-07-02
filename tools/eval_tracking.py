"""Self-supervised tracking-quality metrics — no human labels required.

Physics is the free ground truth: settled balls must not move, balls never
teleport, at most 16 exist, each number exists at most once, and real balls
don't flicker in and out of existence. This harness runs the REAL pipeline
over video segments and scores those invariants, so any change (code or
settings) gets a number — the objective function the autonomous tuning loop
(tools/autotune.py) optimizes.

    python tools/eval_tracking.py                          # default segments
    python tools/eval_tracking.py --segments 100:40 620:40 --json out.json

Metrics (all per-minute rates or fractions; lower is better unless noted):
  jitter_px      mean |dpos| of SETTLED tracks between frames (stutter Joe sees)
  teleports      track jumps faster than a physically possible ball
  phantom_rate   short-lived tracks (<0.5 s) per minute — false-positive churn
  dup_numbers    frames with the same number alive on 2+ confirmed tracks
  overcount      frames with >16 confirmed tracks
  id_flips       committed-number changes per track-minute
  fps            pipeline throughput (higher is better)
"""

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billiards_trainer.config import Settings  # noqa: E402
from billiards_trainer.vision.pipeline import Pipeline  # noqa: E402
from billiards_trainer.vision.tracking import BallTracker  # noqa: E402

# A hard-struck cue moves ~10 m/s; on a ~1000 px table that is ~35 px/frame at
# 30 fps in rectified space. Anything over ~3x that in ONE frame is a teleport
# (association error), not motion.
TELEPORT_PX = 110.0
SETTLED_SPEED = 0.6      # px/frame below which a ball is physically "at rest"
PHANTOM_LIFE_S = 0.5     # tracks living shorter than this are churn


def run_segment(video: str, start_s: float, dur_s: float,
                settings: Settings, tracker_kwargs: dict | None = None) -> dict:
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
    pipe = Pipeline(settings, source=video)
    if tracker_kwargs:
        pipe.tracker = BallTracker(**tracker_kwargs)

    prev: dict[int, tuple] = {}          # id -> (x, y, number)
    first_seen: dict[int, int] = {}
    last_seen: dict[int, int] = {}
    numbers: dict[int, int] = {}         # id -> last committed number
    id_flips = 0
    jitters: list[float] = []
    teleports = 0
    dup_frames = 0
    overcount = 0
    n_frames = int(dur_s * fps)
    t0 = time.perf_counter()
    done = 0
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        res = pipe.process(frame, i / fps)
        done += 1
        if res.status not in ("tracking", "deviated"):
            continue
        cur = {}
        nums_alive = defaultdict(int)
        for tr in res.tracks:
            cur[tr.id] = (tr.x, tr.y, tr.number)
            first_seen.setdefault(tr.id, i)
            last_seen[tr.id] = i
            if tr.number >= 0:
                nums_alive[tr.number] += 1
            old_num = numbers.get(tr.id)
            if old_num is not None and old_num >= 0 and tr.number >= 0 \
                    and tr.number != old_num:
                id_flips += 1
            numbers[tr.id] = tr.number
            if tr.id in prev:
                px, py, _ = prev[tr.id]
                d = ((tr.x - px) ** 2 + (tr.y - py) ** 2) ** 0.5
                if d > TELEPORT_PX:
                    teleports += 1
                elif tr.speed < SETTLED_SPEED and d > 0:
                    jitters.append(d)
        if any(c > 1 for c in nums_alive.values()):
            dup_frames += 1
        if len(res.tracks) > 16:
            overcount += 1
        prev = cur
    wall = time.perf_counter() - t0
    cap.release()

    minutes = max(1e-6, done / fps / 60.0)
    lives = [(last_seen[i] - first_seen[i]) / fps for i in first_seen]
    phantoms = sum(1 for v in lives if v < PHANTOM_LIFE_S)
    track_minutes = max(1e-6, sum(lives) / 60.0)
    # median simultaneous confirmed tracks — the recall signal the pure physics
    # metrics can't see (a never-detected resting ball creates no track and no
    # penalty; an optimizer WILL exploit that without this)
    alive_counts = sorted(
        sum(1 for i in first_seen
            if first_seen[i] <= f <= last_seen[i]) for f in range(done))
    median_tracks = alive_counts[len(alive_counts) // 2] if alive_counts else 0
    return {
        "median_tracks": median_tracks,
        "frames": done,
        "jitter_px": round(statistics.mean(jitters), 3) if jitters else 0.0,
        "jitter_p95": round(sorted(jitters)[int(0.95 * (len(jitters) - 1))], 3)
                      if jitters else 0.0,
        "teleports_per_min": round(teleports / minutes, 2),
        "phantom_rate_per_min": round(phantoms / minutes, 2),
        "dup_number_frac": round(dup_frames / max(1, done), 4),
        "overcount_frac": round(overcount / max(1, done), 4),
        "id_flips_per_track_min": round(id_flips / track_minutes, 2),
        "tracks_total": len(first_seen),
        "fps": round(done / wall, 1),
    }


def score(m: dict, expected_tracks: float | None = None) -> float:
    """Single scalar objective (LOWER is better) for the tuning loop. Weights
    encode what a practice aid must get right: no phantom balls, no stutter,
    stable identities, and EVERY real ball tracked; throughput only matters
    below real-time. ``expected_tracks`` (human-verified ball count for the
    segment) closes the recall blind spot: without it an optimizer can win by
    simply not detecting hard balls."""
    fps_penalty = max(0.0, 30.0 - m["fps"]) * 0.5
    missing = (max(0.0, expected_tracks - m.get("median_tracks", 0)) * 5.0
               if expected_tracks else 0.0)
    return (m["jitter_px"] * 30.0
            + m["teleports_per_min"] * 2.0
            + m["phantom_rate_per_min"] * 3.0
            + m["dup_number_frac"] * 50.0
            + m["overcount_frac"] * 50.0
            + m["id_flips_per_track_min"] * 1.0
            + fps_penalty + missing)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default=str(ROOT / "testVideo.MP4"))
    ap.add_argument("--segments", nargs="*", default=["100:30", "480:30", "1140:30"],
                    metavar="START:DUR", help="segments in seconds")
    ap.add_argument("--settings-json", default="",
                    help="JSON settings overrides, e.g. from autotune")
    ap.add_argument("--json", default="", help="write metrics JSON here")
    args = ap.parse_args()

    settings = Settings()
    if args.settings_json:
        overrides = json.loads(Path(args.settings_json).read_text()
                               if Path(args.settings_json).exists()
                               else args.settings_json)
        for dotted, v in overrides.items():
            obj = settings
            *parents, leaf = dotted.split(".")
            for p in parents:
                obj = getattr(obj, p)
            setattr(obj, leaf, v)

    per_seg = []
    for seg in args.segments:
        parts = [float(x) for x in seg.split(":")]
        start, dur = parts[0], parts[1]
        expected = parts[2] if len(parts) > 2 else None  # human-verified ball count
        m = run_segment(args.video, start, dur, settings)
        m["segment"] = seg
        m["score"] = round(score(m, expected), 3)
        per_seg.append(m)
        print(f"[{seg:>9}] " + "  ".join(
            f"{k}={v}" for k, v in m.items() if k not in ("segment",)))

    agg = {k: round(statistics.mean(s[k] for s in per_seg), 3)
           for k in per_seg[0]
           if k not in ("segment", "frames", "tracks_total", "median_tracks")}
    print("\nAGGREGATE:", "  ".join(f"{k}={v}" for k, v in agg.items()))
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"aggregate": agg, "segments": per_seg}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
