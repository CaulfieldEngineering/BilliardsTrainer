"""Backfill analysis sidecars for existing sessions.

New recordings write their sidecar live; this analyzes OLD session files
once so their playback is instant and smooth too. The autonomy loop runs
it over the library; each file takes a few minutes and never needs doing
again (unless a better model is promoted — then re-run with --force).

    python tools/build_analysis_cache.py --video <session.mp4>
    python tools/build_analysis_cache.py --all [--force]
"""

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billiards_trainer.config import Settings  # noqa: E402
from billiards_trainer.vision.analysis_cache import (  # noqa: E402
    SidecarReader,
    SidecarWriter,
)
from billiards_trainer.vision.pipeline import Pipeline  # noqa: E402


def build(video: Path, force: bool = False) -> bool:
    if SidecarReader.exists(video) and not force:
        print(f"  {video.name}: sidecar exists, skipping")
        return True
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  {video.name}: unreadable", file=sys.stderr)
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pipeline = Pipeline(Settings.load())
    writer = SidecarWriter(video, {"fps": fps})
    n, last_t = 0, -1.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        n += 1
        # analysis cadence: detect every 3rd frame; write states at 10Hz
        res = pipeline.process(frame, t, annotate=False, detect=(n % 3 == 0))
        if res.shot_event is not None:
            writer.add_shot(res.shot_event)
        if res.status == "tracking" and t - last_t >= 0.1:
            last_t = t
            writer.add_frame(t, res.tracks,
                             carried_ids=res.carried_ids,
                             foreign_frac=res.foreign_frac)
        if n % 1800 == 0:
            print(f"    {t/60:.1f} min analyzed...")
    cap.release()
    writer.close()
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.video:
        vids = [Path(args.video)]
    elif args.all:
        d = Path(Settings.load().recording.resolved_dir())
        vids = sorted(d.glob("session-*.mp4"))
    else:
        print("need --video or --all", file=sys.stderr)
        return 2
    ok = True
    for v in vids:
        print(f"[{v.name}]")
        ok = build(v, args.force) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
