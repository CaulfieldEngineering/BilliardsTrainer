"""Sample representative eval clips from a long source video.

Scans the video for motion (cheap frame-diff energy), classifies each second as
idle / active, then cuts a diverse set of 30-45s clips: several IDLE windows
(for false-positive testing) and several ACTIVE windows (shots / motion). Clips
are written native-resolution to ``_eval/clips`` (gitignored).

    python tools/sample_clips.py --video testVideo.MP4
    python tools/sample_clips.py --video testVideo.MP4 --scan-only   # just print timeline

No ffmpeg dependency — uses OpenCV to seek + re-encode (mp4v).
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

try:  # Windows consoles default to cp1252 — make our output UTF-8 safe
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass


def scan_motion(cap, fps, sample_hz=3.0):
    """Return per-second max motion energy (% of centre-ROI pixels changed)."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / sample_hz)))
    prev = None
    per_sec: dict[int, float] = {}
    fidx = 0
    while fidx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        roi = frame[h // 5: h * 4 // 5, w // 5: w * 4 // 5]
        g = cv2.cvtColor(cv2.resize(roi, (160, 90)), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            m = float((cv2.absdiff(g, prev) > 25).mean()) * 100.0
            sec = int(fidx / fps)
            per_sec[sec] = max(per_sec.get(sec, 0.0), m)
        prev = g
        fidx += step
    return per_sec


def pick_clips(per_sec, duration_s, clip_len, idle_thr, active_thr, n_idle, n_active):
    secs = sorted(per_sec)
    idle_runs, cur = [], []
    for s in secs:
        if per_sec[s] < idle_thr:
            cur.append(s)
        else:
            if len(cur) >= clip_len:
                idle_runs.append((cur[0], cur[-1]))
            cur = []
    if len(cur) >= clip_len:
        idle_runs.append((cur[0], cur[-1]))

    # active spikes: local seconds above active threshold, de-clustered
    active_secs = [s for s in secs if per_sec[s] >= active_thr]
    active_centers, last = [], -999
    for s in active_secs:
        if s - last >= clip_len:
            active_centers.append(s)
            last = s

    chosen = []
    # spread idle clips across the available runs
    if idle_runs:
        idx = np.linspace(0, len(idle_runs) - 1, min(n_idle, len(idle_runs)))
        for i in sorted(set(int(round(x)) for x in idx)):
            a, b = idle_runs[i]
            start = a + max(0, (b - a - clip_len) // 2)
            chosen.append(("idle", start))
    # spread active clips
    if active_centers:
        idx = np.linspace(0, len(active_centers) - 1, min(n_active, len(active_centers)))
        for i in sorted(set(int(round(x)) for x in idx)):
            c = active_centers[i]
            chosen.append(("active", max(0, c - clip_len // 3)))
    # de-dup overlapping starts
    chosen.sort(key=lambda x: x[1])
    out, last_start = [], -999
    for kind, start in chosen:
        if start - last_start >= clip_len // 2:
            out.append((kind, start))
            last_start = start
    return out


def cut_clip(cap, fps, start_s, clip_len, dst: Path, w, h):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = int(clip_len * fps)
    written = 0
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    writer.release()
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="testVideo.MP4")
    ap.add_argument("--out", default=str(ROOT / "_eval" / "clips"))
    ap.add_argument("--clip-len", type=int, default=40)
    ap.add_argument("--idle-thr", type=float, default=0.5)
    ap.add_argument("--active-thr", type=float, default=3.0)
    ap.add_argument("--n-idle", type=int, default=6)
    ap.add_argument("--n-active", type=int, default=9)
    ap.add_argument("--scan-only", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open {args.video}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
    print(f"{args.video}: {w}x{h} {fps:.1f}fps {dur/60:.1f}min — scanning motion…")

    per_sec = scan_motion(cap, fps)
    vals = np.array(list(per_sec.values())) if per_sec else np.array([0.0])
    print(f"motion: min={vals.min():.2f} med={np.median(vals):.2f} "
          f"p90={np.percentile(vals,90):.2f} max={vals.max():.2f}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir.parent / "motion_timeline.json").write_text(
        json.dumps({str(k): round(v, 3) for k, v in sorted(per_sec.items())}), encoding="utf-8")

    chosen = pick_clips(per_sec, dur, args.clip_len, args.idle_thr, args.active_thr,
                        args.n_idle, args.n_active)
    print(f"selected {len(chosen)} clips:")
    for kind, start in chosen:
        print(f"  {kind:7s} @ {start//60:02d}:{start%60:02d}  (motion≈{per_sec.get(start,0):.2f})")
    if args.scan_only:
        return 0

    counts: dict[str, int] = {}
    for kind, start in chosen:
        counts[kind] = counts.get(kind, 0) + 1
        name = f"{kind}_{counts[kind]:02d}_{start:04d}s.mp4"
        dst = out_dir / name
        n = cut_clip(cap, fps, start, args.clip_len, dst, w, h)
        print(f"  wrote {name} ({n} frames, {dst.stat().st_size//1_000_000}MB)")
    cap.release()
    print(f"\nDone. {len(chosen)} clips in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
