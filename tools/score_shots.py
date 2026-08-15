"""Validate the shot detector against the session's own audio track.

The vision stack got trustworthy the day it was measured against physics;
the SHOT detector has never been measured at all. Ground truth without
labels comes from a second modality: every real shot begins with a cue-ball
strike — a sharp broadband "clack" the room mic records. Audio onsets and
vision shots are independent witnesses of the same events, so their
agreement bounds precision/recall with no human in the loop.

Honest caveats, by design:
- Audio onsets include ball placements, rail thumps, pocket drops and the
  occasional voice — audio is a SUPERSET of shots. A vision shot with no
  onset inside it is strong evidence of a FALSE shot; an onset with no shot
  around it is only a CANDIDATE miss until a human looks at the dumped frame.
- Shot detection is cadence-sensitive (strike/settle frames are consecutive
  frame counts), so this runs at stride 1 — real-time frame flow, slower to
  process. Use --duration windows, not whole sessions.

    python tools/score_shots.py --video <session.mp4> --start 60 --duration 300
    python tools/score_shots.py --latest --duration 240 --json out.json
"""

import argparse
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billiards_trainer.config import Settings  # noqa: E402
from billiards_trainer.vision.pipeline import Pipeline  # noqa: E402


# --------------------------------------------------------------------------- #
# Audio: impact onsets
# --------------------------------------------------------------------------- #
def audio_onsets(video: Path, start: float, duration: float,
                 min_gap_s: float = 0.30) -> list[float]:
    """Times (absolute, seconds) of sharp transients in the session audio.

    Deliberately primitive — 16 kHz mono, first-difference high-pass, 10 ms
    RMS envelope, adaptive MAD threshold, local-max peak pick. A billiard
    clack is a huge broadband spike; if THIS can't see it, the take has no
    usable audio (also worth knowing)."""
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "a.wav"
        cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-i", str(video),
               "-t", str(duration), "-vn", "-ac", "1", "-ar", "16000",
               "-f", "wav", str(wav), "-y"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0 or not wav.exists():
            return []
        with wave.open(str(wav)) as w:
            sr = w.getframerate()
            x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
    if x.size < sr:
        return []
    hp = np.diff(x)                              # first-difference high-pass
    hop = sr // 100                              # 10 ms
    n = hp.size // hop
    env = np.sqrt((hp[:n * hop].reshape(n, hop) ** 2).mean(axis=1))
    med = float(np.median(env))
    mad = float(np.median(np.abs(env - med))) or 1.0
    thr = med + 8.0 * mad
    gap = int(min_gap_s * 100)
    onsets, last = [], -gap
    for i in range(1, n - 1):
        if env[i] >= thr and env[i] >= env[i - 1] and env[i] >= env[i + 1] \
                and i - last >= gap:
            onsets.append(start + i / 100.0)
            last = i
    return onsets


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default="")
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--start", type=float, default=60.0)
    ap.add_argument("--duration", type=float, default=240.0)
    ap.add_argument("--json", default="")
    ap.add_argument("--dump", action="store_true",
                    help="save frames at disagreements for eyeballing")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    settings = Settings.load()
    if args.video:
        video = Path(args.video)
    else:
        d = Path(settings.recording.resolved_dir())
        vids = sorted(d.glob("session-*.mp4"), key=lambda p: p.stat().st_mtime)
        video = vids[-1] if vids else Path()
    if not video.is_file():
        print(f"no video: {video}", file=sys.stderr)
        return 2

    onsets = audio_onsets(video, args.start, args.duration)
    if not args.quiet:
        print(f"audio: {len(onsets)} onsets in {args.duration:.0f}s")

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)
    pipeline = Pipeline(settings)
    shots = []
    state_frames = {"settled": 0, "moving": 0}
    end_t = args.start + args.duration
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t > end_t:
            break
        res = pipeline.process(frame, t, annotate=False)
        n += 1
        state_frames[res.shot_state] = state_frames.get(res.shot_state, 0) + 1
        if res.shot_event is not None:
            e = res.shot_event
            shots.append({
                "start_t": e.start_t, "end_t": e.end_t, "outcome": e.outcome.value,
                "num_pocketed": e.num_pocketed, "max_travel": e.max_travel,
                "duration_s": e.duration_s,
            })
        if not args.quiet and n % 900 == 0:
            print(f"  {n} frames, {len(shots)} shots so far...")
    cap.release()

    # ---- cross-modal agreement ------------------------------------------- #
    # A shot's strike happens at start_t; allow slack for the state machine's
    # strike_frames latency (it flags "moving" a few frames late).
    def has_onset(s) -> bool:
        return any(s["start_t"] - 1.0 <= o <= s["end_t"] for o in onsets)

    def has_shot(o) -> bool:
        return any(s["start_t"] - 1.0 <= o <= s["end_t"] + 0.5 for s in shots)

    confirmed = [s for s in shots if has_onset(s)]
    unheard = [s for s in shots if not has_onset(s)]          # likely FALSE shots
    unclaimed = [o for o in onsets if not has_shot(o)]        # candidate misses

    print(f"\n=== {video.name}  [{args.start:.0f}s..{end_t:.0f}s] ===")
    print(f"vision shots: {len(shots)}  "
          f"(makes {sum(1 for s in shots if s['outcome'] == 'make')}, "
          f"misses {sum(1 for s in shots if s['outcome'] == 'miss')}, "
          f"scratches {sum(1 for s in shots if s['outcome'] == 'scratch')})")
    print(f"audio onsets: {len(onsets)}")
    print(f"audio-confirmed shots : {len(confirmed)}/{len(shots)}")
    print(f"UNHEARD shots (likely false)   : {len(unheard)}")
    print(f"UNCLAIMED onsets (cand. misses): {len(unclaimed)} "
          "(includes placements/rail/pocket noise)")
    mf = state_frames.get("moving", 0)
    print(f"time in 'moving': {mf / max(1, n):.1%} of frames")
    for s in unheard[:6]:
        print(f"  unheard shot @ {s['start_t']:.1f}s {s['outcome']} travel={s['max_travel']:.0f}")

    if args.json:
        out = {"video": video.name, "start": args.start, "duration": args.duration,
               "shots": shots, "onsets": onsets,
               "confirmed": len(confirmed), "unheard": len(unheard),
               "unclaimed": len(unclaimed)}
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"json -> {p}")

    if args.dump and (unheard or unclaimed):
        out_dir = ROOT / "_eval" / "shots" / video.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video))
        wanted = [("unheard", s["start_t"]) for s in unheard[:8]] + \
                 [("unclaimed", o) for o in unclaimed[:8]]
        for kind, tt in wanted:
            cap.set(cv2.CAP_PROP_POS_MSEC, tt * 1000.0)
            ok, fr = cap.read()
            if ok:
                cv2.putText(fr, f"{kind} @ {tt:.1f}s", (12, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.imwrite(str(out_dir / f"{kind}_{tt:07.1f}s.png"), fr)
        cap.release()
        print(f"disagreement frames -> {out_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
