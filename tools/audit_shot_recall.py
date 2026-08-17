"""G5 recall audit: does the shot detector miss real strokes?

Precision was easy (every vision shot must contain an audio onset). RECALL
is the hard half — proving there are no strokes the detector never saw.
The trick: the analysis sidecar is a second, independent witness. For every
audio onset that no detected shot claims, ask the sidecar what the balls
did next:

    quiet before + balls move right after  ->  MISSED-SHOT candidate
    anything else                          ->  non-shot noise (racking,
                                               pocket drop, chalk, voices)

Every onset in every session gets audited — no sampling — and each
missed-shot candidate dumps an annotated frame so a human can eyeball the
handful that matter instead of listening to hours of audio.

    python tools/audit_shot_recall.py --sessions 8
    python tools/audit_shot_recall.py --video <session.mp4> --json out.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from score_shots import audio_onsets  # noqa: E402

from billiards_trainer.config import Settings  # noqa: E402
from billiards_trainer.vision.analysis_cache import SidecarReader  # noqa: E402

#: An onset this close to a shot START is that shot's strike. The window is
#: asymmetric: the detector stamps start at the strike frame, but audio can
#: lead vision by a beat (sound reaches the mic instantly; motion needs a
#: detection tick at ~10 Hz).
_CLAIM_BEFORE_S = 1.0
_CLAIM_AFTER_S = 1.5


def _positions(reader: SidecarReader, t: float) -> dict[int, tuple[float, float, float]]:
    return {tr.id: (tr.x, tr.y, tr.radius) for tr in reader.tracks_at(t) if tr.active}


def _max_disp(a: dict, b: dict) -> tuple[float, float]:
    """(max displacement, that ball's radius) across ids present in both."""
    best, r_best = 0.0, 10.0
    for tid, (ax, ay, ar) in a.items():
        if tid in b:
            bx, by, br = b[tid]
            d = float(np.hypot(bx - ax, by - ay))
            if d > best:
                best, r_best = d, max(6.0, (ar + br) / 2.0)
    return best, r_best


def audit_session(video: Path, quiet: bool = False) -> dict | None:
    reader = SidecarReader(video)
    if not len(reader):
        return None
    t_end = reader._times[-1]
    onsets = audio_onsets(video, 0.0, t_end + 2.0)
    shots = [(float(s["start"]), float(s["end"])) for s in reader.shots]

    claimed_strike, claimed_body, unclaimed = [], [], []
    for o in onsets:
        if any(s - _CLAIM_BEFORE_S <= o <= s + _CLAIM_AFTER_S for s, _e in shots):
            claimed_strike.append(o)
        elif any(s <= o <= e + 1.0 for s, e in shots):
            claimed_body.append(o)      # rail thump / pocket drop of a shot
        else:
            unclaimed.append(o)

    # Precision half: every vision shot must contain a strike onset.
    unheard_shots = [s for s, _e in shots
                     if not any(s - _CLAIM_BEFORE_S <= o <= s + _CLAIM_AFTER_S
                                for o in onsets)]

    # Recall half: classify unclaimed onsets by what the balls did.
    missed = []
    for o in unclaimed:
        pre_a = _positions(reader, max(0.0, o - 1.6))
        pre_b = _positions(reader, max(0.0, o - 0.3))
        d_pre, r_pre = _max_disp(pre_a, pre_b)
        if d_pre > 0.8 * r_pre:
            continue                    # already in motion — not a fresh stroke
        base = _positions(reader, max(0.0, o - 0.2))
        moved = False
        for dt in (0.7, 1.3, 2.0):
            post = _positions(reader, min(t_end, o + dt))
            d_post, r_post = _max_disp(base, post)
            if d_post > 1.5 * r_post:
                moved = True
                break
        if moved:
            missed.append(round(o, 2))

    return {
        "video": video.name,
        "duration_s": round(t_end, 1),
        "shots": len(shots),
        "onsets": len(onsets),
        "claimed_strike": len(claimed_strike),
        "claimed_body": len(claimed_body),
        "unclaimed_noise": len(unclaimed) - len(missed),
        "missed_shot_candidates": missed,
        "unheard_shots": [round(s, 2) for s in unheard_shots],
    }


def dump_candidate_frames(video: Path, times: list[float], out_dir: Path) -> None:
    if not times:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    for o in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, (o + 0.5) * 1000.0)
        ok, fr = cap.read()
        if not ok:
            continue
        cv2.putText(fr, f"missed-shot candidate @ {o:.1f}s (frame is 0.5s after)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imwrite(str(out_dir / f"{video.stem}_{o:07.1f}s.png"), fr)
    cap.release()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default="", help="audit one session")
    ap.add_argument("--sessions", type=int, default=8,
                    help="audit the N newest sidecar'd sessions")
    ap.add_argument("--json", default="")
    ap.add_argument("--dump-dir", default=str(ROOT / "_eval" / "shot_recall"))
    args = ap.parse_args()

    if args.video:
        vids = [Path(args.video)]
    else:
        d = Path(Settings.load().recording.resolved_dir())
        vids = [v for v in sorted(d.glob("session-*.mp4"),
                                  key=lambda p: p.stat().st_mtime, reverse=True)
                if SidecarReader.exists(v)][:args.sessions]
        vids.reverse()

    reports = []
    for v in vids:
        r = audit_session(v)
        if r is None:
            print(f"{v.name}: empty sidecar, skipped")
            continue
        reports.append(r)
        dump_candidate_frames(v, r["missed_shot_candidates"],
                              Path(args.dump_dir) / v.stem)
        print(f"{r['video'][:44]:46s} shots={r['shots']:3d} onsets={r['onsets']:4d} "
              f"strike={r['claimed_strike']:3d} body={r['claimed_body']:3d} "
              f"noise={r['unclaimed_noise']:4d} "
              f"MISS?={len(r['missed_shot_candidates']):2d} "
              f"UNHEARD={len(r['unheard_shots']):2d}")

    total = {
        "sessions": len(reports),
        "shots": sum(r["shots"] for r in reports),
        "onsets": sum(r["onsets"] for r in reports),
        "missed_candidates": sum(len(r["missed_shot_candidates"]) for r in reports),
        "unheard_shots": sum(len(r["unheard_shots"]) for r in reports),
    }
    print(f"\nTOTAL: {total['sessions']} sessions, {total['shots']} shots, "
          f"{total['onsets']} onsets audited, "
          f"{total['missed_candidates']} missed-shot candidates, "
          f"{total['unheard_shots']} unheard shots")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"aggregate": total, "sessions": reports}, indent=2))
        print(f"json -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
