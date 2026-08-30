"""Draw what JOE sees: the phone's trails and labels, on the video frame.

Joe, 2026-08-31: "Do you not see these artifacts when you personally watch the
videos?" No - and that was a real hole. Every vision check in this campaign has
used tools/debug_overlay.py, which draws the TRACKER's per-frame beliefs. That
verified naming and make/miss, and it is blind to the thing he actually looks
at: the trail polylines and the sentence, which come from <video>.shots.json.
Three defects lived in that gap - a potted ball's trail shooting back across
the table, a pot labelled with the wrong pocket, and junk "rearranging"
entries - and he found all three in ten minutes.

This renders the SUMMARY, not the tracker: same polylines, same per-ball
colours, same text the phone shows.

    python tools/render_shot.py <session.mp4> --at 32 45 85
    python tools/render_shot.py <session.mp4> --shot 6      # by index
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
REC = Path(r"C:/Users/Joe/Dropbox/Billiards/BilliardsTrainer")
OUT = ROOT / "_train" / "bench_fix" / "asjoesees"


def _colour(n):
    from billiards_trainer.core.balls import pool_ball_bgr
    return pool_ball_bgr(n) if n is not None and n >= 0 else (200, 200, 200)


def render(video: Path, shots: list, out_dir: Path) -> list:
    import cv2
    import numpy as np
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    made = []
    for s in shots:
        # the LAST frame of the shot: every trail fully drawn, which is
        # the state the phone leaves on screen at the end of playback
        cap.set(cv2.CAP_PROP_POS_MSEC, float(s["end"]) * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        h, w = fr.shape[:2]
        for tr in (s.get("trails") or []):
            pts = tr.get("p") or []
            col = _colour(tr.get("n"))
            prev = None
            for p in pts:
                # normalized video coords -> pixels
                xy = (int(p[1] * w), int(p[2] * h))
                if prev is not None:
                    cv2.line(fr, prev, xy, col, 3, cv2.LINE_AA)
                prev = xy
            if prev is not None:
                cv2.circle(fr, prev, 7, col, -1)
                cv2.putText(fr, str(tr.get("n")), (prev[0] + 9, prev[1] - 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        head = (f"@{s['start']:.2f}  {s.get('action')}  {s.get('outcome')}"
                f"   {len(s.get('trails') or [])} trails")
        cv2.putText(fr, head, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 255), 2)
        cv2.putText(fr, (s.get("text") or "")[:78], (18, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        p = out_dir / f"asjoesees_{str(s['start']).replace('.', '_')}.png"
        cv2.imwrite(str(p), fr)
        made.append(p)
        print(f"  {p.name}: {head}", flush=True)
    cap.release()
    return made


def strip(video: Path, s: dict, out_dir: Path, n: int = 12) -> Path:
    """A contact sheet ACROSS the shot, trails drawn only up to each
    moment - the closest thing to watching it play.

    Joe, 2026-08-31: "My expectation is that you watch it as a video or
    analyze every frame at least so you see what I see." I cannot press
    play; I can look at stills. So the honest substitute is to step
    through the shot and look at every step, with the overlay exactly as
    the phone draws it."""
    import cv2
    import numpy as np
    out_dir.mkdir(parents=True, exist_ok=True)
    t0, t1 = float(s["start"]), float(s["end"])
    cap = cv2.VideoCapture(str(video))
    tiles = []
    for k in range(n):
        t = t0 + (t1 - t0) * k / max(1, n - 1)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        h, w = fr.shape[:2]
        for tr in (s.get("trails") or []):
            col = _colour(tr.get("n"))
            prev = None
            for p_ in (tr.get("p") or []):
                if p_[0] > t:          # only what has happened BY NOW
                    break
                xy = (int(p_[1] * w), int(p_[2] * h))
                if prev is not None:
                    cv2.line(fr, prev, xy, col, 4, cv2.LINE_AA)
                prev = xy
            if prev is not None:
                cv2.circle(fr, prev, 8, col, -1)
        cv2.putText(fr, f"t={t:.2f}", (18, 44), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (0, 255, 255), 3)
        tiles.append(cv2.resize(fr, (int(w * 460 / h), 460)))
    cap.release()
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles) - 3, 4)]
    if not rows:
        return None
    wid = min(r.shape[1] for r in rows)
    sheet = np.vstack([r[:, :wid] for r in rows])
    out = out_dir / f"strip_{str(s['start']).replace('.', '_')}.png"
    cv2.imwrite(str(out), sheet)
    print(f"  {out.name}: {n} frames across @{t0:.2f}-{t1:.2f}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--at", nargs="*", type=float, default=None,
                    help="seconds; renders the shot containing each")
    ap.add_argument("--shot", type=int, default=None, help="1-based index")
    ap.add_argument("--strip", type=int, default=0,
                    help="render a contact sheet of N frames across the shot")
    a = ap.parse_args()
    video = Path(a.video)
    if not video.is_absolute():
        video = REC / video
    doc = json.loads(Path(str(video) + ".shots.json").read_text(encoding="utf-8"))
    shots = doc["shots"] if isinstance(doc, dict) and "shots" in doc else doc
    if a.shot:
        pick = [shots[a.shot - 1]]
    elif a.at:
        pick = []
        for t in a.at:
            inside = [s for s in shots
                      if float(s["start"]) - 2.0 <= t <= float(s["end"]) + 2.0]
            pick.extend(inside or [min(shots, key=lambda s: abs(s["start"] - t))])
    else:
        pick = shots
    if a.strip:
        for sh in pick:
            strip(video, sh, OUT, a.strip)
    else:
        render(video, pick, OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
