"""One on-demand re-measure job: engine -> gate -> merge, guarded.

The box, runnable by button (Joe: "be able to reprocess a session
with the latest measurement engine, on demand"): the exact pipeline
the autonomous rollout runs, promoted out of session scripts into the
package. One job at a time (RUNNING marker), never beside a live
recording, deterministic output, progress in APP_DIR/m1_progress.json
(the app's status bar polls it), verdict in APP_DIR/m1_result.json.

    python -m billiards_trainer.measure.job <video> [--defer-presence]
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from ..config import APP_DIR, EXPORTS_DIR

log = logging.getLogger("measure.job")

M1_DIR = APP_DIR / "m1"
RUNNING = M1_DIR / "RUNNING"
RESULT = APP_DIR / "m1_result.json"
GATE_MAX = 0.55           # champion's impossible/1k — beat it or no merge


def _gate(sidecar_video: Path) -> dict:
    from collections import Counter
    from types import SimpleNamespace

    from ..core.types import BallClass
    from ..eval.invariants import SequenceScorer
    from ..vision.analysis_cache import SidecarReader
    r = SidecarReader(sidecar_video)
    scorer = SequenceScorer()
    nbf = 0
    for fi, frame in enumerate(r._frames):
        tracks = [SimpleNamespace(id=tr[0], x=tr[1], y=tr[2], radius=tr[3],
                                  number=tr[4], cls=BallClass(tr[5]),
                                  active=bool(tr[6]), misses=0, speed=0.0)
                  for tr in frame]
        nbf += len(tracks)
        scorer.add(tracks, fi, tracking=True)
    rep = scorer.report
    kinds = dict(Counter(v.kind for v in rep.violations).most_common(3))
    return {"impossible_per_1k": round(1000 * rep.impossible / max(1, nbf), 2),
            "kinds": kinds}


def run(video: str, presence_pause: bool = False) -> dict:
    """Engine -> gate -> (prefer-dense) merge for ONE session.
    presence_pause=False is the on-demand default: the click is the
    consent. Autonomous callers pass True."""
    from .engine import reprocess
    from ..vision.shots_export import export_shots_summary

    video_p = Path(video)
    out: dict = {"video": video_p.name}
    if RUNNING.exists():
        out["refused"] = "another heavy job is running"
        return out
    if list(EXPORTS_DIR.glob(".session-*.part.mp4")):
        out["refused"] = "recording live"
        return out
    M1_DIR.mkdir(parents=True, exist_ok=True)
    RUNNING.write_text(f"on-demand re-measure: {video_p.name}")
    try:
        # a fresh run ALWAYS replaces the old sidecar (latest engine,
        # by definition of the button)
        for suff in (".analysis.jsonl", ".analysis.jsonl.prev"):
            (M1_DIR / (video_p.name + suff)).unlink(missing_ok=True)
        res = reprocess(str(video_p), str(M1_DIR),
                        presence_pause=presence_pause)
        out["engine"] = res
        if res.get("aborted"):
            return out
        out["gate"] = _gate(M1_DIR / video_p.name)
        rate = out["gate"]["impossible_per_1k"]
        if rate > GATE_MAX:
            out["merged"] = f"NO - gate red ({rate}/1k > {GATE_MAX})"
            return out
        # REBUILD, DON'T PATCH. This used to merge dense trails into
        # whatever shot list the recording-time pass had produced, so a
        # stroke the engine measured correctly could still be shown as
        # "rearranging" and five real shots were missing entirely. The
        # summary is now written wholly from the sidecar this run just
        # produced - shots, trails and all - so a reprocess depends on
        # the video and the current engine, and on nothing else.
        sj = Path(str(video_p) + ".shots.json")
        if sj.exists():
            bak = Path(str(sj) + ".pre_engine")
            if not bak.exists():
                bak.write_bytes(sj.read_bytes())
        written = export_shots_summary(str(video_p),
                                       sidecar_video=M1_DIR / video_p.name)
        out["summary"] = str(written) if written else "NO - export failed"
        return out
    finally:
        RUNNING.unlink(missing_ok=True)
        try:
            RESULT.write_text(json.dumps(out, default=str))
        except OSError:
            pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    print(json.dumps(run(args[0],
                         presence_pause="--defer-presence" in sys.argv),
                     default=str))
