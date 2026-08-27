"""M2 corpus gate: score the dense engine's output on its OWN merits.

Runs the standard truth instruments (duplicate identities, id-hops,
physics-impossible rate) over a DENSE m1 sidecar and the session's
REAL sparse sidecar side by side. The engine graduates from sparse
supervision only by matching-or-beating these numbers — the same bar
every champion has cleared.

    python tools/m2_gate.py --session session-20260826-002906.mp4 \
        --dense-dir <scratch m1 dir>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _lowprio import demote

demote()

from audit_identity_wander import audit as hops_audit  # noqa: E402
from audit_render import audit as render_audit  # noqa: E402

from billiards_trainer.eval.invariants import SequenceScorer  # noqa: E402
from billiards_trainer.vision.analysis_cache import SidecarReader  # noqa: E402

SESS_DIR = Path("C:/Users/Joe/Dropbox/Billiards/BilliardsTrainer")


def impossible_rate(video: Path) -> dict:
    """Physics-impossible rate per 1k ball-frames via SequenceScorer
    (identical default config both sides = fair comparison even where
    bed-dependent checks are inactive)."""
    from types import SimpleNamespace

    from billiards_trainer.core.types import BallClass
    reader = SidecarReader(video)
    scorer = SequenceScorer()
    n_ballframes = 0
    for fi, frame in enumerate(reader._frames):
        tracks = [SimpleNamespace(id=tr[0], x=tr[1], y=tr[2], radius=tr[3],
                                  number=tr[4],
                                  cls=BallClass(tr[5]) if not isinstance(
                                      tr[5], BallClass) else tr[5],
                                  active=bool(tr[6]), misses=0, speed=0.0)
                  for tr in frame]
        n_ballframes += len(tracks)
        scorer.add(tracks, fi, tracking=True)
    rep = scorer.report
    imp = int(rep.impossible)
    return {"impossible": imp,
            "per_1k": round(1000.0 * imp / max(1, n_ballframes), 2),
            "ball_frames": n_ballframes}


def score(video: Path, label: str) -> dict:
    out = {"label": label}
    r = render_audit(video)
    out["dup_states"] = (r or {}).get("dup_states", (r or {}).get("duplicates"))
    out["states"] = (r or {}).get("states")
    h = hops_audit(video)
    out["id_hops_per_1k"] = h.get("hops_per_1k", h.get("per_1k"))
    out.update({f"imp_{k}": v for k, v in impossible_rate(video).items()})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--dense-dir", required=True)
    args = ap.parse_args()
    real = SESS_DIR / args.session
    dense = Path(args.dense_dir) / args.session
    for v, label in ((real, "SPARSE (live champion)"),
                     (dense, "DENSE (m1 engine)")):
        s = score(v, label)
        print(f"\n{label}:")
        for k, v2 in s.items():
            if k != "label":
                print(f"  {k}: {v2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
