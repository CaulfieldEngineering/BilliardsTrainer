"""Identity-wander audit: how often does a NUMBER teleport between tracks?

Joe's shot-36 verdict exposed the family: the struck 7 was lost mid-roll
and its number landed on garbage tracks across the table, so the outcome
derivation saw a departure that never happened. This audit counts the
signature across the library — a number appearing >WANDER_DIAM ball
diameters away from its previous sighting within <WANDER_DT seconds,
which no rolling ball path explains (interpolated positions would show
intermediates; a hop with none is an assignment jump, not motion).

    python tools/audit_identity_wander.py            # whole library
    python tools/audit_identity_wander.py --video X  # one session
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billiards_trainer.config import Settings  # noqa: E402
from billiards_trainer.vision.analysis_cache import (  # noqa: E402
    SidecarReader,
    sidecar_path,
)

WANDER_DIAM = 8.0     # jump distance that no honest roll explains...
WANDER_DT = 1.2       # ...crossed in under this many seconds
MIN_GAP_DT = 0.05     # ignore same-state duplicates


def audit(video: Path) -> dict:
    r = SidecarReader(video)
    last: dict = {}          # number -> (t, x, y, track_id)
    hops = []
    ball_d = 25.0            # ~2r fallback; refined from data below
    radii = [row[3] for fr in r._frames[:50] for row in fr if row[6]]
    if radii:
        ball_d = 2.0 * (sorted(radii)[len(radii) // 2])
    for t, fr in zip(r._times, r._frames):
        for row in fr:
            tid, x, y, _r, num, _cls, active = row
            if not active or num < 0:
                continue
            if num in last:
                pt, px, py, pid = last[num]
                dt = t - pt
                if MIN_GAP_DT < dt < WANDER_DT:
                    dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
                    if dist > WANDER_DIAM * ball_d and tid != pid:
                        hops.append({"num": num, "t": round(t, 1),
                                     "dist_d": round(dist / ball_d, 1),
                                     "dt": round(dt, 2),
                                     "from": (round(px), round(py)),
                                     "to": (round(x), round(y))})
            last[num] = (t, x, y, tid)
    return {"video": video.name, "states": len(r._times), "hops": hops}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default="")
    args = ap.parse_args()
    if args.video:
        vids = [Path(args.video)]
    else:
        d = Path(Settings.load().recording.resolved_dir())
        vids = sorted(d.glob("session-*.mp4"))
    total_hops = 0
    total_states = 0
    for v in vids:
        if not sidecar_path(v).is_file():
            continue
        try:
            res = audit(v)
        except Exception as e:  # noqa: BLE001
            print(f"{v.name}: audit failed ({e})")
            continue
        total_hops += len(res["hops"])
        total_states += res["states"]
        if res["hops"]:
            worst = max(res["hops"], key=lambda h: h["dist_d"])
            print(f"{res['video']}: {len(res['hops'])} hop(s); worst "
                  f"#{worst['num']} {worst['dist_d']}d in {worst['dt']}s "
                  f"@t={worst['t']}")
    rate = 1000.0 * total_hops / max(1, total_states)
    print(f"\nLIBRARY: {total_hops} identity hops across {total_states} "
          f"states ({rate:.2f}/1k states)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
