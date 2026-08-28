"""Merge M1 dense measurements into a session's replay trails.

Reads the session's REAL shots.json (shots, strokes, aim — untouched)
and replaces each shot's trail polylines with dense samples from the
engine's sidecar, normalized through the ENGINE'S OWN transform (meta
hinv — self-consistent chain to video pixels). Per-shot gates keep it
honest: a dense trail must geometrically agree with the sparse trail it
replaces (same path, same endpoints) or the sparse one stays.

Time base: dense t is VIDEO time; trail points must ship in sidecar
time so the phone's per-shot normalization (o = start - strike) lands
them back on video time. So: t_out = t_video + o.
"""

from __future__ import annotations

import bisect
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("measure.trails_merge")

PRE_S = 0.5          # show the address rest before the strike
MAX_PTS = 400
ENDPOINT_TOL = 0.05  # dense must END where sparse ended (both at rest)
PATH_TOL = 0.035     # every moving sparse point near the dense polyline


def _norm(hinv, w, h, x, y):
    v = hinv @ np.array([x, y, 1.0])
    return (min(1.0, max(0.0, v[0] / v[2] / w)),
            min(1.0, max(0.0, v[1] / v[2] / h)))


def _agrees(dense_pts, sparse_pts) -> bool:
    if len(dense_pts) < 5 or len(sparse_pts) < 3:
        return False
    sx0, sy0 = sparse_pts[0][1], sparse_pts[0][2]
    moving = [(sx, sy) for (_t, sx, sy) in sparse_pts
              if ((sx - sx0) ** 2 + (sy - sy0) ** 2) ** 0.5 > 0.008]
    if len(moving) < 2:
        return False
    for (sx, sy) in moving:
        dd = min(((dx - sx) ** 2 + (dy - sy) ** 2) ** 0.5
                 for (_t, dx, dy) in dense_pts)
        if dd > PATH_TOL:
            return False
    ex, ey = dense_pts[-1][1], dense_pts[-1][2]
    lx, ly = sparse_pts[-1][1], sparse_pts[-1][2]
    return ((ex - lx) ** 2 + (ey - ly) ** 2) ** 0.5 < ENDPOINT_TOL


def merge_trails(video_path, dense_reader, doc: dict,
                 arbiter=None, prefer_dense: bool = False) -> dict:
    """Returns upgrade stats; mutates doc in place (caller writes it).
    arbiter: optional VideoArbiter — when the bootstrap agreement gate
    refuses a dense trail, the VIDEO decides (the ball's actual resting
    place at trail end).

    prefer_dense: the tie-break, and it is EARNED, not assumed. False
    (bootstrap era): ambiguity keeps sparse. True (for sessions whose
    FULL gate beat the champion): ambiguity takes dense; sparse
    survives only when the video ACTIVELY sides with it. Joe caught
    the timid default: 58/125 upgrades left his replays looking
    unchanged - mixed shots kept the very cue-ball trails he watches."""
    meta = dense_reader.meta
    hinv = np.asarray(meta.get("hinv"), dtype=float)
    w, h = float(meta.get("w")), float(meta.get("h"))
    times = dense_reader._times
    frames = dense_reader._frames
    t_lo, t_hi = (min(times), max(times)) if times else (0, 0)

    up = kept = skipped = 0
    for s in doc.get("shots", []):
        sv = s.get("stroke") or {}
        if not s.get("trails") or sv.get("strike") is None:
            skipped += 1
            continue
        strike_v = float(sv["strike"])
        o = float(s["start"]) - strike_v
        end_v = float(s.get("end", s["start"])) - o
        if strike_v - PRE_S < t_lo or end_v > t_hi:
            skipped += 1                  # dense data doesn't cover it
            continue
        k0 = bisect.bisect_left(times, strike_v - PRE_S)
        k1 = bisect.bisect_right(times, end_v + 0.3)
        # collect per-number dense points across the window
        per_n: dict = {}
        for j in range(k0, min(k1, len(frames))):
            for tr in frames[j]:
                n = tr[4]
                if n < 0 or not tr[6]:
                    continue
                x, y = _norm(hinv, w, h, tr[1], tr[2])
                per_n.setdefault(n, []).append(
                    [round(times[j] + o, 3), round(x, 4), round(y, 4)])
        new_trails = []
        any_upgrade = False
        pocketed_ns = {int(pb.get("number", -1)) for pb in
                       (s.get("pocketed_balls") or [])} if isinstance(
                           s.get("pocketed_balls"), list) else set()
        for tr in s["trails"]:
            n = tr.get("n", -1)
            dp = per_n.get(n)
            take = bool(dp and _agrees(dp, tr.get("p", [])))
            if dp and not take and len(dp) >= 5:
                if arbiter is not None and len(tr.get("p", [])) >= 3:
                    # video-truth arbitration: who ends where reality is?
                    t_end_v = dp[-1][0] - o
                    v = arbiter.verdict(t_end_v, (dp[-1][1], dp[-1][2]),
                                        (tr["p"][-1][1], tr["p"][-1][2]),
                                        n in pocketed_ns)
                    # earned tie-break: unknown goes to dense only when
                    # the session's full gate beat the champion
                    take = v == "dense" or (prefer_dense and v != "sparse")
                elif prefer_dense:
                    take = True
            if take:
                pts = dp[:: max(1, len(dp) // MAX_PTS + 1)][:MAX_PTS]
                new_trails.append({"n": n, "p": pts, "dense": True})
                any_upgrade = True
            else:
                new_trails.append(tr)     # abstain: sparse stays
        s["trails"] = new_trails
        if any_upgrade:
            up += 1
        else:
            kept += 1
    return {"shots_upgraded": up, "shots_kept_sparse": kept,
            "skipped_no_coverage": skipped}


def merge_into_session(video_path, dense_sidecar_video,
                       arbitrate: bool = True,
                       prefer_dense: bool = False) -> dict | None:
    """Load, merge, and WRITE the session's shots.json (bumps the
    processed stamp). Returns the stats dict, or None on failure."""
    from datetime import datetime, timezone

    from ..vision.analysis_cache import SidecarReader
    video_path = Path(video_path)
    sj = Path(str(video_path) + ".shots.json")
    doc = json.loads(sj.read_text(encoding="utf-8"))
    dense = SidecarReader(dense_sidecar_video)
    if not dense.meta.get("hinv"):
        log.error("dense sidecar lacks meta hinv - refusing to merge")
        return None
    arbiter = None
    if arbitrate:
        try:
            from ..config import Settings
            from ..vision.pipeline import Pipeline
            from .arbitrate import VideoArbiter
            from .engine import _acquire_calib
            pipe = Pipeline(Settings.load())
            calib = _acquire_calib(video_path, pipe)
            if calib is not None and dense.meta.get("hinv"):
                arbiter = VideoArbiter(video_path, calib,
                                       dense.meta["hinv"],
                                       dense.meta["w"], dense.meta["h"])
                arbiter._pipe = pipe
        except Exception:  # noqa: BLE001 - arbitration is an upgrade path
            log.exception("arbiter unavailable; bootstrap gate only")
    try:
        stats = merge_trails(video_path, dense, doc, arbiter=arbiter,
                             prefer_dense=prefer_dense)
    finally:
        if arbiter is not None:
            arbiter.close()
    doc["exported"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    doc["trails_engine"] = "m1-dense"
    sj.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    log.info("dense trails merged into %s: %s", sj.name, stats)
    return stats
