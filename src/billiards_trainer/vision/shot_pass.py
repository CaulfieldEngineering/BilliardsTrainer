"""THE close pass: one canonical sequence for finishing a session's shots.

Joe's unification mandate ("Each shot needs to be analyzed and processed
the same way"): this sequence used to live as three hand-copied variants
— session close (workers/controller), the library backfill
(tools/build_analysis_cache), and the phone-verdict watcher
(companion/corrections_watcher) — and they had already drifted once (the
stroke pass reached the close path weeks after the others). Every caller
now runs THIS function; a new per-shot stage gets added here exactly
once.

Order matters and is load-bearing:
  1. derive_and_correct — identity-derived outcomes (review verdicts win)
  2. classify_and_mark  — stroke / break / ball_in_hand / nothing labels
  3. annotate_session   — camera stroke metrics (idempotent per version;
                          live-measured shots are skipped)
  4. export_shots_summary — the phone/desktop dossier, incl. the
                          measured-or-abstained contract
  5. (optional) library index + lifetime stats — the cross-session
     surfaces; skipped by callers that batch many sessions and refresh
     these once at the end.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("vision.shot_pass")


def forensic_fill(video) -> int:
    """Corridor re-pass for misses without a trusted tag. Appends
    {type: tag_correction, src: forensic} records (the reader ranks them
    below Joe's review, above the derivation). Returns records written."""
    import json

    from .analysis_cache import SidecarReader, sidecar_path
    from .forensic_repass import repass_shot
    reader = SidecarReader(video)
    todo = []
    for s in reader.shots:
        if s.get("outcome") != "miss":
            continue
        if s.get("_tag_review") or s.get("_tag_forensic"):
            continue                     # already answered at a higher rank
        todo.append((float(s["start"]), float(s.get("end", s["start"] + 8.0))))
    if not todo:
        return 0
    n = 0
    with open(sidecar_path(video), "a", encoding="utf-8") as fh:
        for start, end in todo:
            try:
                r = repass_shot(str(video), start, end)
            except Exception:  # noqa: BLE001 - one shot never stops the fill
                log.exception("forensic repass failed @%.1fs", start)
                continue
            if not (r and r.get("ok") and r.get("side")):
                continue
            rec = {"type": "tag_correction", "start": round(start, 3),
                   "miss_side": r["side"], "src": "forensic"}
            if r.get("cut") in ("left", "straight", "right"):
                rec["cut"] = r["cut"]    # repass_shot already names the side
            fh.write(json.dumps(rec) + "\n")
            n += 1
    if n:
        log.info("forensic fill: %d verdict(s) recovered for %s", n, video)
    return n


def run_close_pass(video, refresh_library: bool = True) -> dict:
    """Run the canonical per-session close sequence. Returns a small
    summary dict; never raises (each stage is individually guarded, and
    a stage failure never blocks the later ones)."""
    video = str(video)
    out: dict = {}
    try:
        from .outcomes import derive_and_correct
        out["outcomes"] = derive_and_correct(video)
    except Exception:  # noqa: BLE001 - derivation must not block export
        log.exception("close pass: outcome derivation failed")
    try:
        from .actions import classify_and_mark
        out["actions"] = classify_and_mark(video)
    except Exception:  # noqa: BLE001
        log.exception("close pass: action labeling failed")
    try:
        from .stroke_vision import annotate_session
        out["strokes"] = annotate_session(video)
    except Exception:  # noqa: BLE001
        log.exception("close pass: stroke metrics failed")
    try:
        from .shots_export import export_shots_summary
        export_shots_summary(video)
        out["exported"] = True
    except Exception:  # noqa: BLE001
        log.exception("close pass: summary export failed")
    # forensic corridor re-pass for misses the derivation abstained on —
    # runs AFTER the first export (it needs the summary's transform) and
    # re-exports when it recovers verdicts. This stage was a one-time
    # backlog tool until 2026-08-26; new sessions never got it (the
    # marathon session: 51 misses, 1 trusted tag before this).
    try:
        n_forensic = forensic_fill(video)
        out["forensic"] = n_forensic
        if n_forensic and out.get("exported"):
            from .shots_export import export_shots_summary
            export_shots_summary(video)
    except Exception:  # noqa: BLE001
        log.exception("close pass: forensic fill failed")
    if refresh_library:
        try:
            from .shots_export import export_library_index, export_lifetime_stats
            rec_dir = Path(video).parent
            export_library_index(rec_dir)
            export_lifetime_stats(rec_dir)
        except Exception:  # noqa: BLE001
            log.exception("close pass: library refresh failed")
    return out
