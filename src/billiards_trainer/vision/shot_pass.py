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
    if refresh_library:
        try:
            from .shots_export import export_library_index, export_lifetime_stats
            rec_dir = Path(video).parent
            export_library_index(rec_dir)
            export_lifetime_stats(rec_dir)
        except Exception:  # noqa: BLE001
            log.exception("close pass: library refresh failed")
    return out
