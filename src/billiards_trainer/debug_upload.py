"""Debug-bundle staging — the test-machine → dev-machine feedback channel (skeleton).

When Joe (on the test machine) sees the detector fail, he taps "Save clip + flag
as failure" in Settings → Debug. The controller zips the recent preview buffer +
detector state, then this module *stages* the zip into a local inbox folder.

For now "upload" == copy into the inbox dir. The eval harness / a sync tool can
pick clips up from there. The single function ``stage_bundle`` is the seam to
later retarget at Supabase Storage (see docs/eval/README.md) without touching the
controller or UI.
"""

import logging
import shutil
import sys
from pathlib import Path

from .config import APP_DIR

log = logging.getLogger("debug_upload")


def inbox_dir() -> Path:
    """Where flagged bundles land. In a source checkout we use the repo's
    gitignored ``_eval/inbox`` so the harness sees them directly; in the frozen
    app we use a writable folder under the app data dir."""
    if not getattr(sys, "frozen", False):
        repo = Path(__file__).resolve().parents[2]
        if (repo / "tools" / "eval_harness.py").exists():
            return repo / "_eval" / "inbox"
    return APP_DIR / "eval_inbox"


def stage_bundle(zip_path: str | Path) -> Path:
    """'Upload' a debug bundle by copying it into the inbox. Returns the dest.

    RETARGET POINT: replace the copy below with a Supabase Storage upload
    (bucket put + row insert) when cloud sync is wired. Keep the signature.
    """
    src = Path(zip_path)
    dst_dir = inbox_dir()
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    log.info("Staged debug bundle -> %s", dst)
    return dst
