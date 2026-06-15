"""Self-update recovery: launch sentinels + frozen-build integrity check.

The self-swap (see ``updater.install_and_relaunch``) relies on a sentinel file to
know whether the freshly-swapped exe actually started:

* The running app writes ``launched_ok.flag`` once Qt is up — proof that the
  Python runtime + Qt DLLs loaded.
* The swap batch deletes that flag before launching the new exe, then waits for
  it to reappear. If it doesn't (the classic "Failed to load Python DLL" case,
  usually antivirus quarantining the unsigned onefile extraction), the batch
  restores the backup and drops ``update_failed.flag`` so the restored app can
  explain what happened.
"""

import logging
import sys
from pathlib import Path

from ..config import APP_DIR

log = logging.getLogger("update.recovery")

LAUNCHED_OK = APP_DIR / "launched_ok.flag"
UPDATE_FAILED = APP_DIR / "update_failed.flag"
UPDATE_PENDING = APP_DIR / "update_pending.json"


def mark_launched_ok() -> None:
    """Signal the swap batch that this process started successfully."""
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        LAUNCHED_OK.write_text("ok", encoding="utf-8")
    except OSError:
        pass


def consume_update_failed() -> bool:
    """Return True (once) if the last update was reverted. Clears the flags."""
    failed = UPDATE_FAILED.exists()
    for f in (UPDATE_FAILED, UPDATE_PENDING):
        try:
            f.unlink(missing_ok=True)
        except OSError:
            pass
    return failed


def _python_dll_name() -> str:
    return f"python{sys.version_info.major}{sys.version_info.minor}.dll"


def verify_frozen_integrity() -> str | None:
    """Proactive check: in a frozen build, confirm the Python runtime DLL was
    extracted. Returns a problem description, or None if all good / not frozen.

    (The pure bootloader DLL-load failure happens before our code runs and is
    handled by the swap batch's rollback; this catches partial-extraction states
    where we *did* start but bundled files went missing.)"""
    if not getattr(sys, "frozen", False):
        return None
    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return None
    dll = Path(meipass) / _python_dll_name()
    if not dll.exists():
        log.warning("Frozen integrity: %s missing from %s", dll.name, meipass)
        return ("Some program files appear to be missing — antivirus may have "
                "removed part of the download.")
    return None


def install_dir() -> Path:
    """Directory the (frozen) exe lives in, for 'open folder' / Defender hints."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd()
