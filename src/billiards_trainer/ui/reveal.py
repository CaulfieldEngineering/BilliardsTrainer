"""Reveal a file or folder in the OS file manager.

Every platform spells this differently, and the wrong spelling fails SILENTLY
from a GUI app (no console to print to): the sessions sidebar shipped
``open -R`` with a "Show in Finder" label, which on the Windows rig meant a
menu item that named the wrong application and did nothing at all.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("ui.reveal")


def reveal_label(kind: str = "file") -> str:
    """Menu text naming the actual file manager (users look for this by name)."""
    what = "Reveal" if sys.platform == "win32" else "Show"
    where = {"win32": "Explorer", "darwin": "Finder"}.get(sys.platform, "File Manager")
    return f"{what} {'in' if kind == 'file' else 'folder in'} {where}"


def reveal(path: str | Path) -> bool:
    """Open the file manager with ``path`` selected (or opened, if a folder).

    Returns True if the command launched. Never raises: a failed reveal must not
    take down the menu action that called it.
    """
    p = Path(path)
    if not p.exists():
        log.warning("reveal: path does not exist: %s", p)
        return False
    target = os.path.normpath(str(p))
    try:
        if sys.platform == "win32":
            # /select, needs the file as ONE argument with no space after the
            # comma. explorer.exe exits non-zero even when it succeeds, so the
            # return code is deliberately ignored.
            if p.is_dir():
                os.startfile(target)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["explorer", f"/select,{target}"])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", target] if p.is_file()
                             else ["open", target])
        else:
            # Linux has no universal "select this file", so open its folder.
            subprocess.Popen(["xdg-open", target if p.is_dir() else str(p.parent)])
        return True
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("reveal failed for %s: %s", target, exc)
        return False
