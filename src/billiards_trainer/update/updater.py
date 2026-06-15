"""Auto-update logic.

Flow:
  1. On launch (if enabled) fetch ``version.json`` from the latest GitHub Release.
  2. Compare its ``version`` to the running ``__version__``.
  3. If newer, prompt the user (UpdateDialog) to download the installer.
  4. Download to a temp dir with progress, then launch it and quit so the
     installer can replace files.

The pure functions here (``parse_version``, ``is_newer``, ``fetch_manifest``)
are unit-tested; the Qt threading wrapper is a thin shell over them.
"""

import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import requests
from PySide6.QtCore import QObject, QThread, Signal

from ..version import UPDATE_MANIFEST_URL, __version__

log = logging.getLogger("updater")
_VERSION_RE = re.compile(r"(\d+)")


def parse_version(v: str) -> tuple[int, ...]:
    """Loose semantic-version parse: '0.1.42' -> (0, 1, 42). Non-numeric junk
    is stripped; missing components are treated as 0."""
    if not v:
        return (0,)
    nums = _VERSION_RE.findall(v.strip().lstrip("vV"))
    return tuple(int(n) for n in nums) if nums else (0,)


def is_newer(candidate: str, current: str) -> bool:
    """True if ``candidate`` is a strictly newer version than ``current``."""
    a, b = parse_version(candidate), parse_version(current)
    length = max(len(a), len(b))
    a = a + (0,) * (length - len(a))
    b = b + (0,) * (length - len(b))
    return a > b


@dataclass
class UpdateInfo:
    version: str
    url: str
    notes: str = ""
    pub_date: str = ""

    @classmethod
    def from_manifest(cls, data: dict) -> "UpdateInfo":
        return cls(
            version=str(data.get("version", "")),
            url=str(data.get("url", "")),
            notes=str(data.get("notes", "")),
            pub_date=str(data.get("pub_date", "")),
        )


def fetch_manifest(url: str = UPDATE_MANIFEST_URL, timeout: float = 6.0) -> UpdateInfo | None:
    """Fetch + parse version.json. Returns None on any network/parse error.

    Logs every step so "I don't see an update" reports have a diagnostic trail
    (this path used to fail silently — e.g. a frozen-build SSL/cert problem would
    leave no trace at all)."""
    log.info("Update check: fetching manifest %s", url)
    try:
        resp = requests.get(url, timeout=timeout, headers={"Accept": "application/json"})
        log.info("Update check: HTTP %s from %s", resp.status_code, resp.url)
        resp.raise_for_status()
        info = UpdateInfo.from_manifest(resp.json())
        log.info("Update check: manifest version=%s url=%s", info.version, info.url)
        return info
    except requests.exceptions.SSLError as exc:
        log.warning("Update check FAILED (SSL/cert): %s — the bundled CA store may "
                    "be missing; updates can't be verified.", exc)
    except requests.RequestException as exc:
        log.warning("Update check FAILED (network): %s", exc)
    except ValueError as exc:
        log.warning("Update check FAILED (bad JSON): %s", exc)
    return None


def check_for_update(current: str = __version__,
                     url: str = UPDATE_MANIFEST_URL) -> UpdateInfo | None:
    """Return UpdateInfo if a newer release is available, else None."""
    info = fetch_manifest(url)
    if not info or not info.version:
        log.info("Update check: no usable manifest; staying on v%s", current)
        return None
    newer = is_newer(info.version, current)
    log.info("Update check: available=%s current=%s -> %s",
             info.version, current, "OFFER UPDATE" if newer else "up to date")
    return info if newer else None


# --------------------------------------------------------------------------- #
# Qt wrappers
# --------------------------------------------------------------------------- #
class UpdateCheckWorker(QObject):
    """Runs the (blocking) network check off the UI thread."""

    finished = Signal(object)  # UpdateInfo | None

    def __init__(self, current: str = __version__, url: str = UPDATE_MANIFEST_URL):
        super().__init__()
        self._current = current
        self._url = url

    def run(self) -> None:
        self.finished.emit(check_for_update(self._current, self._url))


class DownloadWorker(QObject):
    """Streams an installer to a temp file, emitting progress 0..100."""

    progress = Signal(int)
    finished = Signal(str)   # path to downloaded file ("" on failure)
    failed = Signal(str)

    def __init__(self, url: str):
        super().__init__()
        self._url = url

    def run(self) -> None:
        try:
            dest = Path(tempfile.gettempdir()) / Path(self._url).name
            with requests.get(self._url, stream=True, timeout=30) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                written = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if not chunk:
                            continue
                        f.write(chunk)
                        written += len(chunk)
                        if total:
                            self.progress.emit(int(written * 100 / total))
            self.progress.emit(100)
            self.finished.emit(str(dest))
        except (requests.RequestException, OSError) as exc:
            self.failed.emit(str(exc))


def install_and_relaunch(downloaded: str) -> None:
    """Apply a downloaded update.

    For the frozen portable .exe we can't overwrite ourselves while running, so
    we spawn a tiny detached batch that waits for this process to exit, swaps the
    new exe over the current one, and relaunches it. When running from source we
    just open the downloaded file. (The frozen path needs real-world verification
    on a release build — see docs/BLOCKERS.md.)
    """
    import os
    import subprocess
    import sys

    if sys.platform == "win32" and getattr(sys, "frozen", False) and downloaded.lower().endswith(".exe"):
        current = sys.executable
        bat = Path(tempfile.gettempdir()) / "bt_update.bat"
        bat.write_text(
            "@echo off\r\n"
            "ping 127.0.0.1 -n 3 > nul\r\n"
            f'move /Y "{downloaded}" "{current}"\r\n'
            f'start "" "{current}"\r\n'
            'del "%~f0"\r\n',
            encoding="utf-8",
        )
        # DETACHED_PROCESS so the batch survives this process exiting.
        subprocess.Popen(["cmd", "/c", str(bat)], creationflags=0x00000008)
        return
    if sys.platform == "win32":
        os.startfile(downloaded)  # type: ignore[attr-defined]
    else:
        subprocess.Popen([downloaded])


def run_in_thread(worker: QObject) -> QThread:
    """Move ``worker`` to a new QThread and start it. Caller keeps a ref."""
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    thread.start()
    return thread
