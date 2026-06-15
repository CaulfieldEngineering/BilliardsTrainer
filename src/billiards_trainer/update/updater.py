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
    sha256: str = ""

    @classmethod
    def from_manifest(cls, data: dict) -> "UpdateInfo":
        return cls(
            version=str(data.get("version", "")),
            url=str(data.get("url", "")),
            notes=str(data.get("notes", "")),
            pub_date=str(data.get("pub_date", "")),
            sha256=str(data.get("sha256", "")).lower(),
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


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """SHA256 of a file as a lowercase hex string."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class DownloadWorker(QObject):
    """Streams an installer to a temp file, verifies its SHA256, emits progress."""

    progress = Signal(int)
    finished = Signal(str)   # path to verified file
    failed = Signal(str)

    def __init__(self, url: str, expected_sha: str = ""):
        super().__init__()
        self._url = url
        self._expected = (expected_sha or "").lower()

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
        except (requests.RequestException, OSError) as exc:
            log.warning("Update download failed: %s", exc)
            self.failed.emit(str(exc))
            return

        # Integrity check — never swap a corrupt/partial download.
        if self._expected:
            actual = sha256_file(str(dest))
            if actual != self._expected:
                log.warning("Update checksum mismatch: expected %s got %s",
                            self._expected, actual)
                self.failed.emit(
                    "The downloaded update is corrupted (checksum mismatch). "
                    "Please reinstall manually from the GitHub releases page.")
                return
            log.info("Update download verified (sha256 %s…)", actual[:12])
        else:
            log.info("Update download complete (no checksum to verify against)")
        self.finished.emit(str(dest))


# The swap batch: wait for us to exit, clean stale onefile temp dirs, copy the
# new exe in, relaunch, and roll back to a backup if it doesn't confirm startup.
_SWAP_BAT = r"""@echo off
setlocal enabledelayedexpansion
:waitexit
tasklist /FI "PID eq {pid}" 2>nul | find "{pid}" >nul
if not errorlevel 1 (
  timeout /t 1 /nobreak >nul
  goto waitexit
)
for /d %%D in ("%TEMP%\_MEI*") do rd /s /q "%%D" 2>nul
copy /Y "{downloaded}" "{current}" >nul
if errorlevel 1 goto rollback
del /f /q "{flag}" 2>nul
start "" "{current}"
set /a n=0
:waitok
if exist "{flag}" goto success
set /a n+=1
if !n! geq {timeout} goto rollback
timeout /t 1 /nobreak >nul
goto waitok
:success
del /f /q "{backup}" 2>nul
del /f /q "{downloaded}" 2>nul
goto done
:rollback
taskkill /F /IM "{exe_name}" >nul 2>&1
timeout /t 1 /nobreak >nul
if exist "{backup}" copy /Y "{backup}" "{current}" >nul
echo failed > "{failflag}"
start "" "{current}"
:done
del "%~f0"
"""


def install_and_relaunch(downloaded: str, expected_sha: str = "") -> None:
    """Apply a downloaded update with verification, backup and rollback.

    Frozen Windows path: back up the current exe, then spawn a detached batch
    that waits for us to exit, cleans stale ``_MEI*`` temp dirs, copies the new
    exe in, relaunches, and — if the new exe fails to confirm startup within the
    timeout (e.g. AV quarantined a bundled DLL) — restores the backup and flags
    the failure so the restored app can explain it. Source/dev path just opens
    the file.
    """
    import os
    import shutil
    import subprocess
    import sys

    if expected_sha:
        try:
            if sha256_file(downloaded).lower() != expected_sha.lower():
                raise ValueError("checksum mismatch at install time")
        except (OSError, ValueError) as exc:
            log.warning("Refusing to install — %s", exc)
            raise

    if not (sys.platform == "win32" and getattr(sys, "frozen", False)
            and downloaded.lower().endswith(".exe")):
        if sys.platform == "win32":
            os.startfile(downloaded)  # type: ignore[attr-defined]
        else:
            subprocess.Popen([downloaded])
        return

    from .recovery import LAUNCHED_OK, UPDATE_FAILED

    current = sys.executable
    backup = str(Path(current).with_name(Path(current).name + ".bak"))
    try:
        shutil.copy2(current, backup)
    except OSError as exc:
        log.warning("Could not back up current exe (%s); proceeding without rollback", exc)
        backup = ""
    for flag in (LAUNCHED_OK, UPDATE_FAILED):
        try:
            flag.unlink(missing_ok=True)
        except OSError:
            pass

    bat = Path(tempfile.gettempdir()) / "bt_update.bat"
    bat.write_text(_SWAP_BAT.format(
        pid=os.getpid(), downloaded=downloaded, current=current,
        backup=backup, exe_name=Path(current).name,
        flag=str(LAUNCHED_OK), failflag=str(UPDATE_FAILED), timeout=25,
    ), encoding="utf-8")
    log.info("Launching update swap (backup=%s)", backup or "none")
    subprocess.Popen(["cmd", "/c", str(bat)], creationflags=0x00000008)


def run_in_thread(worker: QObject) -> QThread:
    """Move ``worker`` to a new QThread and start it. Caller keeps a ref."""
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    thread.start()
    return thread
