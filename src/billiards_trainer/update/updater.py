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
import sys
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
    url: str                 # Windows .exe (the original single-asset field)
    notes: str = ""
    pub_date: str = ""
    sha256: str = ""
    mac_url: str = ""        # macOS .app zip (empty on pre-Mac releases)
    mac_sha256: str = ""

    @classmethod
    def from_manifest(cls, data: dict) -> "UpdateInfo":
        return cls(
            version=str(data.get("version", "")),
            url=str(data.get("url", "")),
            notes=str(data.get("notes", "")),
            pub_date=str(data.get("pub_date", "")),
            sha256=str(data.get("sha256", "")).lower(),
            mac_url=str(data.get("mac_url", "")),
            mac_sha256=str(data.get("mac_sha256", "")).lower(),
        )

    def asset_for_platform(self, platform: str | None = None) -> tuple[str, str]:
        """(download_url, sha256) for the given (default: current) platform.
        Empty url = no binary for this platform in the release."""
        platform = platform or sys.platform
        if platform == "darwin":
            return self.mac_url, self.mac_sha256
        if platform == "win32":
            return self.url, self.sha256
        return "", ""


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


# Windows process-creation flags for the detached swap. CREATE_NO_WINDOW gives a
# HIDDEN console (inherited by the batch's tasklist/find/ping children, so none of
# them pop a window); CREATE_NEW_PROCESS_GROUP keeps our Ctrl-C out of it. We do
# NOT use DETACHED_PROCESS — a console-less cmd is exactly what made the PID-poll's
# piped `find` open a visible window and block forever.
_CREATE_NO_WINDOW = 0x08000000
_CREATE_NEW_PROCESS_GROUP = 0x00000200
SWAP_CREATIONFLAGS = _CREATE_NO_WINDOW | _CREATE_NEW_PROCESS_GROUP


def _show_messagebox(title: str, text: str) -> None:
    """Best-effort native message box, no Qt needed (safe as the app exits).

    Windows-only; silently no-ops elsewhere or if the call fails — the fallback
    must never itself raise."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        # MB_OK | MB_ICONWARNING | MB_SETFOREGROUND
        ctypes.windll.user32.MessageBoxW(0, text, title, 0x00000030 | 0x00010000)
    except Exception:  # noqa: BLE001 - a failed dialog must not mask the real error
        pass


# The swap batch: wait for us to exit, clean stale onefile temp dirs, swap in the
# new exe, unblock it, relaunch, and roll back to a backup if it never confirms
# startup.
#
# CRITICAL spawn note: this MUST be launched with CREATE_NO_WINDOW, *not*
# DETACHED_PROCESS. A detached process has no console, so the piped
# ``tasklist | find "<pid>"`` poll spawns a brand-new console window whose `find`
# never inherits the pipe and instead blocks reading the (empty) console stdin
# forever — leaving the user staring at a stuck ``find "<pid>"`` cmd window while
# the new app never launches. CREATE_NO_WINDOW gives a hidden console with normal
# handle inheritance: no window, the pipe works. (uses ``ping`` for delays, never
# ``timeout`` which needs an interactive console.) Both wait loops are HARD-bounded
# so the updater can never hang; on failure it restores the backup, writes
# update-failed.log to the app-data dir, and opens recovery instructions in Notepad.
_SWAP_BAT = r"""@echo off
setlocal enabledelayedexpansion
set "LOG={log}"
echo [start] %DATE% %TIME% waiting for pid {pid}> "%LOG%"
set /a w=0
:waitexit
tasklist /FI "PID eq {pid}" /NH 2>nul | find "{pid}" >nul
if errorlevel 1 goto exited
set /a w+=1
if !w! geq 30 ( echo [%TIME%] WARN pid {pid} still present after ~30s; proceeding>> "%LOG%" & goto exited )
ping 127.0.0.1 -n 2 >nul
goto waitexit
:exited
echo [%TIME%] old process gone>> "%LOG%"
for /d %%D in ("%TEMP%\_MEI*") do rd /s /q "%%D" 2>nul
copy /Y "{downloaded}" "{current}" >nul
if errorlevel 1 (
  echo [%TIME%] ERROR copy failed>> "%LOG%"
  goto rollback
)
echo [%TIME%] copied new exe>> "%LOG%"
del "{current}:Zone.Identifier" >nul 2>&1
del /f /q "{flag}" >nul 2>&1
echo [%TIME%] launching new exe>> "%LOG%"
start "" "{current}"
set /a n=0
:waitok
if exist "{flag}" goto success
set /a n+=1
if !n! geq {timeout} goto rollback
ping 127.0.0.1 -n 2 >nul
goto waitok
:success
echo [%TIME%] new exe confirmed (sentinel found)>> "%LOG%"
del /f /q "{backup}" >nul 2>&1
del /f /q "{downloaded}" >nul 2>&1
del /f /q "{recovery}" >nul 2>&1
del /f /q "{failflag}" >nul 2>&1
goto done
:rollback
echo [%TIME%] new exe never confirmed -- rolling back>> "%LOG%"
echo failed> "{failflag}"
copy /Y "%LOG%" "{appfaillog}" >nul 2>&1
taskkill /F /IM "{exe_name}" >nul 2>&1
ping 127.0.0.1 -n 3 >nul
if exist "{backup}" copy /Y "{backup}" "{current}" >nul
echo Billiards Trainer update could not start, so your previous version was restored.> "{recovery}"
echo.>> "{recovery}"
echo If the app still does not open, in the install folder:>> "{recovery}"
echo   1. Delete BilliardsTrainer.exe>> "{recovery}"
echo   2. Rename BilliardsTrainer.exe.bak to BilliardsTrainer.exe>> "{recovery}"
echo   3. Run it.>> "{recovery}"
echo.>> "{recovery}"
echo Or download a fresh copy: {releases}>> "{recovery}"
echo Tip: add the install folder to your antivirus exclusions.>> "{recovery}"
start "" "{current}"
start "" notepad "{recovery}"
:done
echo [%TIME%] updater done>> "%LOG%"
del "%~f0" >nul 2>&1
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

    from ..config import APP_DIR
    from ..version import RELEASES_PAGE_URL
    from .recovery import LAUNCHED_OK, UPDATE_FAILED

    current = sys.executable
    install_dir = Path(current).resolve().parent
    backup = str(Path(current).with_name(Path(current).name + ".bak"))
    recovery = str(install_dir / "RECOVERY.txt")
    swap_log = str(install_dir / "updater.log")
    # Human-readable failure log in %LOCALAPPDATA%\BilliardsTrainer — a known,
    # writable location to diagnose a failed update even if the install dir is
    # read-only / AV-locked.
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    app_fail_log = str(APP_DIR / "update-failed.log")
    try:
        shutil.copy2(current, backup)
    except OSError as exc:
        log.warning("Could not back up current exe (%s); proceeding without rollback", exc)
        backup = ""
    # Strip the mark-of-the-web from the download so Windows doesn't block launch.
    try:
        Path(downloaded + ":Zone.Identifier").unlink(missing_ok=True)
    except OSError:
        pass
    for flag in (LAUNCHED_OK, UPDATE_FAILED):
        try:
            flag.unlink(missing_ok=True)
        except OSError:
            pass

    bat = Path(tempfile.gettempdir()) / "bt_update.bat"
    bat.write_text(_SWAP_BAT.format(
        pid=os.getpid(), downloaded=downloaded, current=current,
        backup=backup, exe_name=Path(current).name,
        flag=str(LAUNCHED_OK), failflag=str(UPDATE_FAILED),
        recovery=recovery, log=swap_log, appfaillog=app_fail_log,
        releases=RELEASES_PAGE_URL, timeout=60,
    ), encoding="utf-8")
    log.info("Launching update swap: new=%s -> %s (backup=%s, log=%s)",
             downloaded, current, backup or "none", swap_log)
    # CREATE_NO_WINDOW (NOT DETACHED_PROCESS): a detached cmd has no console, so its
    # piped ``tasklist | find`` poll spawns a visible console whose `find` blocks on
    # console stdin forever — the stuck ``find "<pid>"`` window bug. CREATE_NO_WINDOW
    # gives a hidden console with normal handle inheritance: no window, the pipe
    # works, and the Popen child still outlives us. NEW_PROCESS_GROUP detaches it
    # from our Ctrl-C group; explicit DEVNULL std handles keep redirects sane.
    try:
        subprocess.Popen(
            ["cmd", "/c", str(bat)],
            creationflags=SWAP_CREATIONFLAGS,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            close_fds=True,
        )
    except OSError as exc:
        # We couldn't even start the swap. Don't strand the user with no app and no
        # message: record it and pop a native dialog pointing at the installer.
        log.error("Could not launch update swap: %s", exc)
        try:
            Path(app_fail_log).write_text(
                f"The update could not be started: {exc}\n\n"
                f"Please re-run the installer from {RELEASES_PAGE_URL}\n",
                encoding="utf-8")
        except OSError:
            pass
        _show_messagebox(
            "Billiards Trainer — update",
            "The update could not be started, so your current version is unchanged.\n\n"
            f"Please download and run the latest installer from:\n{RELEASES_PAGE_URL}")
        raise


def run_in_thread(worker: QObject) -> QThread:
    """Move ``worker`` to a new QThread and start it. Caller keeps a ref."""
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    thread.start()
    return thread
