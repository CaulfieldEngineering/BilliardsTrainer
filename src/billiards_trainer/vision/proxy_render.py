"""Phone playback proxies: 720p ~3.5 Mbps twins of the recordings.

Sessions record at 16-24 Mbps (constant-quality QP20 spends bits on
detail) — heavy for the phone's decoder (the trail-stutter suspect) and
heavier on cellular. A proxy in <recordings>/proxies/ carries the
IDENTICAL timeline, so every overlay, seek, correction and shot ID works
unchanged; /api/link simply prefers it when present.

Encoded with the AMD hardware encoder (h264_amf — the same one the
recorder uses, so a proxy render must NEVER run while a recording is
live: the guard kills it the moment a .part appears and deletes the
partial; the next close pass or backfill retries).
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

log = logging.getLogger("vision.proxy")

PROXY_DIRNAME = "proxies"
# Below this, the original streams fine on the phone and a proxy would
# barely shrink it (short stub sessions) - skip, link.js falls back.
MIN_SOURCE_BYTES = 25_000_000


def proxy_path(video) -> Path:
    video = Path(video)
    return video.parent / PROXY_DIRNAME / video.name


def has_proxy(video) -> bool:
    p = proxy_path(video)
    return p.is_file() and p.stat().st_size > 100_000


def render_proxy(video, timeout_s: int = 5400) -> bool:
    """Render the proxy, recording-guarded. True when a proxy exists on
    return (fresh or already there)."""
    from ..capture.audio import NO_WINDOW, find_ffmpeg
    from ..config import EXPORTS_DIR
    video = Path(video)
    if has_proxy(video):
        return True
    if video.is_file() and video.stat().st_size < MIN_SOURCE_BYTES:
        return True                      # small enough to stream as-is
    ff = find_ffmpeg()
    if ff is None or not video.is_file():
        return False

    def recording_live() -> bool:
        return bool(list(EXPORTS_DIR.glob(".session-*.part.mp4")))

    if recording_live():
        log.info("proxy render deferred: recording live")
        return False
    out = proxy_path(video)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part.mp4")
    cmd = [ff, "-v", "error", "-i", str(video),
           "-vf", "scale=-2:720",
           "-c:v", "h264_amf", "-rc", "cbr", "-b:v", "3500k",
           "-c:a", "aac", "-b:a", "96k",
           "-movflags", "+faststart", "-y", str(tmp)]
    try:
        proc = subprocess.Popen(cmd, creationflags=NO_WINDOW,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        t0 = time.time()
        while proc.poll() is None:
            if recording_live():
                # the recorder owns the AMF encoder — Joe outranks proxies
                proc.kill()
                proc.wait(timeout=15)
                tmp.unlink(missing_ok=True)
                log.info("proxy render killed: recording started")
                return False
            if time.time() - t0 > timeout_s:
                proc.kill()
                proc.wait(timeout=15)
                tmp.unlink(missing_ok=True)
                log.warning("proxy render timed out for %s", video.name)
                return False
            time.sleep(5)
        if proc.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 100_000:
            tmp.replace(out)
            log.info("proxy rendered: %s (%.0f MB)", out.name,
                     out.stat().st_size / 1e6)
            return True
        tmp.unlink(missing_ok=True)
        log.warning("proxy render failed for %s (rc=%s)", video.name,
                    proc.returncode)
        return False
    except Exception:  # noqa: BLE001 - proxies are enrichment
        log.exception("proxy render errored for %s", video.name)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False
