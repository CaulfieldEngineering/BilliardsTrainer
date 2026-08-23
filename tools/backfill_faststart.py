"""One-time backfill: re-container every fragmented session mp4 as faststart.

Every recording before 2026-08-23 was written fragmented (empty front moov)
— unplayable-without-full-download on iOS. New recordings are remuxed at
finalize (controller._remux_faststart); this fixes the existing library.

Safety: skips files modified in the last 10 minutes, stops entirely if a
recording is in progress (a hidden .part exists in exports) or one starts
between files. Remux happens in a temp dir OUTSIDE the Dropbox folder,
then an atomic replace; original mtime is preserved (session tooling
verifies UTC filename stamps against mtime). Already-faststart files are
detected (front moov > 10 KB) and skipped, so reruns are cheap no-ops.
"""
import ctypes
import glob
import os
import struct
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from billiards_trainer.capture.audio import NO_WINDOW, find_ffmpeg  # noqa: E402
from billiards_trainer.config import EXPORTS_DIR  # noqa: E402

SESS_DIR = "C:/Users/Joe/Dropbox/Billiards/BilliardsTrainer"


def front_moov_size(path: str) -> int:
    """Size of the moov box among the first two top-level boxes (0 if absent)."""
    with open(path, "rb") as f:
        pos = 0
        for _ in range(2):
            f.seek(pos)
            hdr = f.read(8)
            if len(hdr) < 8:
                return 0
            size, typ = struct.unpack(">I4s", hdr)
            if typ == b"moov":
                return size
            if size == 1:
                size = struct.unpack(">Q", f.read(8))[0]
            if size <= 0:
                return 0
            pos += size
    return 0


def recording_active() -> bool:
    if list(EXPORTS_DIR.glob(".session-*.part.mp4")):
        return True
    newest = max((os.path.getmtime(f) for f in
                  glob.glob(os.path.join(SESS_DIR, "session-*.mp4"))), default=0)
    return (time.time() - newest) < 600


def main() -> None:
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    ff = find_ffmpeg()
    if not ff:
        sys.exit("no ffmpeg")
    files = sorted(glob.glob(os.path.join(SESS_DIR, "session-*.mp4")),
                   key=os.path.getmtime, reverse=True)  # newest first
    done = skipped = failed = 0
    for p in files:
        if recording_active():
            print("recording active - stopping backfill (rerun later)")
            break
        if front_moov_size(p) > 10_000:
            skipped += 1
            continue
        mtime = os.path.getmtime(p)
        tmp = os.path.join(tempfile.gettempdir(),
                           "bt_fs_" + os.path.basename(p))
        try:
            r = subprocess.run(
                [ff, "-v", "error", "-i", p, "-c", "copy",
                 "-movflags", "+faststart", "-y", tmp],
                capture_output=True, timeout=600, creationflags=NO_WINDOW)
            if (r.returncode == 0
                    and os.path.getsize(tmp) > os.path.getsize(p) * 0.9):
                os.replace(tmp, p)
                os.utime(p, (mtime, mtime))
                done += 1
                print("ok  %s (%.1f GB)" % (os.path.basename(p),
                                            os.path.getsize(p) / 1e9))
            else:
                failed += 1
                err = (r.stderr or b"")[-200:].decode("utf-8", "replace")
                print("FAIL %s: %s" % (os.path.basename(p), err.strip()))
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("FAIL %s: %s" % (os.path.basename(p), exc))
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    print("backfill: %d remuxed, %d already fine, %d failed"
          % (done, skipped, failed))


if __name__ == "__main__":
    main()
