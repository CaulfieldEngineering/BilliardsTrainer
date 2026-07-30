"""Audio sidecar recording for session clips.

OpenCV's VideoWriter is video-only, so audio is captured in parallel by an
ffmpeg subprocess (DirectShow on Windows, avfoundation on macOS) and muxed into
the session mp4 when the recording stops. Everything degrades gracefully: no
ffmpeg, no audio device, or a failed mux all leave the plain video-only mp4
exactly as before.

Pause handling: ffmpeg can't pause a capture, so each unpaused stretch becomes
its own .m4a segment; stop() concatenates the segments before the mux. That
keeps audio aligned with the frames actually written.

Sync: the video's container fps is whatever the writer declared, but frames are
written at the pipeline's real rate — over minutes that skew makes audio drift.
The mux rescales video timestamps by (declared/actual) so the clip plays in
true wall-clock time, which also fixes playback pacing of the video itself.
"""

from __future__ import annotations

import logging
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger("capture.audio")

_FFMPEG_FALLBACKS = ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg")


def find_ffmpeg() -> str | None:
    """Path of an ffmpeg binary, or None. Homebrew paths are checked explicitly
    because a .app launched from Finder doesn't inherit the shell PATH."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    for cand in _FFMPEG_FALLBACKS:
        if Path(cand).is_file():
            return cand
    return None


def find_ffprobe() -> str | None:
    """Path of the ffprobe that ships beside our ffmpeg, or None.

    Derived as a SIBLING of the ffmpeg binary (keeping its suffix) rather than
    by string-substitution on the path: winget installs ffmpeg under a
    directory literally named ``ffmpeg-8.1.2-full_build``, so a naive
    ``replace("ffmpeg", "ffprobe")`` rewrites the directory too and yields a
    path that does not exist. Falls back to PATH lookup.
    """
    ff = find_ffmpeg()
    if ff:
        probe = Path(ff).with_name("ffprobe" + Path(ff).suffix)
        if probe.is_file():
            return str(probe)
    return shutil.which("ffprobe")


def list_audio_devices() -> list[str]:
    """Capture-device names ffmpeg can record from, best-first ([] if none).

    Windows/dshow needs a device NAME (there is no ":default" as on macOS), so
    the rig's USB mic has to be discovered rather than assumed. ffmpeg prints
    the device list to stderr and exits non-zero by design — that is not an
    error here.
    """
    ff = find_ffmpeg()
    if ff is None or sys.platform != "win32":
        return []
    try:
        res = subprocess.run(
            [ff, "-hide_banner", "-list_devices", "true", "-f", "dshow",
             "-i", "dummy"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("could not enumerate audio devices: %s", exc)
        return []
    names = []
    for line in res.stderr.splitlines():
        # rows look like:  [dshow @ ...] "Yeti Stereo Microphone" (audio)
        if not line.rstrip().endswith("(audio)"):
            continue
        start, end = line.find('"'), line.rfind('"')
        if 0 <= start < end:
            names.append(line[start + 1:end])
    return names


class AudioRecorder:
    """Segment-per-unpaused-stretch audio capture via ffmpeg.

    avfoundation on macOS, DirectShow on Windows — the rig PC is Windows, and
    the T3i sends no audio over HDMI, so the source is a separate USB mic.
    """

    def __init__(self, device: str = "default") -> None:
        self._device = device or "default"
        self._ffmpeg = find_ffmpeg()
        self._proc: subprocess.Popen | None = None
        self._segments: list[Path] = []
        self._work_dir: Path | None = None

    @property
    def available(self) -> bool:
        return self._ffmpeg is not None and sys.platform in ("darwin", "win32")

    def _input_args(self) -> list[str] | None:
        """Platform-specific ffmpeg input for the chosen capture device."""
        if sys.platform == "darwin":
            # avfoundation's ":<dev>" form selects audio-only.
            return ["-f", "avfoundation", "-i", f":{self._device}"]
        if sys.platform == "win32":
            device = self._device
            if not device or device == "default":
                found = list_audio_devices()
                if not found:
                    log.warning("no DirectShow audio capture device found")
                    return None
                device = found[0]
                log.info("audio device (auto): %s", device)
            return ["-f", "dshow", "-i", f"audio={device}"]
        return None

    def start(self, work_dir: Path) -> bool:
        """Begin (or resume) capturing a new audio segment under work_dir."""
        if not self.available or self._proc is not None:
            return False
        work_dir.mkdir(parents=True, exist_ok=True)
        self._work_dir = work_dir
        seg = work_dir / f"audio-{len(self._segments):03d}.wav"
        # -thread_queue_size: avfoundation drops input packets when the queue
        # overruns (measured: a 202s session captured only 177s of samples, the
        # PTS gaps then read as "audio early"/drifting). A big queue plus RAW
        # PCM (no AAC encode inside the capture process, which competes with
        # the app's GPU/CPU work) keeps capture real-time.
        # MEASURED on this rig: the USB mic drops ~12% of its samples, so a
        # plain capture yields 10.57s of audio per 12.00s of wall time and the
        # file drifts earlier and earlier. Stamping packets from the WALL CLOCK
        # and running aresample=async=1 at CAPTURE time fills those gaps with
        # silence immediately -> 12.00s captured per 12.00s wall, verified 0.0%
        # deficit. (Tiny inaudible micro-dropouts instead of a sliding clock.)
        source = self._input_args()
        if source is None:
            return False
        cmd = [self._ffmpeg, "-hide_banner", "-loglevel", "error",
               "-thread_queue_size", "4096",
               "-use_wallclock_as_timestamps", "1",
               *source,
               "-af", "aresample=async=1:first_pts=0",
               "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", "-y", str(seg)]
        try:
            self._proc = subprocess.Popen(
                cmd,
                # Windows has no SIGINT for a child process, so the graceful
                # stop is writing "q" to ffmpeg's stdin (see _stop_proc).
                stdin=subprocess.PIPE if sys.platform == "win32"
                else subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            log.warning("audio capture failed to start: %s", exc)
            return False
        self._segments.append(seg)
        log.info("audio capture started -> %s", seg.name)
        return True

    def pause(self) -> None:
        """Finalize the current segment (recording pause)."""
        self._stop_proc()

    def _stop_proc(self) -> None:
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        try:
            if sys.platform == "win32":
                # Windows can't deliver SIGINT to a child (Popen.send_signal
                # raises ValueError for it), and killing outright truncates the
                # WAV header so the segment reads as 0 samples. ffmpeg's own
                # "q" command on stdin is the graceful equivalent.
                if proc.stdin:
                    proc.stdin.write(b"q\n")
                    proc.stdin.flush()
                    proc.stdin.close()
            else:
                # SIGINT lets ffmpeg write the trailer; SIGKILL would corrupt it.
                proc.send_signal(signal.SIGINT)
            proc.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            proc.kill()
            proc.wait(timeout=2)

    def stop_and_mux(self, video_path: str, ts_scale: float = 1.0) -> bool:
        """Finish capture and produce video_path WITH the audio track.

        ts_scale rescales video timestamps (declared fps / actual fps) so audio
        stays in sync over long sessions. Returns True when the mp4 now has
        audio; on any failure the original video-only file is left untouched.
        """
        self._stop_proc()
        segments = [s for s in self._segments if s.is_file() and s.stat().st_size > 0]
        self._segments = []
        if not segments or self._ffmpeg is None:
            self._cleanup(segments)
            return False

        video = Path(video_path)
        work = self._work_dir or video.parent
        try:
            audio = self._concat(segments, work)
            if audio is None:
                return False
            muxed = video.with_name(video.stem + ".muxed.mp4")
            cmd = [self._ffmpeg, "-hide_banner", "-loglevel", "error"]
            if abs(ts_scale - 1.0) > 0.02:
                cmd += ["-itsscale", f"{ts_scale:.6f}"]
            # aresample=async=1 INSERTS SILENCE wherever the capture dropped
            # samples, so the audio timeline matches its (correct) wall-clock
            # timestamps instead of sliding earlier and earlier. apad+shortest
            # then makes the track span exactly the video's length.
            # MEASURED lead-in trim: capture starts when recording is armed,
            # but the video writer only opens on the first frame AND the mic
            # needs ~0.5s to deliver its first buffer. Both streams stop
            # together, so whatever extra length the audio has IS the lead-in
            # — trim exactly that from its start instead of guessing a delay.
            lead = self._duration(audio) - self._duration(video)
            cmd += ["-i", str(video)]
            if lead > 0.05:
                cmd += ["-ss", f"{lead:.3f}"]
                log.info("audio lead-in trim: %.3fs", lead)
            cmd += ["-i", str(audio),
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-af", "aresample=async=1:first_pts=0,apad",
                    "-shortest",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                    "-movflags", "+faststart", "-y", str(muxed)]
            res = subprocess.run(cmd, capture_output=True, timeout=300)
            if res.returncode != 0 or not muxed.is_file() or muxed.stat().st_size == 0:
                log.warning("audio mux failed: %s", res.stderr.decode(errors="replace")[-400:])
                muxed.unlink(missing_ok=True)
                return False
            muxed.replace(video)
            log.info("audio muxed into %s (ts_scale %.3f)", video.name, ts_scale)
            return True
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("audio mux failed: %s", exc)
            return False
        finally:
            self._cleanup(segments)

    def _duration(self, path: Path) -> float:
        """Media duration in seconds via ffprobe (0.0 when unknown)."""
        probe = find_ffprobe()
        if not probe:
            return 0.0
        try:
            res = subprocess.run(
                [probe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(path)],
                capture_output=True, text=True, timeout=30)
            return float(res.stdout.strip() or 0.0)
        except (OSError, ValueError, subprocess.SubprocessError):
            return 0.0

    def _concat(self, segments: list[Path], work: Path) -> Path | None:
        if len(segments) == 1:
            return segments[0]
        listing = work / "audio-concat.txt"
        listing.write_text(
            "".join(f"file '{s.as_posix()}'\n" for s in segments), encoding="utf-8")
        out = work / "audio-full.wav"
        cmd = [self._ffmpeg, "-hide_banner", "-loglevel", "error",
               "-f", "concat", "-safe", "0", "-i", str(listing),
               "-c:a", "pcm_s16le", "-y", str(out)]
        res = subprocess.run(cmd, capture_output=True, timeout=120)
        if res.returncode != 0:
            log.warning("audio concat failed: %s", res.stderr.decode(errors="replace")[-400:])
            return None
        return out

    def _cleanup(self, segments: list[Path]) -> None:
        work = self._work_dir
        for s in segments:
            s.unlink(missing_ok=True)
        if work is not None:
            for name in ("audio-concat.txt", "audio-full.wav"):
                (work / name).unlink(missing_ok=True)
            try:
                work.rmdir()  # only if empty
            except OSError:
                pass
        self._work_dir = None


class _NullAudio:
    """Stand-in when audio is disabled: same surface, does nothing."""

    available = False

    def start(self, work_dir: Path) -> bool:  # noqa: ARG002
        return False

    def pause(self) -> None:
        pass

    def stop_and_mux(self, video_path: str, ts_scale: float = 1.0) -> bool:  # noqa: ARG002
        return False


def make_recorder(enabled: bool, device: str = "default"):
    """AudioRecorder when audio is on and possible, else a no-op stand-in."""
    if not enabled:
        return _NullAudio()
    rec = AudioRecorder(device)
    if not rec.available:
        log.info("audio recording unavailable (no ffmpeg, or unsupported "
                 "platform %s) — video only", sys.platform)
        return _NullAudio()
    return rec


elapsed_monotonic = time.monotonic  # patchable in tests
