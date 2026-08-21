"""On-demand RIFE smooth slow-motion renders (Joe: "Rife looks great").

Casual slow-mo stays the player's ½×/¼× playback; THIS renders a
4×-interpolated clip of one shot on request. Output lands in
<recordings>/slowmo/ — a subfolder the sessions list never sees
(list_folder is non-recursive) — and the watcher appends it to the
"Slow-mo" playlist in playlists.json, which the phone pulls.

The rife binary lives OUTSIDE the repo (~35MB of exe+models):
C:/Users/Joe/.billiards-tools/rife — tools/fetch_rife.py restores it.
Measured on the mini PC's Radeon 780M: ~35s for a 6s clip at half
resolution (full-res OOMs the iGPU while the live app holds it;
half-res matches the phone's display size anyway).
"""

import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

log = logging.getLogger("companion.rife")

RIFE_EXE = Path.home() / ".billiards-tools" / "rife" / "rife-ncnn-vulkan.exe"
_MODEL = "rife-v4.6"
_FACTOR = 4          # 30fps -> effective 1/4-speed at smooth 30fps
_PRE_ROLL = 1.5
_TAIL = 1.0


def _ffmpeg() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


def slowmo_name(video: Path, start: float) -> str:
    return f"slowmo-{video.stem}-{int(round(start))}.mp4"


def render_slowmo(video: Path, start: float, end: float) -> Path | None:
    """Render one shot's smooth slow-mo clip. Returns the output path in
    <recordings>/slowmo/, or None on failure. Synchronous (~35s/6s clip);
    the caller decides threading and recording-guards."""
    if not RIFE_EXE.is_file():
        log.warning("rife binary missing at %s (run tools/fetch_rife.py)",
                    RIFE_EXE)
        return None
    out_dir = video.parent / "slowmo"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / slowmo_name(video, start)
    if out.is_file():
        return out                      # idempotent: re-requests are free
    t0 = max(0.0, float(start) - _PRE_ROLL)
    dur = max(1.0, float(end) - float(start) + _PRE_ROLL + _TAIL)
    dur = min(dur, 20.0)                # bound the render (~2 min worst)
    tick = time.time()
    with tempfile.TemporaryDirectory(prefix="rife-") as td:
        tdp = Path(td)
        (tdp / "in").mkdir()
        (tdp / "out").mkdir()
        try:
            subprocess.run(
                [_ffmpeg(), "-y", "-ss", f"{t0:.2f}", "-i", str(video),
                 "-t", f"{dur:.2f}", "-an", "-vf", "scale=468:820",
                 str(tdp / "in" / "%05d.png")],
                check=True, capture_output=True, timeout=120)
            n_in = len(list((tdp / "in").glob("*.png")))
            if n_in < 8:
                log.warning("rife: too few frames (%d) for %s @%.1f",
                            n_in, video.name, start)
                return None
            subprocess.run(
                [str(RIFE_EXE), "-i", str(tdp / "in"), "-o", str(tdp / "out"),
                 "-m", _MODEL, "-n", str(n_in * _FACTOR), "-j", "1:1:1"],
                check=True, capture_output=True, timeout=600)
            subprocess.run(
                [_ffmpeg(), "-y", "-framerate", "30",
                 "-i", str(tdp / "out" / "%08d.png"),
                 "-c:v", "libx264", "-crf", "19", "-pix_fmt", "yuv420p",
                 str(out)],
                check=True, capture_output=True, timeout=120)
        except (subprocess.SubprocessError, OSError) as exc:
            log.warning("rife render failed for %s @%.1f: %s",
                        video.name, start, exc)
            return None
    log.info("rife: %s rendered in %.0fs", out.name, time.time() - tick)
    return out


def add_to_slowmo_playlist(recordings: Path, clip_name: str,
                           label: str) -> None:
    """Append the rendered clip to the 'Slow-mo' playlist in the synced
    playlists.json (created if absent). The phone pulls this document."""
    import json
    p = recordings / "playlists.json"
    try:
        doc = json.loads(p.read_text(encoding="utf-8")) if p.is_file() \
            else {"mod": 0, "playlists": []}
    except (OSError, ValueError):
        doc = {"mod": 0, "playlists": []}
    now = int(time.time() * 1000)
    pl = next((q for q in doc.get("playlists", [])
               if q.get("name") == "Slow-mo"), None)
    if pl is None:
        pl = {"id": "slowmo", "name": "Slow-mo", "mod": now, "clips": []}
        doc.setdefault("playlists", []).append(pl)
    if not any(c.get("slowmo") == clip_name for c in pl["clips"]):
        pl["clips"].append({"slowmo": clip_name, "label": label,
                            "session": "", "start": 0})
        pl["mod"] = now
        doc["mod"] = now
        p.write_text(json.dumps(doc, separators=(",", ":")),
                     encoding="utf-8")
