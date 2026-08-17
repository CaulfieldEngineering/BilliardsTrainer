"""Per-session analysis sidecar: analyze once, play back forever.

Joe, watching choppy playback: "by the time we're playing a video back, we
shouldn't have to process anything in realtime, right? It should all be
cached data simply being played back." Correct — running 83ms/frame of
inference during playback was the design mistake. The pipeline's output is
recorded ONCE (live while recording, or backfilled offline) into a JSONL
sidecar next to the session file; playback then decodes video, looks up
the cached state, and paints — no models, no tracker, smooth by
construction. This is also the substrate of the shot dossier: the sidecar
IS the per-session tracking record.

Format (one JSON object per line):
    {"type":"meta","v":1,"fps":30,"table":{...},"H":[...],"corners":[...]}
    {"type":"f","t":12.34,"tracks":[[id,x,y,r,num,cls,active],...]}
    {"type":"shot","start":10.2,"end":16.8,"outcome":"make","pocketed":1}
    {"type":"correction","start":10.2,"outcome":"miss"}   # review verdicts

Corrections are APPENDED (the file is a log, not a document): the reader
applies the last correction matching a shot's start time. Joe's review
verdicts therefore survive re-opens and travel with the session file.

Track states land at detection cadence (~7-10 Hz); the reader interpolates
between neighbouring records so overlays glide at display rate.
"""

import json
import logging
from bisect import bisect_right
from pathlib import Path

from .types import BallClass, Track

log = logging.getLogger("vision.cache")

SIDECAR_SUFFIX = ".analysis.jsonl"


def sidecar_path(video_path: str | Path) -> Path:
    return Path(str(video_path) + SIDECAR_SUFFIX)


def append_correction(video_path: str | Path, start: float, outcome: str) -> bool:
    """Persist a review verdict for the shot starting at ``start`` seconds.

    Append-only by design — the sidecar is a log. Returns False when there
    is no sidecar to correct (nothing silently invented)."""
    p = sidecar_path(video_path)
    if not p.is_file():
        return False
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "correction", "start": round(float(start), 3),
                            "outcome": outcome}) + "\n")
    return True


class SidecarWriter:
    """Appends pipeline state while a session records. Cheap: ~3 KB/s."""

    def __init__(self, video_path: str | Path, meta: dict):
        self._path = sidecar_path(video_path)
        self._f = open(self._path, "w", encoding="utf-8")  # noqa: SIM115 - long-lived
        meta = {"type": "meta", "v": 1, **meta}
        self._f.write(json.dumps(meta) + "\n")
        self._n = 0

    def add_frame(self, t: float, tracks: list) -> None:
        rec = [[int(tr.id), round(float(tr.x), 1), round(float(tr.y), 1),
                round(float(tr.radius), 1), int(tr.number),
                tr.cls.value, bool(tr.active)] for tr in tracks]
        self._f.write(json.dumps({"type": "f", "t": round(t, 3), "tracks": rec},
                                 separators=(",", ":")) + "\n")
        self._n += 1
        if self._n % 50 == 0:
            self._f.flush()

    def add_shot(self, event) -> None:
        self._f.write(json.dumps({
            "type": "shot", "start": round(float(event.start_t), 3),
            "end": round(float(event.end_t), 3),
            "outcome": event.outcome.value,
            "pocketed": int(event.num_pocketed)}) + "\n")
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.flush()
            self._f.close()
            log.info("analysis sidecar written: %s (%d states)", self._path, self._n)
        except OSError:
            pass


class SidecarReader:
    """Loads a sidecar and answers 'tracks at time t' with interpolation."""

    def __init__(self, video_path: str | Path):
        self.meta: dict = {}
        self.shots: list[dict] = []
        self._times: list[float] = []
        self._frames: list[list] = []
        p = sidecar_path(video_path)
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("type") == "meta":
                    self.meta = d
                elif d.get("type") == "f":
                    self._times.append(float(d["t"]))
                    self._frames.append(d["tracks"])
                elif d.get("type") == "shot":
                    self.shots.append(d)
                elif d.get("type") == "correction":
                    # last-wins: apply to the shot whose start matches
                    for s in self.shots:
                        if abs(float(s.get("start", -1)) - float(d["start"])) < 0.2:
                            s["outcome"] = d.get("outcome", s.get("outcome"))
                            s["corrected"] = True
                            break
        log.info("analysis sidecar loaded: %s (%d states, %d shots)",
                 p.name, len(self._times), len(self.shots))

    @staticmethod
    def exists(video_path: str | Path) -> bool:
        return sidecar_path(video_path).is_file()

    def __len__(self) -> int:
        return len(self._times)

    # ------------------------------------------------------------------ #
    def tracks_at(self, t: float) -> list[Track]:
        """Interpolated track list for media time ``t`` (seconds)."""
        if not self._times:
            return []
        i = bisect_right(self._times, t)
        if i <= 0:
            return self._to_tracks(self._frames[0])
        if i >= len(self._times):
            return self._to_tracks(self._frames[-1])
        t0, t1 = self._times[i - 1], self._times[i]
        a, b = self._frames[i - 1], self._frames[i]
        if t1 - t0 <= 1e-6 or t1 - t0 > 1.0:
            # a gap (detection pause) — snap, don't tween across it
            return self._to_tracks(a)
        w = (t - t0) / (t1 - t0)
        bmap = {r[0]: r for r in b}
        out = []
        for r in a:
            r2 = bmap.get(r[0])
            if r2 is None:
                out.append(self._to_track(r))
                continue
            blended = [r[0],
                       r[1] + (r2[1] - r[1]) * w,
                       r[2] + (r2[2] - r[2]) * w,
                       r[3], r[4], r[5], r[6]]
            out.append(self._to_track(blended))
        return out

    @staticmethod
    def _to_track(r) -> Track:
        return Track(id=int(r[0]), x=float(r[1]), y=float(r[2]),
                     radius=float(r[3]), number=int(r[4]),
                     cls=BallClass(r[5]), active=bool(r[6]))

    def _to_tracks(self, rows) -> list[Track]:
        return [self._to_track(r) for r in rows]


def clip_export_cmd(video_path: str | Path, start: float, end: float,
                    pre_roll_s: float = 5.0, tail_s: float = 1.0,
                    shot_no: int | None = None) -> tuple[list[str], Path]:
    """(ffmpeg argv, destination) for exporting one shot as a clip.

    STREAM COPY — no re-encode, so an export takes well under a second and
    keeps the recording's exact quality. -ss before -i snaps to the
    previous keyframe, which errs on including a little MORE lead-in: for
    a pre-shot-routine clip that is the right direction to err.
    Destination: <recordings>/clips/<session>_shotNN.mp4.
    """
    video_path = Path(video_path)
    t0 = max(0.0, float(start) - pre_roll_s)
    t1 = float(end) + tail_s
    out_dir = video_path.parent / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_shot{shot_no:02d}" if shot_no is not None else f"_{t0:.0f}s"
    dest = out_dir / f"{video_path.stem}{tag}.mp4"
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t0:.2f}", "-to", f"{t1:.2f}",
           "-i", str(video_path), "-c", "copy", "-y", str(dest)]
    return cmd, dest
