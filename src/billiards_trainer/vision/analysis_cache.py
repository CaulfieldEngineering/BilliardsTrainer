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

from ..core.types import BallClass, Track

log = logging.getLogger("vision.cache")

SIDECAR_SUFFIX = ".analysis.jsonl"


def sidecar_path(video_path: str | Path) -> Path:
    return Path(str(video_path) + SIDECAR_SUFFIX)


def append_correction(video_path: str | Path, start: float, outcome: str,
                      src: str = "review") -> bool:
    """Persist an outcome verdict for the shot starting at ``start``.

    ``src`` ranks the verdict: "review" (a human looked) outranks
    "derived" (recomputed from the identity record) — a re-run of the
    derivation must never clobber a frame-verified human call, which is
    exactly what happened to the 9-ball session's shot 5 before this
    field existed. Append-only by design — the sidecar is a log."""
    p = sidecar_path(video_path)
    if not p.is_file():
        return False
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "correction", "start": round(float(start), 3),
                            "outcome": outcome, "src": src}) + "\n")
    return True


class SidecarWriter:
    """Appends pipeline state while a session records. Cheap: ~3 KB/s."""

    def __init__(self, video_path: str | Path, meta: dict):
        self._path = sidecar_path(video_path)
        self._f = open(self._path, "w", encoding="utf-8")  # noqa: SIM115 - long-lived
        meta = {"type": "meta", "v": 1, **meta}
        self._f.write(json.dumps(meta) + "\n")
        self._n = 0
        # REBASE to recording time: live recordings feed pipeline t (seconds
        # since the SOURCE started), so an evening session's sidecar began at
        # t=1160 while its video starts at 0 — every seek/audit/phone-chip
        # missed (found on Joe's first live-reviewed 9-ball drill). The first
        # frame written defines t=0; shots backdated before it clamp to 0.
        self._t0: float | None = None

    def add_frame(self, t: float, tracks: list,
                  carried_ids: set | None = None,
                  foreign_frac: float = 0.0) -> None:
        if self._t0 is None:
            self._t0 = float(t)
        t = max(0.0, t - self._t0)
        rec = [[int(tr.id), round(float(tr.x), 1), round(float(tr.y), 1),
                round(float(tr.radius), 1), int(tr.number),
                tr.cls.value, bool(tr.active)] for tr in tracks]
        d = {"type": "f", "t": round(t, 3), "tracks": rec}
        # v2 hand-context, omitted when absent so quiet states stay tiny:
        # which balls are hand-adjacent, and how much bed the hand covers.
        # This is what lets the recall audit tell strokes from gathering.
        if carried_ids:
            d["c"] = sorted(int(i) for i in carried_ids)
        if foreign_frac >= 0.005:
            d["ff"] = round(float(foreign_frac), 3)
        self._f.write(json.dumps(d, separators=(",", ":")) + "\n")
        self._n += 1
        if self._n % 50 == 0:
            self._f.flush()

    def add_shot(self, event) -> None:
        t0 = self._t0 or 0.0
        self._f.write(json.dumps({
            "type": "shot", "start": round(max(0.0, float(event.start_t) - t0), 3),
            "end": round(max(0.0, float(event.end_t) - t0), 3),
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


def _shot_for(shots: list, start: float) -> dict | None:
    """The shot a review/derived record at ``start`` belongs to.

    Exact start match first (the phone always anchors verdicts to the
    shot's own start). But segmentation MOVES when the tracker improves
    and the library is re-backfilled — Joe's verdicts must survive that
    (his @214s/@466s rearrange verdicts orphaned when boundaries shifted
    3.1s under the appearance-gated tracker). Fall back to CONTAINMENT
    (the verdict's moment lies inside the re-segmented shot), then to the
    nearest start within 8s; farther than that, stay unattached rather
    than guess."""
    for s in shots:
        if abs(float(s.get("start", -1)) - start) < 0.2:
            return s
    for s in shots:
        if (float(s.get("start", 0)) - 1.0 <= start
                <= float(s.get("end", 0)) + 1.0):
            return s
    best, bd = None, 8.0
    for s in shots:
        d = abs(float(s.get("start", -1)) - start)
        if d < bd:
            best, bd = s, d
    return best


class SidecarReader:
    """Loads a sidecar and answers 'tracks at time t' with interpolation."""

    def __init__(self, video_path: str | Path):
        self.meta: dict = {}
        self.shots: list[dict] = []
        self._times: list[float] = []
        self._frames: list[list] = []
        self._carried: list[list] = []      # v2: hand-adjacent ids per state
        self._foreign: list[float] = []     # v2: bed fraction under hands
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
                    self._carried.append(d.get("c") or [])
                    self._foreign.append(float(d.get("ff", 0.0)))
                elif d.get("type") == "shot":
                    d["_orig_outcome"] = d.get("outcome", "miss")
                    self.shots.append(d)
                elif d.get("type") == "tag_correction":
                    # cut / miss-side verdicts, ranked by source: Joe's
                    # review is final; "forensic" (the corridor re-pass)
                    # outranks the derivation but never Joe. Last wins
                    # within a rank.
                    s2 = _shot_for(self.shots, float(d["start"]))
                    if s2 is not None:
                        slot = ("_tag_review"
                                if d.get("src", "review") == "review"
                                else "_tag_forensic")
                        tr = s2.setdefault(slot, {})
                        for k in ("cut", "miss_side"):
                            if d.get(k):
                                tr[k] = d[k]
                elif d.get("type") == "correction":
                    # last-wins WITHIN a rank, but a human verdict is FINAL:
                    # once a review-source correction lands, derived re-runs
                    # can no longer change the outcome. Legacy corrections
                    # (no src field) are treated as review — the safe rank.
                    s = _shot_for(self.shots, float(d["start"]))
                    if s is not None:
                        csrc = d.get("src", "review")
                        if not (csrc == "derived" and s.get("_reviewed")):
                            s["outcome"] = d.get("outcome", s.get("outcome"))
                            if csrc != "derived":
                                s["corrected"] = True
                                s["_reviewed"] = True
                elif d.get("type") == "reviewed":
                    # Joe confirmed the machine got this one RIGHT: lock
                    # outcome and action at review rank (derived re-runs
                    # stand down) without changing either value.
                    s = _shot_for(self.shots, float(d["start"]))
                    if s is not None:
                        s["_reviewed"] = True
                        s["_action_reviewed"] = True
                        s["reviewed_ok"] = True
                elif d.get("type") == "correction_clear":
                    # Joe removed his verdict: restore the shot-line
                    # original and drop every review flag; derived records
                    # appended AFTER this line re-apply normally (the
                    # watcher re-runs derivation right after a clear).
                    s = _shot_for(self.shots, float(d["start"]))
                    if s is not None:
                        s["outcome"] = s.get("_orig_outcome", "miss")
                        for k in ("corrected", "_reviewed", "action",
                                  "action_corrected", "_action_reviewed",
                                  "note", "reviewed_ok"):
                            s.pop(k, None)
                elif d.get("type") == "split":
                    # HUMAN verdict (Joe: "some shot clips actually
                    # include two shots"): bisect the clip at `at`; each
                    # half re-derives independently (the watcher re-runs
                    # derivation right after appending this record).
                    s = _shot_for(self.shots, float(d["start"]))
                    at = float(d.get("at", -1))
                    if (s is not None
                            and float(s.get("start", 0)) < at
                            < float(s.get("end", 0))):
                        idx = self.shots.index(s)
                        second = {"type": "shot", "start": round(at, 3),
                                  "end": s.get("end"),
                                  "outcome": "miss", "pocketed": 0,
                                  "_orig_outcome": "miss"}
                        s["end"] = round(at, 3)
                        self.shots.insert(idx + 1, second)
                elif d.get("type") == "note":
                    # plain-text review note (phone correction Details)
                    s = _shot_for(self.shots, float(d["start"]))
                    if s is not None:
                        s["note"] = str(d.get("text", ""))[:500]
                elif d.get("type") == "action":
                    # same ranking as outcome corrections: a review-source
                    # action is FINAL; derived re-labels stand down. Legacy
                    # records (no src) are derived — they all came from the
                    # classifier before the field existed.
                    s = _shot_for(self.shots, float(d["start"]))
                    if s is not None:
                        asrc = d.get("src", "derived")
                        if not (asrc == "derived"
                                and s.get("_action_reviewed")):
                            s["action"] = d.get("action", "stroke")
                            if asrc != "derived":
                                s["_action_reviewed"] = True
                                s["action_corrected"] = True
        # LEGACY live sidecars recorded source-uptime, not recording time.
        # A recording's first state lands within ~2s of zero when times are
        # right; a first state minutes in means the offset bug — normalize
        # everything by it so old sessions review correctly too.
        if self._times and self._times[0] > 30.0:
            t0 = self._times[0]
            self._times = [t - t0 for t in self._times]
            for s in self.shots:
                s["start"] = max(0.0, float(s.get("start", 0)) - t0)
                s["end"] = max(0.0, float(s.get("end", 0)) - t0)
            log.info("sidecar times normalized by legacy offset %.1fs", t0)
        log.info("analysis sidecar loaded: %s (%d states, %d shots)",
                 p.name, len(self._times), len(self.shots))

    @staticmethod
    def exists(video_path: str | Path) -> bool:
        return sidecar_path(video_path).is_file()

    def hand_context(self, t0: float, t1: float) -> tuple[set[int], float]:
        """(union of hand-adjacent track ids, peak foreign fraction) over the
        states in [t0, t1] — empty/0.0 on v1 sidecars, which never recorded
        hand context (the caller should treat that as 'unknown', not 'no')."""
        lo = bisect_right(self._times, t0 - 0.15)
        hi = bisect_right(self._times, t1 + 0.15)
        ids: set[int] = set()
        peak = 0.0
        for i in range(lo, min(hi, len(self._times))):
            ids.update(self._carried[i])
            peak = max(peak, self._foreign[i])
        return ids, peak

    @property
    def has_hand_context(self) -> bool:
        return any(self._carried) or any(f > 0 for f in self._foreign)

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
        # The sidecar stores no bgr (3 floats/ball/state saved); playback
        # rendering uses the measured mean colour, so synthesize it from the
        # NUMBER — without this every cached ball wore the default grey and
        # the whole playback schematic looked white (Joe's report).
        num = int(r[4])
        if num == 0:
            bgr = (250, 250, 250)
        elif num > 0:
            from ..core.balls import pool_ball_bgr
            bgr = pool_ball_bgr(num)
        else:
            bgr = (200, 200, 200)
        return Track(id=int(r[0]), x=float(r[1]), y=float(r[2]),
                     radius=float(r[3]), number=num,
                     cls=BallClass(r[5]), active=bool(r[6]), bgr=bgr)

    def _to_tracks(self, rows) -> list[Track]:
        return [self._to_track(r) for r in rows]


def carry_review_verdicts(prev_sidecar: str | Path,
                          new_sidecar: str | Path) -> int:
    """Append the HUMAN records from an old sidecar onto a rebuilt one.

    A --force re-backfill rewrites the sidecar from scratch — machine data
    is recomputable, but Joe's phone verdicts are not: without this, a
    library re-analysis would silently discard every correction, note and
    confirm he ever filed (they'd survive only in the .prev backup).
    Carried in original order (append order IS rank resolution). Only
    explicitly review-sourced corrections/actions move: legacy UNTAGGED
    corrections are the old derived pass wearing review rank for reader
    safety — carrying them would freeze stale machine outcomes over the
    new derivation. Notes, confirms and clears are always human. Shifted
    segmentation is fine: the reader re-attaches by containment."""
    prev_p, new_p = Path(prev_sidecar), Path(new_sidecar)
    if not prev_p.is_file() or not new_p.is_file():
        return 0
    kept = 0
    with open(new_p, "a", encoding="utf-8") as out:
        for line in prev_p.read_text(encoding="utf-8",
                                     errors="replace").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            t = d.get("type")
            keep = (t in ("note", "reviewed", "correction_clear", "split")
                    or (t == "correction" and d.get("src") == "review")
                    or (t == "action" and d.get("src") == "review"))
            if keep:
                out.write(json.dumps(d, separators=(",", ":")) + "\n")
                kept += 1
    return kept


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
