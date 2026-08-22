"""Per-shot DOSSIER export: the data contract for the future VLM coach.

Joe's north star, verbatim: "EVENTUALLY, we'll use these individual shots
to create LLM or VLM analysis. IE: 'On this shot, you missed to the left.
Your head popped up, and your cue swerved to the right.'" This tool builds
the substrate: for each detected shot, one machine-readable JSON (shot
facts + per-ball trajectories + cue-ball stroke metrics) and one rendered
trajectory diagram. A coach — human, LLM, or VLM — reads these and talks
about the shot without touching video.

Everything comes from the analysis sidecar; no inference re-runs.

    python tools/export_shot_dossier.py --video <session.mp4> --shot 3
    python tools/export_shot_dossier.py --video <session.mp4> --all
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billiards_trainer.vision.analysis_cache import SidecarReader  # noqa: E402

#: Trajectory sampling step. The sidecar records ~10 Hz; sampling finer only
#: interpolates between the same states.
_DT = 0.1

#: A ball participates in the shot if it moved at least this many multiples
#: of its own radius during the window.
_PARTICIPANT_TRAVEL_R = 1.5


def _paths(reader: SidecarReader, t0: float, t1: float) -> dict[int, dict]:
    """Per-ball sampled paths over [t0, t1]: id -> {number, cls, points}."""
    out: dict[int, dict] = {}
    t = t0
    while t <= t1 + 1e-9:
        for tr in reader.tracks_at(t):
            e = out.setdefault(tr.id, {"number": tr.number, "cls": tr.cls.value,
                                       "radius": tr.radius, "points": []})
            # number can firm up mid-shot; keep the latest non-unknown
            if tr.number >= 0:
                e["number"] = tr.number
            e["points"].append([round(t - t0, 2), round(tr.x, 1), round(tr.y, 1)])
        t += _DT
    return out


def _travel(points: list) -> float:
    return sum(math.hypot(points[i + 1][1] - points[i][1],
                          points[i + 1][2] - points[i][2])
               for i in range(len(points) - 1))


def _stroke_metrics(points: list) -> dict:
    """Cue-ball stroke numbers a coach actually uses: peak speed, distance,
    direction changes (rail/ball contacts show as bends >25 deg)."""
    if len(points) < 3:
        return {}
    speeds = []
    bends = []
    for i in range(len(points) - 1):
        (t0, x0, y0), (t1, x1, y1) = points[i], points[i + 1]
        d = math.hypot(x1 - x0, y1 - y0)
        if t1 > t0:
            speeds.append(d / (t1 - t0))
    for i in range(1, len(points) - 1):
        ax = points[i][1] - points[i - 1][1]
        ay = points[i][2] - points[i - 1][2]
        bx = points[i + 1][1] - points[i][1]
        by = points[i + 1][2] - points[i][2]
        na, nb = math.hypot(ax, ay), math.hypot(bx, by)
        if na < 2 or nb < 2:
            continue
        cosang = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
        ang = math.degrees(math.acos(cosang))
        if ang > 25:
            bends.append({"t": points[i][0], "angle_deg": round(ang, 1),
                          "at": [points[i][1], points[i][2]]})
    return {
        "peak_speed_px_s": round(max(speeds), 1) if speeds else 0.0,
        "mean_speed_px_s": round(sum(speeds) / len(speeds), 1) if speeds else 0.0,
        "direction_changes": bends[:12],
    }


_BALL_COLOURS = {0: (255, 255, 255), 1: (0, 215, 255), 2: (255, 100, 30),
                 3: (40, 40, 230), 4: (140, 30, 90), 5: (0, 140, 255),
                 6: (60, 150, 30), 7: (30, 30, 140), 8: (30, 30, 30),
                 9: (0, 215, 255), 10: (255, 100, 30), 11: (40, 40, 230),
                 12: (140, 30, 90), 13: (0, 140, 255), 14: (60, 150, 30),
                 15: (30, 30, 140)}


def _diagram(paths: dict[int, dict], size: tuple[int, int],
             out_png: Path) -> None:
    """Schematic table with each participant's path: dot = start, ring =
    end, colour = ball. Scaled to the sidecar's coordinate extents."""
    w, h = size
    img = np.full((h, w, 3), (90, 130, 20), np.uint8)   # felt green-ish
    cv2.rectangle(img, (2, 2), (w - 3, h - 3), (40, 60, 100), 4)
    for e in paths.values():
        pts = e["points"]
        if len(pts) < 2:
            continue
        colour = _BALL_COLOURS.get(e["number"], (200, 200, 200))
        arr = np.int32([[int(p[1] * w / max(1, e.get("_sw", w))),
                         int(p[2] * h / max(1, e.get("_sh", h)))] for p in pts])
        cv2.polylines(img, [arr.reshape(-1, 1, 2)], False, colour, 2, cv2.LINE_AA)
        cv2.circle(img, tuple(arr[0]), 6, colour, -1, cv2.LINE_AA)
        cv2.circle(img, tuple(arr[-1]), 8, colour, 2, cv2.LINE_AA)
        label = str(e["number"]) if e["number"] >= 0 else "?"
        cv2.putText(img, label, (arr[-1][0] + 8, arr[-1][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_png), img)


def _describe(reader: SidecarReader, shot: dict) -> dict:
    """Structured description + one factual line (never fatal)."""
    try:
        from billiards_trainer.vision.describe import compose_text, describe_shot
        d = describe_shot(reader, shot)
        d["text"] = compose_text(d)
        return d
    except Exception:  # noqa: BLE001 - description is enrichment
        return {}


def export_shot(video: Path, reader: SidecarReader, shot_no: int,
                shot: dict, out_root: Path) -> Path:
    t0 = float(shot["start"])
    t1 = float(shot["end"])
    paths = _paths(reader, max(0.0, t0 - 1.0), t1)
    # coordinate extents for the diagram scale
    xs = [p[1] for e in paths.values() for p in e["points"]]
    ys = [p[2] for e in paths.values() for p in e["points"]]
    sw = max(xs) * 1.05 if xs else 675
    sh = max(ys) * 1.05 if ys else 1271
    participants = {}
    cue_metrics = {}
    for tid, e in paths.items():
        trav = _travel(e["points"])
        if trav < _PARTICIPANT_TRAVEL_R * max(6.0, e.get("radius", 10.0)):
            continue
        e["travel_px"] = round(trav, 1)
        e["_sw"], e["_sh"] = sw, sh
        participants[tid] = e
        if e["number"] == 0:
            cue_metrics = _stroke_metrics(e["points"])
    out_dir = out_root / f"{video.stem}_shot{shot_no:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _diagram(participants, (400, int(400 * sh / max(1.0, sw))),
             out_dir / "trajectory.png")
    doc = {
        "session": video.name,
        "shot_no": shot_no,
        "start_s": t0, "end_s": t1,
        "duration_s": round(t1 - t0, 2),
        "outcome": shot.get("outcome"),
        "corrected_by_user": bool(shot.get("corrected")),
        "pocketed": shot.get("pocketed", 0),
        "participants": [
            {"track_id": tid, "number": e["number"], "cls": e["cls"],
             "travel_px": e["travel_px"], "points_txy": e["points"]}
            for tid, e in sorted(participants.items())],
        "cue_stroke": cue_metrics,
        # Hierarchy #3: structured facts + one factual line — the coach's
        # raw material ("on this shot you..." starts from exactly this).
        "description": _describe(reader, shot),
        "media": {"video": str(video),
                  "clip_hint": f"ffmpeg -ss {max(0, t0 - 5):.2f} -to {t1 + 1:.2f}"
                               f" -i \"{video}\" -c copy shot{shot_no:02d}.mp4"},
    }
    (out_dir / "shot.json").write_text(json.dumps(doc, indent=1))
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--shot", type=int, default=0, help="1-based shot number")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "_eval" / "dossiers"))
    args = ap.parse_args()

    video = Path(args.video)
    reader = SidecarReader(video)
    if not reader.shots:
        print("no shots in sidecar", file=sys.stderr)
        return 1
    todo = range(1, len(reader.shots) + 1) if args.all else [args.shot]
    for no in todo:
        if not 1 <= no <= len(reader.shots):
            print(f"shot {no} out of range 1..{len(reader.shots)}", file=sys.stderr)
            return 2
        d = export_shot(video, reader, no, reader.shots[no - 1], Path(args.out))
        print(f"shot {no}: {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
