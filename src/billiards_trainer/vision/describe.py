"""Describe the make/miss — Joe's hierarchy #3, the coach's raw material.

From a sidecar shot window, produce a STRUCTURED description (who moved,
how hard, where things went) plus a one-line natural rendering. Facts
only, from the identity record — interpretation ("you cut it too thin")
is the future coach's job, built on exactly these fields.

Pocket names come from the position envelope (the same bed-corner logic
frame verification validated for outcomes): corners plus long-rail
midpoints, named from the overhead view.
"""

import logging

from .analysis_cache import SidecarReader
from .outcomes import departed_for_shot, stable_numbers

log = logging.getLogger("vision.describe")

_BALL_NAMES = {0: "cue ball", 1: "the 1", 2: "the 2", 3: "the 3",
               4: "the 4", 5: "the 5", 6: "the 6", 7: "the 7",
               8: "the 8", 9: "the 9", 10: "the 10", 11: "the 11",
               12: "the 12", 13: "the 13", 14: "the 14", 15: "the 15"}


def _bed_envelope(reader: SidecarReader) -> tuple:
    xs, ys = [], []
    for fr in reader._frames:
        for row in fr:
            if row[6]:
                xs.append(row[1])
                ys.append(row[2])
    if len(xs) < 50:
        return None
    xs.sort(), ys.sort()
    n = len(xs)
    return (xs[int(0.005 * n)], ys[int(0.005 * n)],
            xs[min(n - 1, int(0.995 * n))], ys[min(n - 1, int(0.995 * n))])


def _pocket_name(x: float, y: float, env: tuple) -> str:
    """Nearest pocket, named from the overhead view (portrait table:
    y grows downward, long axis vertical)."""
    x0, y0, x1, y1 = env
    pockets = {
        "top-left": (x0, y0), "top-right": (x1, y0),
        "bottom-left": (x0, y1), "bottom-right": (x1, y1),
        "left-middle": (x0, (y0 + y1) / 2), "right-middle": (x1, (y0 + y1) / 2),
    }
    return min(pockets, key=lambda k: (pockets[k][0] - x) ** 2
                                      + (pockets[k][1] - y) ** 2)


def _paths(reader: SidecarReader, t0: float, t1: float) -> dict:
    """id -> {"num", "path": [(t,x,y)...], "launch": t|None, "dist": px}."""
    out: dict = {}
    t = t0
    while t <= t1 + 1e-9:
        for tr in reader.tracks_at(t):
            if not tr.active:
                continue
            e = out.setdefault(tr.id, {"num": tr.number, "path": [],
                                       "launch": None, "dist": 0.0})
            if tr.number >= 0:
                e["num"] = tr.number
            if e["path"]:
                _, px, py = e["path"][-1]
                step = ((tr.x - px) ** 2 + (tr.y - py) ** 2) ** 0.5
                e["dist"] += step
                if e["launch"] is None and step / 0.15 > 60.0:
                    e["launch"] = t
                    e["pre_launch"] = len(e["path"])
            e["path"].append((t, tr.x, tr.y))
        t += 0.15
    return out


def describe_shot(reader: SidecarReader, s: dict) -> dict:
    """Structured facts for one shot. Never raises; degrades to fewer
    fields when the record is thin."""
    t0, t1 = float(s["start"]), float(s["end"])
    env = _bed_envelope(reader)
    paths = _paths(reader, t0, t1)
    movers = {i: e for i, e in paths.items() if e["dist"] > 30.0}
    short = min(env[2] - env[0], env[3] - env[1]) if env else 600.0

    cue = next((e for e in movers.values() if e["num"] == 0), None)
    # Contact attribution needs REAL travel with a real launch — a basket
    # blip or jitter track (30-60px of noise, no launch) once read as
    # "contact on the 4" while the 4 sat in a pocket.
    def _near_cue_at(e) -> bool:
        # A REAL contact launches next to the cue ball. A basket blip or a
        # teleporting stale track launches wherever it likes — nowhere near
        # the cue — and once claimed "contact on the 4" while the 4 sat in
        # a pocket. No cue path = no contact claims (facts only).
        if not cue or not cue["path"] or e["launch"] is None:
            return False
        lx, ly = next(((x, y) for t, x, y in e["path"]
                       if t >= e["launch"]), e["path"][-1][1:])
        cx, cy = min(cue["path"], key=lambda p: abs(p[0] - e["launch"]))[1:]
        return ((lx - cx) ** 2 + (ly - cy) ** 2) ** 0.5 < 70.0

    # Only balls that STABLY existed before the strike can be named as
    # the contact: a motion-blurred 7 briefly misread "4" mid-flight
    # produced "contact on the 4" while the 4 sat in a pocket.
    before = stable_numbers(reader, max(0.0, t0 - 1.5), t0 - 0.2)
    objs = sorted((e for e in movers.values()
                   if e["num"] != 0 and e["num"] in before
                   and e["launch"] is not None
                   and e["dist"] > 60.0
                   and e.get("pre_launch", 0) >= 3
                   and _near_cue_at(e)),
                  key=lambda e: e["launch"])
    first_obj = objs[0] if objs else None

    # pace from the cue's peak sampled step speed, normalized to table size
    pace = None
    if cue and len(cue["path"]) >= 2:
        peak = 0.0
        for k in range(1, len(cue["path"])):
            (ta, xa, ya), (tb, xb, yb) = cue["path"][k - 1], cue["path"][k]
            v = ((xb - xa) ** 2 + (yb - ya) ** 2) ** 0.5 / max(1e-3, tb - ta)
            peak = max(peak, v)
        rel = peak / short          # table-widths per second
        pace = ("soft" if rel < 0.6 else
                "medium" if rel < 1.3 else
                "firm" if rel < 2.2 else "power")

    gone, _detail = departed_for_shot(reader, s)
    # The RANKED outcome is truth (human verdicts included): if it says
    # this was no scratch, a phantom cue departure in old tracking data
    # must not put "scratched" in the description.
    if s.get("outcome") != "scratch":
        gone = {n for n in gone if n != 0}
    potted = []
    for num in sorted(gone):
        e = next((e for e in paths.values() if e["num"] == num), None)
        entry = {"ball": num}
        if e and e["path"] and env:
            _, lx, ly = e["path"][-1]
            entry["pocket"] = _pocket_name(lx, ly, env)
        potted.append(entry)

    desc = {
        "outcome": s.get("outcome", "?"),
        "action": s.get("action", "stroke"),
        "pace": pace,
        "n_movers": len(movers),
        "cue_travel_tw": round(cue["dist"] / short, 2) if cue else None,
        "first_object": first_obj["num"] if first_obj is not None else None,
        "object_travel_tw": (round(first_obj["dist"] / short, 2)
                             if first_obj else None),
        "potted": potted,
        "duration_s": round(t1 - t0, 1),
    }
    return desc


def compose_text(d: dict) -> str:
    """One factual line from a description dict."""
    if d.get("action") == "ball_in_hand":
        return "Ball in hand — balls repositioned."
    if d.get("action") == "nothing":
        return "No table action."
    bits = []
    pace = d.get("pace")
    opener = "Break" if d.get("action") == "break" else "Stroke"
    bits.append(f"{opener} at {pace} pace" if pace else opener)
    fo = d.get("first_object")
    if fo is not None and fo >= 0 and fo != 0:
        bits.append(f"contact on {_BALL_NAMES.get(fo, f'ball {fo}')}")
    pots = d.get("potted") or []
    scratch = any(p["ball"] == 0 for p in pots)
    named = [p for p in pots if p["ball"] != 0]
    if named:
        parts = [f"{_BALL_NAMES.get(p['ball'], str(p['ball']))}"
                 + (f" into the {p['pocket']} pocket" if p.get("pocket") else "")
                 for p in named]
        bits.append("potted " + " and ".join(parts))
    elif d.get("outcome") == "miss":
        bits.append("no ball fell")
    if scratch:
        bits.append("cue ball scratched")
    ct = d.get("cue_travel_tw")
    if ct is not None and ct > 0.1:
        bits.append(f"cue travelled {ct:.1f} table-widths")
    return "; ".join(bits).capitalize() + "."
