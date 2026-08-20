"""The review scoreboard: how right is the system, per Joe's own verdicts?

Every phone verdict is ground truth with Joe's name on it: a CONFIRM
(reviewed marker) says the machine's outcome AND action were right; a
CORRECTION says which channel was wrong (the pre-correction value is
recovered from the sidecar log order). This tool aggregates the ledger
across the library into accuracy-by-channel — the number every
classifier/derivation change should move.

    python tools/review_scoreboard.py
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billiards_trainer.config import Settings  # noqa: E402
from billiards_trainer.vision.analysis_cache import sidecar_path  # noqa: E402


def ledger_for(video: Path) -> list[dict]:
    """Per-shot human-review ledger from the raw sidecar log: for every
    shot start, the machine's LAST pre-review value per channel and the
    review events that followed (confirms, corrections, notes)."""
    sc = sidecar_path(video)
    if not sc.is_file():
        return []
    shots: dict[float, dict] = {}

    def entry(start: float) -> dict | None:
        for k in shots:
            if abs(k - start) < 0.2:
                return shots[k]
        return None

    for line in sc.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        t = d.get("type")
        if t == "shot":
            shots[float(d["start"])] = {
                "start": float(d["start"]),
                "machine_outcome": d.get("outcome", "miss"),
                "machine_action": "stroke",
                "events": [],
            }
        elif t in ("correction", "action", "note", "reviewed",
                   "correction_clear"):
            e = entry(float(d.get("start", -1)))
            if e is None:
                continue
            # GROUND TRUTH = explicit src=="review" ONLY. Untagged legacy
            # corrections predate the src field and are overwhelmingly the
            # derived library pass — counting them as human verdicts made
            # the first scoreboard claim 213 reviews when Joe had filed 8.
            src = d.get("src")
            if t == "correction" and src != "review":
                if not any(ev[0] == "outcome" for ev in e["events"]):
                    e["machine_outcome"] = d.get("outcome", e["machine_outcome"])
            elif t == "action" and src != "review":
                if not any(ev[0] == "action" for ev in e["events"]):
                    e["machine_action"] = d.get("action", e["machine_action"])
            elif t == "correction" and src == "review":
                e["events"].append(("outcome", d.get("outcome")))
            elif t == "action" and src == "review":
                e["events"].append(("action", d.get("action")))
            elif t == "note":
                e["events"].append(("note", d.get("text", "")))
            elif t == "reviewed":
                e["events"].append(("confirm", None))
            elif t == "correction_clear":
                e["events"] = [("cleared", None)]
    return [e for e in shots.values() if e["events"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    d = Path(Settings.load().recording.resolved_dir())
    outcome_ok = outcome_bad = action_ok = action_bad = 0
    confirms = notes = 0
    rows = []
    for v in sorted(d.glob("session-*.mp4")):
        for e in ledger_for(v):
            if e["events"] and e["events"][-1][0] == "cleared":
                continue
            kinds = {k for k, _ in e["events"]}
            row = {"session": v.name, "start": e["start"]}
            if "confirm" in kinds:
                confirms += 1
                outcome_ok += 1
                action_ok += 1
                row["verdict"] = "confirmed"
            fixed_out = next((val for k, val in e["events"]
                              if k == "outcome"), None)
            fixed_act = next((val for k, val in e["events"]
                              if k == "action"), None)
            if fixed_out is not None:
                if fixed_out != e["machine_outcome"]:
                    outcome_bad += 1
                    row["outcome"] = f"{e['machine_outcome']} -> {fixed_out}"
                else:
                    outcome_ok += 1
            if fixed_act is not None:
                if fixed_act != e["machine_action"]:
                    action_bad += 1
                    row["action"] = f"{e['machine_action']} -> {fixed_act}"
                else:
                    action_ok += 1
            note = next((val for k, val in e["events"] if k == "note"), None)
            if note:
                notes += 1
                row["note"] = note
            rows.append(row)
    out_n = outcome_ok + outcome_bad
    act_n = action_ok + action_bad
    board = {
        "reviewed_shots": len(rows),
        "confirms": confirms,
        "notes": notes,
        "outcome_accuracy": (f"{outcome_ok}/{out_n}"
                             + (f" ({100 * outcome_ok / out_n:.0f}%)"
                                if out_n else "")),
        "action_accuracy": (f"{action_ok}/{act_n}"
                            + (f" ({100 * action_ok / act_n:.0f}%)"
                               if act_n else "")),
    }
    (ROOT / "_eval" / "review_scoreboard.json").write_text(
        json.dumps({"board": board, "rows": rows}, indent=1))
    if args.json:
        print(json.dumps(board, indent=1))
        return 0
    print("REVIEW SCOREBOARD (human verdicts = ground truth)")
    for k, v_ in board.items():
        print(f"  {k}: {v_}")
    if confirms == 0 and rows:
        print("  NOTE: zero confirms yet — Joe has only filed corrections,"
              " so 'accuracy among reviewed' is selection-biased low."
              " It becomes meaningful as ✓ confirms accumulate.")
    print("\nledger:")
    for r in rows:
        bits = [f"{r['session']} @{r['start']:.0f}s"]
        for k in ("verdict", "outcome", "action", "note"):
            if r.get(k):
                bits.append(f"{k}={r[k]!r}")
        print("  " + "  ".join(bits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
