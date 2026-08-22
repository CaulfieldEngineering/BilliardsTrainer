"""Which sessions are MEASURABLE? Calibration gates over the archive.

Root-cause analytics report inches and degrees; a session whose frame
is wrong yields confident nonsense. This runs the tablespace gates over
every sidecar and says, per session, whether its inch figures can be
trusted — and whether its stored overlay transform maps the bed to the
right part of the video (the recovered-file failure).

    python tools/audit_calibration.py [--json]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billiards_trainer.config import Settings  # noqa: E402
from billiards_trainer.vision.analysis_cache import (  # noqa: E402
    SidecarReader,
    sidecar_path,
)
from billiards_trainer.vision.tablespace import (  # noqa: E402
    audit,
    audit_summary_transform,
    space_for_video,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    st = Settings.load()
    d = Path(st.recording.resolved_dir())
    rows = []
    for v in sorted(d.glob("session-*.mp4")):
        if not sidecar_path(v).is_file():
            continue
        try:
            reader = SidecarReader(v)
        except (OSError, ValueError):
            continue
        ts0 = space_for_video(v, reader, st)
        res = audit(reader, configured_size=st.table.size, space=ts0)
        row = {"session": v.name, "ok": res["ok"],
               "failed": [g[0] for g in res["gates"] if not g[1]],
               "detail": {g[0]: g[2] for g in res["gates"]}}
        ts = res["space"]
        if ts is not None:
            row["size"] = ts.size
            row["px_per_in"] = round(ts.px_per_in, 3)
            row["bed_in"] = [round(ts.bed_short_in, 1), round(ts.bed_long_in, 1)]
            sj = Path(str(v) + ".shots.json")
            if sj.is_file():
                try:
                    tf = json.loads(sj.read_text(encoding="utf-8")).get("transform")
                    if tf:
                        tok, tdet = audit_summary_transform(tf, ts)
                        row["transform_ok"] = tok
                        row["transform"] = tdet
                        if not tok:
                            row["ok"] = False
                            row["failed"].append("summary_transform")
                except (OSError, ValueError):
                    pass
        rows.append(row)
    good = [r for r in rows if r["ok"]]
    out = {"sessions": len(rows), "measurable": len(good), "rows": rows}
    (ROOT / "_eval" / "calibration_audit.json").write_text(
        json.dumps(out, indent=1))
    if args.json:
        print(json.dumps(out, indent=1))
        return 0
    print(f"CALIBRATION AUDIT: {len(good)}/{len(rows)} sessions measurable")
    for r in rows:
        mark = "OK " if r["ok"] else "BAD"
        extra = ("" if r["ok"]
                 else "  failed=" + ",".join(sorted(set(r["failed"]))))
        print(f"  {mark} {r['session']:44s} {r.get('size','?'):4s} "
              f"{str(r.get('bed_in','?')):14s}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
