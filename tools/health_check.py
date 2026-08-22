"""Product health board: every feature area, verified, before anything new.

Joe's rule for the autonomy loop: "ensure we don't add features when others
are still broken or lacking." This board is that rule made executable. Each
area gets an automated check with a verdict:

  GREEN  healthy — feature work is allowed
  AMBER  degraded/lacking — fix before new features (unless blocked on Joe)
  RED    broken — drop everything, this IS the work session

The work loop runs --quick at the start of every session and obeys the
worst verdict. --full adds the test suite. Board persists to
_eval/health.json for trend-keeping.

    python tools/health_check.py --quick
    python tools/health_check.py --full
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from billiards_trainer.config import EXPORTS_DIR, LOGS_DIR, MODELS_DIR, Settings  # noqa: E402

GREEN, AMBER, RED = "GREEN", "AMBER", "RED"


def _check_app_running() -> tuple[str, str]:
    try:
        r = subprocess.run(
            ["powershell", "-Command",
             "(Get-Process pythonw -ErrorAction SilentlyContinue | "
             "Where-Object {$_.MainWindowTitle}).Count"],
            capture_output=True, text=True, timeout=30)
        n = int((r.stdout or "0").strip() or 0)
        if n >= 1:
            return GREEN, "app window present"
        return AMBER, "app not running (fine if Joe closed it)"
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return AMBER, "could not query processes"


def _check_app_log() -> tuple[str, str]:
    log = LOGS_DIR / "billiards_trainer.log"
    if not log.is_file():
        return AMBER, "no app log"
    try:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-400:]
    except OSError:
        return AMBER, "log unreadable"
    errs = [ln for ln in tail if "ERROR" in ln or "Traceback" in ln]
    if len(errs) >= 5:
        return RED, f"{len(errs)} ERROR lines in recent log: {errs[-1][:90]}"
    if errs:
        return AMBER, f"{len(errs)} ERROR line(s): {errs[-1][:90]}"
    return GREEN, "log clean (last 400 lines)"


def _check_recordings() -> tuple[str, str]:
    d = Path(Settings.load().recording.resolved_dir())
    vids = sorted(d.glob("session-*.mp4"), key=lambda p: p.stat().st_mtime)
    if not vids:
        return AMBER, "no recordings found"
    newest = vids[-1]
    from billiards_trainer.vision.analysis_cache import SidecarReader
    if not SidecarReader.exists(newest):
        age_h = (time.time() - newest.stat().st_mtime) / 3600
        if age_h > 1.0:
            return AMBER, f"newest recording lacks a sidecar: {newest.name}"
    return GREEN, f"{len(vids)} recordings; newest has sidecar"


def _check_orphans() -> tuple[str, str]:
    parts = [p for p in EXPORTS_DIR.glob(".session-*.part.mp4")
             if time.time() - p.stat().st_mtime > 3600]
    if parts:
        return RED, f"{len(parts)} orphaned .part recording(s) — a recording died"
    return GREEN, "no orphaned recordings"


def _check_playback_speed() -> tuple[str, str]:
    """Cached playback must beat 30fps — Joe's smooth-by-construction bar."""
    import cv2

    from billiards_trainer.vision.analysis_cache import SidecarReader
    from billiards_trainer.vision.pipeline import Pipeline
    d = Path(Settings.load().recording.resolved_dir())
    vids = [v for v in sorted(d.glob("session-*.mp4"),
                              key=lambda p: p.stat().st_mtime, reverse=True)
            if SidecarReader.exists(v)
            and v.stat().st_size > 20e6]   # skip record-button test stubs
    if not vids:
        return AMBER, "no sidecar'd session to benchmark"
    v = vids[0]
    cap = cv2.VideoCapture(str(v))
    if not cap.isOpened():
        return AMBER, f"could not open {v.name}"
    p = Pipeline(Settings.load())
    n = 0
    while n < 30:                       # warm up calibration
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        p.process(fr, n / 30.0)
    p.playback_cache = SidecarReader(v)
    n, t0 = 0, time.perf_counter()
    while n < 120:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        p.process(fr, 1.0 + n / 30.0, annotate=True)
    cap.release()
    if n < 60:
        return AMBER, f"{v.name} too short to benchmark"
    fps = n / (time.perf_counter() - t0)
    if fps < 30:
        return RED, f"cached playback {fps:.0f}fps (<30) on {v.name}"
    return GREEN, f"cached playback {fps:.0f}fps on {v.name}"


def _check_identity() -> tuple[str, str]:
    """No duplicate identities in the newest sidecars (G1 tripwire)."""
    sys.path.insert(0, str(ROOT / "tools"))
    import audit_render
    d = Path(Settings.load().recording.resolved_dir())
    vids = sorted(d.glob("session-*.mp4"), key=lambda p: p.stat().st_mtime,
                  reverse=True)[:3]
    total_d = total_f = 0
    for v in vids:
        r = audit_render.audit(v)
        if r:
            total_d += r["dupe_frames"]
            total_f += r["frames"]
    if not total_f:
        return AMBER, "no sidecars to audit"
    if total_d:
        return RED, f"{total_d}/{total_f} states with duplicate identities"
    return GREEN, f"0/{total_f} duplicate-identity states (3 newest sessions)"


def _check_champion() -> tuple[str, str]:
    fp = ROOT / "_eval" / "champion.json"
    if not fp.is_file():
        return AMBER, "no champion fingerprint file"
    meta = json.loads(fp.read_text())
    champ = MODELS_DIR / meta["file"]
    if not champ.is_file():
        return RED, "champion model file MISSING"
    if champ.stat().st_size != meta["size"]:
        return RED, (f"champion size {champ.stat().st_size} != recorded "
                     f"{meta['size']} — unexpected model swap")
    bak = MODELS_DIR / (meta["file"] + ".bak")
    if bak.exists():
        return AMBER, "gate .bak present — a gate is running or died"
    return GREEN, f"champion {meta.get('name', '?')} verified ({meta['size']} bytes)"


def _check_autonomy() -> tuple[str, str]:
    hb = ROOT / "_eval" / "heartbeat.txt"
    lock = ROOT / "_eval" / "loop.lock"
    if lock.exists() and time.time() - lock.stat().st_mtime > 3 * 3600:
        return RED, "stale loop.lock (>3h) — a work session died"
    if hb.exists():
        age = (time.time() - hb.stat().st_mtime) / 3600
        if age > 5 and not lock.exists():
            return AMBER, f"heartbeat {age:.1f}h old — loop may be stalled"
        return GREEN, f"heartbeat {age:.1f}h"
    return AMBER, "no heartbeat yet"


def _check_cue_sensor() -> tuple[str, str]:
    s = Settings.load()
    if not s.cue.enabled:
        return GREEN, "cue sensor off by choice (opt-in)"
    return AMBER, "cue sensor enabled but connection unverified (needs Joe + sensor)"


def _check_disk() -> tuple[str, str]:
    import shutil as sh
    d = Path(Settings.load().recording.resolved_dir())
    try:
        free_gb = sh.disk_usage(d).free / 1e9
    except OSError:
        return AMBER, "recording dir unreachable"
    if free_gb < 5:
        return RED, f"{free_gb:.1f} GB free on recording drive"
    if free_gb < 20:
        return AMBER, f"{free_gb:.1f} GB free on recording drive"
    return GREEN, f"{free_gb:.0f} GB free"


def _check_tests() -> tuple[str, str]:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x"],
                       capture_output=True, text=True, timeout=900, cwd=ROOT)
    last = (r.stdout or "").strip().splitlines()[-1] if r.stdout else ""
    if r.returncode == 0:
        return GREEN, f"suite green ({last})"
    return RED, f"TEST FAILURES: {last[:120]}"


def _check_corrections() -> tuple[str, str]:
    """Phone review verdicts (Joe: "the watchdog should review for new
    corrections each time"). Reports fresh verdicts since the last pass
    and — the backstop — applies any STRAGGLERS itself if the companion
    watcher isn't eating the queue (which doubles as the companion
    liveness signal)."""
    import time
    d = Path(Settings.load().recording.resolved_dir())
    box = d / "corrections"
    if not box.is_dir():
        return GREEN, "no corrections yet"
    # stragglers older than 2 min mean the companion watcher is dead;
    # apply them here so Joe's verdicts never rot in the queue
    stale = [f for f in box.glob("*.json")
             if time.time() - f.stat().st_mtime > 120]
    applied = 0
    if stale:
        import sys as _s
        _s.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from billiards_trainer.companion.corrections_watcher import scan_once
        applied = scan_once(d)
    # fresh-verdict count since the last watchdog pass
    stamp = Path(__file__).resolve().parents[1] / "_eval" / "corrections_seen.txt"
    last = 0.0
    try:
        last = float(stamp.read_text().strip())
    except (OSError, ValueError):
        pass
    done = d / "corrections" / "done"
    fresh = [f for f in done.glob("*.json")
             if f.stat().st_mtime > last] if done.is_dir() else []
    try:
        stamp.write_text(str(time.time()))
    except OSError:
        pass
    if stale and applied:
        return AMBER, (f"companion watcher stalled — watchdog applied "
                       f"{applied} verdict(s) itself")
    if fresh:
        return GREEN, f"{len(fresh)} new phone verdict(s) since last pass"
    return GREEN, "queue clear, no new verdicts"


def _check_hygiene() -> tuple:
    """Code hygiene DRIFT (Joe: "make sure... everything's not turning
    into spaghetti"). Absolutes are noise — a 1200-line pipeline is not
    news — so this reports only what got WORSE since the recorded
    baseline. AMBER, never RED: hygiene must not block a fix.
    """
    import subprocess
    try:
        r = subprocess.run([sys.executable, "tools/hygiene_audit.py", "--json"],
                           cwd=ROOT, capture_output=True, text=True,
                           timeout=600)
        d = json.loads(r.stdout or "{}")
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        return "AMBER", f"hygiene audit failed: {exc}"
    worse = d.get("worse") or []
    lint = d.get("lint", {}).get("n", -1)
    if not worse:
        return "GREEN", f"no drift; {lint} lint findings"
    return "AMBER", f"{len(worse)} regressions: " + "; ".join(worse[:3])


def _check_identity_hops() -> tuple[str, str]:
    """Identity-wander tripwire: numbers teleporting between tracks (the
    shot-36 family, gated 2026-08-20). Baseline after the reachability
    gate ~1/1k states; a regression pushes it back toward 2.4+."""
    sys.path.insert(0, str(ROOT / "tools"))
    import audit_identity_wander as aw
    d = Path(Settings.load().recording.resolved_dir())
    vids = sorted(d.glob("session-*.mp4"), key=lambda p: p.stat().st_mtime,
                  reverse=True)[:3]
    hops = states = 0
    for v in vids:
        try:
            r = aw.audit(v)
        except Exception:  # noqa: BLE001
            continue
        hops += len(r["hops"])
        states += r["states"]
    if not states:
        return AMBER, "no sidecars to audit"
    rate = 1000.0 * hops / states
    if rate > 4.0:
        return RED, f"identity hops {rate:.1f}/1k (gate regressed?)"
    if rate > 2.0:
        return AMBER, f"identity hops {rate:.1f}/1k (above gated baseline)"
    return GREEN, f"identity hops {rate:.2f}/1k (3 newest sessions)"


CHECKS = {
    "app": _check_app_running,
    "app_log": _check_app_log,
    "recording": _check_recordings,
    "orphans": _check_orphans,
    "playback": _check_playback_speed,
    "identity": _check_identity,
    "champion": _check_champion,
    "autonomy": _check_autonomy,
    "cue_sensor": _check_cue_sensor,
    "corrections": _check_corrections,
    "id_hops": _check_identity_hops,
    "hygiene": _check_hygiene,
    "disk": _check_disk,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="include the test suite")
    ap.add_argument("--quick", action="store_true", help="(default)")
    args = ap.parse_args()
    checks = dict(CHECKS)
    if args.full:
        checks["tests"] = _check_tests

    board = {}
    worst = GREEN
    order = {GREEN: 0, AMBER: 1, RED: 2}
    for name, fn in checks.items():
        try:
            status, detail = fn()
        except Exception as exc:  # noqa: BLE001 - a broken check is itself a finding
            status, detail = AMBER, f"check crashed: {exc}"
        board[name] = {"status": status, "detail": detail}
        if order[status] > order[worst]:
            worst = status
        mark = {"GREEN": " ", "AMBER": "!", "RED": "X"}[status]
        print(f" [{mark}] {name:10s} {status:5s}  {detail}")

    out = {"at": datetime.now(timezone.utc).isoformat(), "worst": worst,
           "board": board}
    (ROOT / "_eval" / "health.json").write_text(json.dumps(out, indent=2))
    print(f"\nWORST: {worst}")
    msg = {GREEN: "All healthy — feature work allowed.",
           AMBER: "Degraded areas exist — fix before new features "
                  "(unless blocked on Joe).",
           RED: "Something is BROKEN — that is the work session."}
    print(msg[worst])
    return {GREEN: 0, AMBER: 1, RED: 2}[worst]


if __name__ == "__main__":
    raise SystemExit(main())
