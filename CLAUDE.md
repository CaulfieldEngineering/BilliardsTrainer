# BilliardsTrainer — session rules

**Before designing any change, read `docs/ARCHITECTURE.md`** — the
binding design methodology (Joe, 2026-08-27: read on every effort).
UI work additionally obeys `docs/DESIGN.md`. The measurement plan of
record is `docs/MEASUREMENT_CORE.md`.

The five-second version of ARCHITECTURE.md:
1. One opinion per fact — table truth lives in `measure/core.py`
   MeasurementCore; everything else reads it.
2. Measure, don't patch — no downstream bandages on measurement bugs;
   post-only fixes serve nothing (the product is during-play).
3. Subtract before you add — every addition names what it replaces.
4. Gates, not hope — champions change only via the measured scorecard;
   every behavior change lands with its pinning test.
5. Evidence before theory — cheap discriminating checks first; never
   blame the environment while our own code is unmeasured.

Operational hard rules:
- Watchdog/loop sessions: after the liveness checks, run the HYGIENE
  PASS (`docs/AUTONOMY.md` §Hygiene) — diff since the last `hygiene:`
  GOALS line, reviewed against the seven laws. Structure is checked
  hourly, not just uptime.
- Full test suite green BEFORE every commit (pre-push enforces; don't
  make it catch you).
- Edit tool only for source edits — scripted heredocs corrupted files
  twice.
- Never restart the app or start heavy jobs while a recording is live
  (`.session-*.part.mp4`) or a session file changed in the last 3 min.
- One heavy GPU job at a time; heavy jobs are presence-guarded and
  BelowNormal (see `measure/engine.py` for the pattern).
- Secrets live in `C:\Users\Joe\.billiards-secrets\` — never in repo,
  logs, or chat.
- Joe's review verdicts outrank any machine/forensic conclusion.
