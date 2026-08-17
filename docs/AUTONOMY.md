# The autonomous work loop

Joe (2026-08-15): *"Set up some way where you can work on this day and
night until it's fixed... watchdog timers to see if your workflow has
halted... be relentless."*

## How it runs
Two scheduled jobs live in the Claude session attached to this repo:

1. **Work session** — every 2 hours. Reads docs/GOALS.md, takes the
   highest-priority unfinished goal, does one bounded chunk of real work
   (label sessions, train/gate a challenger, build a tool, fix what a
   metric exposed), verifies it, commits, appends one line to the GOALS
   progress log, and refreshes the heartbeat.
2. **Watchdog** — hourly, offset from the work session. Checks the
   heartbeat and the invariants below; recovers and logs an incident if
   anything is wrong.

## Serialization invariants (violations = incident)
- `_eval/loop.lock` exists only while a work session runs. Stale lock
  (> 3 h old) ⇒ the run died; watchdog clears it and recovers.
- `%LOCALAPPDATA%/BilliardsTrainer/models/pool_ballid_r2.onnx.bak`
  exists only during a gate run. Present with no live gate ⇒ a swap died
  mid-run; watchdog restores the champion from the .bak (copy back,
  delete .bak) and logs the incident. CURRENT champion: **c5**, byte size
  12,277,106 (promoted 2026-08-16 — the first challenger to win).
  `pool_ballid_r2.prev.onnx` is the intentional rollback archive of the
  previous champion (12,277,094) — never auto-"restore" it.
- Never run two of: corpus scoring, challenger gate, app restart.
- Never restart the app while Joe is recording (check for a growing
  session-*.mp4 in the recording dir before any restart).

## Heartbeat
`_eval/heartbeat.txt` — one line, ISO timestamp + what the last work
session did. Refreshed at the END of each successful work session.

## Honest limits (told to Joe)
- The jobs live in the Claude session. If VS Code / the session closes,
  the loop stops until a session reopens. Jobs auto-expire after 7 days;
  each work session re-arms them when near expiry.
- Work sessions fire only while the session is idle — they queue behind
  any live conversation with Joe.

## Incident log
(watchdog appends here)
