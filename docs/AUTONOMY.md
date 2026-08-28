# The autonomous work loop

Joe (2026-08-15): *"Set up some way where you can work on this day and
night until it's fixed... watchdog timers to see if your workflow has
halted... be relentless."*

## Scope and the health-first rule (Joe, 2026-08-16)
The loop covers the WHOLE product: UI updates, features, research,
development, training, tooling — "expand this beyond vision training."
With one hard rule: **"ensure we don't add features when others are still
broken or lacking."** Made executable by tools/health_check.py:

- Every work session runs the health board FIRST (--quick).
- RED anywhere ⇒ fixing it IS the session. No exceptions.
- AMBER ⇒ fix before feature work, unless the amber is blocked on Joe
  (e.g. cue sensor hardware) — then note it and proceed to features.
- All GREEN ⇒ feature/research/UI work proceeds by GOALS priority.

The board checks: app process + log errors, recording pipeline +
orphans, cached playback speed (>=30fps bar), identity duplicates (G1
tripwire), champion fingerprint (_eval/champion.json is canonical),
autonomy heartbeat/lock, cue sensor state, disk space; --full adds the
test suite.

## How it runs
Two scheduled jobs live in the Claude session attached to this repo:

1. **Work session** — every 2 hours. Reads docs/GOALS.md, takes the
   highest-priority unfinished goal, does one bounded chunk of real work
   (label sessions, train/gate a challenger, build a tool, fix what a
   metric exposed), verifies it, commits, appends one line to the GOALS
   progress log, and refreshes the heartbeat.
2. **Watchdog** — hourly, offset from the work session. Checks the
   heartbeat and the invariants below; recovers and logs an incident if
   anything is wrong. THEN runs the hygiene pass (below) — the
   operational checks alone let a week of layering grow under green
   tests (Joe, 2026-08-27: "The audit you just did is what the hygiene
   watchdog is supposed to be doing every hour").

## Hygiene pass (every watchdog run — architecture, not just liveness)

Tests measure behavior; nobody was measuring STRUCTURE. This pass is
the hourly structural review, kept cheap by being diff-driven:

1. Find the last `hygiene:` line in docs/GOALS.md — it names the last
   reviewed commit. Diff `<that>..HEAD` (code only; skip docs-only).
2. Review each commit's diff against docs/ARCHITECTURE.md's seven
   laws. Hunt specifically:
   - a new holder of table/session state outside MeasurementCore (L1)
   - a downstream patch on a measurement symptom (L2)
   - a pure addition that names nothing it replaces (L3)
   - a new FramePacket field, time-proximity join, or timebase
     reconciliation (the map's three worst families)
   - a change with no pinning test (L4)
3. Append ONE line to GOALS: `hygiene: clean|N findings through
   <short-hash>`. Findings that need work become demolition-ledger
   entries in ARCHITECTURE.md §5 (or BLOCKERS.md if urgent); serious
   violations of L1-L7 are INCIDENTS (log below), and the fix goes at
   the head of the work queue — before feature goals.
4. WEEKLY (or when Joe asks): a fresh full audit in the style of
   docs/design/state-opinion-map-2026-08-27.md. The site count must
   SHRINK release over release; growth is an incident, not a trend.

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
- 2026-08-27: CHARTER GAP, caught by Joe, not the loop: a week of
  hourly "hygiene" watchdogs verified liveness only (heartbeat, locks,
  champion .bak) because that's all this charter listed — meanwhile
  ~40 private opinion-holders of table state accumulated under green
  tests (see docs/design/state-opinion-map-2026-08-27.md). Tests
  measure behavior; nothing was measuring structure. Structural fix:
  the hygiene pass above is now a standing watchdog duty, diff-driven
  hourly + full audit weekly, with the site count required to shrink.
- 2026-08-17: red push slipped through a PIPED pytest exit code for the
  second time (5b877cf; first was the 088ed55 CI incident). Joe got the
  failure email. Structural fix: a local git pre-push hook now runs the
  full suite UNPIPED and blocks the push on any failure — the loop can
  no longer push red by construction. Rule stays: never gate on a piped
  exit code.
- 2026-08-18: Joe reported mouse lag + empty schematic while playing —
  the loop's verification backfill was running full-tilt inference while
  he was AT THE TABLE, and a dead detect-worker thread had never been
  restarted. New rules: (1) HEAVY JOBS (backfill, corpus scoring,
  training) must not START if any session-*.mp4 was modified in the last
  10 minutes, and must be KILLED if a recording starts — Joe's presence
  outranks loop verification, always. (2) Every fix to something Joe SEES
  ships with a pinned UI test through the real theme (test_transport_bar
  is the pattern). (3) App restarts wait for 5 minutes of recording
  quiet, monitored, not assumed.
- 2026-08-18 (measurement invariant, PERMANENT): per-session physics
  scores are only comparable WITHIN a same-day batch. Proven by worktree
  bisect: the exact loop-13 commit that produced 015737=1.41/6-teleports
  in its own corpus run scores 2.00/12 today — identical code, video,
  and model; the environment (GPU runtime/driver state) moved. Rules:
  (1) promotion gates score champion+challenger same-batch (now the
  score_challenger default); (2) never compare against a saved
  aggregate across days for any decision; (3) corpus dirs are
  era-stamped by date — treat old ones as historical record only.
- 2026-08-18 (invariant REFINED by the ruler_20260818 run): corpus
  batches reproduce across days bit-for-bit (0.50/1k aggregate, worst
  1.41, held-out oog 26.17 — twice, days apart). What does NOT compare
  is SOLO score_session runs vs corpus numbers: a lone run executes on
  an idle GPU; a mid-corpus subprocess runs 16 sessions deep into
  thermal/driver load, and detection cadence shifts either direction
  per session (015737: 2.00 solo vs 1.41 in-corpus, identical code).
  Rules stay: same-batch gates for promotions; solo runs are for
  debugging deltas within one sitting, never for cross-checking corpus
  verdicts. The loop-23 "environmental drift across days" reading was
  this category error — corrected here.
- INCIDENT 2026-08-22: loop.lock orphaned — the 18:38Z session closed out
  (heartbeat, GOALS, push all done) but omitted the lock delete; detected
  by heartbeat-newer-than-lock at the next session start. Cleared. The
  close-out checklist exists for exactly this; follow it in order.

## What's New (Joe-facing changelog — standing requirement 2026-08-23)
Joe: "I just come back to a wall of text I don't understand each time."
GOALS.md and heartbeats are written for the loop; the channel written for
JOE is companion-cloud/public/whatsnew.json, shown in the phone app's
menu with an unread dot. At every session close that changed anything he
can SEE or that changes what his data means:
- prepend an entry: id (increment), date, plain-language title + body —
  what changed FOR HIM, never how. No tracker internals, no module names,
  no metrics he didn't ask for. If it can't be said plainly, it does not
  belong in the file.
- deploy (the file ships with the app).
Sessions that change nothing Joe-visible add nothing — the dot must mean
something.
ALSO (Joe 2026-08-23, refined twice): ONE surface. Each whatsnew.json
entry carries the plain one-liner in `body`, and optionally `details` —
an array of short paragraphs telling the full story (what changed, why it
was wrong before, how the fix works), rendered as a "More details"
dropdown inside the entry. Details are thorough but written for a layman:
analogies over internals, no module names, no jargon. There is no
separate journal file or menu row — that was built and then folded in at
Joe's direction. Same discipline: no Joe-perceptible change, no entry. Separately: when the accuracy chain completes (trajectory fit
validated + library relabelled), notify Joe proactively — he asked to be
told when it's all fixed.

- INCIDENT 2026-08-24: c7 gate run held loop.lock >5h — the champion same-batch corpus pass blew score_challenger's 4h subprocess timeout (CPU inference is ~4x slower than the DML-era runs the timeout was sized for). Champion restored cleanly by the finally block (verified 12,277,106 bytes, no .bak). Challenger pass completed; champion pass resumed standalone. Lock cleared here.
- INCIDENT 2026-08-24 (2): Joe started recording mid-gate; per the kill rule the gate was stopped, which landed mid-challenger-swap — c7 was live in MODELS_DIR with the champion in .bak. Restored by file copy immediately (12,277,106 verified); the running app held c5 in memory throughout, no session was scored or recorded on c7. Champion same-batch aggregate COMPLETE (33 sessions); only the challenger corpus pass remains — resume when the table is quiet.
- INCIDENT 2026-08-26 (3): the stroke-v2 library backfill ran midday and Joe reported session lists taking minutes to load. Immediate response: killed the backfill, verified zero data damage (47/47 shots.json valid, sidecar tails clean), restarted the desktop app. Post-hoc evidence (GetLastInputInfo 14.2h idle matching last night's mic reseat; zero playback lines in the app log) says Joe was NOT at the desktop - the report was almost certainly the PHONE, leading theory: Dropbox sync churn from the backfill's appends saturating the home uplink his phone shares. RULE SHIPPED: heavy jobs now defer on joe_present() (tools/_presence.py, GetLastInputInfo, typed ctypes, fail-toward-present; pinned by tests) IN ADDITION to the table guards. Known gap: guards check between sessions, not mid-session - an hour-long marathon annotation can't yet be interrupted; measure_shot's abort hook is the next step if this recurs. Stroke-v2 backfill deferred to a quiet overnight window.
