# TODO / parked work items

Not a backlog of committed work — a place to capture things deliberately
deferred so they aren't lost. Nothing here is in progress.

## Feature requests (from Joe)

- **Transport: jump back 5 seconds.** Add a "« 5s" button to the video transport
  bar (next to the step ◀|/|▶ buttons in `live_page._transport_bar`) that seeks
  `current_frame − round(5 × fps)`. Reuse the existing `video_seek` signal →
  `controller.video_seek`. *(Requested 2026-06-16; not started.)*

- **Playing area should exclude the rail tops (rail-width inset).** Felt detection
  maps the whole blue area, but the outer margin of blue is the rail/cushion TOP
  cloth, not the bed — the true playing area is inside the cushion noses and is
  slightly smaller. Fix: inset the detected corners toward the centroid by
  `table.nose_inset_frac` (already exists, currently 0.0) in
  `calibration.calibrate()` before `rectify_tabletop`, so the bird's-eye maps the
  bed and pockets/on-table tests use the cushion-nose boundary. The exact inset is
  table-specific, so expose it as a **live "Rail inset" slider in the Sandbox
  tuning panel** (applies via reconfigure → recalibrate) so Joe can dial it while
  watching. Keep default 0.0 to avoid regressing cue tracking until tuned.
  *(Requested 2026-06-16; not started.)*

- **LLM/VLM post-session video analysis (deep, offline second pass).** Real-time
  OpenCV does what it can live; then, once a session ends, a vision-language model
  re-analyzes the *recorded video itself* for depth the frame-by-frame pipeline
  can't reach: miss patterns (which angles/distances/cut directions miss most,
  clustering by pocket and shot type), positional/strategic notes, and a **review
  pass that grades and corrects the real-time results** — confirming or flipping
  each auto make/miss and catching shots the live detector dropped.

  *Model selection (config-gated, mirror `cue`/Supabase optionality):* add an
  `AnalysisSettings` block — `backend = off | local | openrouter`, `model`,
  `api_key`, `endpoint`. **Local** talks to an Ollama-style HTTP endpoint running a
  vision model (Qwen2.5-VL / moondream) to honour the "local & free" ethos;
  **OpenRouter** posts over HTTPS with the user's key (the app already uses
  `requests`). Off by default; degrades silently like the cue sensor.

  *Hooks that already exist* (low-friction path): sessions are recorded (Record →
  `controller._write_recording`; clips land in `exports/`), every shot is logged
  to `logs/shots.jsonl` with timestamps, and the SQLite DB holds sessions+shots.
  So the second pass keys off shot timestamps to sample keyframes around each
  shot, sends frame(s) + context to the VLM, and writes structured JSON verdicts
  back to the DB (new per-shot `grade`/`notes` + a session summary). Trigger it
  from `stop()`/`end_session` as a background job so it never blocks the UI.

  *Surface:* a "Session review" panel (extend the Stats page) showing the model's
  corrections + patterns; corrected make/miss can update the recorded counts.

  *Synergy:* the VLM's per-shot make/miss verdicts are exactly the labels the
  YOLO fine-tune pipeline wants — the review pass doubles as an **auto-labeler**
  for training data (ties into `tools/train_pool_model.py` + the
  Capture-for-analysis flow). *(Requested 2026-07-17; not started.)*

- **HDMI-dongle rig validation (when the capture device arrives).** Joe's ceiling
  rig sends video over an HDMI→USB capture dongle (plain UVC webcam to the app)
  and keeps USB for touchless control (`tether.remote_camera_sync`). To verify on
  hardware: (1) dongle appears in the camera dropdown and streams at its full
  resolution (set Settings→Camera width/height if it defaults to 720p);
  (2) whether the T3i keeps HDMI output alive during a transient USB control
  session, or blanks it for the few seconds (either is acceptable, just confirm);
  (3) the T3i needs its HDMI info overlay OFF (DISP cycling) for a clean feed —
  document the camera-menu steps; (4) `packaging/`: bundle libusb.dylib in the
  PyInstaller spec for the frozen mac build (pyusb needs it).
  *(Planned 2026-07-17.)*

## Infra

### Dev/test machine sync mechanism  *(flagged, not implemented)*

Joe tests on a **different machine** than where the code lives, so debug data
(logs, Capture-for-analysis zips, captured frames, `updater.log`, `shots.jsonl`,
the SQLite DB) currently has no easy path back to the dev box. We need a
low-friction one-way (test → dev) channel. Options, easiest first:

- **Syncthing or a OneDrive/Dropbox shared folder** — point the app's exports/
  logs dir at a synced folder; zero code, near-real-time, offline-friendly.
  Probably the best fit for the "local/free" constraint.
- **Small log/artifact upload endpoint in Joe's existing Calorie Tracker
  Supabase project** — reuse an already-running Supabase; the app already has a
  config-gated Supabase sync skeleton (`src/billiards_trainer/sync/`), so a
  "upload debug bundle" button could post a zip to a storage bucket. Needs
  credentials + a bucket; mild setup.
- **SSH/SCP** if both machines have it — most manual, but trivial and fully
  local if they're on the same LAN.

Recommendation when picked up: start with a synced folder (no code), and only
build an in-app "Send debug bundle" button (zip logs+captures → Supabase
storage) if the folder route proves clumsy. **Do not implement until Joe picks a
direction.**

## Parked ideas from the prior-art survey (`docs/PRIOR_ART.md`)

Surfaced for a decision, not committed:

1. **Detector sprint** — extend `tools/train_pool_model.py` to also train on the
   public *only_balls v3* Roboflow dataset (the model `pool_coach` uses), export
   ONNX, and benchmark on *our* rectified fixtures. Decide if single-class "ball"
   detection on the warped view is good enough to default auto-detection on.
2. **Aim-line / best-shot overlay** — port the MIT ghost-ball + Dijkstra
   geometry from `8-Ball-Pool-Analysis` into our warped space (works off tracked
   or manual ball positions; no detector required).
3. **Calibration hardening** — k-means line-cluster / flood-fill corner fallback
   + geometric-median corner smoothing, A/B'd against the current HSV path.

See `docs/PRIOR_ART.md` for the full analysis, comparison table, and license
caveats (GPL-3.0 / AGPL-3.0 vs MIT).
