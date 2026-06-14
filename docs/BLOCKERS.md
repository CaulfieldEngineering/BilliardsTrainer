# Blockers, decisions & limitations

Written during the autonomous overnight MVP build. Nothing here stopped progress
— each item was resolved with a pragmatic default and is logged for Joe to decide
on next. Constraints honoured: **no paid AI APIs, everything local + free; native
desktop; coach deferred.**

---

## Decisions taken (no blocker, but you should know)

### 1. Ball detection is **classical by default**, YOLO is pluggable
Pre-trained `yolov8n` has **no pool-ball classes** (COCO doesn't include them), and
fine-tuning needs a labelled dataset + training run that wasn't feasible overnight
without your camera. So the **default detector is classical** (Hough circles on the
rectified view + colour classification + a dedicated white-blob pass for the cue
ball). It needs no model and no `torch`, keeping the installer ~150 MB instead of
~2 GB.

**Upgrade path (no code changes needed):**
1. `pip install -e ".[yolo]"` (or bundle it in a "pro" build).
2. Get weights — e.g. search **Roboflow Universe** for "pool balls" / "billiards"
   and export a YOLOv8 `.pt`, or fine-tune `yolov8n` on a few hundred frames from
   your own camera.
3. Drop the file in the app's `models/` folder (named `pool_balls.pt`, `best.pt`,
   or `billiards.pt`) and set **Settings → Ball detection → Backend = yolo**.
   The factory auto-loads it and falls back to classical if anything's missing.

### 2. Distribution is a **portable `.exe`**, not an Inno/NSIS installer
The CI builds a single-file PyInstaller `.exe` and publishes it to GitHub Releases
with `version.json`. This satisfies "download from a URL and run." A *true*
installer (Start-menu shortcut, Program Files, uninstaller) via Inno Setup is a
clean follow-up — it just wasn't worth the unverifiable CI complexity overnight.

### 3. In-app **self-update** is implemented but unverified on a real build
The updater polls `version.json`, prompts, downloads with progress, then (when
frozen) writes a swap batch that replaces the running exe and relaunches. The
download/prompt/version-compare logic is unit-tested; the **frozen swap-and-
relaunch path could not be exercised by the agent** (no signed release build on
hand). Please verify on your machine once the first two releases exist (install
v0.1.N, push a trivial change, confirm it offers + applies v0.1.N+1).

### 4. CI publishes a Release **on every push to main**
Each push → version `0.1.<run_number>` → a new GitHub Release with the exe +
`version.json`. If that's too noisy, we can gate releases behind tags instead.
Chosen this way to match your "push to main triggers a build + publish" request.

---

## Things that need YOUR machine / a real camera

- **Felt HSV defaults** are validated against the project's reference capture
  (corner RMSE ~8 px). Your room's lighting/felt may differ — Settings exposes a
  manual hue range + sensitivity. A **click-to-pick-felt-colour-from-the-frame**
  control is a TODO (the manual sliders work today).
- **Ball-detection tuning.** The classical detector is demo-grade; thresholds
  (`Settings → Ball detection → strictness`, ball-radius fractions in
  `config.py`) will likely want a pass against your actual overhead camera. The
  synthetic **demo** mode proves the full chain works end-to-end regardless.
- **Camera mounting.** Best results come from a fixed overhead view (as the
  original prototype assumed). Calibration locks on the first clear frame.
- **No GUI/camera verification was possible by the agent** — it ran headless.
  All logic is covered by tests + a headless end-to-end smoke run, but please do
  a real run-through (Live → Try demo first, then your camera).

---

## Explicitly out of scope (per your direction)

- **AI coach — skipped entirely** (deferred until after the shot clock, as you said).
- **No app icon yet** — drop a `packaging/app.ico` and the spec will use it.
- **Pose / body fundamentals (Phase 8)** is wired as an optional MediaPipe module
  but is a stretch stub, not a finished feature.
