# 🎱 Billiards Trainer

A local, real-time **pool/billiards practice analyst** for Windows. Point a camera
at your table, and it detects the table, tracks the balls, recognises your shots,
and keeps your **make/miss statistics** — all running **locally and free** (no
cloud APIs, no subscriptions).

> Rebuilt from the original C++/Win32 table-detection prototype into a Python +
> PySide6 application. The hard-won CV (felt → corners → homography) was ported
> faithfully; ball tracking, shot detection, stats, drills, and a shot clock are
> new. The old C++ code is archived under [`legacy/`](legacy/).

---

## Quick start (no camera needed)

1. Download `BilliardsTrainer-<version>.exe` from the
   [latest release](https://github.com/CaulfieldEngineering/BilliardsTrainer/releases/latest).
2. Run it. You land on the **Sandbox** tab — click **Try demo** and a synthetic
   table runs the whole pipeline so you can watch the big **MAKES / MISSES**
   counter move.
3. To use your own table: open **Settings → Camera**, pick your webcam **by name**
   from the dropdown (hit the refresh icon if you just plugged it in), click
   **Test preview** to confirm it's the right one, **Save**, then go to
   **Sandbox → Start**. The app remembers the camera by name, so it still finds
   it even if Windows reshuffles the device order later.

The app checks for updates on launch and prompts you when a newer build exists.

### Sandbox = the core loop

The Sandbox tab is deliberately simple: detect the table, shoot freely, count
makes and misses. The table is detected and **locked once** automatically (and
remembered between launches); the big counter is the only thing you need to
watch. Drills and the shot clock are separate, optional, and off by default.

### Tuning for your real table

If detection looks off under your lighting, the Sandbox control bar has three
tools (top-right):

- **Pick felt** (crosshair) — click it, then tap your cloth in the live view to
  seed the felt colour from your actual table.
- **Toggle overlays** (layers) — hide/show the detection drawing to see the raw
  camera.
- **Recalibrate** (refresh) — re-detect the table if it shifted.

And **Save last 5s** writes a clip of exactly what the detector saw to
`%APPDATA%/BilliardsTrainer/exports/` — handy when something looks wrong. Every
shot is also logged to `…/logs/shots.jsonl`.

### Updating & feedback

- **Settings → Check for updates** — the button is in the Settings header (always
  visible) and in the Updates card. One click checks the latest release; if a
  newer build exists you get a prompt → download with a progress bar → the app
  swaps itself and relaunches. (It also checks silently on launch and logs the
  result to `…/logs/billiards_trainer.log`.)

> **On a build older than v0.1.6?** The in-app "Check for updates" button didn't
> exist before v0.1.6, so you won't see it. Do a **one-time manual download** of
> the newest `BilliardsTrainer-*.exe` from the
> [releases page](https://github.com/CaulfieldEngineering/BilliardsTrainer/releases/latest)
> and run it. From v0.1.6 onward the in-app updater takes over — you won't need
> to visit GitHub again.
- **Settings → Send feedback** — file a bug or feature request from inside the app
  (optionally attaching a screenshot and the last-5s replay). It's saved locally
  and, if you set up cloud backup, synced to Supabase. See
  [`docs/SUPABASE.md`](docs/SUPABASE.md) — the app works fully without it.

## Run from source

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
python -m billiards_trainer        # launch the app
pytest                              # run the test suite
python tools/eval_table.py          # table-detection eval vs the reference capture
```

Optional power-user backends:

```powershell
pip install -e ".[yolo]"    # Ultralytics YOLO ball detector (pulls in torch, ~2 GB)
pip install -e ".[pose]"    # MediaPipe pose / body-fundamentals analysis
```

---

## How it works

```
Camera / video / demo
        │
        ▼
One-shot table calibration            felt HSV → 4 corners → homography (locked)
        │   (re-checked periodically by a deviation watchdog)
        ▼
Per-frame bird's-eye warp  ──►  ball detection  ──►  multi-object tracking
        │                       (Hough + colour,      (ByteTrack-style +
        │                        or optional YOLO)      constant-velocity)
        ▼
Shot state machine             cue strike → motion → pocket entry → make/miss
        │
        ▼
SQLite (sessions, shots)  ──►  live + historical stats, streaks, by-pocket, export
```

Key design principle (a fix for the prototype's biggest cost): **calibrate the
table once, lock the homography, then spend the whole per-frame budget on balls.**
The CV pipeline runs on a dedicated worker thread, never the UI thread.

### Ball detection: classical by default

Pre-trained YOLO has no pool-ball classes, and fine-tuning needs a labelled
dataset, so the **default detector is classical** — Hough circles on the
bird's-eye view, each validated as non-felt and colour-classified (cue / solid /
stripe / 8-ball), with a dedicated white-blob pass for the cue ball. It needs no
model and no torch, so the installer stays ~150 MB. The detector is **pluggable**:
drop fine-tuned weights into the app's `models/` folder and switch the backend to
`yolo` in Settings. See [`docs/BLOCKERS.md`](docs/BLOCKERS.md) for the upgrade path.

---

## Project layout

```
src/billiards_trainer/
  vision/      felt detection, rectification, balls, tracking, calibration, pipeline
  events/      shot / make / miss state machine
  game/        drills, shot clock, modes
  db/          SQLAlchemy models + repository (stats, export)
  ui/          PySide6 dark-themed UI: pages, widgets, dialogs, theme, icons
  capture/     camera / video / image / synthetic-demo frame sources
  workers/     capture+pipeline controller thread
  update/      in-app updater (version.json poll → download → relaunch)
tests/         pytest suite + the labelled reference capture fixtures
tools/         eval_table.py — corner-RMSE eval harness
packaging/     PyInstaller spec, launcher, version.json generator
.github/       CI: build → test → publish installer + version.json to Releases
legacy/cpp/    archived original C++/Win32 prototype
```

## CI/CD

Every push to `main` runs the test suite, builds a Windows `.exe` with
PyInstaller, and publishes it + `version.json` to a GitHub Release. The in-app
updater polls `releases/latest/download/version.json` on launch. See
[`.github/workflows/build.yml`](.github/workflows/build.yml).

## License

MIT.
