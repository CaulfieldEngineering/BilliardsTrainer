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
2. Run it. On the **Live** tab, click **Try demo** — a synthetic table runs the
   whole pipeline and you'll see make/miss stats accumulate.
3. To use your own table: open **Settings**, set **Source** to your camera index
   (usually `0`), Save, then go to **Live → Start**.

The app checks for updates on launch and prompts you when a newer build exists.

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
