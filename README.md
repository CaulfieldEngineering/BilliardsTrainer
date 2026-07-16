# 🎱 Billiards Trainer

A local, real-time **pool/billiards practice analyst** for **Windows and macOS
(Apple silicon)**. Point a camera at your table, and it detects the table,
tracks the balls, recognises your shots, and keeps your **make/miss
statistics** — all running **locally and free** (no cloud APIs, no
subscriptions).

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
   from the dropdown (hit the refresh icon if you just plugged it in). The choice
   is **applied immediately — no Save click needed** — and **Test preview**
   confirms it's the right one. Then go to **Sandbox → Start**. The app remembers
   the camera by name, so it still finds it even if Windows reshuffles the device
   order later.

The app checks for updates on launch and prompts you when a newer build exists.

### macOS (Apple silicon — e.g. a Mac Mini M1 by the table)

1. Download `BilliardsTrainer-mac-<version>.zip` from the
   [latest release](https://github.com/CaulfieldEngineering/BilliardsTrainer/releases/latest),
   unzip, and drag **BilliardsTrainer.app** into Applications.
2. First launch: **right-click → Open** (the app is unsigned; macOS only asks
   once). Grant the **camera** and **Bluetooth** prompts on first use — the
   bundle declares both, so the prompts appear instead of a silent denial.
3. Ball detection runs the same trained ONNX model via **CoreML** on the M-series
   GPU/Neural Engine (falling back to CPU automatically); the cue sensor works
   over CoreBluetooth; shot-clock beeps play natively.

Every release CI-builds and tests on an Apple-silicon runner. One current
difference: on macOS the updater **notifies** you and opens the download page
rather than self-installing — replacing a running `.app` in place is kept
manual until it's been validated on real hardware. Data lives in
`~/Library/Application Support/BilliardsTrainer`.

### Sandbox = the core loop

The Sandbox tab is deliberately simple: detect the table, shoot freely, count
makes and misses. The table is detected and **locked once** automatically (and
remembered between launches); the big counter is the only thing you need to
watch. Drills and the shot clock are separate, optional, and off by default.

The **overhead view is a clean rendered schematic** (proportional felt, rails,
pockets, diamonds, balls) — not the warped camera — so it's easy to read.

**Shot detection** is gated on hard evidence to avoid false counts: a cue ball
must be present, the table must be genuinely in motion (frame-to-frame motion
energy, not fragile per-ball velocity), a ball must travel a real distance, and a
pot only counts when a ball approaches a pocket and drops in — with a warm-up
after Start and a cool-down between shots. Every threshold is tunable in
**Settings → Detection**. Controls in the Sandbox bar/rail:

- **⏸ Pause** counting (video keeps running), **Reset counters**
- **＋Make / －Miss** to log shots by hand any time
- **Confirm shots manually** (Settings → Detection) — auto-detect only *suggests*;
  you tap Make/Miss to commit
- **Debug overlay** (Settings → Detection) to see the raw blobs + shot state
- **Record** the session to a clip for offline analysis

It's still on-device classical CV (demo-grade) — drop fine-tuned YOLO weights in
the models folder and set **Backend = yolo** for a big accuracy jump.

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

#### Antivirus & the auto-updater (Windows)

The app is an **unsigned single-file exe**, which strict antivirus (e.g. Windows
Defender) sometimes treats cautiously — occasionally quarantining a bundled DLL
*while a freshly-updated copy is starting*, which can make an update fail to
launch. v0.2.0+ guards against this: it **verifies the download's SHA256** before
swapping, keeps a **backup and auto-rolls back** if the new version doesn't start
within ~25 s, and then shows a recovery dialog explaining what to do.

To avoid it entirely, **add the install folder to your AV exclusions** once. In an
**admin** PowerShell (replace the path with where you keep the exe):

```powershell
Add-MpPreference -ExclusionPath "C:\Users\<you>\Downloads"
```

The recovery dialog has a button that copies this command (pre-filled with your
actual install folder) to the clipboard.
- **Settings → Send feedback** — file a bug or feature request from inside the app
  (optionally attaching a screenshot and the last-5s replay). It's saved locally
  and, if you set up cloud backup, synced to Supabase. See
  [`docs/SUPABASE.md`](docs/SUPABASE.md) — the app works fully without it.

## Run from source (the fast dev loop)

Iterating on the frozen `.exe` is slow (build + publish + download). For day-to-day
work, **run from source** — edits are testable in seconds:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
python -m billiards_trainer        # launch the app (or: .\run_dev.ps1)
pytest                              # run the test suite
python tools/eval_table.py          # table-detection eval vs the reference capture
python tools/eval_detection.py      # detection false-positive eval (synthetic, no camera)
python tools/eval_detection.py --video clip.mp4   # iterate detection on a recorded clip
python tools/bench_pipeline.py      # per-stage performance benchmark on testVideo.MP4
```

`.\run_dev.ps1` bootstraps the venv on first run, then just launches the app.

**Iterate without the camera.** Detection work does *not* require standing at the
table: `eval_detection.py` runs against the synthetic demo by default, and
`--video <clip.mp4>` runs the full pipeline over any recording (e.g. a
*Capture for analysis* zip's frames, or your own clip). Record once, iterate many
times.

**Behaviour is feature-flagged, not hard-coded.** Big behavioural switches —
auto-detection on/off, real-measured vs. legacy ball colours, allow detection
without a model — are toggles in **Settings** that apply live. No rebuild to
change behaviour.

> **Don't pre-build the `.exe` locally.** CI runs the exact same PyInstaller build
> on every push to `main` and publishes the release — a local build just burns
> 5-10 min duplicating it. Push and let CI build.

Optional ball-detection backends:

```powershell
pip install -e ".[onnx]"    # ONNX ball detector — runs a YOLO .onnx, no torch (~15 MB)
pip install -e ".[yolo]"    # Ultralytics YOLO — only to run/convert .pt weights (~2 GB)
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
The CV pipeline runs on a dedicated worker thread, never the UI thread. A live
camera additionally gets its own **grab thread** that keeps only the newest
frame, so capture I/O never blocks inference and tracking latency stays pinned
at ~1 frame even when processing runs slower than the camera.

**Performance is measured, not guessed:** `python tools/bench_pipeline.py` runs
the real pipeline over a recorded video and prints a per-stage (detect / track /
motion / render) latency table; the same breakdown shows live in the debug
overlay. The heaviest knob is **Settings → AI detection → "Extra far-rail
scan"** — a second inference pass that recovers tiny far-rail balls at ~2× GPU
cost per frame (on: better recall; off: roughly double the frame rate).

### Ball detection: manual-first, AI-ready

On a real overhead camera, classical CV (Hough + colour) is only demo-grade —
wrong colours/sizes, phantom blobs in pockets. And there is **no free
off-the-shelf model that works**: generic COCO YOLO's "sports ball" class does
not fire on top-down pool balls (verified), and the pool-specific community
models are either API-key-gated or unlicensed. So the app is **manual-first**:

- It opens into a **live camera preview** with auto-detection **off** — a clean
  empty overhead and manual **+Make / −Miss**. No fake detections on screen.
- Auto-detection is a toggle that unlocks when a **pool-specific model** is
  present. Drop a YOLO **`.onnx`** into the app's `models/` folder (named e.g.
  `pool_balls.onnx`) and the app runs it via **ONNX Runtime — no torch** (the
  `OnnxYoloDetector`). A `.pt` works too if you install the `[yolo]` extra.
- Balls render in their **real measured colour** (a blue ball looks blue), with a
  grey **?** when the class is uncertain — never a confident wrong colour.

**Getting a model.** No free model detects top-down pool out of the box (generic
COCO weights detect zero pool balls). The Roboflow **Pool V2** dataset (CC BY 4.0)
is a viable base, but Roboflow gates downloads behind a free API key. To build a
local, offline model from it in one command:

```powershell
pip install -e ".[yolo]" roboflow
python tools/train_pool_model.py --api-key <FREE_ROBOFLOW_KEY>   # -> models/pool_balls.onnx
```

See [`docs/CV_ROADMAP.md`](docs/CV_ROADMAP.md) for the full model situation.

### Cue-stroke sensor (Bluetooth IMU)

An optional 6-axis motion sensor on the cue butt (JINOU JO-BEC12-2, ported from
the [pool-stroke-analyzer](../pool-stroke-analyzer) project) measures each
stroke the camera can't see: **impact shock, cue speed at contact, draw length,
steer (cue twist during delivery), backstroke pause, contact cleanliness, and
follow-through stillness**. Enable it in **Settings → Cue stroke sensor**; it
connects automatically and the per-shot stats appear in a **CUE STROKE** card
on the Sandbox rail (peak g immediately at the strike, kinematics ~2.6 s
later). Stroke metrics are joined to each recorded shot in the database, so
make/miss can be correlated against stroke quality — and the strike moment is
the future trigger for shot-clock integration.

Everything degrades gracefully: no sensor, no Bluetooth radio, or no `bleak`
install just shows a status on the card — detection, tracking and scoring are
unaffected. The impact signature is video-validated (14/14 true hits, 0 false
positives on labelled footage) and the app **never writes to the sensor**
(`src/billiards_trainer/cue/`). Raw waveforms live in **Settings → Sensor
diagnostics** for checking the mounting and stream health.

---

## Detector experiments (Phase 1)

A self-healing framework for trying *many* detectors against the eval set and
ranking them — so "better" is always a number, never a vibe. All variants detect
on the **raw oblique frame** (the bird's-eye is display-only).

**Add a new detector = one file.** Drop `src/billiards_trainer/detector_strategies/<name>.py`:

```python
from . import DetectorStrategy
from ..vision.types import Detection

class MyStrategy(DetectorStrategy):
    name = "my_strategy"
    description = "what it does"
    def detect(self, frame_bgr, calib):     # raw BGR frame + table Calibration|None
        # ... return detections in RAW pixel coords; never raise on a normal frame
        return [Detection(x, y, radius, bgr, cls, score)]

STRATEGIES = [MyStrategy()]                 # auto-discovered
```

These modules are **additive** — none is wired into the shipped pipeline, so the
default detector is unchanged. Drop a YOLO `.onnx` in `_eval/models/` and it
auto-registers as an `onnx_<name>` variant with no code at all.

**Run the experiments** (local — needs the gitignored `_eval/clips` + `_eval/labels`):

```powershell
python tools/eval_experiments.py                      # all variants
python tools/eval_experiments.py --only felt_mask_hough,simple_blob
```

Each variant runs in its **own subprocess with a timeout** — a crash, hang, or OOM
in one is captured and marked FAILED in the report; it never blocks the others.
Output lands in `docs/eval/experiments/<timestamp>/`:
- `comparison.md` — ranked table (precision/recall/F1/ID-acc/calib/FPS/status) with
  **CANDIDATE** (beats the Phase-0 baseline F1) / **RECOMMENDED** (beats it by >50%)
  self-evaluation, plus an `idle_02` recall spotlight.
- `visual.html` — every variant's detections side-by-side on the same frames.
- `variants/<name>/result.json` — raw per-variant metrics.

**CI vs local:** experiments are **local-only** — they need the large gitignored
clips/models and (for some variants) ONNX/torch. CI runs the lighter detector
*regression* eval (`.github/workflows/eval.yml`) on the committed synthetic clip;
it does not run the experiment matrix.

---

## Project layout

```
src/billiards_trainer/
  vision/      felt detection, rectification, balls, tracking, calibration, pipeline
  cue/         Bluetooth cue-stroke sensor: protocol, validated analysis, BLE worker
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
