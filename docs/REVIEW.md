# BilliardsTrainer — Code Review & Overhaul Recommendation

**Author:** Claude (code review pass)
**Date:** 2026-06-14
**Scope:** Assess current state, recommend an architecture (incl. web-app question), and propose a phased roadmap toward Joe's vision: a fast, responsive, real-time analytical tool for billiards.
**Status of this doc:** Assessment only. No code changed, nothing committed.

---

## TL;DR

- The repo is a **Windows-only native desktop prototype** (C++17 + OpenCV 4.x + raw Win32/GDI UI). One executable: `table_detector.exe`.
- It is, in reality, a **table-detection + bird's-eye-rectification calibration tool**. That part genuinely works — verified against the captured output frames in `build/Release/captures/`.
- The thing you said matters most — **reliable ball detection and uber-reliable shot/miss detection — does not exist at all.** There is zero ball, cue, motion, tracking, or event-detection code. (Only code comments mention "ball" and "cue.")
- ~62% of the 7,900 lines is a single 4,940-line `main.cpp`, and **~80% of that is hand-rolled Win32 UI plumbing** — which has no reuse value for a cross-platform, multi-device product.
- **The valuable, hard-won asset is the CV pipeline** (felt → corners → homography → rectified view), ~2,400 lines, cleanly separated from the UI.
- **Recommendation: rebuild as a hybrid web app — browser frontend + Python/OpenCV backend over WebSocket — porting the existing CV algorithms to Python.** Rationale below. This is the right call *because of* the code state, not just theory: the UI is throwaway, the CV is portable, and your entire downstream roadmap (ball tracking, pose, coach) lives in the Python ML ecosystem.

---

## 1. What this project currently is

A real-time computer-vision app that points a camera at a pool table and overlays detected geometry:

- **Felt** (playing surface) — color-segmented, with a clean mask and 4 ordered corners.
- **Rectified "bird's-eye" view** — perspective-corrected top-down image (forced 2:1 portrait).
- **Rails** — the cushion rails, detected in the rectified view.
- **Diamonds** — the sight markers along the rails.
- **Playable area** — the "nose line" boundary inset from the felt, computed from table-size + cushion specs.

It is a **calibration / debug instrument**, not an end-user product. The whole UI is sliders and color pickers for tuning detection. There is no game, drill, practice, stats, or coaching concept anywhere in the code.

### Evidence it runs
A built `build/Release/table_detector.exe` (Dec 28) exists, and `build/Release/captures/` contains a real captured session:
- `*-overlay.png` — live perspective view with felt mask (green), rails, yellow diamond markers, corner markers.
- `*-rect.png` — a clean, correct top-down rectification of the table.
- `*-window.png` — the full app: top menu (Debug / Display / Table Setup / Export Captures), small live view, large rectified view, settings sidebar.
- `*.json` — telemetry sidecar (felt corners, HSV params, homography H/Hinv).

The rectification quality in `*-rect.png` is good — minor edge warping, but the homography is sound. **This is the strongest part of the codebase.**

---

## 2. Tech stack

| Layer | Current choice |
|---|---|
| Language | C++17 |
| Build | CMake 3.16 (`find_package(OpenCV REQUIRED)`) |
| CV | OpenCV 4.x — core, imgproc, highgui, imgcodecs, videoio, features2d, calib3d, flann |
| UI | **Raw Win32** (`comctl32`, `comdlg32`, `uxtheme`) + GDI DIB blitting. No UI framework. |
| Capture | OpenCV `VideoCapture` (`CAP_DSHOW`) on a background thread |
| Persistence | Hand-rolled `key=value` `settings.txt` parser |
| ML / DL | **None** |
| Web / mobile | **None** |

> Note: there's a `venv/` folder, but it is an **accidental Inkscape-spawned virtualenv** (`home = C:\Program Files\Inkscape\bin`), gitignored and unused. There is no Python in this project today. Also a stray `nul` file (a Windows redirect accident) — harmless, should be deleted.

---

## 3. Entry point & per-frame pipeline

- **Entry:** `mainCRTStartup` → Win32 message loop (`/SUBSYSTEM:WINDOWS`, no console).
- **Capture:** a background thread (`startCaptureThread`) reads frames into `g_latestCaptureFrame` (sources: `-1` test image, `-2` looping `testVideo.mp4`, `>=0` camera index). Good instinct — keeps the UI from blocking on camera I/O.
- **Processing + render:** happens **on the UI thread** per displayed frame (`main.cpp` ~lines 2160–2266):

```
detectFelt()                  HSV inRange → morphology → largest CC → fill holes
                              → scan 4 extreme edges → robust Huber line fit
                              → intersect lines → 4 corners
  ↓ (EMA corner smoothing)
rectifyTabletop()             homography corners→2:1 portrait, + Hough-line
                              refinement pass to "square up" the rectangle
  ↓
detectRails()                 band around felt edge (rectified) + wood-color filter, per rail
  ↓ (rail-mask temporal smoothing)
detectDiamonds()              top-hat morphology + 90th-pct threshold + NMS + temporal smoothing
  ↓
computePlayableAreaRectified() pure geometry: table size + cushion inset → nose line
  ↓
project * back to perspective via Hinv, then draw overlays on both views
```

Architecturally the pipeline is coherent and the stages are reasonable. The problems are (a) it's color-threshold-bound and brittle, (b) it re-detects the table every single frame even though the table never moves, and (c) CV runs on the UI thread.

---

## 4. Module breakdown

| Module | Lines | State | Notes |
|---|---|---|---|
| `FeltDetection/` | 431 | **Works — strongest module** | HSV segmentation + robust 4-line fit with median outlier rejection (clever handling of pocket curves). |
| `Rectification/` | 461 | **Works — good** | Homography + a Hough-line refinement pass. Hard-codes a 2:1 portrait target. |
| `RailDetection/` | 638 | Works | Band around felt edge in rectified view + color filter. Reasonable. |
| `DiamondDetection/` | 565 | **WIP / noisy** | Real CV (top-hat + percentile + NMS), but cleanup morphology is commented out "until stable" and the overlay is hardcoded to "TEMPORARY" debug yellow. Spams `std::cout` per frame. |
| `CushionDetection/` | 302 | Works, **misnamed** | Untracked. Despite the name it does **no detection** — it's the playable-area / nose-line geometry calc (table + cushion specs). |
| `main.cpp` | 4,940 | Works | ~80% Win32 UI (windows, trackbars, color pickers, sidebar scroll, menus, GDI blitting, settings I/O, capture export). ~20% CV orchestration. |

**Branches:** `main` (current), plus a stale `optimistic-yalow` that's behind main. Recent commit cadence is active and table-detection-focused ("Rail Detection Perfection", "Diamond Detection WIP", "Stable - Overhead View").

---

## 5. What works vs. what doesn't

### ✅ Works (verified against captures)
- Felt/table detection and 4-corner extraction on the test table.
- Perspective rectification to a clean bird's-eye view.
- Rail + diamond + playable-area overlays render in both views.
- Settings persistence, click-to-sample color pickers, dual-view UI, capture export with JSON telemetry.

### ❌ Does not exist (and is the core of your ask)
- **Ball detection** — none.
- **Cue detection** — none.
- **Motion / object tracking** — none (no optical flow, no background subtraction, no Kalman/SORT).
- **Shot / make / miss event detection** — none.
- **Pocket detection** — the README claims Hough-circle pocket detection; **it is not in the code.**
- Stats, database, drills, game modes, coach, shot clock, pose/body analysis — none.

### ⚠️ The README is partly aspirational/inaccurate
It describes pocket detection (absent) and says diamonds are placed "at standard positions" (actually real CV now). Treat the README's feature list as a wish list, not a description.

---

## 6. Correctness gaps & code smells (static review)

1. **Everything is color-threshold dependent.** Felt, rails, and diamonds all key off a picked HSV color + a sensitivity slider. This is brittle to lighting changes, shadows, glare, a different table, or a different felt color, and it implicitly handles only one table at a time. Balls/objects occluding the felt will distort the mask. This is the single biggest reliability risk.
2. **Re-detects the table every frame.** The table doesn't move during a session — detection should be a one-shot (with a "calibration deviated" watchdog, which is even on your own to-do list). Re-detecting every frame wastes the per-frame budget you'll need for balls and adds jitter.
3. **CV runs on the UI thread.** Capture is threaded, but processing + render are not decoupled. Heavy frames will stutter the UI. No proper capture→process→render pipeline.
4. **Per-frame `std::cout` spam** in diamond + cushion detection. The build is `/SUBSYSTEM:WINDOWS`, so output is discarded — but the strings are still formatted every frame. Pure waste + noise.
5. **Diamond detection is mid-surgery:** cleanup morphology commented out, overlay color hardcoded as "TEMPORARY" yellow.
6. **Global mutable static state** inside detection modules (e.g. `static TrackPt track[4][6]`, `static float T_ema[4]` in diamonds; smoothing buffers in main). Not reentrant, hard to test, surprising across source switches.
7. **Forced 2:1 aspect** in rectification. Fine for a 9ft/7ft playing surface, but baked-in assumptions like this will bite when you generalize.
8. **O(W·H) nested pixel loops** for felt edge scanning. Fine at 720p; won't scale and isn't the idiom you'd want in the hot path.
9. **One 4,940-line `main.cpp`** mixing UI + orchestration + file I/O + screenshotting.
10. **No tests, no eval harness, no CI**, and only one fixture image + one video. For "uber reliable," you cannot improve what you cannot measure.
11. Repo hygiene: stray `nul` file, accidental `venv/`, `CushionDetection/` untracked.

**Maturity verdict:** ~60–70% of "rock-solid table detection," partial rails/diamonds, and **0% of the ball/shot/miss core.** The reusable crown jewel is the felt→corners→homography math.

---

## 7. The web-app question

### Recommendation: **Option B (Hybrid) — browser frontend + Python/OpenCV backend over WebSocket — and port the CV to Python.**

#### Why a rewrite of the *shell* is the right call (from the code, not theory)
- The UI is **4,900 lines of throwaway Win32** with zero cross-platform value. It is exactly the thing standing between you and "deliver it to my phone, iPad, and laptop."
- The CV is only **~2,400 lines and cleanly separated** — it ports to Python+OpenCV almost 1:1.
- Your **entire downstream roadmap is Python-native**: ball detection (YOLO/ultralytics), tracking (ByteTrack/SORT), pose/body fundamentals (MediaPipe), and the coach (Claude API) all have first-class Python ecosystems. Staying in C++ taxes every future phase.
- Python iterates **5–10× faster for CV R&D** — and "uber reliable" is an R&D problem (label data, try detectors, measure, repeat), not a one-shot coding problem.

#### Why not the alternatives
- **Option A (pure browser — OpenCV.js / tfjs / MediaPipe Web in WASM/WebGPU):** viable for the table/pose pieces, but "uber-reliable shot/miss detection" needs real multi-object tracking and likely a small detection model. Fighting in-browser latency, threading, and model perf puts the reliability bar out of reach for now. Good *long-term* option for on-device pose, not the place to start.
- **Option C (keep native C++):** the CV is good, but the UI is unsalvageable for your goals and C++ slows every future phase. Keep the C++ only as a **reference oracle** to validate the Python port produces identical homographies.

#### Trade-offs to go in eyes-open
- Hybrid adds a backend to run/host. Mitigation: it's a single local FastAPI process to start; you're the only user.
- Browser→server frame upload adds latency. **Mitigation that fits your setup:** for a fixed overhead tripod, have the **backend read the camera/RTSP directly** and stream annotations to the browser — the browser becomes a thin viewer/controller. Only fall back to browser-captured frames for casual laptop use.

#### Proposed architecture
```
Camera (fixed overhead tripod)
      │  (backend ingests directly — lowest latency)
      ▼
Python CV service
  • one-shot table calibration (cache homography)   ← from existing C++ algos
  • per-frame ball detect + multi-object track
  • event detection (shot / collision / pocket / make-miss)
  • emit structured events
      │  WebSocket (frames as MJPEG/WebRTC, events as JSON)
      ▼
Browser frontend (React/TS, PWA → installs on phone/iPad)
  • live video + Canvas/WebGL overlay (table, balls, trajectories)
  • game / drill / practice modes, table-state viz, stats dashboards
      │
      ▼
SQLite (sessions, shots, drills, stats)  → Postgres later if multi-user
      │
      ▼
Claude API (shot event stream → coaching feedback)
```

**Key principle the current app violates:** calibrate the table **once** per session, lock the homography, then spend the whole per-frame budget on balls.

---

## 8. Proposed roadmap

Effort = focused-engineer days. Dependencies are strict where noted.

| Phase | Goal | Effort | Depends on |
|---|---|---|---|
| **0 — Foundation/scaffold** *(new)* | Repo structure (Python CV service + FastAPI/WS + minimal web client). Port felt→corners→homography from C++. **Build a labeled fixture set + eval harness first** (record real videos: break, run-out, missed shots, varied lighting). | 3–5 | — |
| **1 — Rock-solid table detection** | Harden felt detection, corners, perspective transform. One-shot calibration + "calibration deviated" watchdog. Add **line/edge-based** detection alongside color for robustness. | 5–8 | 0 |
| **2 — Ball detection + tracking** ⭐ | Net-new linchpin. Ball detection (YOLO/seg or classical Hough+color), multi-object tracking (ByteTrack/SORT/Kalman) for stable IDs through occlusion, map to table coords via homography. **Hardest, highest-risk phase.** | 10–15 | 1 |
| **3 — Shot / miss detection** | Event layer: cue-strike, ball motion start/stop, collisions, pocket entry, make/miss classification. **Only as good as Phase 2.** | 7–10 | 2 |
| **4 — UX shell** | Web frontend, live overlay, game/drill/practice modes, table-state viz. | 10–15 | 3 (can start on stubbed events) |
| **5 — Stats + tracking** | DB schema; session/shot logging; drill progress; make/miss rates; trends over time. | 5–8 | 4 |
| **6 — AI coach** | Structured shot data → Claude API → feedback. | 4–6 | 3 + 5 |
| **7 — Autonomous shot clock** | Detect "player at table / shot taken," audio cue, overlay countdown. | 3–5 | 2 |
| **8 — Body fundamentals** | MediaPipe Pose; head-down / stance analysis. Likely needs a **second camera** at player height. Independent track, lower priority. | 8–12 | (independent) |

**Critical path to your stated "first":** reliable table + uber-reliable shot/miss = **Phases 1–3 ≈ 22–33 days**, gated almost entirely on Phase 2 (ball tracking) being hard. Phase 3 quality is capped by Phase 2 quality — invest there.

**To a genuinely useful daily tool (Phases 0–5):** ~45–65 focused days.

### Sequencing advice
- **Do Phase 0's eval harness before anything else.** "Uber reliable" must be a number you can watch move. Record 5–10 real clips and hand-label ball positions / shot outcomes on a sample of frames.
- Keep the **C++ app as a reference oracle** while porting Phase 1 — diff the homographies.
- Phase 2 is where the project succeeds or stalls. Timebox a spike: try ultralytics YOLO on a few hundred labeled frames vs. classical Hough+color, and pick based on measured precision/recall, not vibes.

---

## 9. Immediate, low-risk cleanups (whenever you start)
- Delete the stray `nul` file and the accidental `venv/`.
- Decide the fate of `CushionDetection/` (rename to `PlayableArea/`, and commit or remove).
- Strip per-frame `std::cout` from the hot path.
- Fix the README to describe what exists (no pocket detection; diamonds are real CV).

---

*Next step is yours: pick a path (I recommend Option B + Python port) and I'll scaffold Phase 0 — repo structure, the CV port skeleton, and the fixture/eval harness.*
