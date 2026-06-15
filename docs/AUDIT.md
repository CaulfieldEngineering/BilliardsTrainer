# Speculative-code audit (v0.1.5)

Per the standing review mandate: flag code that isn't pulling its weight yet,
given how little has been tested on a real table. **Nothing here was removed** —
this is a watch-list. Items are ranked by how speculative they are. The reliable
core (table detect → rectify → ball detect → track → make/miss → DB → sandbox
stats) is exercised end-to-end by tests and the demo; everything below is either
secondary to Joe's sandbox-first priority or built ahead of proven need.

## Built ahead of need (keep, but unproven)
- **`pose/` (MediaPipe body fundamentals).** Fully written, optional, and **not
  wired into the UI or pipeline**. Needs a second player-height camera to be
  useful. Lowest-priority surface; safe to leave dormant. Verdict: keep, dormant.
- **`vision/balls.py::YoloBallDetector` + auto-fetch.** Cannot run without the
  `[yolo]` extra (torch) AND a weights file, and no turnkey public pool-ball
  model exists (Roboflow needs a per-user API key). It's a wired-up slot, not a
  working feature. Verdict: keep as the upgrade path; documented in BLOCKERS.
- **Ball class granularity (cue / solid / stripe / 8).** The make/miss core only
  needs "cue vs object" + "which pocket". Stripe-vs-solid-vs-8 classification is
  extra precision that is unverified on real balls and not required by sandbox.
  Verdict: keep (cheap), but don't trust the stripe/solid split yet.

## Secondary to sandbox (off by default, less tested)
- **Drills (`game/drills.py`, Drills page, practice/drill modes).** Wired but
  untested by Joe; sandbox is the priority. Verdict: keep, secondary.
- **Shot clock.** Now a true no-op when disabled (off by default) and hidden in
  the sandbox rail. Verdict: keep, opt-in.

## Dead / unwired settings (decide: implement or drop)
These config fields exist but nothing consumes them yet — they're inert toggles:
- **`UiSettings.mirror_preview`** — never applied (no horizontal flip happens).
- **`BallSettings.cue_speed_strike`** — the shot detector keys off `stop_speed`,
  not this. Currently unused.
- **`TableSettings.nose_inset_frac`** — playable-area inset is computed nowhere.
- **`RectifySettings.margin_scale`** — rectification uses pad + aspect only.
- **`PoseSettings.enabled`** — pose isn't wired, so this does nothing.

Recommendation: leave them (harmless, forward-compatible in the settings schema)
but don't surface `mirror_preview`/`cue_speed_strike` in the UI until wired, to
avoid implying they work. They are intentionally NOT exposed in the v0.1.5
settings page except where functional.

### Source/camera UX (reviewed in v0.1.7 — flag only, not fixing)
While replacing the integer source field with the camera dropdown:
- **Resolution is fixed.** `CameraSource` requests 1280×720 and takes whatever the
  camera clamps to — there is no resolution selector. Fine for most webcams; a
  4K cam would be downscaled-by-request. Low priority; add a resolution combo
  later if a camera needs it.
- **No rotation / flip control.** `mirror_preview` exists but is still inert (see
  above), and there's no 90°/180° rotation for an oddly-mounted camera. If Joe's
  overhead rig ends up rotated, this becomes worth wiring (rotation must also be
  applied before calibration so the homography is consistent).
- **`Use a file…`** in the new picker still accepts video/image paths (kept for
  testing with `testVideo.mp4`) — working, just secondary to the camera flow.

None block the core camera-selection fix; noting them per the review mandate.

## Reliability risk to watch (not dead code, but worth a real-table pass)
- **Phantom shots in free play.** The shot detector counts a MISS whenever balls
  move and stop with nothing pocketed — including hand-repositioning the cue
  ball. On a real table this can inflate the miss count. Tuning (`stop_speed`,
  `min_shot_frames`, a cue-strike signature) needs Joe's footage. The structured
  shot log (`%APPDATA%/BilliardsTrainer/logs/shots.jsonl`) + Save-last-5s replay
  exist specifically to gather that data.
