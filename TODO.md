# TODO / parked work items

Not a backlog of committed work — a place to capture things deliberately
deferred so they aren't lost. Nothing here is in progress.

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
