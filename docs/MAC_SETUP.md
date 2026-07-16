# Mac Mini setup — the billiards-room appliance

Turning a Mac Mini (Apple silicon) into the dedicated, always-on machine for
Billiards Trainer. One-time setup, then it just runs.

## Install

1. Download `BilliardsTrainer-mac-<version>.zip` from the
   [latest release](https://github.com/CaulfieldEngineering/BilliardsTrainer/releases/latest).
2. Unzip and drag **BilliardsTrainer.app** to **Applications**.
3. First launch: **right-click the app → Open → Open** (unsigned app; macOS
   asks exactly once).
4. Approve the two permission prompts on first use:
   - **Camera** — table tracking.
   - **Bluetooth** — the cue-stroke sensor.
   If a prompt was dismissed, re-enable under **System Settings → Privacy &
   Security → Camera / Bluetooth**.

## Camera

Any UVC USB camera works (AVFoundation). Mount it overhead/end-on as on the
Windows rig, plug it in, then pick it in **Settings → Camera**. On macOS
cameras are listed by index ("Camera 0…"), not by friendly name — if the wrong
one opens, try the next index.

## Appliance mode (recommended for a dedicated machine)

- **Auto-start**: System Settings → General → Login Items → **+** → BilliardsTrainer.
- **Never sleep**: System Settings → Energy → prevent automatic sleeping
  (or run `caffeinate -dis` at login). A sleeping Mini kills the camera feed.
- **Auto-login**: System Settings → Users & Groups → automatically log in —
  so a power blip brings the whole rig back without a keyboard.

## Updates

The app checks on launch and pops the release page when a newer build exists —
download the new zip and replace the app (drag over the old one). Self-install
is Windows-only for now; automating the `.app` swap is planned once it can be
validated on this machine.

## First-run checklist (things only real hardware can prove)

Verified in CI on Apple-silicon runners: the full test suite, the frozen
`.app` build, ONNX CPU inference. Check these on the Mini itself:

- [ ] Log line `ONNX detector … on CoreMLExecutionProvider` (in
      `~/Library/Application Support/BilliardsTrainer/logs/billiards_trainer.log`).
      If it says `CPUExecutionProvider`, report the fps — CPU may still be
      real-time on M-series; if not, turn **Settings → AI detection → Extra
      far-rail scan** off to halve detector cost.
- [ ] Live camera at ≥ 25–30 fps in the header readout.
- [ ] Cue sensor connects (Settings → Cue stroke sensor → enable; status shows
      *Connected* with battery). macOS identifies BLE devices by UUID rather
      than MAC — the first scan may take a few extra seconds.
- [ ] Shot-clock beeps audible through the connected speakers/display.
- [ ] Table calibration locks and survives an app restart.
