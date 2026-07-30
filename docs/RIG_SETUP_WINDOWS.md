# Windows rig setup — the billiards-room appliance

Turning the wall-mounted mini PC into the dedicated, zero-friction machine for
Billiards Trainer. Written for the **GMKtec K8 Plus** (Ryzen 7 8845HS, Radeon
780M, 32 GB DDR5, 512 GB) bought 2026-07-29 after the Mac mini M1 died of ANS2
storage panics — see `STATE-2026-07-24.md` for that decision record. Anything
here applies to any Windows mini PC.

**In the box** (verified from the spec sheet): the PC, a 120 W power brick, a
**VESA mount + screws**, an HDMI cable, manual. The only extra worth having is
an **HDMI dummy plug (~$8)** if it will run headless — without a display
attached Windows can pick a silly resolution and Remote Desktop gets awkward.

---

## 0. Acceptance test FIRST (inside the 30-day return window)

Do this before mounting anything. The question is only: *can this hardware run
the pipeline we already have?*

```powershell
git clone https://github.com/CaulfieldEngineering/BilliardsTrainer
cd BilliardsTrainer
python -m venv .venv
.\.venv\Scripts\pip install -r requirements-dev.txt
.\.venv\Scripts\pip install -e .
# put a test clip at testVideo.MP4 (or pass --video <path>)
.\.venv\Scripts\python tools\bench_pipeline.py
```

`bench_pipeline` prints per-stage latency (detect / track / motion / render).

**Keep/return line: ~30 fps end to end with the far-rail rescan ON.**
If it lands short, re-run with `--no-rescan` to see how much the second
inference pass is costing before deciding — the reference numbers from the
RTX 4060 laptop are ~19 ms/frame with rescan and ~10 ms without.

Also confirm, while the box is still returnable:

- **Windows activation**: Settings → System → Activation must read *"activated
  with a digital license."* Grey-market keys turn up in this product category
  and are a clean return reason. (One reviewer received a Korean-language
  install — check the language too.)
- **Leave it running 24 h** under load. The 12 % one-star rate on these boxes
  is mostly DOA/unstable units; find that out now, not in month two.
- **Fan noise** at the table. Reports vary unit to unit. If it whines, return it.

---

## 1. BIOS

Reboot and tap `Esc`/`Del` (GMKtec: `Esc`).

- **Auto Power ON → Enabled** — the camera rig hard-cycles (smart plug,
  overheating), so the PC must come back by itself. This is a listed feature on
  the K8 Plus.
- **Restore on AC Power Loss → Power On** (same idea, other vendors' wording).
- **UMA Frame Buffer Size → 4 GB** (Advanced → GFX Configuration → iGPU
  Configuration → `UMA_SPECIFIED`). Gives the 780M dedicated headroom for ONNX
  without starving the OS. Raise toward 8–16 GB only if you later run local
  LLM inference on the iGPU.
- **Power/Performance mode → Balanced** (54 W). Wall-mounted in warm air,
  Performance (70 W) buys little and heats more. Revisit if the benchmark is
  borderline.

## 2. Windows, appliance-ified

```powershell
# never sleep, never blank, no hibernation games
powercfg /change standby-timeout-ac 0
powercfg /change monitor-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /h off
```

- **USB selective suspend OFF** — Control Panel → Power Options → Change plan
  settings → Advanced → USB settings → USB selective suspend → **Disabled**.
  This one matters: it can drop the Cam Link mid-session.
- **Auto-login**: `netplwiz` → uncheck *"Users must enter a user name and
  password."* Needed so a power cycle lands on a running app, not a lock screen.
- **App at startup**: put a shortcut in
  `shell:startup` (`%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup`).
- **Windows Update**: set **active hours** and turn off auto-restart, so it
  never reboots mid-session.
- **Wired Ethernet, WiFi off.** A reviewer measured this model's WiFi card at
  105 °C under sustained wireless load; it has dual 2.5 GbE, and wired is what
  an appliance wants anyway. Bluetooth stays on for the cue sensor.

## 3. Install the app + migrate the data

Either install the released exe (self-updating, checksum-verified — the
Windows build swaps itself and rolls back on failure) or run from source with
`run_dev.ps1`.

Restore into `%LOCALAPPDATA%\BilliardsTrainer\`:

| File | What it is |
|---|---|
| `billiards.db` | every session and shot ever recorded |
| `calibration.json` | table locks (per source; will re-lock anyway, see §5) |
| `settings.json` | all app settings |
| `models\*.onnx` | the fine-tuned ball-ID + position models |

Then point recordings at synced storage: **Settings → Recording → directory**
(`RecordingSettings.directory`) → your Dropbox folder. Keeps the 512 GB SSD
from filling and backs sessions up for free. Note: **the T3i sends no audio
over HDMI**, so without a separate mic the audio track is silence.

## 4. Camera rig

The camera side needs nothing new — the HDMI fix lives on the SD card. Just
confirm, per `STATE-2026-07-24.md`:

- ML → **Modules → Modules debug → "Load modules after crash" = ON**
  (this is what makes forced 1080i survive the rig's hard power cuts)
- ML → **HDMI output = ON, 1080i 60 Hz**
- ML → **Prefs → Config files → Save config now** (plug-cuts skip the clean
  shutdown that would otherwise persist settings)
- **Never plug USB into the camera while HDMI is the feed** — on this body USB
  kills HDMI output.

Verify the wire is HD before trusting anything:

```powershell
.\.venv\Scripts\python tools\feedmeter.py --list
.\.venv\Scripts\python tools\feedmeter.py --source 0 --headless 10 --label "rig install"
```

Healthy: container `1920x1080@60`, **active picture ~1633x928** (or 1621x1080 —
full-height 3:2). A letterboxed `1730x757` means the wire fell back to 480p.
The app also flags this itself: the feed-stats chip on the camera view
(`ui.feed_stats`, on by default) turns amber on SD fallback.

## 5. Recalibrate, then the two open measurements

The active picture geometry changed with the 1080i fix, so the table wants a
fresh lock — just let felt detection run, or use **Pick felt** if the read
looks off. Expect a new calibration entry; the old Mac-era lock won't fit.

Still owed from the HDMI work, both one command each:

1. **Effective-lines verdict** on a focused, well-lit table:
   `feedmeter --source 0 --headless 10 --label "on-table focused"`.
   Baselines: ~330 (old 480p wire), ~704 (USB tether), **700+ = HDMI wins**.
2. **Comb-under-motion**: run the meter while balls roll. A `comb` spike means
   forced 1080i is arriving weaved and the pipeline needs a deinterlace step.

## 6. Mounting

- VESA bracket is in the box.
- **Leave breathing room** — these pull intake air from the bottom/side; flush
  against a wall restricts it.
- **Mount low if you can.** Ceiling-height air plus sustained GPU inference is
  how you lose detection frames quietly. The M1's death was helped along by
  heat and constant power cycling; don't repeat the pattern.
- Plan a home for the external power brick and the Cam Link's USB run.

## 7. Growth path (why this box)

- **RAM**: 32 GB now, two SO-DIMM slots, 96 GB ceiling.
- **Storage**: second M.2 slot free, 8 TB ceiling.
- **GPU**: **OCuLink** (PCIe 4.0 ×4, 64 Gbps) — a dock plus any desktop card
  when live inference outgrows the 780M. Budget for dock **and** its own PSU,
  not just the GPU. USB4 ×2 is the slower alternative.
- **Local LLM** (fits the project's no-cloud rule): ~10–15 tok/s on 8 B Q4,
  30 B MoE around 15–20 tok/s. Run it **post-session or idle**, never during
  live tracking — LLM generation and ONNX inference contend for the same
  memory bandwidth.
