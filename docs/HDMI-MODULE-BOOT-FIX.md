> **SUPERSEDED 2026-07-29 — DO NOT IMPLEMENT.** The 480p reversions were NOT a
> module defect. Root cause (found by Joe, bench-reproduced): ML's crash guard.
> A hard power cut leaves `ML/MODULES/LOADING.LCK` behind, so the next boot
> reads it as "a module crashed me" and skips module loading entirely — no
> hdmi_out, no ROM hook, camera negotiates its default 480p. FIX (stock
> setting, already applied + verified by battery-pull): Modules tab -> Modules
> debug (Q) -> "Load modules after crash" = ON, then Prefs -> Config files ->
> Save config now. See docs/STATE-2026-07-24.md. Kept below for history only.

# hdmi_out.mo — make forced 1080i survive a camera power-cycle

**Repo:** `CaulfieldEngineering/ml-hdmi-out-600d` (private, CI cross-build)
**Body:** Canon 600D / T3i, firmware 102, ML build `magiclantern_simplified @ ed3e7c0dfa`
**Hook:** `Set_HDMI_Code = 0xFF1ED008` (dannephoto ROM-hook variant)

## The defect (measured 2026-07-28, Mac side)

The forced mode is applied by the MENU SELECT handler only. Config persists
(the menu still reads 1080i after a reboot) but nothing re-applies it when
Canon re-initializes the HDMI path at boot, so the wire falls back to 480p.

| State | Active picture in the 1920x1080 container |
|---|---|
| Module armed (healthy) | **1621x1080** — 3:2 image pillarboxed, fills FULL height |
| After camera power-cycle | 1710x875 (letterboxed) |
| After Cam Link USB replug | 1730x757 — classic 480p signature |

The capture device is exonerated: the replug forced a fresh EDID handshake
from the sink side and the camera still offered 480p.

## The change

1. **Factor the apply out of the menu handler.** Whatever the select/`update`
   callback runs today to poke `Set_HDMI_Code`, move its body into:

   ```c
   static void hdmi_apply_forced_mode(void)
   {
       /* existing poke: write the configured mode code via the ROM hook */
   }
   ```
   The menu handler now just calls `hdmi_apply_forced_mode()`.

2. **Re-apply after Canon's own HDMI init — DELAYED, not straight-line.**
   Poking during `init()` is what Canon overwrites. Use a task so the write
   lands after the firmware has finished bringing the port up:

   ```c
   static void hdmi_boot_apply_task(int unused)
   {
       msleep(4000);                 /* let Canon finish HDMI/LV init */
       if (hdmi_force_enabled)       /* the persisted config value */
           hdmi_apply_forced_mode();
   }

   static unsigned int hdmi_out_init(void)
   {
       /* ...existing init... */
       task_create("hdmi_boot", 0x1e, 0x1000, hdmi_boot_apply_task, 0);
       return 0;
   }
   ```
   If 4000 ms proves too early on a cold boot, retry a few times instead of
   guessing one delay:

   ```c
   for (int i = 0; i < 6; i++) { msleep(2000); if (hdmi_force_enabled) hdmi_apply_forced_mode(); }
   ```
   (Re-poking an already-correct mode is harmless.)

3. **Also re-arm on hot-plug** (cable/monitor reconnect re-runs Canon's init).
   Add a property handler on the HDMI-connect property and call the same
   function — e.g.:

   ```c
   PROP_HANDLER(PROP_HDMI_CHANGE)      /* or PROP_HDMI_CHANGE_CODE on this body */
   {
       if (hdmi_force_enabled)
           hdmi_apply_forced_mode();   /* schedule via task if the handler must stay short */
   }
   ```
   Property-handler context is restricted — if the poke is heavy, set a flag
   and let a small polling task in step 2 do the write.

4. **Config persistence:** confirm the mode is a `CONFIG_INT` so ML restores
   it at load; step 2 reads that restored value. (The menu already shows the
   right value after reboot, so this part is already working.)

## Build + install

CI cross-build as before -> `hdmi_out.mo` -> copy to the card's
`ML/modules/`, boot the camera, verify the module loads and the menu still
shows 1080i60.

## Acceptance test (Mac side, no camera contact)

With the camera on the ceiling and the app closed or idle:

```bash
cd ~/Documents/GitHub/_CaulfieldEngineering/BilliardsTrainer
.venv/bin/python tools/feedmeter.py --source 0 --headless 8 --label "post-boot no toggle"
```

**PASS:** `active=1621x1080` (full height) with NO menu toggle after a power
cycle. **FAIL:** any letterboxed geometry (1730x757, 1710x875, ...).

The app also flags this automatically now: any feed that is not full-height
turns the LIVE pill and the corner stats chip amber within ~5 s.

## Note: possible root cause upstream of all this

Camera-direct-to-TV read 480p in ALL modes including playback (a healthy 600D
does 1080i playback) = the camera cannot read the sink's EDID at all. Prime
suspect is the mini-HDMI cable's DDC line. If a known-good cable restores
native 1080i negotiation, this module patch becomes a belt-and-braces backup
rather than the primary fix. Worth swapping the cable during the same trip.
