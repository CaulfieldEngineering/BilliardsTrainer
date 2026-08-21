# BilliardsTrainer Player Redesign Spec

Report keys: **R1** = sports-analysis apps (OnForm/CoachNow/SwingVision/Hudl/Dartfish), **R2** = consumer player gestures (YouTube/TikTok/Netflix/VLC/iOS Photos/NN-g), **R3** = Apple HIG / Material 3, **R4** = pro review tools (Frame.io/LumaFusion/Resolve/CapCut).

## 1. Core model: four states, one surface

The player is a state machine keyed to playback state — not four screens:

- **WATCH** — playing. Nearly chrome-free. Default.
- **INSPECT** — paused. Tap-to-pause *is* the mode switch: pausing reveals the precision chrome (R2: never auto-hide controls while paused — paused is the primary inspection state; R4 CapCut: selection state, not navigation, decides the tools).
- **ANNOTATE** — explicit modal mode entered via the pen; toolbar swap per R3 HIG ("photo-markup" is the canonical modal-toolbar example) and R1 (tools behind a single pen icon).
- **VERDICT** — medium-detent sheet over the paused frame (R3 detents = the platform's named progressive-disclosure mechanism; R4 Frame.io: verdict is one-tap structured metadata, separate from notes).

Opening ANNOTATE or VERDICT while playing **auto-pauses at the current frame** (R4 Frame.io auto-pause-on-analysis-intent); confirming a verdict **auto-advances to the next shot and resumes play** (R4: send-resumes; Resolve Fast Review throughput). At clip end, WATCH auto-advances to the next shot — the session plays as one continuous tape (R4 Resolve source tape kills list-tap-back-tap triage).

## 2. Persistent buttons (exactly 4, visible in all states except Annotate)

| Button | Position | Action |
|---|---|---|
| **‹ Back** | top-left | exit to session list (absorbs title-bar hamburger's nav role) |
| **Outcome badge** | top-center, next to "Shot 7/23" | tap → Verdict sheet (R4 Frame.io one-tap status; "a verdict that takes more than one tap will simply not get recorded") |
| **Pen** | floating, right edge lower third, translucent | enter Annotate; auto-pauses if playing (R4) |
| **… More** | top-right | More sheet (R1: file/metadata actions live in a top-corner overflow) |

Small, translucent, always visible — matches short-form players (TikTok/Reels keep their minimal action buttons up during playback, R2). Everything else is contextual.

## 3. Gesture map (all conventional, none invented)

| Gesture | Zone | Action | Replaces | Convention source |
|---|---|---|---|---|
| Tap | center/anywhere on video | play ⇄ pause (pause reveals Inspect chrome) | current tap — **kept** | short-form tap=pause (TikTok, Instagram 2026) — R2 |
| Double-tap | left third | jump back ~1s (floor: shot start) — i.e. instant replay of the stroke | **Repeat button** (replay use) | YT/Netflix side double-tap; increment scaled to content, "10s traverses an entire pool shot" — R2 |
| Double-tap | right third | jump forward ~1s | — | same — R2 |
| Long-press (hold) | anywhere on video | ¼× slow-mo **while held**, "¼×" pill, release restores 1× | **Speed button** (transient use) | YT/TikTok/IG hold-speed-modifier, direction inverted for analysis — R2 |
| Long-press → drag | anywhere on video | gated scrub-anywhere; full screen width = full clip | coarse scrubbing | YT slide-to-seek gate; iOS 17 Photos removed the *ungated* version — R2 |
| Horizontal flick | video | previous / next shot | **Prev-shot / Next-shot buttons** | Frame.io iOS: swipe between assets vs. scrub within one — two axes, two gestures — R4 |
| Pinch; drag while zoomed | video | zoom / pan (verify pocket rattles, read spin) | new capability | YT zoom-to-fill; OnForm/Hudl headline feature — R2 |
| Tap | drawn object | select → floating style pill | **kept as-is** | R1: annotations are editable objects |

Skipped deliberately: vertical edge swipes (VLC volume/brightness lineage — collides with OS gestures, never generalized, R2). Every gesture gets a one-time single-line transient hint on first use (NN/g, R2; R1 discoverability antipattern) and has a visible ≤2-tap fallback (§7).

## 4. Layouts, top to bottom

### WATCH (playing — default)
1. Top overlay row (translucent gradient, safe-area): `‹` · `Shot 7/23` + outcome badge · `…`
2. Full-bleed portrait video; AI overlays (aim line / ball paths) render per layer settings.
3. Pen button floating on right edge (R1: annotation lives on a side edge; keeps top-of-frame — far rail and pockets — clear of chrome, R2 thumb-zone).
4. Bottom of video: **2px shot-scoped progress hairline** with a tiny playhead. Nothing else (TikTok hairline, R2).

That is the entire watch UI. Chrome-free per requirement; the footage is the workspace (R1).

### INSPECT (paused)
1. Top row: unchanged.
2. Video: unchanged; pinch-zoom + pan active.
3. Thin translucent shot-info strip over the video's bottom edge: ball badges + one-line description (R1: thin auto-metadata overlay only; details live in the sheet).
4. **Bottom-anchored precision cluster** (thumb zone — 96% vs 61% tap accuracy, R2), replacing the hairline:
   - **Session strip** — every shot as a state-colored chip/tick (grey = unreviewed, green/red = outcome, dot = has note), current shot enlarged; tap to jump. Absorbs the current shot-chip strip. (R4 Resolve dual timeline: fixed overview, never pinch-zoom; R4 LumaFusion: media wears its reviewed state.)
   - **Shot scrub lane** (coarse; full width = clip) with **event markers** (cue contact, first object-ball contact, pocket/rest) and note dots. Drag-only — no tap-to-seek (YT removed it: stray taps must never teleport the playhead, R2). Absorbs the current scrub lane. (R2 chapters-as-seek-targets; R4 Frame.io timecode markers.)
   - **Frame lane** (fine; 1 tick = 1 frame, haptic detent per frame, frame#/ms readout while dragging). (R1 OnForm two-resolution scrubbing — one scrubber cannot serve both "get to the shot" and "find the contact frame"; R2: never demand frame accuracy from a bare 1:1 drag.)
   - **Transport row, 4 buttons ≥44pt:** `[−1 frame] [play] [+1 frame] [speed chip]`. Frame-step buttons are mandatory — "every app studied has them" (R1). Speed chip: tap cycles 1× → ½ → ¼ → ⅛ (discrete presets, never a slider — all five R1 apps converged); **long-press opens a popover with the four speeds + Loop toggle** (long-press = secondary-action verb, R4 Resolve iPad). The Loop toggle is where the current **Repeat button** lands (OnForm keeps loop in the bottom time strip, R1).

### ANNOTATE (pen)
1. Top row swaps (R3 modal toolbar swap; visually distinct tint per M3 "vibrant" edit-mode): `[Undo]` left · **"Annotate"** label center (visible mode indicator — R4 CapCut antipattern: silent contextual morphing confuses) · `[Done]` right (persistent escape hatch — R4 "stuck in full-screen" antipattern).
2. Video = **canvas only**. Hard boundary: the canvas never scrubs, the lanes never draw — this is the exact OnForm flywheel/draw-collision complaint (R1 antipattern).
3. Right-edge vertical rail (expands from the pen's position): `line` · `protractor` · `show/hide` · `clear`. Entering arms the last-used tool so pen-tap = draw immediately. (R1: CoachNow right-side rail, undo opposite; nested disclosure.)
4. Drawn objects keep draggable endpoints, live degree readout, and the existing floating style pill (colors/weight/opacity/delete) on selection — already matches R1's editable-objects pattern; keep verbatim.
5. Bottom: frame lane + `[−1][+1]` remain, so you can step to the contact frame mid-markup without touching the canvas.

*Deliberate deviation, flagged:* Frame.io hides annotations except when their comment is active (R4). Here drawings stay visible during replay **of their own shot**, because the user's core annotate use is "draw the reference line, then replay against it." Mitigation: drawings are shot-scoped, never leak across shots, and show/hide + clear are one tap inside the rail.

### VERDICT (sheet — tap outcome badge from any state)
Medium detent (~45%), grabber, paused frame visible above (R3: medium detent is Apple's literal progressive-disclosure example; video never scrimmed by a companion surface — M3 standard-sheet model):
1. Outcome chips row — one-tap correction (R4).
2. Action / shot-type correction chips.
3. Note field — note is pinned to the current frame and becomes a dot on the scrub lane; tapping the dot later seeks to it (R4 Frame.io timecode-anchored comments).
4. **`Confirm reviewed`** — the single prominent action (R3: one primary action per surface). Confirm → dismiss → auto-advance → play (R4).
5. Drag to full detent: `Correct this clip` (boundary/detection fix — the mandatory correction affordance per R1's SwingVision antipattern), `Remove shot` (red, confirm via action sheet, R3).

### MORE sheet (…)
Modal bottom sheet, full-width rows (R3: menus become bottom sheets on compact screens):
`Save clip to playlist` · `Aim line` ✓ · `Ball paths` ✓ · `Correct this clip` · `Export/share clip` · `Remove shot` (red, last). Overlay rows are full-row tap targets with checkmarks, **not** inline switches (M3: no live controls inside menu items, R3). Inapplicable rows disable, never disappear (R3). This sheet *is* the old hamburger, minus the drawing tools it wrongly contained.

## 5. Disposition of every current control

| Current control | New home |
|---|---|
| Tap video = play/pause | **Kept** (already the short-form convention, R2) |
| Prev / next shot buttons | Horizontal flick; fallback: session strip (Inspect) |
| Frame-back / frame-fwd | Kept as buttons in Inspect transport (R1/R2 insist) |
| Play button | Kept (center of Inspect transport) |
| Repeat button | Double-tap-left replay + Loop toggle in speed popover |
| Speed cycle | Speed chip (Inspect) + long-press-hold ¼× gesture |
| Scrub lane | Kept, Inspect-only, + event markers + frame lane below |
| Shot-chip strip | Session strip in Inspect cluster, state-colored |
| Details button (full-width) | Deleted; outcome badge tap opens Verdict sheet |
| "Correct this clip" | Verdict sheet (full detent) + More sheet |
| Shot info row | Split: counter+badge top overlay; badges+description in Inspect strip; full detail in Verdict sheet |
| Hamburger: playlist, overlays | More sheet |
| Hamburger: drawing tools | Annotate mode rail via pen button |
| Style pill | Kept verbatim |
| Title-bar hamburger | ‹ Back + More sheet |

## 6. Convention conflicts in the current design (flagged)

1. **Full-width Details button** — a once-per-shot action holding the screen's highest-frequency position; violates frequency-ranked toolbars (R3) and eats video pixels, "the most-punished sin in this category" (R1 Dartfish).
2. **Drawing tools inside the hamburger sheet** — conflicts with the universal pen-icon-enters-a-mode convention (R1 OnForm/CoachNow) and with R3's nonmodal-palette rule: a modal sheet occludes the very canvas it edits.
3. **Permanently visible 7-button transport + chip strip + Details row** — every modern player hides chrome during playback (R2); persistent stacks read as pre-2015.
4. **Repeat as a top-level button** — no studied player surfaces loop at top level; loop belongs with speed in the time strip (R1 OnForm).
5. **Overlay toggles as switches in a menu** — M3 explicitly forbids embedded controls in menu items (R3); use tappable checkmark rows.
6. **Already compliant, keep:** tap=play/pause (R2), discrete speed presets (R1), bottom-anchored scrub lane (R2), protractor live degrees + object-style pill (R1 editable-objects baseline).

## 7. ≤2-tap audit (from WATCH, worst case)

play/pause 1 · replay stroke 1 (double-tap) · prev/next shot 1 (flick) · verdict sheet 1 · outcome correction 2 · confirm reviewed 2 · frame-step 2 (tap-pause, tap-step) · speed change 2 · loop 2 (long-press chip, tap) · jump to any shot 2 (pause, chip) · draw (last tool) 2 (pen auto-arms) · show/hide / clear drawings 2 · style pill 1 (tap object) · playlist / overlay toggles / correct clip / remove / export 2 (More or badge, then row). Temporary slow-mo, scrub-anywhere, zoom: 0 taps (gestures) with the chip/lanes as visible fallbacks — no gesture-only capability exists (NN/g, R2).
