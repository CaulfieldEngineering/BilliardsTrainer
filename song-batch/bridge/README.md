# Cubase MIDI Remote bridge

**Not built yet.** Build-order item 6, and the gate on all of Tier 3.

## Why this needs designing before it needs writing

The MIDI Remote script sandbox (Cubase 15, API 1.3) has **no HTTP and no IPC**.
MIDI is the only transport in and out. Everything Tier 3 wants to send -
plugin UIDs, slot object IDs, parameter names, displayed values like `"8000 Hz"`
- is wider than the 7 bits a CC gives you.

So the transport is **multi-byte SysEx with explicit command/response
correlation**. Not one-CC-per-command. A developer building a similar controller
reportedly lost two days to exactly this. Design it once, write it once, never
touch it again.

## Constraints to design against

- 7-bit data bytes. Anything with the high bit set must be encoded (7-in-8
  packing, or nibble-split - pick one and document it).
- SysEx has no built-in request/response pairing. Every command carries a
  sequence number; every response echoes it. Without this, an async reply from
  one command gets attributed to another.
- The script is ES5. No `let`, no arrow functions, no `Promise`.
- Commands can fail silently inside the host. Every command needs a timeout on
  the daemon side and an explicit `ERR` response shape.

## Known hard blockers in API 1.3 - do not design around these, they are walls

- No track creation.
- No audio region rendering.
- No external process communication beyond MIDI.

Because of the last one, **the audio feedback loop does not run through Cubase
at all** - it runs through our own headless host. See CLAUDE.md.

## To verify before writing code

- Exact `trySetSlotPlugin` signature and calling convention. The reference docs
  lag the capability and the discoverable methods do not obviously lead there.
  Route is: `da.mPluginManager` on the selected track's DirectAccess ->
  `getPluginCollectionByIndex` etc. for enumeration; `makeInsertEffectViewer`
  -> `accessSlotAtIndex` / `getRuntimeID` for slot object IDs.
  Budget real time for the first working call.
- Whether the docs' browser Playground (alpha) is usable from a phone. If it is,
  it is the fastest iteration loop available for this tier.

Docs: https://steinbergmedia.github.io/midiremote_api_doc/
