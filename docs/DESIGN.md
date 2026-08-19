# Design language

Distilled from a review of current dark-UI and data-table practice
(LogRocket dark-mode guide, Setproduct data-table reference, UX
Collective dark-UI principles) after Joe's Aug 19 feedback rounds.
These are BINDING rules for UI work in this repo — new UI ships only if
it obeys them, and several are pinned by tests.

## Surfaces
- FLAT panels: one background step (`bg_elevated`), hairline border
  (`border_soft`), near-square corners (2px). No gradients, no floating
  rounded cards, no drop shadows.
- Regions read as LABELED PANES divided by hairlines — every pane gets a
  small-caps caption (BIRD'S-EYE, SHOT TIMELINE...), content inside.
- Boxes never nest visibly: a tile inside a panel has NO chrome of its
  own (transparent, borderless). One container level per region.

## Selection & hover — ONE surface, ever
- Selection is a single subtle full-row tint (`surface_hi`). Hover is a
  lighter tint (`surface`). That is the whole vocabulary.
- NEVER a second highlight inside a highlight (the blue-box-on-blue-row
  method is banned — Joe: "never use this text-highlight background
  color method anywhere ever again").
- No focus rectangles on list cells; `outline: none` on views.
- Loud accent backgrounds are for primary actions only, never for rows.

## Data tables & lists
- Text left-aligned; numbers right-aligned so digits line up by place
  value. Dates one consistent format per table.
- NO redundant columns: if two columns always say the same thing, one
  of them dies (session Name duplicated Date — it died).
- Truncation is a design failure, not a text mode: if a column clips at
  the pane's minimum width, redesign the columns. Tooltips may carry
  the long form (full filename lives in the row tooltip).
- Sorting stays on headers; sort by hidden numeric roles, never by the
  display string.

## Density & type
- Tight by default: page margins 10/8, panel padding 6-8, list rows one
  line. Whitespace is spent INSIDE content, not around chrome.
- Hierarchy through size/weight steps (12px caps captions, 13-14px
  body, 18px stat values), not through boxes or colour blocks.
- Values dim to `text_faint` when inactive; colour only when live.

## Process
- Every UI change ships with a pinned test for the rule it enforces.
- Verify layout with an offscreen full-window render BEFORE shipping —
  agreement between the code and my intent is not verification; only
  pixels are.
