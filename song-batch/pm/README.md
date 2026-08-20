# Project management sync

**Not built yet.** Build-order item 4. This file records the decisions already
made so that whoever builds it (probably Claude, from a phone) does not have to
re-derive them.

## Settled

- **`spec.yaml` is the master. The PM tool is a view.** Status lives in git next
  to the song. One-way sync, repo -> PM, by default.
- Two-way sync only if it can be made genuinely reliable. It probably cannot -
  reconciling a Notion edit against a git edit needs a conflict story, and there
  isn't one. Assume one-way.
- **The phone-first views must work from the repo alone**, with the PM backend
  unreachable. `./sb status`, `--blocked`, `--ready-for <stage>` and `--json`
  already do this. Anything the PM layer adds is a convenience on top, never a
  dependency.
- Backend-agnostic behind a thin interface here in `pm/`. Notion first, because
  it is already connected as an MCP. ClickUp second if ever.

## To build

- `pm/backend.py` - the interface: `push(specs) -> None`, `describe() -> str`.
- `pm/notion.py` - the Notion implementation.
- `pm/dashboard.py` - static HTML/markdown generator. Worth doing *first*: it
  has no external dependency, no auth, and answers the same four questions.
- `./sb pm push` / `./sb pm dashboard` wired into `lib/cli.py`.

## The four questions it must answer

1. What is the state of all 20?
2. What is blocked?
3. What is ready for guitar tracking?
4. What changed since yesterday?  (git log over `songs/*/spec.yaml` answers this
   one for free - do that rather than storing history in the PM tool.)
