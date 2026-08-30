"""Trail overlay rules, from Joe's 2026-08-30 report.

  "First clip of pinned session has wild inaccurate trails"  (shot 1/12,
  REARRANGING - long straight lines across the whole table, which are
  Joe's HANDS carrying balls plus the identity swaps that causes, drawn
  with the same glowing lines that everywhere else mean "a struck ball
  went here")

  "trails still do not follow when I'm scrubbing the transport playhead.
  They need to stay in sync during scrubbing"  (the render loop blanked
  the overlay for the whole drag, because the video clock is parked
  while the cached-frame cover owns the picture)

Both verified with tools/phone_view.py --local before shipping.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "companion-cloud" / "public" / "app.js"


@pytest.fixture(scope="module")
def app_js() -> str:
    return APP.read_text(encoding="utf-8")


def _code(js: str) -> str:
    return "\n".join(ln.split("//")[0] for ln in js.splitlines())


def test_no_shot_trails_on_a_non_stroke_event(app_js):
    """A table change has no shot, so it has no shot trails."""
    i = app_js.find("function drawTrails(")
    assert i > 0, "drawTrails vanished"
    body = _code(app_js[i:i + 2600])
    for act in ("rearrange", "ball_in_hand", "nothing"):
        assert f'"{act}"' in body or f"'{act}'" in body, (
            f"drawTrails no longer refuses {act!r} - the rearranging clip "
            f"will draw hand-carries as if they were a shot again")
    assert "return" in body.split("rearrange")[1][:120], (
        "the non-stroke check must RETURN, not merely branch")


def test_scrubbing_draws_at_the_shown_frames_time(app_js):
    """The overlay must follow the thumb, not blank for the whole drag."""
    code = _code(app_js)
    assert "shownTime" in code, (
        "FrameCache no longer publishes the frame it painted, so the "
        "overlay has no time to draw at while scrubbing")
    i = code.find("const scrubT")
    assert i > 0, "the render loop no longer reads a scrub time"
    seg = code[i:i + 700]
    assert "FrameCache.shownTime()" in seg
    assert "drawTrails(scrubT" in code, (
        "drawTrails is not given the scrub time - trails will lag or blank")


def test_blanking_still_applies_when_the_time_is_unknown(app_js):
    """The protection this relaxes must survive: a shot swap in flight,
    or a seek with no cover up, still blanks - a trail at the wrong time
    is worse than no trail."""
    code = _code(app_js)
    i = code.find("const picReady")
    assert i > 0, "picReady vanished"
    seg = code[i:i + 400]
    assert "swapInFlight" in seg, "swap-in-flight blanking was dropped"
    assert "readyState" in seg, "the readyState guard was dropped"


def test_the_cover_forgets_its_time_when_hidden(app_js):
    """Otherwise a stale shownTime outlives the cover and the overlay
    draws at a moment that is no longer on screen."""
    code = _code(app_js)
    i = code.find("hide: ()")
    assert i > 0, "FrameCache.hide vanished"
    assert "shownT = null" in code[i:i + 200], (
        "hiding the cover must clear the published time")
