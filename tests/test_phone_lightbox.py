"""A picture must never be able to strand the reader.

Joe, 2026-08-30: "In the dev journal, when I click into a picture
there's no Back or exit button. I'm stuck."

Both journal image paths called `window.open(src, "_blank")`. In the
installed app that hands the reader a chromeless view with no browser
chrome and no way back. The fix is an in-page lightbox with FOUR
independent exits - the Close button, tapping outside the picture,
Escape, and the phone's own Back button (which works because opening
pushes a history entry).

These pin the exits, not the styling. If any one of them is removed the
reader can be trapped again, which is the whole bug.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "companion-cloud" / "public" / "app.js"
HTML = ROOT / "companion-cloud" / "public" / "index.html"


@pytest.fixture(scope="module")
def app_js() -> str:
    return APP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def html() -> str:
    return HTML.read_text(encoding="utf-8")


def _code_only(js: str) -> str:
    """Drop // comments - the rule is about what RUNS, and the comments
    quote the very call they exist to warn about."""
    return "\n".join(ln.split("//")[0] for ln in js.splitlines())


def test_journal_pictures_never_navigate_away(app_js):
    """No image handler may hand the reader off to a bare browser view."""
    code = _code_only(app_js)
    for m in re.finditer(r"window\.open\([^)]*\)", code):
        line_start = code.rfind("\n", 0, m.start()) + 1
        context = code[max(0, line_start - 600):m.end()]
        assert "openLightbox" in context, (
            "an image path calls window.open again - that is the trap: "
            "in the installed app it opens a chromeless view with no "
            "back button. Only the no-modal fallback inside openLightbox "
            "may call it.")


def test_both_image_paths_use_the_one_opener(app_js):
    """The rich-HTML images and the attached figures share ONE opener."""
    assert app_js.count("openLightbox(") >= 3, (
        "expected both journal image paths plus the definition")
    assert "function openLightbox(" in app_js


def test_all_four_exits_are_wired(app_js):
    """Close button, tap-outside, Escape, and the phone's Back."""
    assert 'lb.addEventListener("click"' in app_js, "tap-outside exit gone"
    assert 'ev.key === "Escape"' in app_js, "Escape exit gone"
    assert 'window.addEventListener("popstate"' in app_js, (
        "the phone's Back button no longer closes the picture - it will "
        "leave the app instead")
    assert 'history.pushState({ lb: 1 }' in app_js, (
        "without a pushed history entry, Back cannot close the picture")
    assert 'getElementById("lb-close")' in app_js, "Close button unwired"


def test_the_close_control_exists_and_says_so(html):
    """The complaint was a MISSING control, so it must be visible text."""
    assert 'id="lb-close"' in html
    m = re.search(r'id="lb-close"[^>]*>([^<]*)<', html)
    assert m and "Close" in m.group(1), (
        "the exit control must be labelled, not an unmarked glyph")


def test_opening_a_picture_locks_the_page_behind_it(app_js):
    """The article must not scroll under the picture."""
    assert 'document.body.style.overflow = "hidden"' in app_js
    assert 'document.body.style.overflow = ""' in app_js, "scroll never restored"


def test_overlay_is_opaque(html):
    """DESIGN.md flat surfaces - and the offscreen render showed the page
    reading through a translucent scrim, which looks broken."""
    m = re.search(r"#lb \{[^}]*\}", html)
    assert m, "#lb rule missing"
    assert "rgba(" not in m.group(0), (
        "the lightbox background went translucent again; the page shows "
        "through it")


def test_no_focus_rectangle_on_the_close_button(html):
    """DESIGN.md: no focus rectangles. The render showed a loud yellow
    ring because openLightbox focuses the button."""
    assert re.search(r"#lb-close:focus\s*\{[^}]*outline:\s*none", html), (
        "focus ring returns on the Close button")
