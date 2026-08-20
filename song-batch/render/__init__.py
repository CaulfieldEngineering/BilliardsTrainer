"""Audio rendering - the half of the loop that makes phone operation work.

Without audio you can read a diff from a phone but cannot judge whether the
groove is any good. Two implementations, in ascending order of fidelity:

* :mod:`render.fluidsynth` - crude, same-day, GM soundfont. Judges groove and
  arrangement. Shipping today.
* ``render/juce-host/`` - a headless VST3 host that loads SSD itself and renders
  offline. Real tone, real kit. Not built yet; see CLAUDE.md.
"""

from .fluidsynth import RenderResult, preflight, render  # noqa: F401
