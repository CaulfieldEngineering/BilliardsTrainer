"""song-batch transform library.

Tier 1 of the pipeline: pure, deterministic MIDI file transforms that need no
DAW and no human present. Everything here must be reproducible - same input,
same output, byte for byte - so that a build is reviewable as a git diff.
"""

__all__ = ["midi_io", "drummap", "remap", "transforms", "sections", "spec"]
