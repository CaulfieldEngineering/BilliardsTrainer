"""Measurement substrate for the vision stack.

Everything in here answers one question: *did that change make it better or
worse?* Until this package existed the honest answer was "nobody knows" — the
detector was only ever judged by eye, so quality random-walked.

The centrepiece is :mod:`invariants`, which scores tracking WITHOUT human
labels by checking physical laws that pool tables obey unconditionally. That is
what makes unattended iteration possible over hours of footage.
"""

from .invariants import (
    InvariantReport,
    SequenceScorer,
    Violation,
    check_frame,
    check_sequence,
)

__all__ = [
    "InvariantReport",
    "SequenceScorer",
    "Violation",
    "check_frame",
    "check_sequence",
]
