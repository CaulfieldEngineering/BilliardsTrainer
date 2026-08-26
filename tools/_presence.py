"""Joe-presence signal for heavy background jobs.

INCIDENT 2026-08-26: the stroke-v2 library backfill ran while Joe was
at the machine — his session lists took minutes to load. The recording
guards (no .part, 10-min quiet) only cover Joe AT THE TABLE; this
covers Joe AT THE MACHINE: system-wide keyboard/mouse idle time via
GetLastInputInfo. Heavy jobs defer until the PC has been untouched for
IDLE_MIN minutes.

Typed ctypes throughout — bare windll calls fail silently (the
SetPriorityClass lesson, pinned by tests).
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

IDLE_MIN = 15.0


class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]


_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.GetLastInputInfo.restype = wintypes.BOOL
_user32.GetLastInputInfo.argtypes = [ctypes.POINTER(_LASTINPUTINFO)]
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.GetTickCount.restype = wintypes.DWORD
_kernel32.GetTickCount.argtypes = []


def idle_seconds() -> float:
    """Seconds since the last keyboard/mouse input, system-wide.
    Returns 0.0 (assume present — the safe direction) on API failure."""
    info = _LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(_LASTINPUTINFO)
    if not _user32.GetLastInputInfo(ctypes.byref(info)):
        return 0.0
    # Both are 32-bit tick counts; unsigned wrap subtraction stays correct
    # across the 49.7-day rollover.
    return ((_kernel32.GetTickCount() - info.dwTime) & 0xFFFFFFFF) / 1000.0


def joe_present(idle_min: float = IDLE_MIN) -> bool:
    """True while the machine has seen input within idle_min minutes."""
    return idle_seconds() < idle_min * 60.0
