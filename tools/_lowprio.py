"""Demote the current process to BELOW_NORMAL (Windows, typed ctypes).

Bare ctypes.windll mangles the 64-bit pseudo-handle and SetPriorityClass
fails SILENTLY (returns 0) - measured 2026-08-23; every tool that used the
untyped call was running at Normal priority the whole time.
"""
import ctypes
import sys


def demote() -> bool:
    if sys.platform != "win32":
        return False
    try:
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        return bool(k.SetPriorityClass(k.GetCurrentProcess(), 0x4000))
    except Exception:  # noqa: BLE001 - best-effort
        return False
