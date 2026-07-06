"""Cue-stroke sensing: a Bluetooth 6-axis IMU on the cue butt.

Ported from the pool-stroke-analyzer project (see its Documentation/
INTEGRATION.md for provenance). Layout:

  protocol.py  JINOU JO-BEC12-2 frame parsing/decoding — pure stdlib, no BLE
  analysis.py  the validated science: impact detection, fusion, metrics
  detector.py  live streaming impact detector + per-shot metrics (pure Python)
  worker.py    the only BLE-touching module (bleak, imported lazily)

The whole feature is optional at runtime: no sensor, no Bluetooth radio, or no
bleak install all degrade to a status message — never an error dialog, never a
crash. Everything else in the app works identically without it.
"""
