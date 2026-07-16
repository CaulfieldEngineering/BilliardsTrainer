# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — Windows single-file exe, or a macOS .app bundle.

Heavy optional backends (torch/ultralytics/mediapipe) are excluded to keep the
download small (~150 MB instead of ~2 GB); the app degrades to the classical
ball detector without them. Run from the repo root:

    pyinstaller packaging/billiards_trainer.spec --noconfirm

On macOS this produces dist/BilliardsTrainer.app with the camera + Bluetooth
usage strings in Info.plist (without them, macOS silently denies the webcam
and the cue sensor instead of prompting). The bundle is unsigned: first launch
on a new Mac is right-click → Open.
"""

import os
import sys

from PyInstaller.utils.hooks import collect_all

block_cipher = None

# SPECPATH is injected by PyInstaller and is the directory containing this spec
# (``packaging/``); the repo root is its parent. Using it makes paths robust
# regardless of the current working directory.
ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

# onnxruntime ships native DLLs that PyInstaller's modulegraph misses — collect_all
# grabs its binaries/data/submodules so the trained YOLO (.onnx) detector — the
# DEFAULT — runs in the frozen build. On Windows this is onnxruntime-directml, so
# collect_all also pulls the DirectML provider DLLs for GPU inference. Imported
# lazily by onnx_model, so the app still launches if onnxruntime is somehow absent.
try:
    _ort_datas, _ort_bins, _ort_hidden = collect_all("onnxruntime")
except Exception:  # onnxruntime not installed in this build env -> ship without it
    _ort_datas, _ort_bins, _ort_hidden = [], [], []

# bleak's Windows backend rides on the winrt namespace packages, which
# modulegraph misses (runtime-generated bindings). Collect both so the cue
# sensor works in the frozen exe; a build env without them just ships without
# BLE (the worker degrades to an 'unavailable' status).
_ble_datas, _ble_bins, _ble_hidden = [], [], []
for _pkg in ("bleak", "winrt"):
    try:
        _d, _b, _h = collect_all(_pkg)
        _ble_datas += _d; _ble_bins += _b; _ble_hidden += _h
    except Exception:
        pass

a = Analysis(
    [os.path.join(ROOT, "packaging", "launch.py")],
    pathex=[os.path.join(ROOT, "src")],
    binaries=list(_ort_bins) + list(_ble_bins),
    datas=([(os.path.join(ROOT, "packaging", "app.ico"), ".")]
           if os.path.exists(os.path.join(ROOT, "packaging", "app.ico")) else [])
          + list(_ort_datas) + list(_ble_datas),
    hiddenimports=[
        "sqlalchemy.dialects.sqlite",
        "billiards_trainer",
        "billiards_trainer.detector_strategies.onnx_model",
        # Detectors are imported via the package's static core + dynamic discovery;
        # name them explicitly so the frozen bundle always contains them. Without
        # this, frozen discovery (iter_modules finds nothing on disk) would have no
        # detector at all. There are two now: the trained ONNX model + the cue-ball
        # heuristic fallback.
        "billiards_trainer.detector_strategies",
        "billiards_trainer.detector_strategies.cue_ball_white",
        # in-app Ball-ID trainer (lazily imported from the settings page) + its
        # data store — name them so the frozen bundle includes them.
        "billiards_trainer.train",
        "billiards_trainer.train.store",
        "billiards_trainer.ui.dialogs.ball_trainer_dialog",
        # friendly camera names on Windows (DirectShow via comtypes)
        "pygrabber",
        "pygrabber.dshow_graph",
        "comtypes",
        "comtypes.client",
        "comtypes.stream",
        # cue-stroke sensor: worker + diagnostics dialog are imported lazily, so
        # name the whole package explicitly (frozen discovery finds nothing on
        # disk — the same lesson as the detector strategies above).
        "billiards_trainer.cue",
        "billiards_trainer.cue.protocol",
        "billiards_trainer.cue.analysis",
        "billiards_trainer.cue.detector",
        "billiards_trainer.cue.worker",
        "billiards_trainer.ui.dialogs.cue_diagnostics_dialog",
        # ensure HTTPS verification works in the frozen build (update check)
        "certifi",
        *_ort_hidden,
        *_ble_hidden,
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # torch/ultralytics stay OUT (huge + opencv DLL conflict). The YOLO11 path
        # ships via ONNX + onnxruntime instead (bundled above via collect_all).
        "torch", "torchvision", "ultralytics", "mediapipe",
        "matplotlib", "tkinter", "PySide6.QtWebEngineCore",
        "PySide6.Qt3DCore", "PySide6.QtCharts", "PySide6.QtDataVisualization",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="BilliardsTrainer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,            # GUI app, no console window
    disable_windowed_traceback=False,
    icon=(os.path.join(ROOT, "packaging", "app.ico")
          if os.path.exists(os.path.join(ROOT, "packaging", "app.ico")) else None),
)

if sys.platform == "darwin":
    _icns = os.path.join(ROOT, "packaging", "app.icns")
    app = BUNDLE(
        exe,
        name="BilliardsTrainer.app",
        icon=_icns if os.path.exists(_icns) else None,
        bundle_identifier="com.caulfieldengineering.billiardstrainer",
        info_plist={
            # TCC permission strings — REQUIRED, or macOS denies the hardware
            # without ever prompting the user.
            "NSCameraUsageDescription":
                "Billiards Trainer watches your table through the overhead "
                "camera to track balls and score shots.",
            "NSBluetoothAlwaysUsageDescription":
                "Billiards Trainer connects to the motion sensor on your cue "
                "to measure each stroke.",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
        },
    )
