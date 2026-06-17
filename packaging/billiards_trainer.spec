# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — builds a single-file Windows executable.

Heavy optional backends (torch/ultralytics/mediapipe) are excluded to keep the
download small (~150 MB instead of ~2 GB); the app degrades to the classical
ball detector without them. Run from the repo root:

    pyinstaller packaging/billiards_trainer.spec --noconfirm
"""

import os

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

a = Analysis(
    [os.path.join(ROOT, "packaging", "launch.py")],
    pathex=[os.path.join(ROOT, "src")],
    binaries=list(_ort_bins),
    datas=([(os.path.join(ROOT, "packaging", "app.ico"), ".")]
           if os.path.exists(os.path.join(ROOT, "packaging", "app.ico")) else [])
          + list(_ort_datas),
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
        # friendly camera names on Windows (DirectShow via comtypes)
        "pygrabber",
        "pygrabber.dshow_graph",
        "comtypes",
        "comtypes.client",
        "comtypes.stream",
        # ensure HTTPS verification works in the frozen build (update check)
        "certifi",
        *_ort_hidden,
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
