# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — builds a single-file Windows executable.

Heavy optional backends (torch/ultralytics/mediapipe) are excluded to keep the
download small (~150 MB instead of ~2 GB); the app degrades to the classical
ball detector without them. Run from the repo root:

    pyinstaller packaging/billiards_trainer.spec --noconfirm
"""

import os

block_cipher = None

# SPECPATH is injected by PyInstaller and is the directory containing this spec
# (``packaging/``); the repo root is its parent. Using it makes paths robust
# regardless of the current working directory.
ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

a = Analysis(
    [os.path.join(ROOT, "packaging", "launch.py")],
    pathex=[os.path.join(ROOT, "src")],
    binaries=[],
    datas=([(os.path.join(ROOT, "packaging", "app.ico"), ".")]
           if os.path.exists(os.path.join(ROOT, "packaging", "app.ico")) else []),
    hiddenimports=[
        "sqlalchemy.dialects.sqlite",
        "billiards_trainer",
        # friendly camera names on Windows (DirectShow via comtypes)
        "pygrabber",
        "pygrabber.dshow_graph",
        "comtypes",
        "comtypes.client",
        "comtypes.stream",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
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
