# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file shared by the Windows and macOS builds.

Both `build_windows.ps1` and `build_macos.sh` invoke:
    pyinstaller packaging/spacr.spec

The macOS branch also produces a .app bundle (BUNDLE at the bottom).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Repo root is one dir up from packaging/
ROOT = Path(SPECPATH).resolve().parent
ENTRY = str(ROOT / "packaging" / "spacr_launcher.py")

# ------------------------------------------------------------------
# Hidden imports: PyInstaller's static analysis misses lazy imports
# used by cellpose, matplotlib, statsmodels, ...
# ------------------------------------------------------------------
hiddenimports = []
for pkg in (
    "spacr", "cellpose", "cellpose.io", "cellpose.models",
    "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets",
    "vispy",
    "torch", "torchvision",
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_agg",
    "scipy", "scipy.spatial.transform._rotation_groups",
    "sklearn", "sklearn.utils._typedefs", "sklearn.neighbors._partition_nodes",
    "skimage", "skimage.feature._orb_descriptor_positions",
    "statsmodels", "seaborn",
    "mahotas",
    "tables",           # HDF5 storage for sequencing pipeline
    "huggingface_hub",
):
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass

# ------------------------------------------------------------------
# Data files: cellpose model weights (bundled if present in cache),
# spacr's own resources/ (icons, fonts, sample settings CSVs), the
# spacr version file.
# ------------------------------------------------------------------
datas = []
datas += collect_data_files("spacr", includes=["resources/**/*", "fonts/**/*"])
datas += collect_data_files("cellpose", includes=["*.txt", "*.md"])


# ------------------------------------------------------------------
# Analysis / bundling
# ------------------------------------------------------------------
block_cipher = None

a = Analysis(
    [ENTRY],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tensorflow", "tensorboard", "keras",   # explicitly banned; see spacr's no-TF rule
        "notebook", "IPython.html",
        "PySide2", "PyQt6", "PyQt5",
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
    [],
    exclude_binaries=True,
    name="spacr",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,       # UPX conflicts with the numpy .so's on macOS
    console=False,   # windowed app (no terminal)
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="spacr",
)

# --- macOS .app bundle ---
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="spaCR.app",
        icon=None,
        bundle_identifier="com.einarolafsson.spacr",
        info_plist={
            "NSHighResolutionCapable": "True",
            "LSMinimumSystemVersion": "11.0",
        },
    )
