# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file shared by the Windows and macOS builds.

Both `build_windows.ps1` and `build_macos.sh` invoke:
    pyinstaller packaging/spacr.spec

The macOS branch also produces a .app bundle (BUNDLE at the bottom).
"""
from __future__ import annotations

import sys
from pathlib import Path

from packaging.version import Version
from PyInstaller import __version__ as PYINSTALLER_VERSION
from PyInstaller.utils.hooks import (
    collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata,
)

if Version(PYINSTALLER_VERSION) < Version("6.10"):
    raise RuntimeError("The independent updater helper requires PyInstaller >= 6.10")

# Repo root is one dir up from packaging/
ROOT = Path(SPECPATH).resolve().parent
ENTRY = str(ROOT / "packaging" / "spacr_launcher.py")

# ``collect_submodules`` probes packages in an isolated interpreter before
# ``Analysis(pathex=...)`` exists.  Pin that probe to this checkout now;
# otherwise a globally installed ``spacr`` can win and freshly added runtime
# modules disappear from an apparently successful bundle.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------
# Hidden imports: collect spaCR and Cellpose's runtime modules because both
# dispatch by dotted name.  Let PyInstaller's maintained hooks and ordinary
# static analysis handle third-party packages; recursively collecting all of
# torch, scipy, sklearn, statsmodels, VisPy, and their test suites made the
# desktop bundle enormous without making it more complete.
# ------------------------------------------------------------------
_NON_RUNTIME_PARTS = frozenset({
    "benchmarks", "demos", "examples", "tests", "testing",
})


def _is_runtime_module(name):
    return _NON_RUNTIME_PARTS.isdisjoint(name.split("."))


def _is_cellpose_runtime_module(name):
    # spaCR uses Cellpose's models, I/O, training and inference modules.  Its
    # standalone GUI and distributed-contrib front ends are separate extras;
    # collecting them makes a core bundle depend on whichever Dask/Qt stack
    # happens to be installed on the build machine.
    return (_is_runtime_module(name)
            and not name.startswith(("cellpose.contrib", "cellpose.gui")))


hiddenimports = collect_submodules(
    "spacr", filter=_is_runtime_module, on_error="raise",
)
hiddenimports += collect_submodules(
    "cellpose", filter=_is_cellpose_runtime_module, on_error="raise",
)
hiddenimports += [
    "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets",
    "PySide6.QtOpenGL", "PySide6.QtOpenGLWidgets",
    # VisPy chooses both of these by string at runtime.  spaCR explicitly
    # selects PySide6 and VisPy's default desktop GL2 implementation.
    "vispy", "vispy.app.backends._pyside6",
    "vispy.gloo.gl.gl2", "vispy.gloo.gl._gl2",
    "torch", "torchvision",
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_agg",
    "scipy.spatial.transform._rotation_groups",
    "sklearn.utils._typedefs", "sklearn.neighbors._partition_nodes",
    "skimage.feature._orb_descriptor_positions",
    "statsmodels", "seaborn",
    "tables",           # HDF5 storage for sequencing pipeline
    "huggingface_hub",
]
hiddenimports = list(dict.fromkeys(hiddenimports))

# The desktop builders install spaCR's declared core environment.  PyInstaller
# nevertheless follows guarded imports when one of these optional packages is
# already present in a developer's environment, which used to make two builds
# of the same commit differ by gigabytes.  Excluding non-core roots preserves
# the same graceful "install the extra" paths as a clean core installation.
_NON_CORE_IMPORTS = [
    # spaCR extras
    "anndata", "btrack", "catboost", "cuml", "cupy", "jax",
    "lightgbm", "mahotas", "napari", "numcodecs", "numpyro", "omero",
    "piper", "pylibCZIrw", "pymc", "torchcam", "trackastra", "ultrack",
    "zarr",
    # development/documentation-only packages
    "_pytest", "black", "docutils", "hypothesis", "mypy", "pingouin",
    "pyarrow", "pytest", "ruff", "sphinx", "xenon", "yapf",
    # optional branches of core dependencies, discovered only when this
    # workstation happened to have them installed
    "altair", "astropy", "bokeh", "dask", "distributed", "intake",
    "nbconvert", "nltk", "onnxruntime", "panel", "plotly", "rdflib",
    "selenium", "spacy", "timm", "transformers", "xarray",
]

# ------------------------------------------------------------------
# Data files: cellpose model weights (bundled if present in cache),
# spacr's own resources/ (icons, fonts, sample settings CSVs), the
# spacr version file.
# ------------------------------------------------------------------
datas = []
datas += collect_data_files("spacr", includes=["resources/**/*", "fonts/**/*"])
datas += collect_data_files("cellpose", includes=["*.txt", "*.md"])
datas += copy_metadata("spacr")
# ImageIO reads its distribution version at import time. Its data hook does
# not retain dist-info, so otherwise a frozen Measure run fails on spacr.io.
datas += copy_metadata("imageio")


def _torchvision_binaries():
    """Retain torchvision's directly loaded operator library and dependencies."""
    libraries = collect_dynamic_libs(
        "torchvision", search_patterns=["*.so", "*.so.*", "*.pyd", "*.dll", "*.dylib"],
    )
    # The maintained hook asks for hidden import torchvision._C, but these
    # wheels load it with torch.ops.load_library instead. Native CI found no
    # _C in the resulting bundle and failed while registering torchvision::nms.
    if not any(Path(source).name.startswith(("_C.", "_C_stable."))
               for source, _ in libraries):
        raise RuntimeError("torchvision's _C operator library was not collected")
    return libraries


binaries = _torchvision_binaries()


# ------------------------------------------------------------------
# Analysis / bundling
# ------------------------------------------------------------------
block_cipher = None

helper_analysis = Analysis(
    [str(ROOT / "spacr" / "install_cleanup.py")],
    pathex=[], binaries=[], datas=[], hiddenimports=[],
    hookspath=[], runtime_hooks=[],
    excludes=["spacr", "PySide6", "torch", "numpy"],
    noarchive=False,
)
helper_pyz = PYZ(helper_analysis.pure)
helper_exe = EXE(
    helper_pyz, helper_analysis.scripts,
    helper_analysis.binaries, helper_analysis.datas, [],
    name="spacr-update-helper", exclude_binaries=False,
    debug=False, bootloader_ignore_signals=False,
    strip=False, upx=False, console=True,
)
binaries += [(helper_exe.name, ".")]

a = Analysis(
    [ENTRY],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={
        # spaCR renders either into Qt or directly to an image.  Naming the
        # two backends prevents an installed but unrelated GUI toolkit from
        # being swept into the frozen application.
        "matplotlib": {"backends": ["QtAgg", "Agg"]},
    },
    runtime_hooks=[],
    excludes=[
        "tensorflow", "keras",   # explicitly banned; see spaCR's no-TF rule
        "notebook", "IPython.html",
        "PySide2", "PyQt6", "PyQt5",
        *_NON_CORE_IMPORTS,
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
