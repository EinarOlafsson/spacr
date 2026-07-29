"""Contracts that keep optional native features out of core imports."""

from __future__ import annotations

import builtins
import subprocess
import sys

import numpy as np
import pytest


def test_core_modules_do_not_eagerly_import_optional_native_features():
    code = """
import sys
import numpy
import spacr.io
import spacr.timelapse
assert "pylibCZIrw" not in sys.modules
assert "btrack" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def _block_import(monkeypatch, package):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == package or name.startswith(package + "."):
            raise ImportError(f"blocked optional dependency: {package}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_missing_pylibczirw_error_names_the_czi_extra(monkeypatch):
    from spacr import io

    _block_import(monkeypatch, "pylibCZIrw")
    with pytest.raises(ImportError, match=r"spacr\[czi\]"):
        io._load_pylibczi()


def test_missing_btrack_error_names_the_btrack_extra(monkeypatch, tmp_path):
    from spacr import timelapse

    _block_import(monkeypatch, "btrack")
    with pytest.raises(ImportError, match=r"spacr\[btrack\]"):
        timelapse._btrack_track_cells(
            src=str(tmp_path),
            name="synthetic",
            batch_filenames=[],
            object_type="cell",
            plot=False,
            save=False,
            masks_3D=np.zeros((1, 2, 2), dtype=np.uint16),
            mode="btrack",
            timelapse_remove_transient=False,
        )
