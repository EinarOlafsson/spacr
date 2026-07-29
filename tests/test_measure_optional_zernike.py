"""Mahotas is optional; only the Zernike feature boundary may require it."""
from __future__ import annotations

import builtins
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


def _block_mahotas(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "mahotas" or name.startswith("mahotas."):
            raise ModuleNotFoundError("blocked Mahotas for test")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_measure_module_imports_without_mahotas():
    code = """
import importlib.abc
import sys
class BlockMahotas(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mahotas" or fullname.startswith("mahotas."):
            raise ModuleNotFoundError("blocked Mahotas for test")
        return None
sys.meta_path.insert(0, BlockMahotas())
import spacr.measure
print("measure imported")
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "measure imported" in proc.stdout


def test_explicit_zernike_request_has_actionable_missing_extra(monkeypatch):
    from spacr import measure

    _block_mahotas(monkeypatch)
    mask = np.zeros((16, 16), dtype=np.int32)
    mask[4:12, 4:12] = 1
    with pytest.raises(ImportError, match=r"spacr\[zernike\]"):
        measure._calculate_zernike(
            mask, pd.DataFrame({"label": [1]}), degree=4)


def test_empty_and_3d_masks_do_not_require_mahotas(monkeypatch):
    from spacr import measure

    _block_mahotas(monkeypatch)
    frame = pd.DataFrame()
    assert measure._calculate_zernike(
        np.zeros((8, 8), dtype=np.int32), frame) is frame
    assert measure._calculate_zernike(
        np.zeros((2, 8, 8), dtype=np.int32), frame) is frame


def test_automatic_morphology_skips_zernike_when_extra_is_missing(
        monkeypatch, capsys):
    from spacr import measure

    _block_mahotas(monkeypatch)
    cell = np.zeros((20, 20), dtype=np.int32)
    cell[4:16, 4:16] = 1
    empty = np.zeros_like(cell)
    settings = {
        "cell_mask_dim": 0,
        "nucleus_mask_dim": None,
        "pathogen_mask_dim": None,
        "organelle_mask_dim": None,
        "cytoplasm": False,
    }

    cell_df, *_ = measure._morphological_measurements(
        cell, empty, empty, empty, empty, settings, zernike=None)

    assert not any(column.startswith("cell_zernike_")
                   for column in cell_df.columns)
    assert "spacr[zernike]" in capsys.readouterr().out
