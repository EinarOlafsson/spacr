"""Coverage-fill batch 4 for spacr.io channel-merge / MIP helpers."""
from __future__ import annotations

import numpy as np
import cv2
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr import io as IO


# ---------------------------------------------------------------------------
# _merge_file
# ---------------------------------------------------------------------------

def test_merge_file(tmp_path):
    chan_dirs = []
    for c in range(3):
        d = tmp_path / f"chan{c}"; d.mkdir()
        img = (np.random.default_rng(c).random((16, 16)) * 255).astype(np.uint16)
        cv2.imwrite(str(d / "f.tif"), img)
        chan_dirs.append(str(d))
    stack_dir = tmp_path / "stack"
    IO._merge_file(chan_dirs, str(stack_dir), "f.tif")
    out = stack_dir / "f.npy"
    assert out.exists()
    stack = np.load(str(out))
    assert stack.shape == (16, 16, 3)


def test_merge_file_missing_channel(tmp_path):
    # one channel dir is missing the file → skipped with warning, others merge
    chan_dirs = []
    for c in range(2):
        d = tmp_path / f"chan{c}"; d.mkdir()
        chan_dirs.append(str(d))
    img = (np.random.default_rng(0).random((8, 8)) * 255).astype(np.uint16)
    cv2.imwrite(str(tmp_path / "chan0" / "f.tif"), img)
    # chan1 has no f.tif
    stack_dir = tmp_path / "stack"
    IO._merge_file(chan_dirs, str(stack_dir), "f.tif")
    stack = np.load(str(stack_dir / "f.npy"))
    assert stack.shape[-1] == 1   # only the one valid channel


# ---------------------------------------------------------------------------
# _mip_all
# ---------------------------------------------------------------------------

def test_mip_all_3d(tmp_path):
    arr = (np.random.default_rng(0).random((16, 16, 3)) * 255).astype(np.uint16)
    np.save(str(tmp_path / "f.npy"), arr)
    IO._mip_all(str(tmp_path), include_first_chan=True)
    out = np.load(str(tmp_path / "f.npy"))
    assert out.shape[-1] == 4   # original 3 + MIP


def test_mip_all_exclude_first(tmp_path):
    arr = (np.random.default_rng(1).random((16, 16, 3)) * 255).astype(np.uint16)
    np.save(str(tmp_path / "f.npy"), arr)
    IO._mip_all(str(tmp_path), include_first_chan=False)
    out = np.load(str(tmp_path / "f.npy"))
    assert out.shape[-1] == 4


def test_mip_all_non_3d(tmp_path):
    # 2-D array → zero-array concatenation branch
    arr = (np.random.default_rng(2).random((16, 16)) * 255).astype(np.uint16)
    np.save(str(tmp_path / "f.npy"), arr)
    IO._mip_all(str(tmp_path))
    out = np.load(str(tmp_path / "f.npy"))
    assert out.ndim == 3
