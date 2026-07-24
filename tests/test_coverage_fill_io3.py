"""Coverage-fill batch 3 for spacr.io array-merge helper."""
from __future__ import annotations

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

from spacr import io as IO


def _setup_src(tmp_path, mask_shape=(16, 16)):
    src = tmp_path / "exp"
    (src / "stack").mkdir(parents=True)
    (src / "masks" / "cell_mask_stack").mkdir(parents=True)
    (src / "masks" / "nucleus_mask_stack").mkdir(parents=True)
    stack = (np.random.default_rng(0).random((16, 16, 3)) * 255).astype(np.uint16)
    cell = np.zeros(mask_shape, dtype=np.uint16); cell[2:8, 2:8] = 1
    nucleus = np.zeros(mask_shape, dtype=np.uint16); nucleus[3:6, 3:6] = 1
    np.save(str(src / "stack" / "f.npy"), stack)
    np.save(str(src / "masks" / "cell_mask_stack" / "f.npy"), cell)
    np.save(str(src / "masks" / "nucleus_mask_stack" / "f.npy"), nucleus)
    return src


def test_load_and_concatenate_arrays(tmp_path):
    src = _setup_src(tmp_path)
    IO._load_and_concatenate_arrays(
        str(src), channels=[0, 1, 2], cell_chann_dim=0,
        nucleus_chann_dim=1, pathogen_chann_dim=None, organelle_chann_dim=None)
    out = src / "merged" / "f.npy"
    assert out.exists()
    merged = np.load(str(out))
    # 3 image channels + cell + nucleus = 5
    assert merged.shape[-1] == 5


def test_load_and_concatenate_arrays_padding(tmp_path):
    # mask stacks with a different X/Y size exercise the padding branch
    src = _setup_src(tmp_path, mask_shape=(20, 18))
    IO._load_and_concatenate_arrays(
        str(src), channels=None, cell_chann_dim=0,
        nucleus_chann_dim=1, pathogen_chann_dim=None, organelle_chann_dim=None)
    out = src / "merged" / "f.npy"
    assert out.exists()
    merged = np.load(str(out))
    # padded to the max X,Y across all arrays
    assert merged.shape[0] == 20 and merged.shape[1] == 18
