"""Coverage-fill for spacr.submodules directly-testable classes/helpers."""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

import matplotlib
matplotlib.use("Agg")

from spacr import submodules as S


def _make_pair(tmp_path, i, size=32, rgb=False):
    img = (np.random.default_rng(i).random(
        (size, size, 3) if rgb else (size, size)) * 1000).astype(np.uint16)
    lbl = np.zeros((size, size), dtype=np.uint16)
    lbl[4:12, 4:12] = 1
    lbl[20:28, 20:28] = 2
    ip = tmp_path / f"img{i}.tif"; lp = tmp_path / f"lbl{i}.tif"
    tifffile.imwrite(str(ip), img); tifffile.imwrite(str(lp), lbl)
    return str(ip), str(lp)


# ---------------------------------------------------------------------------
# CellposeLazyDataset
# ---------------------------------------------------------------------------

def test_lazy_dataset_len_and_getitem(tmp_path):
    imgs, lbls = zip(*[_make_pair(tmp_path, i) for i in range(3)])
    ds = S.CellposeLazyDataset(
        list(imgs), list(lbls),
        {"normalize": True, "percentiles": (2, 99), "target_size": 16},
        randomize=False, augment=False)
    assert len(ds) == 3
    img, lbl = ds[0]
    assert img.shape == (16, 16) and img.dtype == np.float32
    assert lbl.shape == (16, 16) and lbl.dtype == np.uint16


def test_lazy_dataset_augment_len(tmp_path):
    imgs, lbls = zip(*[_make_pair(tmp_path, i) for i in range(2)])
    ds = S.CellposeLazyDataset(
        list(imgs), list(lbls), {"target_size": 16},
        randomize=True, augment=True)
    assert len(ds) == 16   # 2 * 8
    # touch every augmentation branch
    for k in range(8):
        img, lbl = ds[k]
        assert img.shape == (16, 16)


def test_lazy_dataset_rgb_input(tmp_path):
    ip, lp = _make_pair(tmp_path, 9, rgb=True)
    ds = S.CellposeLazyDataset(
        [ip], [lp], {"normalize": False, "target_size": 8},
        randomize=False)
    img, _ = ds[0]
    assert img.shape == (8, 8)   # grayscaled from RGB


def test_lazy_dataset_length_mismatch():
    with pytest.raises(ValueError):
        S.CellposeLazyDataset(["a"], ["b", "c"], {"target_size": 8})


def test_lazy_dataset_empty():
    with pytest.raises(ValueError):
        S.CellposeLazyDataset([], [], {"target_size": 8})


def test_lazy_dataset_static_helpers():
    rgb = np.zeros((4, 4, 3), dtype=np.float32); rgb[..., 0] = 3.0
    gray = S.CellposeLazyDataset._to_grayscale(rgb)
    assert gray.shape == (4, 4)
    # already 2-D → unchanged
    assert S.CellposeLazyDataset._to_grayscale(gray).shape == (4, 4)
    scaled = S.CellposeLazyDataset._scale_to_unit_interval(
        np.array([[0, 255]], dtype=np.float32))
    assert scaled.max() <= 1.0
    # max<=1 → returned unchanged
    small = np.array([[0.0, 0.5]], dtype=np.float32)
    assert S.CellposeLazyDataset._scale_to_unit_interval(small).max() == 0.5


# ---------------------------------------------------------------------------
# CellposeLazyDataset_v1 (legacy)
# ---------------------------------------------------------------------------

def test_lazy_dataset_v1(tmp_path):
    imgs, lbls = zip(*[_make_pair(tmp_path, i) for i in range(2)])
    ds = S.CellposeLazyDataset(
        list(imgs), list(lbls),
        {"normalize": True, "percentiles": [2, 99], "target_size": 16},
        randomize=False, augment=True)
    assert len(ds) == 16
    for k in range(8):
        img, lbl = ds[k]
        assert img.shape == (16, 16) and lbl.shape == (16, 16)
        assert lbl.dtype == np.uint16
