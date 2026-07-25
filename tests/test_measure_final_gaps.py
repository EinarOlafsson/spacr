"""Final measure.py gaps: _save_object_crop's per-channel-count branches
and measure_crop's fov-normalisation / 2-channel PNG paths.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

from spacr.settings import get_measure_crop_settings


# ---------------------------------------------------------------------------
# _save_object_crop channel-count branches
# ---------------------------------------------------------------------------

def _crop(n_chan, size=16):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (size, size, n_chan)).astype(np.uint8)


def test_save_object_crop_single_channel_grayscale(tmp_path):
    from spacr.measure import _save_object_crop
    p = tmp_path / "one.png"
    out = _save_object_crop(_crop(1), (0,), str(p), [16, 16])
    assert os.path.isfile(out or p)
    assert Image.open(p).mode in ("L", "RGB")


def test_save_object_crop_two_channels_padded(tmp_path):
    from spacr.measure import _save_object_crop
    p = tmp_path / "two.png"
    _save_object_crop(_crop(2), (0, 1), str(p), [16, 16])
    assert Image.open(p).mode == "RGB"


def test_save_object_crop_three_channels_rgb(tmp_path):
    from spacr.measure import _save_object_crop
    p = tmp_path / "three.png"
    _save_object_crop(_crop(3), (0, 1, 2), str(p), [16, 16])
    assert Image.open(p).size == (16, 16)


def test_save_object_crop_more_than_three_writes_npy(tmp_path):
    """>3 channels can't be a PNG: the full stack goes to .npy with a
    3-channel PNG preview alongside."""
    from spacr.measure import _save_object_crop
    p = tmp_path / "many.png"
    out = _save_object_crop(_crop(5), (0, 1, 2, 3, 4), str(p), [16, 16])
    assert str(out).endswith(".npy")
    assert os.path.isfile(out)
    assert os.path.isfile(p)          # preview still written
    assert np.load(out).shape[2] == 5


# ---------------------------------------------------------------------------
# measure_crop fov normalisation + 2-channel PNG assembly
# ---------------------------------------------------------------------------

def _merged(tmp_path, rng, n_chan=4):
    merged = tmp_path / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    (tmp_path / "measurements").mkdir(parents=True, exist_ok=True)
    H = W = 32
    layers = [rng.integers(50, 400, (H, W)).astype(np.uint16) for _ in range(n_chan)]
    cell = np.zeros((H, W), np.uint16); cell[4:20, 4:20] = 1
    nuc = np.zeros((H, W), np.uint16); nuc[8:14, 8:14] = 1
    pat = np.zeros((H, W), np.uint16); pat[10:12, 10:12] = 1
    data = np.stack(layers + [cell, nuc, pat], axis=-1).astype(np.uint16)
    np.save(merged / "plate1_A01_F001.npy", data)
    return merged


def _settings(merged, **over):
    s = get_measure_crop_settings(settings={})
    s.update({
        "src": str(merged), "channels": [0, 1, 2, 3],
        "cell_mask_dim": 4, "nucleus_mask_dim": 5, "pathogen_mask_dim": 6,
        "png_dims": [0, 1, 2], "png_size": [16, 16],
        "save_measurements": False, "save_png": True, "save_arrays": False,
        "plot": False, "verbose": False, "timelapse": False,
        "crop_mode": ["cell"], "normalize": [1, 99], "normalize_by": "png",
        "experiment": "exp", "n_jobs": 1, "test_mode": False,
        "cytoplasm": True,
    })
    s.update(over)
    return s


def test_measure_core_normalize_by_fov(tmp_path, rng):
    """normalize_by='fov' computes percentiles across the field first."""
    from spacr.measure import _measure_crop_core
    merged = _merged(tmp_path, rng)
    idx, _t, _cells, _figs = _measure_crop_core(
        0, [], "plate1_A01_F001.npy",
        _settings(merged, normalize_by="fov"))
    assert idx == 0
    assert list(tmp_path.rglob("*.png"))


def test_measure_core_two_channel_png_gets_dummy_third(tmp_path, rng):
    """A 2-entry png_dims is padded with a zero channel before saving."""
    from spacr.measure import _measure_crop_core
    merged = _merged(tmp_path, rng)
    idx, _t, _cells, _figs = _measure_crop_core(
        0, [], "plate1_A01_F001.npy",
        _settings(merged, png_dims=[0, 1]))
    assert idx == 0
    assert list(tmp_path.rglob("*.png"))


def test_measure_core_normalize_false_uses_full_range(tmp_path, rng):
    """normalize=False falls back to a 0-100 percentile stretch."""
    from spacr.measure import _measure_crop_core
    merged = _merged(tmp_path, rng)
    idx, _t, _cells, _figs = _measure_crop_core(
        0, [], "plate1_A01_F001.npy",
        _settings(merged, normalize=False))
    assert idx == 0
