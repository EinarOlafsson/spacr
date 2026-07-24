"""Tests for the lossless mask-TIFF helpers and the merged-array reader's
tif/npy flexibility (object labels must be preserved exactly)."""
from __future__ import annotations

import os
import numpy as np


def test_save_object_mask_roundtrip_preserves_labels(tmp_path):
    from spacr.io import save_object_mask, _load_array_any
    mask = np.zeros((16, 16), np.uint16)
    mask[2:6, 2:6] = 3
    mask[10:14, 10:14] = 41000   # large label id — must survive uint16
    out = save_object_mask(str(tmp_path), "plate1_A01_f1.npy", mask, compression="lzw")
    assert out.endswith(".tif")
    back = _load_array_any(out)
    assert back.dtype == np.uint16
    assert np.array_equal(back, mask)   # labels EXACTLY preserved


def test_mask_variant_path_prefers_tif_then_npy(tmp_path):
    from spacr.io import _mask_variant_path
    d = str(tmp_path)
    # only npy present
    np.save(os.path.join(d, "x.npy"), np.zeros((2, 2), np.uint16))
    assert _mask_variant_path(d, "x.npy").endswith("x.npy")
    # add a tif -> tif preferred
    import tifffile
    tifffile.imwrite(os.path.join(d, "x.tif"), np.zeros((2, 2), np.uint16))
    assert _mask_variant_path(d, "x.npy").endswith("x.tif")
    assert _mask_variant_path(d, "missing.npy") is None


def _build(root, mask_ext):
    """stack/ (2 img channels) + cell + nucleus mask folders, one FOV."""
    stack = os.path.join(root, "stack")
    os.makedirs(stack)
    H = W = 12
    arr = np.zeros((H, W, 2), np.float32)
    arr[..., 0] = 5.0
    arr[..., 1] = 9.0
    np.save(os.path.join(stack, "fov.npy"), arr)
    cell = np.zeros((H, W), np.uint16); cell[1:6, 1:6] = 7
    nuc = np.zeros((H, W), np.uint16); nuc[7:11, 7:11] = 2
    for name, m in (("cell_mask_stack", cell), ("nucleus_mask_stack", nuc)):
        d = os.path.join(root, "masks", name)
        os.makedirs(d)
        if mask_ext == ".tif":
            import tifffile
            tifffile.imwrite(os.path.join(d, "fov.tif"), m, compression="lzw")
        else:
            np.save(os.path.join(d, "fov.npy"), m)
    return cell, nuc


def _run_merged(root):
    from spacr.io import _load_and_concatenate_arrays
    _load_and_concatenate_arrays(root,
                                 channels=[0, 1], cell_chann_dim=None,
                                 nucleus_chann_dim=None, pathogen_chann_dim=None,
                                 organelle_chann_dim=None)
    return np.load(os.path.join(root, "merged", "fov.npy"))


def test_merged_reader_reads_npy_masks(tmp_path):
    cell, nuc = _build(str(tmp_path), ".npy")
    merged = _run_merged(str(tmp_path))
    # 2 image channels + cell + nucleus mask slices
    assert merged.shape[-1] == 4
    assert np.array_equal(merged[..., 2], cell)   # labels preserved
    assert np.array_equal(merged[..., 3], nuc)


def test_merged_reader_reads_tif_masks(tmp_path):
    cell, nuc = _build(str(tmp_path), ".tif")
    merged = _run_merged(str(tmp_path))
    assert merged.shape[-1] == 4
    assert np.array_equal(merged[..., 2], cell)
    assert np.array_equal(merged[..., 3], nuc)
