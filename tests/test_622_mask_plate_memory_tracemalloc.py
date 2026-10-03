"""Mask generation's peak memory must not grow with the number of fields.

Runs the whole v1 Mask pipeline (ingest, normalisation, Cellpose for three
objects, cell adjustment, merging) under tracemalloc on synthetic plates of
two sizes. A plate four times larger must not raise the peak by more than a
small part of its extra pixels: anything holding the whole plate would.
"""
import tracemalloc
import types

import numpy as np
import pytest
import tifffile

from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call

SIDE = 256
CHANNELS = ("01", "02", "03")


@pytest.fixture
def fake_cellpose(monkeypatch):
    """CPU stand-in for Cellpose: two square objects per field."""
    import torch
    import spacr.object as O
    import spacr.plot as PL

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(PL, "plot_cellpose4_output", lambda *a, **k: None)

    class _Model:
        def __init__(self, *args, **kwargs):
            pass

        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
                 normalize=True, rescale=None, diameter=None,
                 flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
                 anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
                 min_size=15, max_size_fraction=0.4, niter=None,
                 augment=False, tile_overlap=0.1, bsize=None,
                 compute_masks=True, progress=None):
            check_cellpose_eval_call(x, channel_axis, z_axis=z_axis,
                                     do_3D=do_3D,
                                     stitch_threshold=stitch_threshold)
            masks = []
            for image in x:
                mask = np.zeros(np.asarray(image).shape[:2], np.uint16)
                mask[20:60, 20:60] = 1
                mask[100:150, 100:150] = 2
                masks.append(mask)
            return masks, [np.zeros_like(m, np.float32) for m in masks], None

    monkeypatch.setattr(O, "cp_models", types.SimpleNamespace(CellposeModel=_Model))


def _plate(folder, n_fields):
    folder.mkdir()
    rng = np.random.default_rng(n_fields)
    for index in range(n_fields):
        well, field = f"A{index // 4 + 1:02}", f"{index % 4 + 1:03}"
        for channel in CHANNELS:
            tifffile.imwrite(
                folder / f"plate1_{well}_T0001F{field}L01A01Z01C{channel}.tif",
                rng.integers(0, 4000, (SIDE, SIDE), dtype=np.uint16))
    return folder


def _peak_bytes(src):
    from spacr.core import preprocess_generate_masks
    from spacr.settings import set_default_settings_preprocess_generate_masks

    settings = set_default_settings_preprocess_generate_masks({
        "src": str(src), "metadata_type": "cellvoyager", "custom_regex": None,
        "channels": [0, 1, 2], "nucleus_channel": 0, "cell_channel": 1,
        "pathogen_channel": 2, "organelle_channel": None, "plot": False,
        "batch_size": 2, "test_mode": False, "timelapse": False,
        "normalize": True, "randomize": False, "verbose": False,
        "preprocess": True, "masks": True, "save": True, "adjust_cells": True,
        "keep_intermediate": True, "keep_original_images": True, "n_jobs": 1,
        "cell_diameter": 40, "nucleus_diameter": 20, "pathogen_diameter": 20,
        "magnification": 20})
    tracemalloc.start()
    try:
        preprocess_generate_masks(settings)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.slow
def test_mask_generation_peak_memory_does_not_scale_with_the_plate(
        tmp_path, fake_cellpose):
    _peak_bytes(_plate(tmp_path / "warmup", 2))
    small_fields, large_fields = 3, 12
    small = _peak_bytes(_plate(tmp_path / "small", small_fields))
    large_src = _plate(tmp_path / "large", large_fields)
    large = _peak_bytes(large_src)

    merged = sorted(p.name for p in (large_src / "merged").glob("*.npy"))
    assert len(merged) == large_fields
    stack = np.load(large_src / "merged" / merged[0])
    assert stack.shape[:2] == (SIDE, SIDE)
    assert stack.shape[2] >= len(CHANNELS) + 3

    field_bytes = SIDE * SIDE * len(CHANNELS) * np.dtype(np.float32).itemsize
    extra_plate = (large_fields - small_fields) * field_bytes
    assert large - small < 0.25 * extra_plate, (
        f"peak grew by {(large - small) / 1e6:.1f} MB for "
        f"{extra_plate / 1e6:.1f} MB of extra fields")
    assert large < large_fields * field_bytes + 64e6
