"""Crops are exported raw, and every spaCR reader still sees the old stretch.

The maintainer's rule (2026-11-05, restated 2026-11-07 for the BMC gene-class
benchmark): crop files keep the source dtype and intensities, with only the
outside-object pixels set to 0. Normalisation belongs to the model's input
transform. A per-crop 0-100 stretch on export erased the intensity phenotype.

What these tests pin:

* a raw export keeps uint16 and the stored values (no stretch), in Measure and
  in ``extract_crop(raw=True)``;
* the folder is marked crop format 4, and ``read_crop_png`` and both
  classification decoding policies return the same 8-bit pixels for a raw
  folder as for the old stretched folder of the same objects;
* ``crop_intensity='auto'`` keeps an explicit ``normalize`` percentile pair
  stretched, so an older settings file writes what it wrote before;
* with flat-field correction on, crops are cut from the corrected field, so
  correcting first and cropping second gives the crop on disk.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from spacr import crops
from spacr.classification_pixels import (
    DECLARED_UINT8, STORED_PIL, read_classification_image,
)
from spacr.settings import get_measure_crop_settings

SIGNAL = 3000


def _field(size=96, n_channels=3):
    rng = np.random.default_rng(1)
    cell = np.zeros((size, size), np.uint16)
    yy, xx = np.mgrid[:size, :size]
    for i, (cy, cx) in enumerate([(28, 28), (28, 68), (68, 40)], start=1):
        cell[(yy - cy) ** 2 + (xx - cx) ** 2 <= 14 ** 2] = i
    nucleus = np.zeros_like(cell)
    pathogen = np.zeros_like(cell)
    planes = []
    for c in range(n_channels):
        base = rng.integers(150, 400, size=(size, size)).astype(np.uint16)
        base[cell > 0] += rng.integers(0, SIGNAL * (c + 1),
                                       size=int((cell > 0).sum())).astype(np.uint16)
        planes.append(base)
    return np.stack(planes + [cell, nucleus, pathogen], axis=-1).astype(np.uint16)


def _settings(merged, **over):
    s = get_measure_crop_settings(settings={})
    s.update({
        "src": str(merged), "channels": [0, 1, 2],
        "cell_mask_dim": 3, "nucleus_mask_dim": None, "pathogen_mask_dim": None,
        "png_dims": [0, 1, 2], "png_size": [40, 40],
        "save_measurements": True, "save_png": True, "save_arrays": False,
        "plot": False, "verbose": False, "timelapse": False,
        "crop_mode": ["cell"], "normalize": False, "normalize_by": "png",
        "experiment": "exp", "n_jobs": 1, "test_mode": False,
        "cytoplasm": False, "cell_min_size": 1, "nucleus_min_size": 1,
        "pathogen_min_size": 1, "cytoplasm_min_size": 1,
    })
    s.update(over)
    return s


def _run(tmp_path, tag, data, **over):
    from spacr.measure import _measure_crop_core

    root = tmp_path / tag
    merged = root / "merged"
    merged.mkdir(parents=True)
    (root / "measurements").mkdir(parents=True)
    name = "plate1_A01_F001.npy"
    np.save(merged / name, data)
    crops.clear_crop_format_cache()
    _measure_crop_core(0, [], name, _settings(merged, **over))
    pngs = sorted(
        os.path.join(d, f) for d, _, files in os.walk(root)
        for f in files if f.endswith(".png") and "cell_png" in d)
    assert pngs, "the run wrote no crop PNGs"
    return pngs


def test_resolve_crop_intensity():
    assert crops._resolve_crop_intensity({}) == "raw"
    assert crops._resolve_crop_intensity({"normalize": False}) == "raw"
    assert crops._resolve_crop_intensity({"normalize": [1, 99]}) == "stretch"
    assert crops._resolve_crop_intensity({"normalize": "1,99"}) == "stretch"
    assert crops._resolve_crop_intensity(
        {"normalize": [1, 99], "crop_intensity": "raw"}) == "raw"
    assert crops._resolve_crop_intensity({"crop_intensity": "stretch"}) == "stretch"
    with pytest.raises(crops.CropError):
        crops._resolve_crop_intensity({"crop_intensity": "minmax"})


def test_measure_writes_raw_uint16_crops_without_a_stretch(tmp_path):
    data = _field()
    pngs = _run(tmp_path, "raw", data)
    folder = os.path.dirname(pngs[0])
    marker = crops.read_crop_folder_marker(folder, use_cache=False)
    assert marker["spacr_crop_format"] == 4
    for path in pngs:
        arr = crops._read_crop_array(path)
        assert arr.dtype == np.uint16
        nonzero = arr[arr > 0]
        assert nonzero.max() < 65535, "a stretch would reach the dtype maximum"
        assert set(np.unique(nonzero)) <= set(np.unique(data[..., :3])), \
            "every written value must be a stored field value"


def test_raw_folder_reads_back_like_the_old_stretched_folder(tmp_path):
    data = _field()
    raw = _run(tmp_path, "raw", data, crop_intensity="raw")
    old = _run(tmp_path, "stretch", data, crop_intensity="stretch")
    assert [os.path.basename(p) for p in raw] == [os.path.basename(p) for p in old]
    assert crops.read_crop_folder_marker(os.path.dirname(old[0]))[
        "spacr_crop_format"] == crops.CROP_FORMAT_CURRENT
    for r, o in zip(raw, old):
        assert np.array_equal(crops.read_crop_png(r), crops.read_crop_png(o))
        assert np.array_equal(crops.read_crop_png(r, fmt=3), crops.read_crop_png(o))
        for policy in (DECLARED_UINT8, STORED_PIL):
            a = np.asarray(read_classification_image(r, policy))
            b = np.asarray(read_classification_image(o, policy))
            assert np.array_equal(a, b), policy
        with open(r, "rb") as handle:
            from io import BytesIO
            streamed = read_classification_image(BytesIO(handle.read()),
                                                 DECLARED_UINT8, fmt=4)
        assert np.array_equal(np.asarray(streamed),
                              np.asarray(read_classification_image(o, DECLARED_UINT8)))


def test_an_explicit_percentile_pair_still_stretches_under_auto(tmp_path):
    pngs = _run(tmp_path, "pct", _field(), normalize=[1, 99])
    marker = crops.read_crop_folder_marker(os.path.dirname(pngs[0]), use_cache=False)
    assert marker["spacr_crop_format"] == crops.CROP_FORMAT_CURRENT


def test_extract_crop_raw_keeps_dtype_and_values(tmp_path):
    data = _field()
    path = tmp_path / "plate1_A01_F001.npy"
    np.save(path, data)
    mask_dims = {"cell": 3}
    raw = crops.extract_crop(str(path), "cell", 1, channels=(0, 1, 2),
                             size=(40, 40), mask_dims=mask_dims, raw=True)
    stretched = crops.extract_crop(str(path), "cell", 1, channels=(0, 1, 2),
                                   size=(40, 40), mask_dims=mask_dims)
    assert raw.dtype == np.uint16
    region = data[..., 3] == 1
    assert raw.max() == data[..., :3][region].max()
    assert stretched.max() == 65535
    assert np.array_equal(crops._stretched_view(raw), crops.png_view(stretched))


def test_flat_field_is_applied_to_the_field_before_cropping(tmp_path):
    from spacr.measure_hooks import (
        register_preprocessing_hook, unregister_preprocessing_hook,
    )

    data = _field()
    size = data.shape[0]
    gain = np.linspace(0.6, 1.4, size, dtype=np.float32)[None, :, None]

    def flat_field(channel_arrays, context):
        out = np.rint(channel_arrays.astype(np.float32) * gain)
        return np.clip(out, 0, 65535).astype(channel_arrays.dtype)

    name = register_preprocessing_hook(flat_field, name="test_flat_field_gain")
    try:
        pngs = _run(tmp_path, "ff", data, crop_intensity="raw",
                    illumination_correction=True)
    finally:
        unregister_preprocessing_hook(name)
    corrected = data.copy()
    corrected[..., :3] = flat_field(data[..., :3], None)
    np.save(tmp_path / "corrected.npy", corrected)
    np.save(tmp_path / "uncorrected.npy", data)
    kwargs = dict(channels=(2, 1, 0), size=(40, 40), mask_dims={"cell": 3}, raw=True)

    def views(path):
        out = []
        for label in (1, 2, 3):
            crop = crops.extract_crop(str(path), "cell", label, **kwargs)
            out.extend([crop, crop[:, :, ::-1]])
        return out

    want = views(tmp_path / "corrected.npy")
    unwanted = views(tmp_path / "uncorrected.npy")
    for png in pngs:
        on_disk = crops._read_crop_array(png)
        assert any(np.array_equal(on_disk, w) for w in want), png
        assert not any(np.array_equal(on_disk, u) for u in unwanted), png
