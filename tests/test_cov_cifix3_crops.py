"""Measured-mask records and raw crop decoding in the crops module.

Measure records the label planes it measured beside the merged field, and
raw crops are 16-bit PNGs. These tests hold the record's cytoplasm plane, its
refusal of unreadable or out-of-range records, the cache dropping a field
whose record changed, and the raw-crop readers for every channel layout.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from spacr import crops

cv2 = pytest.importorskip("cv2")


def _merged(tmp_path, name="plate1_A01_F001.npy"):
    cell = np.zeros((20, 20), np.uint16)
    cell[2:10, 2:10] = 1
    cell[12:18, 12:18] = 2
    intensity = np.full((20, 20), 500, np.uint16)
    path = tmp_path / "merged" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.stack([intensity, cell], axis=-1))
    return str(path), cell


def test_a_cytoplasm_alone_is_recorded_and_served(tmp_path):
    path, cell = _merged(tmp_path)
    cytoplasm = np.zeros_like(cell)
    cytoplasm[3:6, 3:6] = 1
    record = crops._write_measured_masks(path, {1: cell}, cytoplasm=cytoplasm)
    assert record is not None and os.path.isfile(record)
    field = crops.open_merged_field(path, mask_dims={"cell": 1}, use_cache=False)
    np.testing.assert_array_equal(field.mask_plane("cytoplasm"), cytoplasm)
    assert "cytoplasm" in field._derived


def test_a_failed_record_write_leaves_no_temp_file(tmp_path, monkeypatch):
    path, cell = _merged(tmp_path)
    changed = cell.copy()
    changed[12:18, 12:18] = 0

    def boom(handle, **arrays):
        raise RuntimeError("disk full")

    monkeypatch.setattr(np, "savez_compressed", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        crops._write_measured_masks(path, {1: changed})
    folder = os.path.dirname(crops._measured_masks_path(path))
    assert os.listdir(folder) == []


def test_a_record_naming_a_plane_the_field_lacks_is_ignored(tmp_path):
    path, cell = _merged(tmp_path)
    target = crops._measured_masks_path(path)
    os.makedirs(os.path.dirname(target))
    np.savez(target, plane_5=cell, source_dims=np.asarray([5]),
             source_digest=np.asarray("x"))
    array = np.load(path, mmap_mode="r")
    assert crops._read_measured_masks(path, array) == {}


def test_an_unreadable_record_is_a_corrupt_merged_file(tmp_path):
    path, _cell = _merged(tmp_path)
    target = crops._measured_masks_path(path)
    os.makedirs(os.path.dirname(target))
    with open(target, "wb") as handle:
        handle.write(b"not an npz archive")
    array = np.load(path, mmap_mode="r")
    with pytest.raises(crops.CorruptMergedFile, match="cannot read measured-mask"):
        crops._read_measured_masks(path, array)


def test_a_cached_field_is_reopened_when_its_record_changed(tmp_path):
    crops.clear_field_cache()
    path, cell = _merged(tmp_path)
    first = crops.open_merged_field(path, mask_dims={"cell": 1})
    np.testing.assert_array_equal(first.mask_plane("cell"), cell)
    changed = cell.copy()
    changed[12:18, 12:18] = 0
    crops._write_measured_masks(path, {1: changed})
    second = crops.open_merged_field(path, mask_dims={"cell": 1})
    assert second is not first
    np.testing.assert_array_equal(second.mask_plane("cell"), changed)
    crops.clear_field_cache()


def _png(tmp_path, array, name="crop.png"):
    path = str(tmp_path / name)
    assert cv2.imwrite(path, array)
    return path


def test_raw_crops_decode_in_every_channel_layout(tmp_path):
    gray = np.arange(16, dtype=np.uint16).reshape(4, 4) * 1000
    assert crops._read_crop_array(_png(tmp_path, gray)).shape == (4, 4, 1)
    bgra = np.zeros((4, 4, 4), np.uint16)
    bgra[..., 0], bgra[..., 2], bgra[..., 3] = 100, 300, 65535
    four = crops._read_crop_array(_png(tmp_path, bgra, "four.png"))
    assert four.shape == (4, 4, 3)
    assert (four[0, 0] == [300, 0, 100]).all()


def test_unreadable_crop_bytes_raise_a_crop_error(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"nope")
    with pytest.raises(crops.CropError, match="not a readable image"):
        crops._read_crop_array(str(bad))
    import io
    with pytest.raises(crops.CropError, match="not a readable image"):
        crops._read_crop_array(io.BytesIO(b"nope"))

    class ReadOnly:
        def read(self):
            return b"nope"

    with pytest.raises(crops.CropError, match="not a readable image"):
        crops._read_crop_array(ReadOnly())


def test_the_stretched_view_handles_one_channel_and_float_crops():
    plane = np.linspace(0, 1, 36, dtype=np.float32).reshape(6, 6) * 4000
    view = crops._stretched_view(plane)
    assert view.shape == (6, 6, 3) and view.dtype == np.uint8
    assert (view[..., 0] == view[..., 1]).all()
    single = crops._stretched_view(plane[:, :, None].astype(np.uint16))
    assert single.shape == (6, 6, 3)


def _raw_png(tmp_path):
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 4000, (8, 8, 3)).astype(np.uint16)
    path = _png(tmp_path, arr[:, :, ::-1], "raw.png")
    return path, arr


def test_decode_crop_image_reads_a_raw_crop_from_its_file_or_pixels(tmp_path):
    from PIL import Image

    path, arr = _raw_png(tmp_path)
    expected = crops._stretched_view(arr)
    with Image.open(path) as image:
        from_file = crops.decode_crop_image(image, fmt=4, as_format=3)
    np.testing.assert_array_equal(from_file, expected)
    flipped = Image.fromarray(np.ascontiguousarray(expected))
    np.testing.assert_array_equal(
        crops.decode_crop_image(flipped, fmt=4, as_format=3),
        crops._stretched_view(np.asarray(flipped)))
    with Image.open(path) as image:
        reversed_order = crops.decode_crop_image(image, fmt=4, as_format=2)
    np.testing.assert_array_equal(reversed_order, expected[:, :, ::-1])


def test_read_crop_png_returns_a_raw_folder_in_the_requested_order(tmp_path):
    path, arr = _raw_png(tmp_path)
    crops.write_crop_folder_marker(str(tmp_path), crops._CROP_FORMAT_DECLARED_RAW)
    crops.clear_crop_format_cache() if hasattr(crops, "clear_crop_format_cache") else None
    expected = crops._stretched_view(arr)
    np.testing.assert_array_equal(crops.read_crop_png(path), expected)
    np.testing.assert_array_equal(
        crops.read_crop_png(path, fmt=3, as_format=2), expected[:, :, ::-1])
