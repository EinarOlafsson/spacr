"""Mask saving must reject uint16 overflow before replacing an existing file."""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from spacr.qt import mask_engine as engine
from spacr.tiff_io import write_tiff


def _binary_components(count):
    mask = np.zeros((513, 513), np.uint8)
    sites = mask[::2, ::2]
    rows, columns = np.unravel_index(np.arange(count), sites.shape)
    sites[rows, columns] = 1
    return mask


def _oversized_ids():
    return np.array([[0, 1, 0], [0, 2**63 + 7, 0]], np.uint64)


def test_binary_mask_at_exact_uint16_capacity_preserves_every_object():
    mask = _binary_components(65535)
    before = mask.copy()
    result = engine.canonical_labels(mask)
    assert result.dtype == np.uint16
    np.testing.assert_array_equal(result > 0, mask > 0)
    np.testing.assert_array_equal(np.unique(result), np.arange(65536))
    np.testing.assert_array_equal(mask, before)


@pytest.mark.parametrize("count", [65536, 66049])
def test_excess_binary_components_are_rejected_without_changing_input(count):
    mask = _binary_components(count)
    before = mask.copy()
    with pytest.raises(ValueError, match="uint16"):
        engine.canonical_labels(mask)
    np.testing.assert_array_equal(mask, before)


@pytest.mark.parametrize("large", [65536, 2**63 + 7, 2**64 - 1])
def test_multilabel_uint64_ids_cannot_wrap_into_smaller_ids(large):
    mask = np.array([[0, 1, 0], [0, large, 0]], np.uint64)
    before = mask.copy()
    with pytest.raises(ValueError, match="uint16"):
        engine.canonical_labels(mask)
    np.testing.assert_array_equal(mask, before)


def test_single_foreground_uint64_value_retains_binary_interpretation():
    mask = np.array([[0, 2**64 - 1, 0, 2**64 - 1]], np.uint64)
    result = engine.canonical_labels(mask)
    np.testing.assert_array_equal(result, [[0, 1, 0, 2]])
    with pytest.raises(ValueError, match="uint16"):
        engine.canonical_labels(mask, preserve_ids=True)


@pytest.mark.parametrize("kind", ["tiff", "bundle"])
@pytest.mark.parametrize("overflow", ["components", "uint64_ids"])
def test_failed_save_keeps_existing_file_and_metadata_byte_for_byte(tmp_path, kind, overflow):
    attempted = _binary_components(65536) if overflow == "components" else _oversized_ids()
    original = np.zeros(attempted.shape, np.uint16)
    original[0, 0] = 9
    if kind == "tiff":
        name = "field.tif"
        target = tmp_path / "masks/field.tif"
        target.parent.mkdir()
        write_tiff(str(target), original)
    else:
        name = "field_seg.npy"
        target = tmp_path / name
        np.save(target, {"masks": original, "img": np.zeros(original.shape, np.uint16),
                         "ismanual": np.array([False] * 9),
                         "outlines": original.copy(), "curator_note": "keep this"})
    before = target.read_bytes()
    with pytest.raises(ValueError, match="uint16"):
        engine.save_mask(str(tmp_path), name, attempted)
    assert target.read_bytes() == before
    assert not list(tmp_path.rglob("*.tmp"))
    if kind == "tiff":
        np.testing.assert_array_equal(tifffile.imread(target), original)
    else:
        saved = np.load(target, allow_pickle=True).item()
        np.testing.assert_array_equal(saved["masks"], original)
        assert saved["curator_note"] == "keep this"
        assert not saved["ismanual"].any()
