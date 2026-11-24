"""Item 983: Spatial Mask Generation writes its fields in OME-Zarr like Mask Generation."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("zarr")

from spacr import ops_engine as oe
from spacr.image_store import read_image, read_labels, split_field_name
from tests.test_spatial_submodules import _mask_settings, xenium_slide


def test_the_default_writes_only_merged_npy(tmp_path):
    from spacr.ops_settings import SPATIAL_MASK_DEFAULTS

    assert SPATIAL_MASK_DEFAULTS["mask_image_format"] == "npy"
    source = xenium_slide(tmp_path / "bundle")
    report = oe.run_spatial_masks(_mask_settings(source, tmp_path / "out"))
    assert report["image_format"] == "npy" and report["image_store"] is None
    assert not (tmp_path / "out" / "images").exists()


def test_ome_zarr_fields_and_masks_equal_the_merged_stacks(tmp_path):
    """Item 0aj: spatial OME-Zarr writes fov.ome.zarr through the shared writer, equal to the npy run."""
    from spacr.image_store import _list_fields, _read_field

    source = xenium_slide(tmp_path / "bundle")
    dst, ref = tmp_path / "out", tmp_path / "ref"
    report = oe.run_spatial_masks(_mask_settings(source, dst,
                                                 mask_image_format="ome-zarr"))
    oe.run_spatial_masks(_mask_settings(xenium_slide(tmp_path / "bundle2"), ref))
    merged = sorted((ref / "merged").glob("*.npy"))
    assert merged and report["image_format"] == "ome-zarr"
    assert not list((dst / "merged").glob("*.npy")) if (dst / "merged").exists() else True
    layout = json.loads((ref / "merged" / ".spacr_plane_layout.json").read_text())
    store = dst / "fov.ome.zarr"
    assert _list_fields(store) == sorted(p.name for p in merged)
    for path in merged:
        plate, well, field = split_field_name(path.name)
        stack = np.load(path)
        assert np.array_equal(_read_field(store / path.name), stack)
        image = read_image(store, key=(well, field))
        assert np.array_equal(image, np.moveaxis(stack[..., layout["intensity_channels"]], -1, 0))
        for role, dim in layout["mask_dims"].items():
            assert np.array_equal(read_labels(store, role, key=(well, field)), stack[..., dim])
    assert sum(report["image_store"].values()) == len(merged)
