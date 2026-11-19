"""Item 984: ragged object crops in per-well pixel pools, read back pixel-exact."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("zarr")

from spacr.crop_store import (CROP_COLUMNS, CropPoolWriter, cut_crops,
                              export_crops, read_crop)


def _field(seed=0):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 60000, (3, 120, 160)).astype(np.uint16)
    labels = np.zeros((120, 160), np.int32)
    labels[2:20, 3:40] = 1
    labels[50:90, 60:75] = 2
    labels[100:120, 140:160] = 3
    labels[30:45, 100:130] = 5
    return image, labels


def _write(tmp_path, per_shard=2, margin=4, zero=False):
    image, labels = _field()
    writer = CropPoolWriter(tmp_path, "plate1", "B03", "cell", 3,
                            crops_per_shard=per_shard)
    writer.add(cut_crops(image, labels, margin=margin, zero_outside=zero),
               field=1, timepoint=0)
    return image, labels, writer.close()


def test_ragged_crops_read_back_exactly_the_box_plus_margin(tmp_path):
    image, labels, rows = _write(tmp_path)
    assert [r["object_label"] for r in rows] == [1, 2, 3, 5]
    assert set(CROP_COLUMNS) <= set(rows[0])
    assert [r["crop_array"] for r in rows] == ["cell/pool/0", "cell/pool/0",
                                               "cell/pool/1", "cell/pool/1"]
    for row in rows:
        y0, x0, y1, x1 = (row[k] for k in ("bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1"))
        pixels, mask = read_crop(tmp_path, row)
        assert pixels.dtype == image.dtype
        assert np.array_equal(pixels, image[:, y0:y1, x0:x1])
        assert np.array_equal(mask, labels[y0:y1, x0:x1] == row["object_label"])
        assert (row["height"], row["width"]) == (y1 - y0, x1 - x0)
    by_label = {r["object_label"]: r for r in rows}
    assert by_label[1]["clipped_at_edge"] and by_label[3]["clipped_at_edge"]
    assert not by_label[2]["clipped_at_edge"]
    assert (by_label[2]["bbox_y0"], by_label[2]["bbox_y1"]) == (46, 94)
    assert by_label[2]["crop_store"] == "crops/plate=plate1/well=B03.zarr"


def test_object_only_views_and_deterministic_resize(tmp_path):
    image, labels, rows = _write(tmp_path)
    row = rows[1]
    pixels, mask = read_crop(tmp_path, row, context=False)
    assert (pixels[:, ~mask] == 0).all()
    assert np.array_equal(pixels[:, mask], read_crop(tmp_path, row)[0][:, mask])
    small_a, _m = read_crop(tmp_path, row, size=(32, 32))
    small_b, _m = read_crop(tmp_path, row, size=(32, 32))
    assert small_a.shape == (3, 32, 32) and np.array_equal(small_a, small_b)
    _i, _l, zeroed = _write(tmp_path / "z", zero=True)
    stored, mask = read_crop(tmp_path / "z", zeroed[1])
    assert (stored[:, ~mask] == 0).all()


def test_exports_round_trip_pixel_exact(tmp_path):
    import tifffile

    image, labels, rows = _write(tmp_path)
    tiffs = export_crops(tmp_path, rows, tmp_path / "out", fmt="tiff")
    npys = export_crops(tmp_path, rows, tmp_path / "npy", fmt="npy")
    for row, tif, npy in zip(rows, tiffs, npys):
        pixels, mask = read_crop(tmp_path, row)
        assert np.array_equal(tifffile.imread(tif), pixels)
        assert np.array_equal(np.load(npy), pixels)
        assert np.array_equal(tifffile.imread(tif.replace(".tif", "_mask.tif")) > 0, mask)
    with pytest.raises(ValueError, match="8-bit"):
        export_crops(tmp_path, rows[:1], tmp_path / "png", fmt="png")


def test_shards_are_written_whole_and_never_rewritten(tmp_path):
    _image, _labels, rows = _write(tmp_path, per_shard=3)
    pool = tmp_path / "crops" / "plate=plate1" / "well=B03.zarr" / "cell" / "pool"
    first = (pool / "0" / "c").stat().st_mtime_ns if (pool / "0" / "c").exists() else None
    image, labels = _field(1)
    writer = CropPoolWriter(tmp_path, "plate1", "B03", "cell", 3, crops_per_shard=3)
    writer.add(cut_crops(image, labels), field=2)
    more = writer.close()
    assert {r["crop_array"] for r in more} == {"cell/pool/2", "cell/pool/3"}
    if first is not None:
        assert (pool / "0" / "c").stat().st_mtime_ns == first
    for row in rows + more:
        read_crop(tmp_path, row)
