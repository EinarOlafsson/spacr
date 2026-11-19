"""Item 984: ragged object crops in per-well pixel pools, read back pixel-exact."""
from __future__ import annotations

import json

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


def test_commit_records_pools_and_rows_and_a_stopped_run_is_discarded(tmp_path):
    from spacr.crop_store import discard_uncommitted, pool_attributes, read_commit

    image, labels = _field()
    attrs = {"channel_names": ["DAPI", "GFP", "AF750"], "pixel_size_um": 0.65,
             "settings_hash": "abc", "relationship_columns": ["nucleus_id"]}
    assert read_commit(tmp_path, "plate1", "B03", "cell") is None
    writer = CropPoolWriter(tmp_path, "plate1", "B03", "cell", 3, crops_per_shard=2,
                            attributes=attrs)
    writer.add(cut_crops(image, labels), field=1)
    record = writer.commit()
    assert record["pools"] == ["0", "1"] and record["n_rows"] == 4
    again = read_commit(tmp_path, "plate1", "B03", "cell")
    assert again["rows"] == json.loads(json.dumps(record["rows"])) and again["pools"] == ["0", "1"]
    stored = pool_attributes(tmp_path, "plate1", "B03", "cell")
    assert stored["channel_names"] == ["DAPI", "GFP", "AF750"]
    assert stored["dtype"] == "<u2" and stored["channels"] == 3 and stored["spacr_version"]
    stopped = CropPoolWriter(tmp_path, "plate1", "B03", "cell", 3, crops_per_shard=2,
                             attributes=attrs)
    stopped.add(cut_crops(image, labels), field=2)
    stopped.close()
    pool = tmp_path / "crops" / "plate=plate1" / "well=B03.zarr" / "cell" / "pool"
    (pool / ".9.tmp").mkdir()
    assert discard_uncommitted(tmp_path, "plate1", "B03", "cell") == ["2", "3"]
    assert sorted(p.name for p in pool.iterdir() if p.name != "zarr.json") == ["0", "1"]
    resumed = CropPoolWriter(tmp_path, "plate1", "B03", "cell", 3, crops_per_shard=2,
                             attributes=attrs)
    resumed.add(cut_crops(image, labels), field=2)
    final = resumed.commit()
    assert final["pools"] == ["0", "1", "2", "3"] and final["n_rows"] == 8
    for row in final["rows"]:
        pixels, _mask = read_crop(tmp_path, row)
        y0, x0, y1, x1 = (row[k] for k in ("bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1"))
        assert np.array_equal(pixels, image[:, y0:y1, x0:x1])
    with pytest.raises(ValueError, match="channel_names"):
        CropPoolWriter(tmp_path, "plate1", "B03", "cell", 3,
                       attributes={**attrs, "channel_names": ["GFP", "DAPI", "AF750"]})


def test_fields_with_time_or_depth_are_refused_not_flattened(tmp_path):
    from spacr.image_store import PlateWriter

    writer = PlateWriter(tmp_path, "P1")
    with pytest.raises(ValueError, match="only \\(C, Y, X\\) or \\(Y, X\\)"):
        writer.write_field("A01", 0, np.zeros((2, 3, 8, 8), np.uint16))
    with pytest.raises(ValueError, match="label image 'cell'"):
        writer.write_field("A01", 0, np.zeros((3, 8, 8), np.uint16),
                           {"cell": np.zeros((2, 8, 8), np.int32)})
    assert not (tmp_path / "P1.ome.zarr" / "A" / "1" / "0").exists()


def test_a_stale_measured_mask_record_is_reported(tmp_path):
    from spacr.crops import _write_measured_masks, measured_masks_state

    merged = tmp_path / "merged"
    merged.mkdir()
    stack = np.zeros((20, 20, 2), np.uint16)
    stack[2:8, 2:8, 1] = 1
    path = str(merged / "plate1_A01_1.npy")
    np.save(path, stack)
    assert measured_masks_state(path) == "none"
    measured = stack[..., 1].copy()
    measured[2:4, 2:8] = 0
    _write_measured_masks(path, {1: measured})
    assert measured_masks_state(path) == "current"
    stack[10:15, 10:15, 1] = 2
    np.save(path, stack)
    assert measured_masks_state(path) == "stale"
