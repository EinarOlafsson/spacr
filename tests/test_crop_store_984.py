"""Item 984: ragged object crops in per-well pixel pools, read back pixel-exact."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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
        assert mask.dtype == np.dtype(bool)
        assert np.array_equal(pixels, image[:, y0:y1, x0:x1])
        assert np.array_equal(mask, labels[y0:y1, x0:x1] == row["object_label"])
        assert (row["height"], row["width"]) == (y1 - y0, x1 - x0)
    by_label = {r["object_label"]: r for r in rows}
    assert by_label[1]["clipped_at_edge"] and by_label[3]["clipped_at_edge"]
    assert not by_label[2]["clipped_at_edge"]
    assert (by_label[2]["bbox_y0"], by_label[2]["bbox_y1"]) == (46, 94)
    assert by_label[2]["crop_store"] == "crop.ome.zarr/plate=plate1/well=B03"


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
    pool = tmp_path / "crop.ome.zarr" / "plate=plate1" / "well=B03" / "cell" / "pool"
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
    assert "mask_dtype" not in stored
    assert stored["channel_names"] == ["DAPI", "GFP", "AF750"]
    assert stored["dtype"] == "<u2" and stored["channels"] == 3 and stored["spacr_version"]
    stopped = CropPoolWriter(tmp_path, "plate1", "B03", "cell", 3, crops_per_shard=2,
                             attributes=attrs)
    stopped.add(cut_crops(image, labels), field=2)
    stopped.close()
    pool = tmp_path / "crop.ome.zarr" / "plate=plate1" / "well=B03" / "cell" / "pool"
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


@pytest.mark.parametrize("positions", [(3, 0, 2, 1), (2, 0, 2, 3)],
                         ids=["dense", "sparse-duplicate"])
def test_batched_reads_equal_one_by_one_reads_in_any_row_order(tmp_path, positions):
    """Real pools retain pixels and row order for dense and sparse duplicate requests."""
    from spacr.crop_store import OmeZarrCropSource, read_crops

    _image, _labels, rows = _write(tmp_path, per_shard=3)
    order = [rows[i] for i in positions]
    for kwargs in ({}, {"context": False}, {"size": (16, 20)},
                   {"context": False, "size": (16, 20)}):
        batch = read_crops(tmp_path, order, **kwargs)
        assert len(batch) == len(order)
        for row, (pixels, mask) in zip(order, batch):
            want_pixels, want_mask = read_crop(tmp_path, row, **kwargs)
            assert pixels.dtype == want_pixels.dtype and np.array_equal(pixels, want_pixels)
            assert mask.dtype == want_mask.dtype and np.array_equal(mask, want_mask)
    source = OmeZarrCropSource(str(tmp_path))
    assert all(np.array_equal(a, source.get(r)) for a, r in zip(source.get_many(order), order))


def _ternary_crops():
    """Two differently sized crops with background, cell and nucleus labels."""
    masks = (np.array([[0, 1, 2, 0], [1, 2, 2, 1], [0, 1, 0, 2]], np.uint8),
             np.array([[2, 1, 0], [0, 2, 1]], np.uint8))
    crops = []
    for label, mask in enumerate(masks, start=1):
        height, width = mask.shape
        pixels = np.arange(2 * height * width, dtype=np.uint16).reshape(2, height, width)
        crops.append({"object_label": label, "pixels": pixels + 100 * label,
                      "mask": mask, "bbox_y0": 0, "bbox_x0": 0,
                      "bbox_y1": height, "bbox_x1": width, "margin_px": 0,
                      "clipped_at_edge": False})
    return crops


def test_uint8_masks_keep_labels_through_scalar_batch_raw_context_and_resize(tmp_path):
    """Every public raw read preserves ternary labels and treats both labels as foreground."""
    from spacr.crop_store import OmeZarrCropSource, read_crops

    crops = _ternary_crops()
    writer = CropPoolWriter(tmp_path, "plate1", "B03", "cell", 2, mask_dtype=np.uint8)
    writer.add(crops, field=1)
    record = writer.commit()
    assert record["attributes"]["mask_dtype"] == np.dtype(np.uint8).str
    order = (1, 0, 1)
    rows = [record["rows"][index] for index in order]
    for context in (True, False):
        for size in (None, (6, 8)):
            source = OmeZarrCropSource(str(tmp_path), context=context, size=size)
            batch = read_crops(tmp_path, rows, context=context, size=size)
            for index, row, batched in zip(order, rows, batch):
                crop = crops[index]
                expected_pixels, expected_mask = crop["pixels"].copy(), crop["mask"]
                if not context:
                    expected_pixels[:, expected_mask == 0] = 0
                if size is not None:
                    ys = [0, 0, 1, 1, 2, 2] if index == 0 else [0, 0, 0, 1, 1, 1]
                    xs = [0, 0, 1, 1, 2, 2, 3, 3] if index == 0 else [0, 0, 0, 1, 1, 1, 2, 2]
                    expected_pixels = expected_pixels[:, ys][:, :, xs]
                    expected_mask = expected_mask[ys][:, xs]
                for pixels, mask in (batched, source.get_raw(row),
                                     read_crop(tmp_path, row, context=context, size=size)):
                    assert pixels.dtype == np.dtype(np.uint16)
                    assert mask.dtype == np.dtype(np.uint8)
                    np.testing.assert_array_equal(pixels, expected_pixels)
                    np.testing.assert_array_equal(mask, expected_mask)
                    assert set(np.unique(mask)) == {0, 1, 2}


@pytest.mark.parametrize("absent_flag,present_flag", [(False, True), (np.bool_(False), np.bool_(True)),
                                                     (0, 1)], ids=["json", "numpy", "sqlite"])
def test_absent_mask_placeholders_keep_pixels_in_scalar_batch_and_source_views(
        tmp_path, absent_flag, present_flag):
    """Explicit absent masks preserve pixels under context removal without changing genuine masks."""
    from spacr.crop_store import OmeZarrCropSource, read_crops

    absent, present = _ternary_crops()
    absent.update(mask=np.zeros_like(absent["mask"]), has_mask=False)
    present["has_mask"] = True
    writer = CropPoolWriter(tmp_path, "plate1", "B03", "cell", 2, mask_dtype=np.uint8)
    writer.add([absent, present], field=1)
    rows = writer.commit()["rows"]
    rows[0]["has_mask"], rows[1]["has_mask"] = absent_flag, present_flag
    order = (0, 1, 0)
    for size in (None, (6, 8)):
        source = OmeZarrCropSource(str(tmp_path), context=False, size=size)
        batch = read_crops(tmp_path, [rows[index] for index in order], context=False, size=size)
        for index, batched in zip(order, batch):
            crop = (absent, present)[index]
            expected, mask = crop["pixels"].copy(), crop["mask"]
            if index == 1:
                expected[:, mask == 0] = 0
            if size is not None:
                ys = [0, 0, 1, 1, 2, 2] if index == 0 else [0, 0, 0, 1, 1, 1]
                xs = [0, 0, 1, 1, 2, 2, 3, 3] if index == 0 else [0, 0, 0, 1, 1, 1, 2, 2]
                expected, mask = expected[:, ys][:, :, xs], mask[ys][:, xs]
            for pixels, actual_mask in (batched, source.get_raw(rows[index]),
                                       read_crop(tmp_path, rows[index], context=False, size=size)):
                assert pixels.dtype == np.dtype(np.uint16)
                assert actual_mask.dtype == np.dtype(np.uint8)
                np.testing.assert_array_equal(pixels, expected)
                np.testing.assert_array_equal(actual_mask, mask)
                if index == 0:
                    assert pixels.any() and not actual_mask.any()
    for flag in (None, "", "False", 0.0):
        pixels, mask = read_crop(tmp_path, {**rows[0], "has_mask": flag}, context=False)
        assert not pixels.any() and not mask.any()


@pytest.mark.parametrize("mask_dtype, conflicting", [(bool, np.uint8), (np.uint8, bool)],
                         ids=["legacy-bool", "labelled-uint8"])
def test_mask_dtype_resume_preserves_old_pools_and_refuses_conflicts(tmp_path, mask_dtype, conflicting):
    """A resumed class keeps its mask representation, including legacy implicit Boolean metadata."""
    from spacr.crop_store import class_dir, pool_attributes

    crops = _ternary_crops()
    writer = CropPoolWriter(tmp_path, "plate1", "B03", "cell", 2, mask_dtype=mask_dtype)
    writer.add(crops[:1], field=1)
    first = writer.commit()
    base = class_dir(tmp_path, "plate1", "B03", "cell")
    original = {path: path.read_bytes() for kind in ("pool", "mask")
                for path in (base / kind / "0").rglob("*") if path.is_file()}
    metadata = (base / "zarr.json").read_bytes()
    with pytest.raises(ValueError, match="mask_dtype"):
        CropPoolWriter(tmp_path, "plate1", "B03", "cell", 2, mask_dtype=conflicting)
    with pytest.raises(ValueError, match="mask_dtype"):
        CropPoolWriter(tmp_path, "plate1", "B03", "cell", 2, mask_dtype=mask_dtype,
                       attributes={"mask_dtype": np.dtype(conflicting).str})
    assert (base / "zarr.json").read_bytes() == metadata
    resumed = CropPoolWriter(tmp_path, "plate1", "B03", "cell", 2, mask_dtype=mask_dtype)
    resumed.add(crops[1:], field=2)
    final = resumed.commit()
    assert final["n_rows"] == 2 and final["pools"] == ["0", "1"]
    assert final["rows"][0] == first["rows"][0]
    assert all(path.read_bytes() == contents for path, contents in original.items())
    assert ("mask_dtype" in pool_attributes(tmp_path, "plate1", "B03", "cell")) == (mask_dtype is np.uint8)
    for row, crop in zip(final["rows"], crops):
        pixels, mask = read_crop(tmp_path, row)
        assert mask.dtype == np.dtype(mask_dtype)
        np.testing.assert_array_equal(pixels, crop["pixels"])
        np.testing.assert_array_equal(mask, crop["mask"].astype(mask_dtype))


def test_measure_sink_keeps_uint8_masks_of_untouched_fields(tmp_path):
    """Replacing another field cannot collapse carried cell/nucleus labels to Boolean."""
    from spacr.crop_store import CropSink

    crops = _ternary_crops()
    writer = CropPoolWriter(tmp_path, "plate1", "B03", "cell", 2, mask_dtype=np.uint8)
    writer.add(crops, field=1)
    writer.commit()
    rows = CropSink(tmp_path, ["plate1_B03_2.npy"]).close()
    assert len(rows) == len(crops)
    for row, crop in zip(rows, crops):
        pixels, mask = read_crop(tmp_path, row)
        assert row["field"] == 1 and mask.dtype == np.dtype(np.uint8)
        np.testing.assert_array_equal(pixels, crop["pixels"])
        np.testing.assert_array_equal(mask, crop["mask"])


@pytest.mark.parametrize("invalid", [-1, 256, 1.5])
def test_uint8_masks_refuse_values_that_would_be_lost_on_cast(tmp_path, invalid):
    """Opt-in labelled masks reject wraparound and fractional truncation before buffering."""
    crop = _ternary_crops()[0]
    crop["mask"] = np.full(crop["mask"].shape, invalid)
    writer = CropPoolWriter(tmp_path, "plate1", "B03", "cell", 2, mask_dtype=np.uint8)
    with pytest.raises(ValueError, match="integers from 0 to 255"):
        writer.add([crop], field=1)
    assert writer.close() == []


class _RecordedCropArray:
    """An array whose public slices are recorded before returning independent read buffers."""

    def __init__(self, values, chunks):
        """Keep deterministic stored pixels and an empty slice log."""
        self.values = values
        self.shape = values.shape
        self.chunks = chunks
        self.reads = []

    def __getitem__(self, selection):
        """Record the exact public array selection, as a Zarr read would receive it."""
        self.reads.append(selection)
        return self.values[selection].copy()


@pytest.fixture
def recorded_crop_pools(monkeypatch, tmp_path):
    """Replace only array opening with two deterministic pools and independent mask offsets."""
    from spacr import ome_zarr

    arrays, opened = {}, []
    for number, store in enumerate(("a.zarr", "b.zarr")):
        pixels = np.arange(256, dtype=np.uint16).reshape(2, 128) + 1000 * number
        mask = (np.arange(128) * 3 + number) % 7 < 3
        arrays[store + "/cell/pool/0"] = _RecordedCropArray(pixels, (2, 8))
        arrays[store + "/cell/mask/0"] = _RecordedCropArray(mask, (4,))

    def open_array(path, mode):
        """Resolve the requested array while recording exactly one open per caller."""
        assert mode == "r"
        relative = Path(path).relative_to(tmp_path).as_posix()
        opened.append(relative)
        return arrays[relative]

    def require_zarr():
        """Provide the recorded public array opener without importing a storage backend."""
        return SimpleNamespace(open_array=open_array)

    monkeypatch.setattr(ome_zarr, "require_zarr", require_zarr)
    return arrays, opened


def _pool_row(store, offset, mask_offset, *, height=2, width=2):
    """Build one public crop row with separately addressable image and mask ranges."""
    return {"crop_store": store, "crop_array": "cell/pool/0", "offset": offset,
            "mask_offset": mask_offset, "height": height, "width": width}


def test_dense_crop_ranges_use_one_slice_with_overlap_and_duplicates(tmp_path, recorded_crop_pools):
    """Unordered contiguous requests keep one read per array and every requested result."""
    from spacr.crop_store import read_crops

    arrays, opened = recorded_crop_pools
    rows = [_pool_row("a.zarr", offset, mask_offset)
            for offset, mask_offset in ((8, 24), (0, 16), (4, 20), (4, 20), (2, 18))]
    actual = read_crops(tmp_path, iter(rows))
    assert opened == ["a.zarr/cell/pool/0", "a.zarr/cell/mask/0"]
    assert arrays[opened[0]].reads == [(slice(None), slice(0, 12))]
    assert arrays[opened[1]].reads == [slice(16, 28)]
    assert len(actual) == len(rows)
    for row, (pixels, mask) in zip(rows, actual):
        want_pixels, want_mask = read_crop(tmp_path, row)
        assert pixels.dtype == want_pixels.dtype and np.array_equal(pixels, want_pixels)
        assert mask.dtype == want_mask.dtype and np.array_equal(mask, want_mask)


@pytest.mark.parametrize("context,size", [(True, None), (False, None),
                                          (True, (3, 5)), (False, (3, 5))])
def test_sparse_crop_ranges_skip_gaps_and_keep_independent_mask_order(
        tmp_path, recorded_crop_pools, context, size):
    """Read sparse ranges once, preserving cross-pool order, duplicates and finishing options."""
    from spacr.crop_store import read_crops

    arrays, opened = recorded_crop_pools
    rows = [_pool_row("a.zarr", 96, 20),
            _pool_row("b.zarr", 4, 64, width=3),
            _pool_row("a.zarr", 0, 96),
            _pool_row("a.zarr", 4, 0),
            _pool_row("a.zarr", 96, 20),
            _pool_row("b.zarr", 10, 0)]
    actual = read_crops(tmp_path, iter(rows), context=context, size=size)
    assert opened == ["a.zarr/cell/pool/0", "a.zarr/cell/mask/0",
                      "b.zarr/cell/pool/0", "b.zarr/cell/mask/0"]
    assert arrays["a.zarr/cell/pool/0"].reads == [(slice(None), slice(0, 8)),
                                                (slice(None), slice(96, 100))]
    assert arrays["a.zarr/cell/mask/0"].reads == [slice(0, 4), slice(20, 24), slice(96, 100)]
    assert arrays["b.zarr/cell/pool/0"].reads == [(slice(None), slice(4, 14))]
    assert arrays["b.zarr/cell/mask/0"].reads == [slice(0, 4), slice(64, 70)]
    assert len(actual) == len(rows)
    for row, (pixels, mask) in zip(rows, actual):
        want_pixels, want_mask = read_crop(tmp_path, row, context=context, size=size)
        assert pixels.dtype == want_pixels.dtype and np.array_equal(pixels, want_pixels)
        assert mask.dtype == want_mask.dtype and np.array_equal(mask, want_mask)


def test_disjoint_crops_share_reads_only_across_requested_inner_chunks(
        tmp_path, recorded_crop_pools):
    """Merge shared/adjacent chunk requests while leaving a complete unused chunk unread."""
    from spacr.crop_store import read_crops

    arrays, opened = recorded_crop_pools
    rows = [_pool_row("a.zarr", offset, offset, height=1, width=2)
            for offset in (24, 6, 16, 1, 6)]
    actual = read_crops(tmp_path, rows)
    assert opened == ["a.zarr/cell/pool/0", "a.zarr/cell/mask/0"]
    assert arrays["a.zarr/cell/pool/0"].reads == [(slice(None), slice(1, 8)),
                                                (slice(None), slice(16, 26))]
    assert arrays["a.zarr/cell/mask/0"].reads == [slice(1, 8), slice(16, 18), slice(24, 26)]
    assert len(actual) == len(rows)
    for row, (pixels, mask) in zip(rows, actual):
        want_pixels, want_mask = read_crop(tmp_path, row)
        assert pixels.dtype == want_pixels.dtype and np.array_equal(pixels, want_pixels)
        assert mask.dtype == want_mask.dtype and np.array_equal(mask, want_mask)
