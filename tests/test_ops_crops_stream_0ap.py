"""OPS ph_crops streams its rows, works in the workers and follows save_png (item 0ap)."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
from scipy import ndimage

from spacr import ops_engine

pytest.importorskip("zarr")


def _field(seed=0, shape=(60, 70), channels=2):
    """A field with a few cells, some at the edge, their nuclei and noisy pixels."""
    rng = np.random.default_rng(seed)
    stack = rng.integers(0, 4000, shape + (channels,)).astype(np.uint16)
    cells = np.zeros(shape, np.int32)
    nuclei = np.zeros(shape, np.int32)
    for label, (y, x, h, w) in enumerate(((0, 0, 7, 9), (20, 30, 11, 8), (52, 61, 8, 9),
                                          (30, 2, 6, 5), (5, 50, 9, 12)), start=1):
        cells[y:y + h, x:x + w] = label
        nuclei[y + h // 3:y + h // 3 + 2, x + w // 3:x + w // 3 + 2] = label
    return stack, cells, nuclei


@pytest.mark.parametrize("size", [8, 12, 21])
def test_the_one_cut_gives_the_fixed_square_of_the_reference(size):
    """_ph_cut over the fixed window equals _ph_crop_cell, edges included."""
    stack, cells, nuclei = _field()
    boxes = ndimage.find_objects(cells)
    for label in range(1, int(cells.max()) + 1):
        expected, expected_mask, centre, edge = ops_engine._ph_crop_cell(
            stack, cells, nuclei, boxes[label - 1], label, size, True)
        (top, left), got_centre, got_edge = ops_engine._ph_fixed_window(
            cells, boxes[label - 1], label, size)
        crop, mask = ops_engine._ph_cut(np.moveaxis(stack, -1, 0), cells, nuclei, label,
                                        (top, left, top + size, left + size), True, True)
        assert got_centre == centre and got_edge == edge
        assert crop.dtype == expected.dtype and np.array_equal(crop, expected)
        assert np.array_equal(mask, expected_mask)


def _write(project, crops, workers, keep_rows, cap=3):
    from spacr.crop_store import CropPoolWriter

    writer = CropPoolWriter(project, "P1", "A1", "cell", 2, np.uint16, mask_dtype=np.uint8,
                            crops_per_shard=cap)
    if not keep_rows:
        writer._streaming(workers=workers)
    writer.add(crops)
    return writer.commit()


def _crops(n=10):
    rng = np.random.default_rng(3)
    out = []
    for i in range(n):
        h, w = 3 + i % 4, 4 + i % 3
        out.append({"object_label": i + 1, "field": 1 + i // 4,
                    "pixels": rng.integers(0, 60000, (2, h, w)).astype(np.uint16),
                    "mask": rng.integers(0, 3, (h, w)).astype(np.uint8),
                    "bbox_y0": i, "bbox_x0": 2 * i, "bbox_y1": i + h, "bbox_x1": 2 * i + w,
                    "margin_px": 0, "clipped_at_edge": False, "reason": "linked"})
    return out


def _files(root):
    found = {}
    for here, _dirs, names in os.walk(root):
        for name in names:
            path = os.path.join(here, name)
            with open(path, "rb") as handle:
                found[os.path.relpath(path, root)] = handle.read()
    return found


def test_streamed_threaded_and_prepared_writes_are_byte_identical(tmp_path):
    """Streamed rows, writer threads and worker-prepared crops change no byte of the store."""
    from spacr.crop_store import _prepare_crops, _read_commit_record, read_commit

    crops = _crops()
    reference = _write(tmp_path / "a", crops, 1, True)
    _write(tmp_path / "b", crops, 3, False)
    from spacr.crop_store import CropPoolWriter

    writer = CropPoolWriter(tmp_path / "c", "P1", "A1", "cell", 2, np.uint16,
                            mask_dtype=np.uint8, crops_per_shard=3)._streaming(workers=2)
    writer._add_prepared(_prepare_crops("P1", "A1", "cell", crops, channels=2,
                                        mask_dtype=np.uint8))
    record = writer.commit()
    assert "rows" not in record and record["n_rows"] == 10 and len(record["pools"]) == 4
    first = _files(tmp_path / "a")
    assert first == _files(tmp_path / "b") == _files(tmp_path / "c")
    assert not any(name.endswith(".tmp") for name in first)
    assert read_commit(tmp_path / "c", "P1", "A1", "cell")["rows"] == json.loads(
        json.dumps(reference["rows"], default=int))
    assert _read_commit_record(tmp_path / "c", "P1", "A1", "cell")["n_rows"] == 10


def test_an_appending_commit_keeps_the_earlier_rows_unparsed(tmp_path):
    """A second writer on a committed class streams the earlier rows first, as before."""
    from spacr.crop_store import read_commit

    crops = _crops()
    _write(tmp_path / "a", crops[:4], 1, True)
    second = _write(tmp_path / "a", crops[4:], 2, True)
    _write(tmp_path / "b", crops[:4], 1, False)
    _write(tmp_path / "b", crops[4:], 1, False)
    assert second["n_rows"] == 10 and len(second["rows"]) == 10
    assert _files(tmp_path / "a") == _files(tmp_path / "b")
    assert len(read_commit(tmp_path / "b", "P1", "A1", "cell")["rows"]) == 10


def test_record_only_reading_checks_the_row_count(tmp_path):
    """The record-only read refuses a row file that does not hold the committed rows."""
    from spacr.crop_store import _read_commit_record, class_dir

    _write(tmp_path, _crops(5), 1, False)
    rows = class_dir(tmp_path, "P1", "A1", "cell") / "rows.jsonl"
    rows.write_text("".join(rows.read_text().splitlines(True)[:4]))
    with pytest.raises(ValueError, match="rows.jsonl holds 4"):
        _read_commit_record(tmp_path, "P1", "A1", "cell")


def _task(tmp_path, monkeypatch, **extra):
    stack, cells, nuclei = _field(seed=1)
    masks = tmp_path / "masks.npz"
    np.savez(masks, cells=cells, nuclei=nuclei)
    monkeypatch.setattr(ops_engine, "_ph_read_field",
                        lambda files, channels, unreadable, secondary=None:
                        stack.astype(np.float32))
    task = {"site": 7, "files": {"DAPI": "a.tif"}, "channels": ("DAPI", "CY3"),
            "crop_channels": ("DAPI", "CY3"), "mask_path": str(masks),
            "labels": {label: ("linked", 100 + label) for label in range(1, 6)},
            "size": 12, "flat_folder": None, "masks": True, "secondary": None,
            "plate": "P1", "well": "A1", "crop_format": "ome-zarr", **extra}
    return task


def _source(task):
    return {"read": {"channels": list(task["channels"]),
                     "crop_channels": list(task["crop_channels"]),
                     "flat_folder": None, "flat_mean_one": False},
            "sites": {str(task["site"]): {"files": task["files"], "secondary": None,
                                          "mask_path": task["mask_path"]}}}


@pytest.mark.parametrize("extra", [{}, {"margin": 3, "zero_outside": False},
                                   {"margin": 30, "zero_outside": True},
                                   {"masks": False}])
def test_streamed_crops_equal_the_saved_crops(tmp_path, monkeypatch, extra):
    """save_png off records the selection only; the reader cuts the saved crops' bytes."""
    from spacr.crop_store import class_dir, read_commit, read_crop, read_crops

    task = _task(tmp_path, monkeypatch, **extra)
    saved = ops_engine._ph_crop_field(task)
    streamed = ops_engine._ph_crop_field({**task, "save": False})
    assert "lines" in streamed and not streamed["members"]
    ops_engine._ph_write_crop_pool(str(tmp_path / "on"), "P1", "A1", [saved],
                                   ("DAPI", "CY3"), 2, {}, workers=2)
    ops_engine._ph_write_crop_pool(str(tmp_path / "off"), "P1", "A1", [streamed],
                                   ("DAPI", "CY3"), 2, {"streamed": True},
                                   source=_source(task))
    on = read_commit(str(tmp_path / "on"), "P1", "A1", "cell")
    off = read_commit(str(tmp_path / "off"), "P1", "A1", "cell")
    assert off["pools"] == [] and len(on["pools"]) == 3
    assert not (class_dir(str(tmp_path / "off"), "P1", "A1", "cell") / "pool").exists()
    location = ("crop_store", "crop_array", "crop_index", "offset", "mask_offset",
                "source_field", "zero_outside")
    for one, two in zip(on["rows"], off["rows"]):
        assert {k: v for k, v in one.items() if k not in location} == \
            {k: v for k, v in two.items() if k not in location}
        assert two["crop_array"] is None and two["source_field"] == "ops:7"
    stored = read_crops(str(tmp_path / "on"), on["rows"])
    cut = read_crops(str(tmp_path / "off"), off["rows"])
    for (pixels, mask), (pixels2, mask2), row in zip(stored, cut, off["rows"]):
        assert pixels.dtype == pixels2.dtype and np.array_equal(pixels, pixels2)
        assert mask.dtype == mask2.dtype and np.array_equal(mask, mask2)
        one = read_crop(str(tmp_path / "off"), row)
        assert np.array_equal(one[0], pixels) and np.array_equal(one[1], mask)


def test_the_tile_index_is_walked_once_per_run_folder(tmp_path, monkeypatch):
    """A second process reuses the saved index; a new tile walks again."""
    tiles = tmp_path / "tiles" / "sub"
    tiles.mkdir(parents=True)
    (tiles / "20X_DAPI-GFP_A1_DAPI-GFP_Site-0.tif").write_bytes(b"")
    first = ops_engine._cached_tile_index(str(tmp_path / "run"), str(tmp_path / "tiles"),
                                          "uncycled")
    assert first == ops_engine._index_tiles(str(tmp_path / "tiles"), "uncycled") != {}
    walks = []
    real = ops_engine._index_tiles
    monkeypatch.setattr(ops_engine, "_index_tiles",
                        lambda *a, **k: walks.append(1) or real(*a, **k))
    assert ops_engine._cached_tile_index(str(tmp_path / "run"), str(tmp_path / "tiles"),
                                         "uncycled") == first and walks == []
    (tiles / "20X_DAPI-GFP_A1_DAPI-GFP_Site-1.tif").write_bytes(b"")
    os.utime(tiles, ns=(1, 1))
    again = ops_engine._cached_tile_index(str(tmp_path / "run"), str(tmp_path / "tiles"),
                                          "uncycled")
    assert walks == [1] and sorted(again["A1"][1]) == [0, 1]


def test_ops_follows_the_one_save_crops_switch():
    """OPS reads Measure's save_png, off by default, rather than an OPS-only switch."""
    from spacr.ops_settings import OPS_DEFAULTS

    assert OPS_DEFAULTS["save_png"] is False
    assert not any(key.startswith("ops_") and "save_crop" in key for key in OPS_DEFAULTS)
