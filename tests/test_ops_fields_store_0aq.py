"""OPS phenotype fields in fov.ome.zarr, the OPS crop index in the one store (items 0aq, 0ar, s4b).

ph_crops writes each phenotype field into the project's fov.ome.zarr
through Mask's own writer (raw planes, sha256, verify_store), every OPS crop
is cut from there, and the crop rows go to the crop_index table of the
project's one store through the writer Measure uses. Older runs still read:
their rows from rows.jsonl, their streamed crops from the tiles.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from spacr import ops_engine, store

pytest.importorskip("zarr")
tifffile = pytest.importorskip("tifffile")


def _tiles(folder, site=3, shape=(40, 50)):
    """One site's tiles as OPS reads them: a ZCYX two-channel file and a ZYX file."""
    rng = np.random.default_rng(site)
    two = rng.integers(0, 60000, (4, 2) + shape).astype(np.uint16)
    one = rng.integers(0, 60000, (4,) + shape).astype(np.uint16)
    folder.mkdir(parents=True, exist_ok=True)
    first, second = folder / f"DAPI-GFP_{site}.tif", folder / f"A594_{site}.tif"
    tifffile.imwrite(first, two, metadata={"axes": "ZCYX"})
    tifffile.imwrite(second, one, metadata={"axes": "ZYX"})
    return {"DAPI-GFP": str(first), "A594": str(second)}, two, one


CHANNELS = ("DAPI", "GFP", "A594")


def test_the_fields_are_stored_exactly_as_acquired_and_verified(tmp_path):
    from spacr.image_store import _image_entries, read_frame_channels, verify_store

    files, two, one = _tiles(tmp_path / "tiles")
    report = ops_engine._ph_store_fields(str(tmp_path), "P_1", "A1", {3: files}, CHANNELS, {})
    assert report["written"] == 1 and report["fields"] == {3: "P%5F1_A1_3.npy"}
    store_path = tmp_path / "fov.ome.zarr"
    entry = _image_entries(store_path)["P%5F1_A1_3.npy"]
    assert entry["source_channels"] == list(CHANNELS)
    raw = read_frame_channels(store_path / entry["key"], entry, z_projection=None)
    assert raw.dtype == np.uint16 and raw.shape == (3, 4, 40, 50)
    for got, want in zip(raw, (two[:, 0], two[:, 1], one)):
        assert got.tobytes() == want.tobytes()
    checked = verify_store(store_path)
    assert checked["fields"] == 1 and checked["planes"] == 12 and not checked["mismatches"]
    before = sorted((p.relative_to(store_path), p.read_bytes()) for p in store_path.rglob("*")
                    if p.is_file())
    again = ops_engine._ph_store_fields(str(tmp_path), "P_1", "A1", {3: files}, CHANNELS, {})
    assert again["written"] == 0 and again["skipped"] == 1
    assert before == sorted((p.relative_to(store_path), p.read_bytes())
                            for p in store_path.rglob("*") if p.is_file())


@pytest.mark.parametrize("secondary", [None, {"channels": ["A594"], "dy": 3, "dx": -2}])
def test_the_stored_field_reads_as_the_tiles_read(tmp_path, secondary):
    files, _two, _one = _tiles(tmp_path / "tiles")
    report = ops_engine._ph_store_fields(str(tmp_path), "P1", "A1", {3: files}, CHANNELS,
                                         {3: secondary} if secondary else {})
    task = {"channels": CHANNELS, "crop_channels": ("GFP", "A594"), "secondary": secondary,
            "flat_folder": None}
    from_tiles = ops_engine._ph_crop_stack({**task, "files": files}, [])
    from_store = ops_engine._ph_crop_stack({**task, "field": {
        "store": report["store"], "name": report["fields"][3]}}, [])
    assert from_store.dtype == from_tiles.dtype == np.uint16
    assert from_store.tobytes() == from_tiles.tobytes()


def _crops(n=7):
    rng = np.random.default_rng(1)
    out = []
    for i in range(n):
        h, w = 3 + i % 3, 4 + i % 2
        out.append({"object_label": i + 1, "field": 1 + i // 3, "object_id": f"P1_A1_o{i}",
                    "pixels": rng.integers(0, 60000, (2, h, w)).astype(np.uint16),
                    "mask": rng.integers(0, 3, (h, w)).astype(np.uint8),
                    "bbox_y0": i, "bbox_x0": i, "bbox_y1": i + h, "bbox_x1": i + w,
                    "margin_px": 0, "clipped_at_edge": False})
    return out


@pytest.mark.parametrize("workers", [1, 2])
def test_the_pool_writer_puts_its_rows_in_the_store_crop_index(tmp_path, workers):
    from spacr.crop_store import (CropPoolWriter, _read_commit_record, class_dir, read_commit,
                                  read_crops)

    crops = _crops()
    folder = tmp_path / "measurements"
    writer = CropPoolWriter(tmp_path, "P1", "A1", "cell", 2, np.uint16, mask_dtype=np.uint8,
                            crops_per_shard=3)._streaming(workers=workers, index=folder)
    writer.add(crops)
    record = writer.commit()
    base = class_dir(tmp_path, "P1", "A1", "cell")
    assert not (base / "rows.jsonl").exists()
    assert record["n_rows"] == 7 and record["index"]["table"] == "crop_index"
    assert _read_commit_record(tmp_path, "P1", "A1", "cell")["n_rows"] == 7
    rows = read_commit(tmp_path, "P1", "A1", "cell")["rows"]
    assert [r["object_id"] for r in rows] == [c["object_id"] for c in crops]
    for (pixels, mask), crop in zip(read_crops(tmp_path, rows), crops):
        assert pixels.tobytes() == crop["pixels"].tobytes()
        assert mask.tobytes() == crop["mask"].tobytes()
    assert sorted(tmp_path.rglob(store.CATALOG)) == [folder / store.CATALOG]


def test_an_older_root_catalog_crop_index_still_reads(tmp_path):
    """A project from before item 0ar keeps its index at the root; it is read, never rewritten."""
    from spacr.crop_store import CROP_INDEX_TABLE, _read_crop_index, crop_rows

    frame = pd.DataFrame({"object_id": ["a", "b"], "object_type": ["cell", "cell"],
                          "plate": ["P1", "P1"], "well": ["A1", "A1"], "field": [1, 1],
                          "object_label": [1, 2], "bbox_y0": [0, 5]})
    store.write_part(str(tmp_path), CROP_INDEX_TABLE, frame, plate="P1", well="A1",
                     sort_by=("object_type", "field", "object_label"))
    store.register(str(tmp_path))
    before = (tmp_path / store.CATALOG).read_bytes()
    assert list(_read_crop_index(tmp_path)["object_id"]) == ["a", "b"]
    assert [r["bbox_y0"] for r in crop_rows(tmp_path, ["b", "a"])] == [5, 0]
    assert (tmp_path / store.CATALOG).read_bytes() == before
    assert not (tmp_path / "measurements").exists()


def test_a_streamed_ops_row_needs_no_tiles_once_the_field_is_stored(tmp_path):
    """The streamed reader cuts from fov.ome.zarr: the tiles may be gone."""
    from spacr.crop_store import CropPoolWriter, _index_lines, read_crops

    files, two, one = _tiles(tmp_path / "tiles", shape=(30, 30))
    report = ops_engine._ph_store_fields(str(tmp_path), "P1", "A1", {3: files}, CHANNELS, {})
    cells = np.zeros((30, 30), np.int32)
    nuclei = np.zeros((30, 30), np.int32)
    cells[5:15, 6:20], nuclei[8:11, 9:13] = 1, 1
    cells[18:28, 2:9] = 2
    masks = tmp_path / "masks.npz"
    np.savez(masks, cells=cells, nuclei=nuclei)
    name = report["fields"][3]
    rows = [{"object_label": 1, "label": 1, "site": 3, "field": 3, "ops_crop": True,
             "has_mask": True, "source_field": name, "zero_outside": False,
             "bbox_y0": 2, "bbox_x0": 3, "bbox_y1": 18, "bbox_x1": 23, "margin_px": 3,
             "clipped_at_edge": False, "object_id": "o1"},
            {"object_label": 2, "label": 2, "site": 3, "field": 3, "ops_crop": True,
             "has_mask": False, "source_field": name, "zero_outside": True,
             "bbox_y0": 15, "bbox_x0": 0, "bbox_y1": 30, "bbox_x1": 12, "margin_px": 3,
             "clipped_at_edge": True, "object_id": "o2"}]
    writer = CropPoolWriter(tmp_path, "P1", "A1", "cell", 3, np.uint16, mask_dtype=np.uint8,
                            attributes={"source": "ops_ph_crops"})._streaming(
        index=tmp_path / "measurements", index_schema=ops_engine._ph_crop_index_schema())
    source = {"read": {"channels": list(CHANNELS), "crop_channels": list(CHANNELS),
                       "flat_folder": None, "flat_mean_one": False},
              "store": "fov.ome.zarr",
              "sites": {"3": {"field": name, "secondary": None, "mask_path": str(masks)}}}
    (writer.path / "cell" / ops_engine.OPS_CROP_SOURCE).write_text(json.dumps(source))
    writer._add_index_lines(_index_lines("P1", "A1", "cell", rows))
    writer.commit()
    for path in files.values():
        os.remove(path)
    ops_engine._PH_SOURCE_FIELDS.clear()
    stored = ops_engine._ph_crop_stack({"channels": CHANNELS, "crop_channels": CHANNELS,
                                        "field": {"store": report["store"], "name": name}}, [])
    stack = np.moveaxis(stored, -1, 0)
    from spacr.crop_store import read_commit

    got = read_crops(tmp_path, read_commit(tmp_path, "P1", "A1", "cell")["rows"])
    for (pixels, mask), row in zip(got, rows):
        want, want_mask = ops_engine._ph_cut(
            stack, cells, nuclei, row["object_label"],
            (row["bbox_y0"], row["bbox_x0"], row["bbox_y1"], row["bbox_x1"]),
            row["zero_outside"], row["has_mask"])
        assert pixels.tobytes() == want.tobytes()
        if want_mask is not None:
            assert mask.tobytes() == want_mask.tobytes()
        else:
            assert not mask.any()


@pytest.fixture
def half_map(tmp_path, monkeypatch):
    """A map that starts two float columns at float16."""
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"version": 1, "columns": {
        n: {"kind": "float", "dtype": "float16"} for n in ("cell_area", "cell_ratio")}}))
    monkeypatch.setattr(store, "_DTYPE_MAP_PATH", str(path))
    store._read_dtype_map.cache_clear()
    yield
    store._read_dtype_map.cache_clear()


def test_the_streamed_table_path_narrows_by_the_values_not_the_range(tmp_path, half_map):
    """s4b: Parquet statistics alone let 2049..3000 into float16; the values refuse it."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = 500
    area = np.linspace(2049, 3000, n).round().astype(np.float64)
    ratio = (np.arange(n) % 7 / 4.0).astype(np.float64)
    source = tmp_path / "cells.parquet"
    pq.write_table(pa.table({"cell_area": area, "cell_ratio": ratio}), source)
    from_stats = store._column_ranges(pq.ParquetFile(source).metadata)
    from_values = store._file_ranges(str(source))
    schema = pa.schema([pa.field("cell_area", pa.float64()), pa.field("cell_ratio", pa.float64())])
    narrowed_stats, _ = store._stored_types(schema, from_stats)
    narrowed_values, promoted = store._stored_types(schema, from_values)
    assert narrowed_stats["cell_area"] == pa.float16()
    assert narrowed_values["cell_area"] == pa.float32()
    assert narrowed_values["cell_ratio"] == pa.float16()
    assert promoted == [{"column": "cell_area", "recorded": "float16", "stored": "float32",
                         "reason": "integers not exact"}]


@pytest.mark.parametrize("secondary", [None, {"channels": ["A594"], "dy": -4, "dx": 5}])
def test_a_crop_window_reads_the_bytes_of_the_whole_field(tmp_path, secondary):
    """One crop reads only its window; a batch reads the field once; the bytes agree, edges too."""
    from spacr.crop_store import CropPoolWriter, _index_lines

    files, _two, _one = _tiles(tmp_path / "tiles", shape=(40, 50))
    report = ops_engine._ph_store_fields(str(tmp_path), "P1", "A1", {3: files}, CHANNELS,
                                         {3: secondary} if secondary else {})
    cells = np.zeros((40, 50), np.int32)
    nuclei = np.zeros((40, 50), np.int32)
    cells[0:9, 0:12], cells[15:30, 20:35], cells[33:40, 41:50] = 1, 2, 3
    nuclei[2:5, 2:5], nuclei[20:23, 25:28], nuclei[35:38, 44:47] = 1, 2, 3
    masks = tmp_path / "masks.npz"
    np.savez(masks, cells=cells, nuclei=nuclei)
    name = report["fields"][3]
    boxes = {1: (-6, -6, 14, 18), 2: (10, 15, 35, 40), 3: (30, 38, 46, 58)}
    rows = [{"object_label": k, "label": k, "site": 3, "field": 3, "ops_crop": True,
             "has_mask": True, "source_field": name, "zero_outside": k == 3,
             "bbox_y0": b[0], "bbox_x0": b[1], "bbox_y1": b[2], "bbox_x1": b[3],
             "margin_px": 0, "clipped_at_edge": True, "object_id": f"o{k}"}
            for k, b in boxes.items()]
    writer = CropPoolWriter(tmp_path, "P1", "A1", "cell", 3, np.uint16, mask_dtype=np.uint8,
                            attributes={"source": "ops_ph_crops"})._streaming(
        index=tmp_path / "measurements", index_schema=ops_engine._ph_crop_index_schema())
    source = {"read": {"channels": list(CHANNELS), "crop_channels": ["GFP", "A594"],
                       "flat_folder": None, "flat_mean_one": False},
              "store": "fov.ome.zarr",
              "sites": {"3": {"field": name, "secondary": secondary, "mask_path": str(masks)}}}
    (writer.path / "cell" / ops_engine.OPS_CROP_SOURCE).write_text(json.dumps(source))
    writer._add_index_lines(_index_lines("P1", "A1", "cell", rows))
    writer.commit()
    ops_engine._PH_SOURCE_FIELDS.clear()
    from spacr.crop_store import read_commit

    for row in read_commit(tmp_path, "P1", "A1", "cell")["rows"]:
        window = ops_engine._ph_stream_crop(tmp_path, row)
        whole = ops_engine._ph_stream_crop(tmp_path, row, whole=True)
        assert window[0].dtype == whole[0].dtype == np.uint16
        assert window[0].tobytes() == whole[0].tobytes()
        assert window[1].tobytes() == whole[1].tobytes()
