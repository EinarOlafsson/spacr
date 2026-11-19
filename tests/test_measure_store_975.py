"""Measure writes the spaCR store (item 975, alpha).

A small ``measurements.db`` with two object tables and a ``png_list`` of
16-bit crop PNGs is written to the store with ``_write_measure_store``: every
row and column comes back through ``spacr.store``, every crop pixel for pixel
at its stored bit depth, a missing PNG is counted and gets no pointer, a
rerun replaces the well's files, and the sqlite-only paths are unchanged.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")
pytest.importorskip("zarr")
cv2 = pytest.importorskip("cv2")

from spacr import store
from spacr.crops import _read_crop_array, stamp_crop_folder
from spacr.measure import _measure_object_keys, _write_measure_store
from spacr.utils import _measurement_store_for


def _db(tmp_path, n=12, fmt=None, drop_png=None):
    """A run folder: cell and nucleus tables, crops for the cells."""
    run = tmp_path / "run" / "measurements"
    run.mkdir(parents=True)
    db = run / "measurements.db"
    rng = np.random.default_rng(0)
    keys = [f"plate1_r{1 + i % 2}_c{1 + i % 3}_f{1 + i % 4}_o{i + 1}"
            for i in range(n)]
    cell = pd.DataFrame({
        "prcfo": keys, "plateID": "plate1",
        "rowID": [k.split("_")[1] for k in keys],
        "columnID": [k.split("_")[2] for k in keys],
        "fieldID": [k.split("_")[3] for k in keys],
        "condition": rng.choice(["ctrl", "drug"], n),
        "cell_area": rng.integers(50, 500, n).astype(float),
        "cell_channel_0_mean_intensity": rng.random(n)})
    nucleus = cell[["prcfo", "plateID", "rowID", "columnID", "fieldID"]].assign(
        nucleus_area=rng.random(n))
    folder = tmp_path / "run" / "data" / "cell_png"
    folder.mkdir(parents=True)
    if fmt is not None:
        stamp_crop_folder(str(folder), fmt)
    paths, pixels = [], {}
    for key in keys:
        image = rng.integers(0, 65535, (8, 8, 3), dtype=np.uint16)
        path = folder / f"{key}.png"
        cv2.imwrite(str(path), image)
        paths.append(str(path))
        pixels[key] = image
    png = pd.DataFrame({"png_path": paths, "prcfo": keys, "cell_id": 1.0})
    with sqlite3.connect(str(db), timeout=30) as conn:
        cell.to_sql("cell", conn, index=False)
        nucleus.to_sql("nucleus", conn, index=False)
        png.to_sql("png_list", conn, index=False)
    if drop_png:
        os.remove(paths[drop_png])
    return db, cell, nucleus, pixels


def test_every_row_and_column_and_crop_comes_back(tmp_path):
    db, cell, nucleus, _pixels = _db(tmp_path)
    report = _write_measure_store(str(db), {"measurement_backend": "spacr_store"})
    assert report["missing_crops"] == 0
    assert set(report["tables"]) == {"cell", "nucleus"}
    with store.open(db.parent, verify="full") as opened:
        assert opened.tables() == ["cell", "nucleus"]
        found, crops = opened.query(table="cell", crops=True)
        nuclei = opened.query(table="nucleus")
    found = found.set_index("prcfo").loc[cell["prcfo"]].reset_index()
    for column in cell.columns:
        if column == "plateID":
            assert (found["plate"] == cell[column]).all()
            continue
        left, right = found[column].to_numpy(), cell[column].to_numpy()
        if right.dtype.kind == "f":
            assert np.array_equal(left, right), column
        else:
            assert left.tolist() == right.tolist(), column
    assert found["object_key"].is_unique
    assert len(nuclei) == len(nucleus)
    assert "crop_store" not in nuclei.columns
    with store.open(db.parent) as opened:
        result, stack = opened.query(table="cell", crops=True)
    for position, key in enumerate(result["prcfo"]):
        path = tmp_path / "run" / "data" / "cell_png" / f"{key}.png"
        expected = np.moveaxis(_read_crop_array(str(path)), -1, 0)
        assert stack.dtype == np.uint16
        assert np.array_equal(stack[position], expected)
    genes = result["condition"].tolist()
    for well in set(result["well"]):
        mine = [c for c, w in zip(genes, result["well"]) if w == well]
        assert mine == sorted(mine)


def test_a_reversed_crop_folder_is_stored_in_declared_order(tmp_path):
    from spacr.crops import CROP_FORMAT_RGB

    db, _cell, _nucleus, _pixels = _db(tmp_path, n=4, fmt=CROP_FORMAT_RGB)
    _write_measure_store(str(db), {"measurement_backend": "spacr_store"})
    with store.open(db.parent) as opened:
        result, stack = opened.query(table="cell", features=[], crops=True)
    key = result["prcfo"].iloc[0]
    path = tmp_path / "run" / "data" / "cell_png" / f"{key}.png"
    expected = np.moveaxis(_read_crop_array(str(path))[:, :, ::-1], -1, 0)
    assert np.array_equal(stack[0], expected)


def test_a_missing_png_is_counted_and_has_no_pointer(tmp_path):
    db, cell, _nucleus, _pixels = _db(tmp_path, n=6, drop_png=2)
    report = _write_measure_store(str(db), {"measurement_backend": "spacr_store"})
    assert report["missing_crops"] == 1
    with store.open(db.parent) as opened:
        found = opened.query(table="cell", features=[])
    missing = found[found["prcfo"] == cell["prcfo"].iloc[2]]
    assert missing["crop_store"].isna().all()
    assert found["crop_store"].notna().sum() == 5


def test_a_rerun_replaces_the_wells_files(tmp_path):
    db, _cell, _nucleus, _pixels = _db(tmp_path, n=6)
    settings = {"measurement_backend": "spacr_store"}
    _write_measure_store(str(db), settings)
    before = sorted(p.name for p in (db.parent / "tables" / "cell").rglob("*.parquet"))
    second = _write_measure_store(str(db), settings)
    assert second["registered"]["superseded"] > 0
    after = sorted(p.name for p in (db.parent / "tables" / "cell").rglob("*.parquet"))
    assert len(after) == len(before) and after != before
    with store.open(db.parent, verify="full") as opened:
        assert len(opened.query(table="cell")) == 6


def test_object_keys_are_integer_cell_keys_with_a_fallback():
    keys = _measure_object_keys(["plate1_r1_c2_f3_o4", "not a key",
                                 "plate1_r1_c2_f3_o5"])
    assert keys[0] != keys[2] and keys[1] >= 2 ** 62
    from spacr.ops_run_infra import integer_cell_key

    assert keys[0] == int(integer_cell_key(1, 2, 3, 4))


def test_the_store_backend_keeps_measurements_db_the_live_writer():
    assert _measurement_store_for("x/measurements.db",
                                  {"measurement_backend": "spacr_store"}) is None
    assert _measurement_store_for("x/measurements.db",
                                  {"measurement_backend": "sqlite"}) is None


@pytest.mark.parametrize("fmt", [None, 2, 3, 4])
def test_the_crop_loader_reads_the_store_as_it_reads_the_pngs(tmp_path, fmt):
    from spacr.crop_loader import CropQuery, _store_source, _StoreCropSource
    from spacr.crops import PngCropSource

    db, cell, _nucleus, _pixels = _db(tmp_path, n=5, fmt=fmt)
    _write_measure_store(str(db), {"measurement_backend": "spacr_store"})
    with sqlite3.connect(str(db), timeout=30) as conn:
        rows = [dict(zip(("png_path", "prcfo"), r)) for r in conn.execute(
            "SELECT png_path, prcfo FROM png_list")]
    source = _store_source(CropQuery(path=str(db), object_type="cell"),
                           str(db))
    assert isinstance(source, _StoreCropSource)
    stored = source.get_many(rows)
    pngs = PngCropSource(db_path=str(db)).get_many(rows)
    for got, want in zip(stored, pngs):
        assert got.dtype == want.dtype and got.shape == want.shape
        assert np.array_equal(got, want)
    assert _store_source(CropQuery(path=str(db), object_type="cell",
                                   prefer="png"), str(db)) is None
