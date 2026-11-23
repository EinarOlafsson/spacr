"""The ph_tables phase writes the spaCR store, streamed, in the smallest dtypes.

A small run folder is built by hand: four per-field phenotype tables (the
``ph_measure`` output) and their well table, a mask pass marking two edge
cells, and ``ops_merge`` links for most cells. The columns are named as
OPS measures them, so the recorded dtype map (``store_dtypes.json``) picks
float16, float32, float64 and integer types for them.

Checked: the streamed tables hold the rows, keys and values the whole-well
frames give (each feature equal to the old value cast to its stored dtype);
the store layout and keys are Measure's; a second run writes byte-identical
parts; reading the well table instead of the per-field tables changes no
byte; a value outside the recorded dtype's range widens that column and is
reported; float16 columns read back as float32; and an older folder holding
only ``measurements.db`` still opens.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3

import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("duckdb")

from spacr import ops_engine, store                                # noqa: E402
from spacr.ops_store import write_table                            # noqa: E402

PLATE = "P_1"
SITES = (3, 7, 12, 40)
SETTINGS = {"plate": PLATE, "ops_ph_channels": "DAPI,GFP"}


#: The nested helper of the ph_tables SQLite writer, retired with it on
#: 2026-11-23; the frozen 871 arrival contract still names it.
RETIRED_SQLITE_HELPER = "spacr.ops_engine._ph_insert_rows.declared"


def _current_phtables_api(entries):
    """Drop only the retired SQLite helper from a frozen arrival contract."""
    return {key: value for key, value in entries.items()
            if key != RETIRED_SQLITE_HELPER}


def test_the_retired_sqlite_helper_is_gone_and_only_it_is_dropped():
    import json
    from pathlib import Path

    path = (Path(__file__).resolve().parent / "data" / "release_contracts"
            / "871_911_bmc_paper_items_api_arrivals_2026-11-09.json")
    original = json.loads(path.read_text(encoding="utf-8"))
    assert original[RETIRED_SQLITE_HELPER] == "The SQLite type of a column."
    current = _current_phtables_api(original)
    assert original.keys() - current.keys() == {RETIRED_SQLITE_HELPER}
    assert not hasattr(ops_engine, "_ph_insert_rows")


def _field(site: int, rng) -> pd.DataFrame:
    """One field's cell table with OPS column names; labels out of order."""
    count = 25 + site % 4
    labels = rng.permutation(np.arange(1, count + 1))
    frame = pd.DataFrame({"site": np.int64(site), "label": labels.astype(np.int64),
                          "centroid_row": rng.uniform(0, 500, count),
                          "centroid_col": rng.uniform(0, 500, count)})
    for compartment, scale in (("cell", 1.0), ("nucleus", 0.4),
                               ("cytoplasm", 0.6)):
        frame[f"{compartment}_area"] = rng.uniform(200, 3000, count) * scale
        frame[f"{compartment}_euler_number"] = rng.integers(0, 3, count)
        frame[f"{compartment}_zernike_0"] = 1 / np.pi + rng.normal(0, 1e-13, count)
        frame[f"{compartment}_eccentricity"] = rng.uniform(0, 1, count)
        for channel in ("dapi", "gfp"):
            frame[f"{compartment}_{channel}_mean_intensity"] = rng.gamma(
                4, 250, count)
            frame[f"{compartment}_{channel}_skew_intensity"] = rng.normal(
                0, 1, count)
            frame[f"{compartment}_{channel}_pftas_0"] = rng.uniform(0, 1, count)
    frame.loc[0, "nucleus_dapi_skew_intensity"] = np.nan
    frame.loc[1, "cell_gfp_skew_intensity"] = np.inf
    return frame


def _build(root, sites=SITES, seed=0):
    """A run folder holding everything ph_tables reads."""
    rng = np.random.default_rng(seed)
    run = root / "run"
    folder = run / "A1" / "phenotype_cells"
    folder.mkdir(parents=True)
    frames = []
    for site in sites:
        frame = _field(site, rng)
        frame.to_parquet(folder / f"site-{site}.parquet", index=False)
        frames.append(frame)
    cells = pd.concat(frames, ignore_index=True)
    cells.insert(0, "well", "A1")
    cells.insert(0, "plate", PLATE)
    cells.to_parquet(ops_engine._ph_cells_path(str(run), "A1"), index=False)
    edge = cells[["site", "label"]].copy()
    edge["touches_field_edge"] = False
    edge.loc[[2, 30], "touches_field_edge"] = True
    edge.to_parquet(ops_engine._ph_maskpass_path(str(run), "A1"), index=False)
    links = cells.loc[cells.index % 5 != 0, ["site", "label"]].copy()
    links.insert(0, "well", "A1")
    links.insert(0, "plate", PLATE)
    links["object_id"] = np.arange(1, len(links) + 1, dtype=np.int64)
    write_table(str(run / "measurements.db"), "ops_merge", links)
    return run, cells


def _tables(run, **extra):
    """Run the phase on well A1."""
    settings = {**SETTINGS, "dst_root": str(run), **extra}
    return ops_engine._ph_tables(str(run / "measurements.db"), PLATE, ["A1"],
                                 settings, str(run))


def _hashes(folder):
    """``relative path -> sha256`` of every table file of a store."""
    out = {}
    for here, _dirs, files in os.walk(os.path.join(folder, "tables")):
        for name in files:
            path = os.path.join(here, name)
            with open(path, "rb") as handle:
                out[os.path.relpath(path, folder)] = hashlib.sha256(
                    handle.read()).hexdigest()
    return out


@pytest.fixture
def built(tmp_path):
    """The run folder and its whole-well cell table."""
    return _build(tmp_path)


def test_the_streamed_tables_hold_the_whole_well_rows_in_their_dtypes(built):
    run, cells = built
    report = _tables(run)
    folder = str(run / "measurements")
    assert report["store"] == folder
    assert report["rows"] == {"cell": len(cells) - 2, "nucleus": len(cells) - 2,
                              "cytoplasm": len(cells) - 2}
    assert report["wells"]["A1"]["source"] == "fields"
    owned = ops_engine._ph_owned_cells(str(run), "A1", cells, SETTINGS)
    links = pd.read_sql_query("SELECT * FROM ops_merge", sqlite3.connect(
        str(run / "measurements.db")))
    keys, features = ops_engine._ph_table_frames(
        PLATE, "A1", cells, links, None, owned,
        ops_engine._ph_genotype_rule(SETTINGS))
    with store.open(folder) as opened:
        assert opened.tables() == ["cell", "cytoplasm", "nucleus"]
        for compartment in ("cell", "nucleus", "cytoplasm"):
            found = opened.query(table=compartment)
            reference = pd.concat([keys, features[compartment]], axis=1)
            found = found.set_index("prcfo").loc[reference["prcfo"]].reset_index()
            assert list(found.columns[:13]) == list(ops_engine._PH_TABLE_KEYS)
            if compartment != "cell":
                assert (found["parent_cell_key"] == found["object_key"]).all()
            assert found["timeID"].isna().all()
            assert set(found["plate"]) == {PLATE} and set(found["well"]) == {"A1"}
            assert "plateID" not in found.columns
            for name in ("rowID", "columnID", "fieldID", "genotype_rule"):
                assert found[name].tolist() == reference[name].tolist()
            assert found["object_label"].tolist() == reference["object_label"].tolist()
            linked = reference["object_id"].notna().to_numpy()
            assert found["object_id"].isna().tolist() == (~linked).tolist()
            assert (found["object_id"][linked].astype(np.int64).tolist()
                    == reference["object_id"][linked].astype(np.int64).tolist())
            part = pq.ParquetFile(os.path.join(folder, [
                f["path"] for f in opened._files
                if f["table_name"] == compartment][0]))
            stored = part.schema_arrow
            for name in features[compartment].columns:
                kind = np.dtype(stored.field(name).type.to_pandas_dtype())
                expected = reference[name].to_numpy().astype(kind).astype(float)
                got = found[name].to_numpy(dtype=float)
                assert np.array_equal(got, expected, equal_nan=True), name
                if kind == np.float16:
                    assert found[name].dtype == np.float32
            types = {n: str(stored.field(n).type) for n in stored.names}
            assert types[f"{compartment}_zernike_0"] == "double"
            assert types[f"{compartment}_area"] == "float"
            assert types[f"{compartment}_dapi_skew_intensity"] == "halffloat"
            assert types["object_key"] == "int64" and types["mitotic"] == "int8"
            assert types["object_label"] == "uint32"
            assert types["timeID"] == "uint16"
            object_key = found["object_key"].to_numpy()
            assert len(set(object_key)) == len(object_key)
            catalog = opened._tables[compartment]
            assert catalog["sort_by"] == ["object_key", "timeID"]
            assert catalog["key_columns"][:13] == list(ops_engine._PH_TABLE_KEYS)
    raw = pq.read_table(os.path.join(folder, "tables", "cell", f"plate={PLATE}",
                                     "well=A1", "part-0.parquet"))
    keys_in_order = raw.column("object_key").to_numpy()
    assert np.all(np.diff(keys_in_order) > 0)


def test_a_rerun_writes_byte_identical_parts(built):
    run, _cells = built
    _tables(run)
    folder = str(run / "measurements")
    first = _hashes(folder)
    with store.open(folder) as opened:
        registry = sorted((f["path"], f["rows"], f["sha256"])
                          for f in opened._files)
    _tables(run)
    assert _hashes(folder) == first
    with store.open(folder, verify="full") as opened:
        assert sorted((f["path"], f["rows"], f["sha256"])
                      for f in opened._files) == registry
    assert not os.path.isdir(os.path.join(folder, "_pending"))or not os.listdir(
        os.path.join(folder, "_pending"))


def test_the_well_table_source_gives_the_same_bytes(built, tmp_path):
    run, _cells = built
    _tables(run)
    streamed = _hashes(str(run / "measurements"))
    shutil.rmtree(run / "A1" / "phenotype_cells")
    report = _tables(run)
    assert report["wells"]["A1"]["source"] == "well_table"
    assert _hashes(str(run / "measurements")) == streamed


def test_a_value_beyond_the_recorded_dtype_widens_its_column(tmp_path):
    run, cells = _build(tmp_path)
    cells.loc[4, "cell_dapi_skew_intensity"] = 1.0e6
    cells.loc[5, "cell_euler_number"] = -2
    cells.to_parquet(ops_engine._ph_cells_path(str(run), "A1"), index=False)
    shutil.rmtree(run / "A1" / "phenotype_cells")
    report = _tables(run)
    promoted = {p["column"]: (p["recorded"], p["stored"])
                for p in report["wells"]["A1"]["promoted"]}
    assert promoted["cell_dapi_skew_intensity"] == ("float16", "float32")
    assert promoted["cell_euler_number"] == ("uint8", "int8")
    with store.open(str(run / "measurements")) as opened:
        found = opened.query(table="cell", features=["cell_dapi_skew_intensity",
                                                     "cell_euler_number"])
    assert found["cell_dapi_skew_intensity"].max() == 1.0e6
    assert found["cell_euler_number"].min() == -2


def test_the_rebuild_drops_the_older_tables_and_database(built):
    run, _cells = built
    folder = run / "measurements"
    folder.mkdir()
    with sqlite3.connect(folder / "measurements.db") as conn:
        conn.execute("CREATE TABLE cell (prcfo TEXT)")
    _tables(run, ops_tables_layout="wide")
    assert not (folder / "measurements.db").exists()
    with store.open(str(folder)) as opened:
        assert "cell_wide" in opened.tables()
        wide = opened.query(table="cell_wide")
        cell = opened.query(table="cell")
    assert len(wide) == len(cell)
    assert {"cell_area", "nucleus_area", "cytoplasm_area"} <= set(wide.columns)
    _tables(run)
    with store.open(str(folder)) as opened:
        assert "cell_wide" not in opened.tables()
    assert not (folder / "tables" / "cell_wide").exists()


def test_an_older_measurements_db_folder_still_opens(tmp_path):
    folder = tmp_path / "measurements"
    folder.mkdir()
    with sqlite3.connect(folder / "measurements.db") as conn:
        conn.execute("CREATE TABLE cell (prcfo TEXT, cell_area REAL)")
        conn.execute("INSERT INTO cell VALUES ('P_r1_c1_f0_o1', 12.5)")
    with store.open(str(folder)) as opened:
        assert opened.legacy.endswith("measurements.db")
        assert opened.query(table="cell")["cell_area"].tolist() == [12.5]


def test_the_part_writer_refuses_values_its_types_cannot_hold(tmp_path):
    schema = pa.schema([("x", pa.float16()), ("n", pa.int8())])
    writer = store.writer(tmp_path, "t", plate="P", well="A1",
                          schema=schema)
    with pytest.raises(store.StoreError, match="beyond the range"):
        writer.append(pd.DataFrame({"x": [1.0e6], "n": [1]}))
    with pytest.raises(store.StoreError, match="field 7: .*does not fit"):
        writer.append(pd.DataFrame({"x": [1.0], "n": [300]}), field=7)
    with pytest.raises(store.StoreError, match="not in the part's schema"):
        writer.append(pd.DataFrame({"y": [1.0]}))
    writer.abort()
    assert not list((tmp_path / "tables" / "t").rglob("*.tmp-*"))


def test_the_part_writer_cuts_the_store_row_groups_from_small_batches(tmp_path):
    with store.writer(tmp_path, "t", plate="P", well="A1",
                      row_group_rows=1024, sort_by=("k",)) as writer:
        for start in range(0, 5000, 300):
            writer.append(pd.DataFrame({
                "k": np.arange(start, min(start + 300, 5000)),
                "x": np.ones(min(300, 5000 - start))}))
    manifest = writer.manifest
    part = pq.ParquetFile(tmp_path / manifest["files"][0]["path"])
    assert [part.metadata.row_group(g).num_rows
            for g in range(part.metadata.num_row_groups)] == [1024] * 4 + [904]
    assert manifest["sort_by"] == ["k"]
    store.register(tmp_path)
    with store.open(tmp_path) as opened:
        assert opened.query(table="t")["k"].tolist() == list(range(5000))
    unordered = store.writer(tmp_path, "u", plate="P", well="A1",
                             sort_by=("k",))
    unordered.append(pd.DataFrame({"k": [3, 1]}))
    assert unordered.finish()["sort_by"] == []


def test_float16_reads_back_identically_through_duckdb_pyarrow_and_pandas(
        tmp_path):
    duckdb = pytest.importorskip("duckdb")
    values = np.array([0.0, 1.5e-6, -3.25, 65504.0, np.nan, np.inf, -np.inf,
                       0.1, 1234.5], dtype=np.float64)
    with store.writer(tmp_path, "t", plate="P", well="A1",
                      schema=pa.schema([("x", pa.float16())])) as writer:
        writer.append(pd.DataFrame({"x": np.tile(values, 500)}))
    manifest = writer.manifest
    path = str(tmp_path / manifest["files"][0]["path"])
    expected = np.tile(values, 500).astype(np.float16).astype(np.float32)
    through_duckdb = duckdb.sql(
        f"SELECT x FROM read_parquet('{path}')").fetchnumpy()["x"]
    through_duckdb = np.ma.filled(np.ma.asarray(through_duckdb, np.float32),
                                  np.nan)
    through_arrow = pq.read_table(path).column("x").to_numpy().astype(np.float32)
    through_pandas = pd.read_parquet(path)["x"].to_numpy().astype(np.float32)
    for got in (through_duckdb, through_arrow, through_pandas):
        assert np.array_equal(np.asarray(got, np.float32), expected,
                              equal_nan=True)
    counted = duckdb.sql(f"SELECT count(*) FILTER (WHERE x IS NULL), "
                         f"count(*) FILTER (WHERE isinf(x)) FROM "
                         f"read_parquet('{path}')").fetchall()[0]
    assert counted == (500, 1000)
    store.register(tmp_path)
    with store.open(tmp_path) as opened:
        assert opened.query(table="t")["x"].dtype == np.float32
        assert opened.sql("SELECT x FROM t")["x"].dtype == np.float32


def test_crops_appended_per_batch_land_in_row_order(tmp_path):
    crops = np.arange(4 * 1 * 2 * 2, dtype=np.uint16).reshape(4, 1, 2, 2)
    with store.writer(tmp_path, "cell", plate="P", well="A1") as writer:
        writer.append(pd.DataFrame({"object_key": [1, 2, 3]}),
                      crops=crops[:2], crop_rows=[2, 0])
        writer.append(pd.DataFrame({"object_key": [4, 5]}), crops=crops[2:],
                      crop_rows=[1, 0])
    store.register(tmp_path)
    with store.open(tmp_path) as opened:
        found, stack = opened.query(table="cell", crops=True)
    assert found["object_key"].tolist() == [1, 2, 3, 4, 5]
    assert found["crop_index"].isna().tolist() == [False, True, False,
                                                     False, False]
    for row, crop in ((0, 1), (2, 0), (3, 3), (4, 2)):
        assert np.array_equal(stack[row], crops[crop])


def test_the_dtype_family_is_named_without_compartment_channel_or_number():
    assert store._dtype_family("nucleus_gfp_percentile_95", ["DAPI", "GFP"]) == \
        "{compartment}_{channel}_percentile_{n}"
    assert store._dtype_family("cell_zernike_0", [], literal_numbers=True) == \
        "{compartment}_zernike_0"
    assert store._fitted_dtype("float16", "float", -1.0, 70000.0) == "float32"
    assert store._fitted_dtype("uint8", "int", -1, 5) == "int8"
    assert store._fitted_dtype("float16", "float", None, None) == "float32"
