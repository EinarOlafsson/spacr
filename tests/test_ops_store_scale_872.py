"""The crop store, the measurement tables and the stress report at OPS scale (item 872, B16, B17, B20).

On the synthetic two-field well of item 871's test (eight owned, linked
cells): the streaming crop loader must give every crop once, pixel for
pixel what the per-crop reader gives, in a seeded order; integer join keys
must join the cell, nucleus and cytoplasm tables to the same rows as
``prcfo``; the wide table must equal the three-table join; and the stress
phase must write the reference columns with exact counts.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("tifffile")

from spacr import ops_engine, ops_run_infra                          # noqa: E402
from tests.test_ops_phenotype_run_v2 import (                        # noqa: E402,F401
    FULL, PLATE, _run, measured, models)


@pytest.fixture
def stored(measured):
    """The well through every phase, with integer keys and the wide table."""
    out = _run(measured, (*FULL, "stress"), ops_crop_control_fraction=0.0,
               ops_crop_masks=True, ops_tables_integer_keys=True,
               ops_tables_layout="wide", ops_max_files_per_folder=3)
    return measured, out


def _rows(db, sql):
    """Every row of one query, as a frame."""
    with sqlite3.connect(db) as conn:
        cursor = conn.execute(sql)
        names = [d[0] for d in cursor.description]
        return pd.DataFrame(cursor.fetchall(), columns=names)


def test_the_streaming_loader_reads_every_crop_once_and_pixel_identical(stored):
    root, out = stored
    store = str(root / "run")
    index = ops_run_infra.crop_index(store, PLATE)
    assert len(index) == 8 and index["shard"].nunique() == 3
    for shard, group in index.groupby("shard"):
        with open(shard, "rb") as handle:
            blob = handle.read()
        assert all(blob[o:o + 4] in (b"II*\x00", b"MM\x00*")
                   for o in group["offset_data"])
    streamed = list(ops_run_infra.stream_crops(store, PLATE, seed=3, buffer=3))
    assert sorted(r["member"] for r, _i, _m in streamed) == sorted(index["member"])
    for record, image, mask in streamed:
        expected, expected_mask = ops_engine._ph_read_crop(
            record["shard"], record["offset_data"], record["size"])
        assert np.array_equal(image, expected)
        assert np.array_equal(mask, expected_mask)
    order = [r["member"] for r, _i, _m in streamed]
    again = [r["member"] for r, _i, _m in
             ops_run_infra.stream_crops(store, PLATE, seed=3, buffer=3)]
    other = [r["member"] for r, _i, _m in
             ops_run_infra.stream_crops(store, PLATE, seed=4, buffer=3)]
    plain = [r["member"] for r, _i, _m in
             ops_run_infra.stream_crops(store, PLATE, shuffle=False)]
    assert order == again and order != other
    assert plain == list(index["member"])
    assert ops_run_infra.crop_index(store, PLATE, ["B2"]).empty
    assert ops_run_infra.crop_index(str(root / "nowhere"), PLATE).empty


def test_integer_keys_join_as_prcfo_does_and_the_wide_table_is_the_join(stored):
    _root, out = stored
    db = out["ph_tables"]["db"]
    assert out["ph_tables"]["integer_keys"] is True
    assert out["ph_tables"]["layout"] == "wide"
    assert out["ph_tables"]["rows"]["cell_wide"] == 8
    by_text = _rows(db, "SELECT c.prcfo, n.nucleus_area, y.cytoplasm_area "
                        "FROM cell c JOIN nucleus n ON c.prcfo = n.prcfo "
                        "JOIN cytoplasm y ON c.prcfo = y.prcfo ORDER BY c.prcfo")
    by_key = _rows(db, "SELECT c.prcfo, n.nucleus_area, y.cytoplasm_area "
                       "FROM cell c JOIN nucleus n ON c.cell_key = n.cell_key "
                       "JOIN cytoplasm y ON c.cell_key = y.cell_key "
                       "ORDER BY c.prcfo")
    assert len(by_text) == 8
    pd.testing.assert_frame_equal(by_text, by_key)
    keys = _rows(db, "SELECT prcfo, cell_key FROM cell ORDER BY prcfo")
    assert keys["cell_key"].is_unique
    first = keys.iloc[0]
    assert first["prcfo"] == "P_1_r1_c1_f0_o1"
    assert first["cell_key"] == ((1 * 100 + 1) * 2 ** 24 + 0) * 2 ** 20 + 1
    wide = _rows(db, "SELECT * FROM cell_wide ORDER BY prcfo")
    joined = _rows(db, "SELECT * FROM cell JOIN nucleus USING (prcfo) "
                       "JOIN cytoplasm USING (prcfo) ORDER BY prcfo")
    features = [c for c in wide.columns
                if c.startswith(("cell_", "nucleus_", "cytoplasm_"))
                and c != "cell_key"]
    assert len(features) > 30
    pd.testing.assert_frame_equal(wide[["prcfo", *features]],
                                  joined[["prcfo", *features]])
    indexes = set(_rows(db, "SELECT name FROM sqlite_master WHERE "
                            "type='index'")["name"])
    assert {"ix_cell_cell_key", "ix_nucleus_cell_key",
            "ix_cytoplasm_cell_key", "ix_cell_wide_prcfo"} <= indexes


def test_the_default_tables_carry_no_integer_key(measured):
    out = _run(measured, ("ph_masks", "merge", "aggregate", "ph_tables"))
    db = out["ph_tables"]["db"]
    columns = list(_rows(db, "SELECT * FROM cell LIMIT 1").columns)
    assert "cell_key" not in columns
    assert out["ph_tables"]["layout"] == "compartments"
    with pytest.raises(ValueError, match="compartments or wide"):
        _run(measured, ("ph_tables",), ops_tables_layout="tall")


def test_the_stress_report_has_the_reference_columns_and_exact_counts(stored):
    root, out = stored
    path = root / "run" / "results" / "ops_stress" / "stress_test.csv"
    assert out["stress"] == {"csv": str(path), "rows": 2}
    from spacr.tabular import read_table

    table = read_table(str(path), canonicalise=False)
    assert list(table.columns) == list(ops_run_infra.STRESS_COLUMNS)
    assert len(ops_run_infra.STRESS_COLUMNS) == 58
    well = table[table["level"] == "well"].iloc[0]
    assert well["well"] == "A1" and well["fields"] == 2
    assert well["cells"] == 8 and well["linked"] == 8
    assert well["measure_rows"] == 24 and well["query_join3_rows"] == 8
    assert well["query_whole_well_rows"] == 8 and well["query_one_field_rows"] == 4
    assert well["query_one_gene"] in ("GENE1", "GENE2")
    assert well["query_one_gene_rows"] == 2
    assert well["tar_shards"] == 3 and well["tar_crops"] == 8
    assert well["tar_random_read_n"] == 8
    assert well["files_in_well_dir"] > 4 and well["nas_well_bytes"] > 0
    assert pd.isna(well["png_crops"])
    plate = table[table["level"] == "plate"].iloc[0]
    assert plate["cells_typed"] == 8 and plate["cells_segmented"] == 8
    assert plate["nas_plate_files"] == well["files_in_well_dir"] + 3
    lines = (path.parent / "stress_rows.jsonl").read_text().splitlines()
    assert len(lines) == 2 and '"png_crops"' not in lines[0]


def test_the_stress_report_without_tables_or_crops(measured):
    out = _run(measured, ("ph_masks",), ops_stress_report=True)
    rows = out["stress"]["rows"]
    assert rows == 2
    row = ops_run_infra.stress_rows(str(measured / "run"), PLATE, ["A1"], out)[0]
    assert row["measure_rows"] is None and row["tar_crops"] is None


def test_integer_cell_keys_are_distinct_and_range_checked():
    keys = ops_run_infra.integer_cell_key([1, 1, 2, 16], [1, 2, 1, 24],
                                          [0, 0, 0, 2 ** 24 - 1],
                                          [1, 1, 1, 2 ** 20 - 1])
    assert keys.dtype == np.int64 and len(set(keys.tolist())) == 4
    assert keys.max() < 2 ** 63
    with pytest.raises(ValueError, match="sites below"):
        ops_run_infra.integer_cell_key(1, 1, 2 ** 24, 1)
    with pytest.raises(ValueError, match="labels below"):
        ops_run_infra.integer_cell_key(1, 1, 0, -1)
