"""``call_source``: whether the reads or the pixel fallback made a genotype call.

The B1 pixel read gate keeps the pixel-fallback calls, so every per-cell
table that carries the call also says where it came from: ``ops_barcodes``
(decode), ``ops_cells_<plate>.parquet`` (aggregate) and the store tables
``cell``, ``nucleus`` and ``cytoplasm`` (``ph_tables``). The column is
additive: no call, count or other column moves, which the runs below check
by writing the same well with and without the column in its calls.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")
pytest.importorskip("pyarrow")
pytest.importorskip("tifffile")

from spacr import ops_decode, ops_engine
from spacr.ops_store import read_table, write_table
from tests.test_ops_phenotype_run_v2 import (PLATE, _placed, _run, _sequencing,
                                             _write_tiles, models)  # noqa: F401
from tests.test_ops_pixel_read_gate_964 import _calls

BARCODE_COLUMNS = ["plate", "well", "object_id", "barcode", "quality", "n_reads",
                   "n_agreeing", "fraction", "n_cycles", "mapped_guide"]


@pytest.mark.parametrize("gate, floor", [(True, 2.0), (True, 0.0), (False, 2.0)])
def test_barcode_rows_say_read_or_pixel_exactly_as_the_call_was_made(gate, floor):
    calls = _calls(gate, single_read_quality=floor)
    rows = ops_decode.barcode_rows("P", "A1", [{"rule_a": calls, "n_cycles": 6}])
    found = {row["object_id"]: row["call_source"] for row in rows}
    expected = {o: {"primary": "read", "pixel_fallback": "pixel"}[h["source"]]
                for o, h in calls["calls"].items() if not h["decoy"]}
    assert found == expected
    assert "pixel" in found.values() and "read" in found.values()
    assert sum(v == "pixel" for v in found.values()) == calls["assigned_pixel_real"]
    for row in rows:
        hit = calls["calls"][row["object_id"]]
        assert list(row) == BARCODE_COLUMNS + ["call_source"]
        assert {k: row[k] for k in BARCODE_COLUMNS} == {
            "plate": "P", "well": "A1", "object_id": row["object_id"],
            "barcode": hit["barcode"], "quality": hit["quality"],
            "n_reads": hit["n_reads"], "n_agreeing": hit["n_agreeing"],
            "fraction": hit["fraction"], "n_cycles": 6, "mapped_guide": hit["guide"]}


def test_a_genotype_table_passes_its_call_source_through(tmp_path):
    path = tmp_path / "calls.csv"
    pd.DataFrame({"plate": PLATE, "well": "A1", "object_id": [1, 2, 3],
                  "mapped_guide": ["AAAA", "CCCC", ""],
                  "call_source": ["read", "pixel", "read"]}).to_csv(path, index=False)
    called = ops_engine._ph_genotype_table(str(path), PLATE, ["A1"])
    assert called["call_source"].tolist() == ["read", "pixel"]
    pd.DataFrame({"well": "A1", "object_id": [1, 2],
                  "mapped_guide": ["AAAA", "CCCC"]}).to_csv(path, index=False)
    called = ops_engine._ph_genotype_table(str(path), PLATE, ["A1"])
    assert called["call_source"].isna().all()
    assert list(called.columns) == ["well", "object_id", "mapped_guide", "call_source"]


def _well(base, with_source: bool):
    """A segmented, measured synthetic well whose calls say read or pixel.

    Odd sequencing nuclei are pixel calls, even ones read calls; nucleus 3
    has no call at all. Without ``with_source`` the calls carry no
    ``call_source``, as a run decoded before the column existed.
    """
    _write_tiles(base / "tiles")
    run = base / "run"
    run.mkdir(parents=True)
    db = str(run / "measurements.db")
    _placed(db)
    _run(base, ("ph_segment", "ph_measure"))
    cells = pd.read_parquet(ops_engine._ph_cells_path(str(run), "A1"))
    _sequencing(db, base, cells)
    barcodes = read_table(db, "ops_barcodes")
    barcodes = barcodes[barcodes["object_id"] != 3].reset_index(drop=True)
    if with_source:
        barcodes["call_source"] = np.where(barcodes["object_id"] % 2 == 1,
                                           "pixel", "read")
    write_table(db, "ops_barcodes", barcodes, if_exists="replace")
    out = _run(base, ("ph_masks", "merge", "aggregate", "ph_tables"),
               ops_ph_measure_gpu=False)
    return out, db


@pytest.fixture
def two_runs(tmp_path, models, monkeypatch):  # noqa: F811
    """The same well run with and without ``call_source`` in its calls.

    The segmentation's free-RAM wait is switched off: it keeps 2 GiB free
    under the process's cgroup limit and waits up to 900 s per batch when
    the limit is tighter, and this 64 x 64 well needs a few MB.
    """
    from spacr import _mask_batches

    monkeypatch.setattr(_mask_batches, "_available_ram_bytes", lambda: None)
    return {flag: _well(tmp_path / str(flag), flag) for flag in (True, False)}


def _store_tables(out):
    from spacr import store

    with store.open(out["ph_tables"]["store"]) as opened:
        return {name: opened.query(table=name).sort_values("prcfo").reset_index(drop=True)
                for name in ("cell", "nucleus", "cytoplasm")}


def test_every_per_cell_table_carries_the_call_source(two_runs):
    out, db = two_runs[True]
    barcodes = read_table(db, "ops_barcodes")
    source_of = dict(zip(barcodes["object_id"].astype(int), barcodes["call_source"]))
    assert set(source_of.values()) == {"read", "pixel"}
    cells = pd.read_parquet(out["aggregate"]["outputs"]["cells_parquet"])
    assert list(cells.columns)[-1] == "call_source"
    assert len(cells) and cells["call_source"].tolist() == [
        source_of[int(o)] for o in cells["object_id"]]
    assert set(cells["call_source"]) == {"read", "pixel"}
    assert ops_engine._PH_TABLE_KEYS[-1] == "call_source"
    for name, table in _store_tables(out).items():
        assert "call_source" in table.columns, name
        called = table["object_id"].notna() & table["object_id"].isin(list(source_of))
        assert table.loc[called, "call_source"].tolist() == [
            source_of[int(o)] for o in table.loc[called, "object_id"]], name
        assert table.loc[~called, "call_source"].isna().all(), name
        assert (~called).any() and table.loc[~called, "sgRNA"].isna().all(), name
        assert set(table.loc[called, "call_source"]) == {"read", "pixel"}, name


def test_the_column_is_additive_and_an_old_decode_gives_nulls(two_runs):
    (new, new_db), (old, old_db) = two_runs[True], two_runs[False]
    assert "call_source" not in read_table(old_db, "ops_barcodes").columns
    a = pd.read_parquet(new["aggregate"]["outputs"]["cells_parquet"])
    b = pd.read_parquet(old["aggregate"]["outputs"]["cells_parquet"])
    assert b["call_source"].isna().all()
    pd.testing.assert_frame_equal(a.drop(columns="call_source"),
                                  b.drop(columns="call_source"))
    for suffix in ("guides", "genes"):
        for kind in ("csv", "parquet"):
            key = f"{suffix}_{kind}"
            with open(new["aggregate"]["outputs"][key], "rb") as x, \
                    open(old["aggregate"]["outputs"][key], "rb") as y:
                assert x.read() == y.read(), key
    new_tables, old_tables = _store_tables(new), _store_tables(old)
    for name in new_tables:
        assert old_tables[name]["call_source"].isna().all(), name
        pd.testing.assert_frame_equal(
            new_tables[name].drop(columns="call_source"),
            old_tables[name].drop(columns="call_source"), check_like=False)
    volatile = ("seconds", "outputs", "store", "registered", "peak_rss_gb")
    for report in ("aggregate", "ph_tables"):
        keep = {k: v for k, v in new[report].items() if k not in volatile}
        other = {k: v for k, v in old[report].items() if k not in volatile}
        assert set(keep) == set(other), report
        for key in keep:
            if isinstance(keep[key], (int, float, bool)):
                assert keep[key] == other[key], (report, key)
    assert os.path.isfile(new["aggregate"]["outputs"]["cells_parquet"])
