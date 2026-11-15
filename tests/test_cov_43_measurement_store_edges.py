"""The measurement stores behind spacr.tabular, at the edges the ratchet found.

The PostgreSQL type mapping, an interrupted publish, legacy column names on
the way to Parquet and RDS, a non-cloud address and a driver that yields no
chunk for an empty table. The store round trips themselves are tested in
``tests/test_tabular_measurement_stores.py``.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from spacr import tabular

pytest.importorskip("pyarrow")


def test_a_postgres_timestamp_column_is_typed_as_one():
    assert tabular._postgres_type(pd.Series(
        pd.to_datetime(["2026-09-30"])).dtype) == "TIMESTAMP"


def test_a_write_that_fails_leaves_no_pending_file(tmp_path):
    target = tmp_path / "out.csv"

    def half_written(pending):
        with open(pending, "w") as handle:
            handle.write("partial")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        tabular._publish(str(target), half_written)
    assert list(tmp_path.iterdir()) == []


def test_legacy_names_are_canonicalised_on_the_way_to_parquet_and_rds(
        tmp_path):
    frame = pd.DataFrame({"plate": ["plate1"], "column_name": ["c1"],
                          "value": [1.0]})
    out = tabular._write_parquet(frame, tmp_path / "t.parquet")
    written = pd.read_parquet(out)
    assert {"plateID", "columnID"} <= set(written.columns)
    assert "plate" not in written.columns
    pytest.importorskip("pyreadr")
    rds = tabular._write_rds(frame, tmp_path / "t.rds")
    import pyreadr

    (back,) = pyreadr.read_r(rds).values()
    assert {"plateID", "columnID"} <= set(back.columns)


def test_an_address_that_is_not_a_cloud_store_is_left_as_given():
    assert tabular._fetched("sqlite://not-a-cloud-store/m.db") == (
        "sqlite://not-a-cloud-store/m.db")


def test_a_driver_that_yields_no_chunk_for_an_empty_table_still_gives_the_header(
        tmp_path, monkeypatch):
    """Some pandas versions yield no chunk at all for an empty result; the
    read then asks for the header on its own."""
    db = tmp_path / "m.db"
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE cell (prcfo TEXT, cell_area INTEGER)')
    real = pd.read_sql_query

    def no_chunks(sql, conn, chunksize=None, **kwargs):
        if chunksize is not None:
            return iter(())
        return real(sql, conn, **kwargs)

    monkeypatch.setattr(tabular.pd, "read_sql_query", no_chunks)
    (frame,) = list(tabular._iter_chunks(str(db), "cell", chunksize=10))
    assert frame.empty and list(frame.columns) == ["prcfo", "cell_area"]


