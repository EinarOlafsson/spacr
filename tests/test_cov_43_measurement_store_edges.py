"""The measurement stores behind spacr.tabular, at the edges the ratchet found.

PostgreSQL is driven through an in-memory stand-in for psycopg that keeps
real tables, so a write, an append that adds a column, a replace, a chunked
read, a migration and a query go through the same funnel a server would see.
The SQLite, Parquet and DuckDB edges use the real libraries.
"""
from __future__ import annotations

import re
import sqlite3
import sys
import types

import numpy as np
import pandas as pd
import pytest

from spacr import tabular

pytest.importorskip("pyarrow")


class _Server:
    """Tables kept as ``{name: (columns, rows)}``."""

    def __init__(self):
        self.tables = {}


class _Result:
    def __init__(self, rows, names=()):
        self._rows = list(rows)
        self.description = [types.SimpleNamespace(name=n) for n in names]

    def fetchall(self):
        return list(self._rows)


class _Connection:
    """Just enough of a psycopg connection for spacr.tabular."""

    def __init__(self, server):
        self.server = server

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _select(self, sql):
        match = re.search(r'FROM "([^"]+)"(?: LIMIT (\d+))?', sql)
        columns, rows = self.server.tables[match.group(1)]
        if match.group(2) is not None:
            rows = rows[:int(match.group(2))]
        return columns, rows

    def execute(self, sql, params=None):
        tables = self.server.tables
        if "information_schema.tables" in sql:
            return _Result([(name,) for name in tables])
        if "information_schema.columns" in sql:
            return _Result([(c,) for c in tables[params[0]][0]])
        created = re.match(r'CREATE TABLE "([^"]+)" \((.*)\)$', sql)
        if created:
            names = re.findall(r'"([^"]+)" [A-Z ]+', created.group(2))
            tables[created.group(1)] = (names, [])
            return _Result([])
        dropped = re.match(r'DROP TABLE "([^"]+)"', sql)
        if dropped:
            del tables[dropped.group(1)]
            return _Result([])
        added = re.match(r'ALTER TABLE "([^"]+)" ADD COLUMN "([^"]+)"', sql)
        if added:
            columns, rows = tables[added.group(1)]
            tables[added.group(1)] = (columns + [added.group(2)],
                                      [row + (None,) for row in rows])
            return _Result([])
        columns, rows = self._select(sql)
        return _Result(rows, columns)

    def cursor(self, name=None):
        connection = self

        class _Copy:
            def __init__(self, table, header):
                self.table, self.header = table, header

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def write_row(self, row):
                columns, rows = connection.server.tables[self.table]
                value = dict(zip(self.header, row))
                rows.append(tuple(value.get(c) for c in columns))

        class _Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def copy(self, sql):
                match = re.match(r'COPY "([^"]+)" \((.*)\) FROM STDIN', sql)
                header = re.findall(r'"([^"]+)"', match.group(2))
                return _Copy(match.group(1), header)

            def execute(self, sql):
                self.columns, self.rows = connection._select(sql)
                self.description = [types.SimpleNamespace(name=c)
                                    for c in self.columns]

            def fetchmany(self, size):
                taken, self.rows = self.rows[:size], self.rows[size:]
                return taken

        return _Cursor()


@pytest.fixture
def postgres(monkeypatch):
    server = _Server()
    fake = types.ModuleType("psycopg")
    fake.connect = lambda dsn: _Connection(server)
    monkeypatch.setitem(sys.modules, "psycopg", fake)
    return server


def _cells(n, extra=None):
    frame = pd.DataFrame({"prcfo": [f"plate1_r1_c{i}_1_{i}" for i in range(n)],
                          "cell_area": np.arange(n, dtype=np.int64) + 10,
                          "seen": pd.to_datetime(["2026-09-30"] * n)})
    if extra:
        frame[extra] = np.linspace(0, 1, n)
    return frame


def test_a_postgres_store_round_trips_appends_replaces_and_answers_sql(
        postgres, tmp_path):
    dsn = "postgresql://"
    tabular.write_database(_cells(3), dsn, "cell", canonicalise=False)
    tabular.write_database(_cells(2, extra="cell_mean"), dsn, "cell",
                           if_exists="append", canonicalise=False)
    assert postgres.tables["cell"][0] == ["prcfo", "cell_area", "seen",
                                          "cell_mean"]
    assert tabular.database_tables(dsn) == ("cell",)
    (frame,) = tabular.read_database(dsn, "cell", canonicalise=False,
                                     chunksize=2, report=None)
    assert len(frame) == 5 and frame["cell_mean"].isna().sum() == 3

    tabular.write_database(_cells(1), dsn, "cell", if_exists="replace",
                           canonicalise=False)
    assert len(postgres.tables["cell"][1]) == 1

    answer = tabular._query_store(dsn, 'SELECT * FROM "cell"',
                                  canonicalise=False, report=None)
    assert answer["cell_area"].tolist() == [10]

    postgres.tables["empty"] = (["a", "b"], [])
    (empty,) = tabular.read_database(dsn, "empty", canonicalise=False,
                                     report=None)
    assert empty.empty and list(empty.columns) == ["a", "b"]

    copied = tabular._migrate_database(dsn, str(tmp_path / "m.parquetdb"),
                                       tables="cell", report=None)
    assert copied == ("cell",)


def test_a_postgres_timestamp_column_is_typed_as_one():
    assert tabular._postgres_type(pd.Series(
        pd.to_datetime(["2026-09-30"])).dtype) == "TIMESTAMP"


def test_an_unknown_if_exists_is_refused(tmp_path):
    with pytest.raises(ValueError, match="if_exists must be"):
        tabular.write_database(_cells(1), str(tmp_path / "m.parquetdb"),
                               "cell", if_exists="merge", canonicalise=False)


def test_a_store_table_that_exists_is_not_overwritten_by_fail(tmp_path):
    store = str(tmp_path / "m.parquetdb")
    tabular.write_database(_cells(1), store, "cell", canonicalise=False)
    with pytest.raises(ValueError, match="already exists"):
        tabular.write_database(_cells(1), store, "cell", if_exists="fail",
                               canonicalise=False)


def test_an_empty_sqlite_table_still_gives_its_header(tmp_path):
    db = tmp_path / "m.db"
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE cell (prcfo TEXT, cell_area INTEGER)')
    assert tabular.table_columns(str(db), table="cell",
                                 canonicalise=False) == ("prcfo", "cell_area")
    answer = tabular._query_store(str(db), "SELECT COUNT(*) AS n FROM cell",
                                  canonicalise=False, report=None)
    assert answer["n"].tolist() == [0]


def test_a_parquet_store_table_is_replaced_whole(tmp_path):
    store = str(tmp_path / "m.parquetdb")
    tabular.write_database(_cells(3), store, "cell", canonicalise=False)
    tabular.write_database(_cells(1), store, "cell", if_exists="replace",
                           canonicalise=False)
    (frame,) = tabular.read_database(store, "cell", canonicalise=False,
                                     report=None)
    assert len(frame) == 1


def test_a_parquet_read_stops_at_its_limit_across_parts(tmp_path):
    store = str(tmp_path / "m.parquetdb")
    for _ in range(3):
        tabular.write_database(_cells(4), store, "cell", if_exists="append",
                               canonicalise=False)
    chunks = list(tabular._iter_chunks(store, "cell", limit=6, chunksize=4))
    assert sum(len(chunk) for chunk in chunks) == 6
    assert tabular._store_tables(str(tmp_path / "absent.parquetdb"),
                                 "parquet") == ()


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


def test_duckdb_edges(tmp_path):
    pytest.importorskip("duckdb")
    db = str(tmp_path / "m.duckdb")
    assert tabular._store_tables(db, "duckdb") == ()
    tabular.write_database(_cells(2), db, "cell", canonicalise=False)
    tabular.write_database(_cells(1), db, "cell", if_exists="replace",
                           canonicalise=False)
    (frame,) = tabular.read_database(db, "cell", canonicalise=False,
                                     report=None)
    assert len(frame) == 1
    import duckdb

    with duckdb.connect(db) as conn:
        conn.execute("CREATE TABLE empty (a INTEGER)")
    (empty,) = tabular.read_database(db, "empty", canonicalise=False,
                                     report=None)
    assert empty.empty and list(empty.columns) == ["a"]


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


def test_an_empty_batch_inside_a_parquet_part_is_not_passed_on(tmp_path,
                                                               monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    store = str(tmp_path / "m.parquetdb")
    tabular.write_database(_cells(3), store, "cell", canonicalise=False)
    real = pq.ParquetFile.iter_batches

    def with_an_empty_batch(self, *args, **kwargs):
        batches = list(real(self, *args, **kwargs))
        yield from batches
        yield pa.RecordBatch.from_pandas(
            batches[0].to_pandas().head(0), preserve_index=False)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", with_an_empty_batch)
    chunks = list(tabular._iter_chunks(store, "cell"))
    assert [len(chunk) for chunk in chunks] == [3]
