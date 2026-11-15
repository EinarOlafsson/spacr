"""Table-driven tests of the measurement stores behind ``spacr.tabular``.

``write_database`` is walked over every store kind -- SQLite, a Parquet
store folder, DuckDB, and PostgreSQL through an in-memory stand-in for
psycopg that keeps real tables -- for create, append (adding a column),
replace and its refusals, and each store is read back with ``read_database``
and the chunk iterator.  ``table_columns`` reads headers of empty tables and
Feather files, and ``read_table`` warns when two spellings of one key column
disagree.  All data is synthetic and local.
"""
from __future__ import annotations

import importlib.util
import re
import sqlite3
import sys
import types
import warnings

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
    """Route ``psycopg.connect`` to an in-memory server; yields the server."""
    server = _Server()
    fake = types.ModuleType("psycopg")
    fake.connect = lambda dsn: _Connection(server)
    monkeypatch.setitem(sys.modules, "psycopg", fake)
    return server


def _cells(n, extra=None):
    """``n`` cell rows; ``extra`` adds a float column spanning 0..1."""
    frame = pd.DataFrame({"prcfo": [f"plate1_r1_c{i}_1_{i}" for i in range(n)],
                          "cell_area": np.arange(n, dtype=np.int64) + 10,
                          "seen": pd.to_datetime(["2026-09-30"] * n)})
    if extra:
        frame[extra] = np.linspace(0, 1, n)
    return frame


def _read(store, name="cell", **kwargs):
    """The single frame ``read_database`` gives for ``name``."""
    (frame,) = tabular.read_database(store, name, canonicalise=False,
                                     report=None, **kwargs)
    return frame


def test_write_database_creates_appends_replaces_and_refuses_on_every_store(
        tmp_path, postgres, subtests):
    """Every store kind round-trips the same writes and refuses the same bad requests."""
    stores = [
        ("sqlite", str(tmp_path / "m.db")),
        ("parquet", str(tmp_path / "m.parquetdb")),
        ("postgres", "postgresql://"),
    ]
    if importlib.util.find_spec("duckdb") is not None:
        stores.append(("duckdb", str(tmp_path / "m.duckdb")))
    for kind, store in stores:
        with subtests.test(msg=kind):
            if kind != "postgres":
                assert tabular._store_tables(store, kind) == ()
            tabular.write_database(_cells(3), store, "cell",
                                   canonicalise=False)
            assert tabular.database_tables(store) == ("cell",)
            assert tabular.table_columns(store, table="cell",
                                         canonicalise=False) == (
                "prcfo", "cell_area", "seen")
            # Every store, SQLite included, adds the column an append brings.
            tabular.write_database(
                _cells(2, extra="cell_mean"), store,
                "cell", if_exists="append", canonicalise=False)
            frame = _read(store, chunksize=2)
            assert frame["cell_area"].tolist() == [10, 11, 12, 10, 11]
            assert list(frame.columns) == ["prcfo", "cell_area", "seen",
                                           "cell_mean"]
            assert frame["cell_mean"].isna().tolist() == [True] * 3 + \
                [False] * 2
            assert frame["cell_mean"].tolist()[3:] == [0.0, 1.0]

            with pytest.raises(ValueError) as refused:
                tabular.write_database(_cells(1), store, "cell",
                                       if_exists="fail", canonicalise=False)
            assert str(refused.value) == (
                f"Table 'cell' already exists in {store!r}.")
            with pytest.raises(ValueError) as refused:
                tabular.write_database(_cells(1), store, "cell",
                                       if_exists="merge", canonicalise=False)
            assert str(refused.value) == (
                "if_exists must be 'append', 'replace' or 'fail', not "
                "'merge'.")

            tabular.write_database(_cells(1), store, "cell",
                                   if_exists="replace", canonicalise=False)
            frame = _read(store)
            assert frame["prcfo"].tolist() == ["plate1_r1_c0_1_0"]
            assert frame["cell_area"].tolist() == [10]
            answer = tabular._query_store(store, 'SELECT * FROM "cell"',
                                          canonicalise=False, report=None)
            assert answer["cell_area"].tolist() == [10]

    with subtests.test(msg="postgres-schema-empty-table-and-migration"):
        assert postgres.tables["cell"][0] == ["prcfo", "cell_area", "seen"]
        assert len(postgres.tables["cell"][1]) == 1
        postgres.tables["empty"] = (["a", "b"], [])
        empty = _read("postgresql://", "empty")
        assert empty.empty and list(empty.columns) == ["a", "b"]
        target = str(tmp_path / "copy.parquetdb")
        assert tabular._migrate_database("postgresql://", target,
                                         tables="cell", report=None) == (
            "cell",)
        assert _read(target)["cell_area"].tolist() == [10]

    with subtests.test(msg="duckdb-empty-table-keeps-its-header"):
        duckdb = pytest.importorskip("duckdb")
        db = str(tmp_path / "m.duckdb")
        with duckdb.connect(db) as conn:
            conn.execute("CREATE TABLE empty (a INTEGER)")
        empty = _read(db, "empty")
        assert empty.empty and list(empty.columns) == ["a"]


def test_parquet_chunks_stop_at_the_limit_and_skip_empty_batches(
        tmp_path, monkeypatch, subtests):
    """A limited read stops across parts; an empty record batch is never yielded."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    store = str(tmp_path / "m.parquetdb")
    for _ in range(3):
        tabular.write_database(_cells(4), store, "cell", if_exists="append",
                               canonicalise=False)
    cases = [
        ("limit-6-of-12", {"limit": 6, "chunksize": 4}, [4, 2]),
        ("limit-4-of-12", {"limit": 4, "chunksize": 4}, [4]),
        ("no-limit", {"chunksize": 4}, [4, 4, 4]),
    ]
    for case_id, kwargs, sizes in cases:
        with subtests.test(msg=case_id):
            chunks = list(tabular._iter_chunks(store, "cell", **kwargs))
            assert [len(chunk) for chunk in chunks] == sizes
            assert pd.concat(chunks)["cell_area"].tolist() == \
                ([10, 11, 12, 13] * 3)[:sum(sizes)]

    with subtests.test(msg="empty-batch-dropped"):
        single = str(tmp_path / "one.parquetdb")
        tabular.write_database(_cells(3), single, "cell", canonicalise=False)
        real = pq.ParquetFile.iter_batches

        def with_an_empty_batch(self, *args, **kwargs):
            batches = list(real(self, *args, **kwargs))
            yield from batches
            yield pa.RecordBatch.from_pandas(
                batches[0].to_pandas().head(0), preserve_index=False)

        monkeypatch.setattr(pq.ParquetFile, "iter_batches",
                            with_an_empty_batch)
        chunks = list(tabular._iter_chunks(single, "cell"))
        assert [len(chunk) for chunk in chunks] == [3]


def test_table_columns_reads_the_header_of_any_table_kind(tmp_path, subtests):
    """An empty SQLite table and a Feather file give their header, canonicalised on request."""
    db = tmp_path / "m.db"
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE cell (prcfo TEXT, cell_area INTEGER)')
    feather = tmp_path / "scores.feather"
    pd.DataFrame({"prcfo": ["p1_r1_c1_f1_o1"], "score": [0.5],
                  "gene": ["ABC"], "plate": ["x"]}).to_feather(feather)
    cases = [
        ("empty-sqlite", str(db), {"table": "cell", "canonicalise": False},
         ("prcfo", "cell_area")),
        ("feather-raw", str(feather), {"canonicalise": False},
         ("prcfo", "score", "gene", "plate")),
        ("feather-canonical", str(feather), {},
         ("prcfo", "score", "gene", "plateID")),
    ]
    for case_id, path, kwargs, header in cases:
        with subtests.test(msg=case_id):
            assert tuple(tabular.table_columns(path, **kwargs)) == header
    answer = tabular._query_store(str(db), "SELECT COUNT(*) AS n FROM cell",
                                  canonicalise=False, report=None)
    assert answer["n"].tolist() == [0]


def test_read_table_warns_when_two_spellings_of_a_key_column_disagree(
        tmp_path, subtests):
    """Disagreeing ``row``/``row_name`` warn with the count that differ; agreeing ones merge silently."""
    cases = [
        ("disagree", ["WRONG"] * 3, ["WRONG"] * 3,
         ["rowID metadata columns row, row_name do not agree (3 of 3 rows "
          "differ); row is being used and the rest are dropped"]),
        ("one-differs", ["r1", "r2", "r9"], ["r1", "r2", "r9"],
         ["rowID metadata columns row, row_name do not agree (1 of 3 rows "
          "differ); row is being used and the rest are dropped"]),
        ("agree", ["r1", "r2", "r1"], ["r1", "r2", "r1"], []),
    ]
    for case_id, row, kept, said in cases:
        with subtests.test(msg=case_id):
            path = tmp_path / f"{case_id}.csv"
            pd.DataFrame({"plateID": ["plate1"] * 3,
                          "fieldID": ["f0", "f1", "f2"],
                          "object": [1, 2, 3], "score": [0, 1, 0],
                          "row": row,
                          "row_name": ["r1", "r2", "r1"]}).to_csv(
                path, index=False)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                frame = tabular.read_table(str(path), report=None)
            assert [str(w.message) for w in caught
                    if w.category is UserWarning] == said
            assert list(frame.columns) == ["plateID", "fieldID", "object",
                                           "score", "rowID"]
            assert frame["rowID"].tolist() == kept


def test_sqlite_append_adds_new_columns_with_their_types(tmp_path):
    """An SQLite append that brings a column widens the table, typed like to_sql.

    ``B`` is the existing ``b``: SQLite column names ignore case.
    """
    db = str(tmp_path / "m.db")
    tabular.write_database(pd.DataFrame({"a": [1], "b": ["x"]}), db, "cell")
    tabular.write_database(
        pd.DataFrame({"a": [2], "B": ["y"], "c": [0.5]}),
        db, "cell", if_exists="append")
    with sqlite3.connect(db) as conn:
        info = conn.execute('PRAGMA table_info("cell")').fetchall()
    assert [(row[1], row[2]) for row in info] == [
        ("a", "INTEGER"), ("b", "TEXT"), ("c", "REAL")]
