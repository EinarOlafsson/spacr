"""DuckDB, Parquet and PostgreSQL measurement stores behind spacr.tabular.

SQLite stays the default. The same read/write funnel accepts a ``.duckdb``
file, a ``.parquetdb`` folder or a ``postgresql://`` string, and
``_migrate_database`` copies tables between any two of them. DuckDB and
Parquet run in-process. PostgreSQL runs against a real server only when
``SPACR_TEST_POSTGRES_DSN`` names one; otherwise it is exercised through a
recording stand-in for psycopg.
"""
from __future__ import annotations

import os
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest

from spacr import tabular

pytest.importorskip("pyarrow")


def _measurements(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "prcfo": [f"plate1_A{i % 3 + 1}_{i}_1" for i in range(n)],
        "plateID": ["plate1"] * n,
        "rowID": ["r1"] * n,
        "columnID": [f"c{i % 3 + 1}" for i in range(n)],
        "cell_area": rng.integers(50, 500, n),
        "cell_channel_0_mean_intensity": rng.random(n),
    })


def _stores(tmp_path):
    stores = {"parquet": str(tmp_path / "m.parquetdb")}
    try:
        import duckdb  # noqa: F401
        stores["duckdb"] = str(tmp_path / "m.duckdb")
    except ImportError:
        pass
    dsn = os.environ.get("SPACR_TEST_POSTGRES_DSN")
    if dsn:
        stores["postgres"] = dsn
    return stores


def test_locators_name_their_backend():
    assert tabular._backend_of("~/x/measurements.db") == "sqlite"
    assert tabular._backend_of("x/measurements.duckdb") == "duckdb"
    assert tabular._backend_of("x/measurements.parquetdb/") == "parquet"
    assert tabular._backend_of("postgresql://user@host/db") == "postgres"
    assert tabular._backend_of(object()) == "sqlite"


def test_importing_the_funnel_does_not_import_the_drivers():
    code = ("import sys, spacr.tabular; "
            "print('duckdb' in sys.modules, 'psycopg' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout.split()
    assert out == ["False", "False"]


@pytest.mark.parametrize("backend", ["parquet", "duckdb", "postgres"])
def test_every_store_round_trips_through_the_funnel(tmp_path, backend):
    store = _stores(tmp_path).get(backend)
    if store is None:
        pytest.skip(f"{backend} store not available here")
    frame = _measurements()
    tabular.write_database(frame, store, "cell", if_exists="replace")
    assert "cell" in tabular.database_tables(store)
    back = tabular.read_database(store, "cell", report=None)[0]
    pd.testing.assert_frame_equal(back[frame.columns], frame,
                                  check_dtype=False)
    assert tabular.read_table(store, table="cell", report=None).shape == \
        frame.shape
    assert tabular.table_columns(store, table="cell") == tuple(frame.columns)
    assert len(tabular.read_database(store, "cell", limit=7,
                                     report=None)[0]) == 7

    extra = _measurements(5, seed=1).assign(cell_perimeter=1.5)
    tabular.write_database(extra, store, "cell")
    grown = tabular.read_database(store, "cell", report=None)[0]
    assert len(grown) == 55
    assert grown["cell_perimeter"].notna().sum() == 5

    with pytest.raises(ValueError, match="Table not found"):
        tabular.read_database(store, "nucleus", report=None)
    with pytest.raises(ValueError, match="already exists"):
        tabular.write_database(frame, store, "cell", if_exists="fail")

    if "duckdb" not in _stores(tmp_path) and backend == "parquet":
        with pytest.raises(ImportError, match=r"spacr\[databases\]"):
            tabular._query_store(store, 'SELECT 1', report=None)
        return
    totals = tabular._query_store(
        store, 'SELECT "columnID", COUNT(*) AS n FROM "cell" '
               'GROUP BY "columnID" ORDER BY "columnID"', report=None)
    assert totals["n"].sum() == 55


@pytest.mark.parametrize("backend", ["parquet", "duckdb", "postgres"])
def test_migration_goes_both_ways(tmp_path, backend):
    store = _stores(tmp_path).get(backend)
    if store is None:
        pytest.skip(f"{backend} store not available here")
    source = str(tmp_path / "measurements.db")
    frame = _measurements(120)
    tabular.write_database(frame, source, "cell")
    tabular.write_database(frame.head(0), source, "empty")
    copied = tabular._migrate_database(source, store, chunksize=50,
                                       report=None)
    assert set(copied) == {"cell", "empty"}
    assert len(tabular.read_database(store, "cell", report=None)[0]) == 120

    back = str(tmp_path / "back.db")
    tabular._migrate_database(store, back, report=None)
    again = tabular.read_database(back, "cell", migrate=False,
                                  report=None)[0]
    pd.testing.assert_frame_equal(again[frame.columns], frame,
                                  check_dtype=False)
    assert tabular.table_columns(back, table="empty") == tuple(frame.columns)


def test_measure_copies_a_finished_run_to_the_chosen_store(tmp_path):
    from spacr.measure import (_copy_to_measurement_backend,
                               _measurement_backend_target)

    db = str(tmp_path / "measurements" / "measurements.db")
    tabular.write_database(_measurements(), db, "cell")
    settings = {"measurement_backend": "parquet",
                "measurement_backend_target": ""}
    target = _copy_to_measurement_backend(db, settings)
    assert target == str(tmp_path / "measurements" / "measurements.parquetdb")
    assert tabular.database_tables(target) == ("cell",)
    assert _measurement_backend_target(
        db, {"measurement_backend": "postgres",
             "measurement_backend_target": "host=db dbname=screens"}
    ).startswith("postgresql://")
    with pytest.raises(ValueError):
        _measurement_backend_target(db, {"measurement_backend": "oracle"})


class _Recorder:
    """A stand-in psycopg connection that records what it is asked."""

    def __init__(self, log):
        self.log = log
        self.tables = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.log.append(sql)
        return types.SimpleNamespace(fetchall=lambda: [])

    def cursor(self, name=None):
        log = self.log

        class _Copy:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def write_row(self, row):
                log.append(("row", row))

        class _Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def copy(self, sql):
                log.append(sql)
                return _Copy()

        return _Cursor()


def test_postgres_writes_are_typed_and_copied_mocked(monkeypatch):
    log = []
    fake = types.ModuleType("psycopg")
    fake.connect = lambda dsn: _Recorder(log)
    monkeypatch.setitem(sys.modules, "psycopg", fake)
    frame = pd.DataFrame({"plateID": ["p1", None], "cell_area": [3, 4],
                          "ok": [True, False], "mean": [0.5, np.nan]})
    assert tabular.write_database(frame, "postgresql://", "cell") == \
        "postgresql://"
    create = next(sql for sql in log if isinstance(sql, str)
                  and sql.startswith("CREATE TABLE"))
    assert '"plateID" TEXT' in create and '"cell_area" BIGINT' in create
    assert '"ok" BOOLEAN' in create and '"mean" DOUBLE PRECISION' in create
    rows = [entry[1] for entry in log if isinstance(entry, tuple)]
    assert rows == [("p1", 3, True, 0.5), (None, 4, False, None)]


def test_a_missing_driver_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "psycopg", None)
    with pytest.raises(ImportError, match=r"spacr\[databases\]"):
        tabular.database_tables("postgresql://")
