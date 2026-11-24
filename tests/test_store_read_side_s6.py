"""The store's read side for the GUI (unified-store step S6).

One shared :class:`spacr.store.Store` per project, a per-thread twin, the
path locator the screens use, distinct values for filter lists, a
read-only SQLite connection for an older file, and the streamed export
(CSV, Parquet, SQLite split by feature family above 2,000 columns).
"""
from __future__ import annotations

import os
import sqlite3
import threading

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from spacr import store


def _project(root, wells=("A01", "A02")):
    """A small store project with a cell table in two wells."""
    project = str(root / "measurements")
    for i, well in enumerate(wells):
        frame = pd.DataFrame({
            "object_key": np.arange(10, dtype=np.int64) + 100 * i,
            "gene": ["g1", "NT"] * 5,
            "cell_area": np.arange(10, dtype=np.float32) + i})
        store.write_table(project, "cell", frame, plate="p1", well=well,
                          run_id="t", sort_by=("object_key",),
                          key_columns=("object_key", "gene"))
    store.register(project)
    return project


@pytest.fixture(autouse=True)
def _fresh():
    """No shared store survives a test."""
    store.release_shared()
    yield
    store.release_shared()


def test_locate_accepts_every_spelling_of_a_project(tmp_path):
    project = _project(tmp_path)
    for spelling in (project, os.path.join(project, store.CATALOG),
                     str(tmp_path)):
        assert store.locate(spelling) == project
    legacy = tmp_path / "old" / "measurements" / "measurements.db"
    legacy.parent.mkdir(parents=True)
    sqlite3.connect(legacy).close()
    assert store.locate(tmp_path / "old") == str(legacy)
    assert store.locate(legacy.parent) == str(legacy)
    with pytest.raises(FileNotFoundError):
        store.locate(tmp_path / "nothing")


def test_shared_is_one_store_until_the_catalog_changes(tmp_path, monkeypatch):
    project = _project(tmp_path)
    first = store.shared(project)
    assert store.shared(str(tmp_path)) is first
    monkeypatch.setattr(store, "_SHARED_CHECK", 0.0)
    assert store.shared(project) is first
    frame = pd.DataFrame({"object_key": np.arange(3, dtype=np.int64) + 900,
                          "gene": ["g1"] * 3,
                          "cell_area": np.ones(3, np.float32)})
    store.write_table(project, "cell", frame, plate="p1", well="B01",
                      run_id="t", sort_by=("object_key",),
                      key_columns=("object_key", "gene"))
    store.register(project)
    second = store.shared(project)
    assert second is not first and second.count("cell") == 23
    assert first.count("cell") == 20


def test_a_fork_reads_on_another_thread(tmp_path):
    project = _project(tmp_path)
    base = store.shared(project)
    found = {}

    def _read():
        """Page the table through a twin."""
        twin = store.fork(base)
        try:
            found["frame"] = twin.page("cell", columns=["cell_area"],
                                       limit=5)[0]
        finally:
            twin.close()

    worker = threading.Thread(target=_read)
    worker.start()
    worker.join()
    assert len(found["frame"]) == 5
    assert base.count("cell") == 20
    store.interrupt(base)


def test_distinct_reads_the_key_table_and_an_older_file(tmp_path):
    project = _project(tmp_path)
    opened = store.shared(project)
    assert store.distinct(opened, "cell", "gene") == ["NT", "g1"]
    assert store.distinct(opened, "cell", "well") == ["A01", "A02"]
    assert store.distinct(opened, "cell", "gene", where={"gene": "g1"}) == [
        "g1"]
    with pytest.raises(store.StoreError):
        store.distinct(opened, "cell", "nope")
    legacy = tmp_path / "m.db"
    with sqlite3.connect(legacy) as con:
        pd.DataFrame({"gene": ["b", "a", None, "a"]}).to_sql(
            "cell", con, index=False)
    assert store.distinct(store.open(legacy), "cell", "gene") == ["a", "b"]


def test_legacy_connection_is_read_only_and_only_for_sqlite(tmp_path):
    legacy = tmp_path / "m.db"
    with sqlite3.connect(legacy) as con:
        con.execute("CREATE TABLE t (a INTEGER)")
    with store.legacy_connection(store.open(legacy)) as con:
        with pytest.raises(sqlite3.Error):
            con.execute("INSERT INTO t VALUES (1)")
    with pytest.raises(store.StoreError):
        with store.legacy_connection(store.open(_project(tmp_path))):
            pass


@pytest.mark.parametrize("fmt", ["csv", "parquet", "sqlite"])
def test_export_streams_the_filtered_rows(tmp_path, fmt):
    project = _project(tmp_path)
    target = tmp_path / f"out.{fmt}"
    done = store.export(project, target, fmt=fmt, tables="cell",
                        where="gene = ?", params=["NT"],
                        features=["object_key", "cell_area"], batch_size=3)
    assert done["rows"] == 10 and not os.path.exists(str(target) + ".partial")
    if fmt == "csv":
        got = pd.read_csv(target)
    elif fmt == "parquet":
        got = pd.read_parquet(target)
        assert str(got["cell_area"].dtype) == "float32"
    else:
        with sqlite3.connect(target) as con:
            got = pd.read_sql_query("SELECT * FROM cell", con)
    assert got["object_key"].tolist() == [1, 3, 5, 7, 9, 101, 103, 105, 107,
                                          109]


def test_export_refuses_what_it_cannot_write(tmp_path):
    project = _project(tmp_path)
    with pytest.raises(store.StoreError):
        store.export(project, tmp_path / "x.txt", fmt="xlsx", tables="cell")
    with pytest.raises(store.StoreError):
        store.export(project, tmp_path / "x.csv", fmt="csv",
                     tables=["cell", "cell"])
    with pytest.raises(store.StoreError):
        store.export(project, tmp_path / "x.csv", fmt="csv", tables="cell",
                     features=["missing"])


def test_export_of_an_older_file_keeps_its_rows(tmp_path):
    legacy = tmp_path / "m.db"
    frame = pd.DataFrame({"prc": ["a", "b", "c"], "n": [1, None, 3],
                          "mixed": ["1", 2, "x"]})
    with sqlite3.connect(legacy) as con:
        con.execute("CREATE TABLE cell (prc TEXT, n INTEGER, mixed INTEGER)")
        con.executemany("INSERT INTO cell VALUES (?, ?, ?)",
                        frame.astype(object).where(frame.notna(), None
                                                   ).values.tolist())
    done = store.export(legacy, tmp_path / "x.parquet", fmt="parquet",
                        tables="cell")
    got = pd.read_parquet(tmp_path / "x.parquet")
    assert done["rows"] == 3
    assert got["prc"].tolist() == ["a", "b", "c"]
    assert got["mixed"].tolist() == ["1", "2", "x"]


# the arrival contract of the S6 public API (2026-11-24)

from tests.store_read_side_s6_api_hooks import (  # noqa: E402,F401
    S6_BIRTH,
    _s6_source_docstring,
    _s6_birth_arrivals,
    _validated_s6_arrivals,
    S6_DOCS_BIRTH,
    _s6_docs_arrivals,
    S6_RETIREMENT,
    _retired_s6_browser_api)


def test_the_s6_arrivals_match_their_source():
    entries = _s6_birth_arrivals()
    assert "spacr.store.shared" in entries
    assert "spacr.store_browse.StoreDb.chunk" in entries
