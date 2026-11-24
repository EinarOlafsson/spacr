"""spacr.tabular reads every spaCR measurement format through spacr.store.

Migration step S3 of the unified store: ``read_database``,
``database_tables``, ``table_columns`` and ``read_table(..., table=)`` give
the same frame for one measurement table whether it is stored in an older
``measurements.db``, in a store project or in the ``.parquetdb`` folder that
``measurement_backend: parquet`` still writes, because all three are read by
:meth:`spacr.store.Store._frames`.
"""
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from spacr import store, tabular  # noqa: E402

KEYS = ["prcfo", "rowID", "columnID", "fieldID", "object_label"]


def _cells():
    """Twelve cells in two wells, with float, integer and text features."""
    rng = np.random.default_rng(3)
    rows = []
    for column in ("c1", "c2"):
        for label in range(1, 7):
            rows.append({"prcfo": f"plate1_r1_{column}_f1_o{label}",
                         "rowID": "r1", "columnID": column, "fieldID": "f1",
                         "object_label": label})
    frame = pd.DataFrame(rows)
    frame["cell_area"] = rng.uniform(50, 500, len(frame))
    frame["cell_channel_1_mean_intensity"] = rng.uniform(0, 1, len(frame))
    frame["cell_count_marker"] = rng.integers(0, 9, len(frame))
    frame["condition"] = np.where(frame["columnID"] == "c1", "ctrl", "drug")
    return frame


def _legacy(root, frame):
    """``<root>/measurements/measurements.db`` holding ``cell``."""
    db = os.path.join(root, "measurements", "measurements.db")
    tabular.write_database(frame, db, "cell", if_exists="replace",
                           canonicalise=False)
    return db


def _store_project(root, frame):
    """A store project in ``<root>/measurements`` holding ``cell``, one part
    per well."""
    project = os.path.join(root, "measurements")
    for column, part in frame.groupby("columnID", sort=True):
        store.write_part(project, "cell", part.reset_index(drop=True),
                         plate="plate1", well=f"r1{column}",
                         sort_by=("prcfo",), key_columns=tuple(KEYS),
                         run_id="s3-test")
    store.register(project)
    return project


def _parquet_folder(root, frame):
    """A ``.parquetdb`` folder holding ``cell`` as two appended parts."""
    folder = os.path.join(root, "measurements.parquetdb")
    tabular.write_database(frame.iloc[:5], folder, "cell",
                           if_exists="replace", canonicalise=False)
    tabular.write_database(frame.iloc[5:], folder, "cell",
                           if_exists="append", canonicalise=False)
    return folder


def _comparable(frame, columns):
    """``columns`` of ``frame`` in ``prcfo`` order, with a fresh index."""
    return (frame[columns].sort_values("prcfo")
            .reset_index(drop=True))


@pytest.fixture
def three_formats(tmp_path):
    """The same cells in a legacy database, a store project and a Parquet
    folder, each in its own root."""
    frame = _cells()
    legacy = _legacy(str(tmp_path / "legacy"), frame)
    project = _store_project(str(tmp_path / "store"), frame)
    parquet = _parquet_folder(str(tmp_path / "parquet"), frame)
    return frame, {"legacy": legacy, "store": project, "parquet": parquet}


def test_read_database_gives_equivalent_frames_from_all_three_formats(
        three_formats):
    frame, sources = three_formats
    columns = list(frame.columns)
    expected = _comparable(frame, columns)
    for name, source in sources.items():
        (read,) = tabular.read_database(source, ["cell"], canonicalise=False,
                                        report=None, migrate=False)
        got = _comparable(read, columns)
        pd.testing.assert_frame_equal(got, expected, check_dtype=False,
                                      check_exact=False, rtol=1e-6,
                                      obj=f"cell from {name}")


def test_canonical_reads_agree_across_formats(three_formats):
    frame, sources = three_formats
    reads = {name: tabular.read_database(source, "cell", report=None,
                                         migrate=False)[0]
             for name, source in sources.items()}
    shared = sorted(set.intersection(*(set(f.columns) for f in reads.values())))
    assert set(frame.columns) <= set(shared)
    baseline = _comparable(reads["legacy"], shared)
    for name in ("store", "parquet"):
        pd.testing.assert_frame_equal(_comparable(reads[name], shared),
                                      baseline, check_dtype=False,
                                      check_exact=False, rtol=1e-6,
                                      obj=f"canonical cell from {name}")


def test_tables_and_columns_come_from_the_store_for_every_format(
        three_formats):
    frame, sources = three_formats
    for name, source in sources.items():
        assert tabular.database_tables(source) == ("cell",), name
        columns = tabular.table_columns(source, table="cell",
                                        canonicalise=False)
        assert set(frame.columns) <= set(columns), name


def test_limit_and_chunks_are_honoured_by_every_format(three_formats):
    _frame, sources = three_formats
    for name, source in sources.items():
        (head,) = tabular.read_database(source, "cell", report=None,
                                        migrate=False, limit=3, chunksize=2)
        assert len(head) == 3, name
        (empty,) = tabular.read_database(source, "cell", report=None,
                                         migrate=False, limit=0)
        assert empty.empty and "cell_area" in empty.columns, name
        chunks = list(tabular._iter_chunks(source, "cell", chunksize=5))
        assert sum(len(c) for c in chunks) == 12, name


def test_project_folder_locators_open_the_store(tmp_path):
    frame = _cells()
    root = str(tmp_path / "run")
    project = _store_project(root, frame)
    for locator in (root, project, os.path.join(project, store.CATALOG)):
        read = tabular.read_table(locator, table="cell", canonicalise=False,
                                  report=None)
        assert len(read) == len(frame), locator
        assert tabular.database_tables(locator) == ("cell",), locator


def test_a_legacy_project_folder_opens_its_measurements_db(tmp_path):
    frame = _cells()
    root = str(tmp_path / "old")
    _legacy(root, frame)
    read = tabular.read_table(root, table="cell", canonicalise=False,
                              report=None)
    pd.testing.assert_frame_equal(_comparable(read, list(frame.columns)),
                                  _comparable(frame, list(frame.columns)),
                                  check_dtype=False)


def test_a_missing_table_names_itself_in_every_format(three_formats):
    _frame, sources = three_formats
    for source in sources.values():
        with pytest.raises(ValueError, match="Table not found in database: "
                                             "nucleus"):
            tabular.read_database(source, "nucleus", report=None,
                                  migrate=False)


def test_a_missing_sqlite_file_reads_as_empty_without_creating_it(tmp_path):
    db = str(tmp_path / "absent.db")
    assert tabular.database_tables(db) == ()
    with pytest.raises(ValueError, match="Table not found"):
        tabular.read_database(db, "cell", report=None, migrate=False)
    assert not os.path.exists(db)


def test_legacy_sqlite_keeps_sqlite_value_types_and_skips_duckdb(tmp_path):
    db = str(tmp_path / "mixed.db")
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE annotations (png_path TEXT, annotate)")
        conn.executemany("INSERT INTO annotations VALUES (?, ?)",
                         [("a.png", 1), ("b.png", "x"), ("c.png", None)])
    (read,) = tabular.read_database(db, "annotations", canonicalise=False,
                                    report=None, migrate=False)
    assert read["annotate"].tolist()[:2] == [1, "x"]
    with store.open(db) as opened:
        list(opened._frames("annotations"))
        assert opened._duck is None
        assert list(opened._table_columns("annotations")["name"]) == [
            "png_path", "annotate"]
        counted = opened.sql("SELECT COUNT(*) AS n FROM annotations")
        assert int(counted["n"].iloc[0]) == 3
        assert opened._duck is not None


def test_store_frames_refuse_an_unknown_table(tmp_path):
    project = _store_project(str(tmp_path / "s"), _cells())
    with store.open(project) as opened:
        with pytest.raises(store.StoreError, match="no table"):
            list(opened._frames("nucleus"))
        described = opened._table_columns("cell")
        assert {"name", "type"} <= set(described.columns)
        assert "cell_area" in set(described["name"])


def test_query_store_reads_parquet_and_duckdb_through_the_store(tmp_path):
    frame = _cells()
    folder = _parquet_folder(str(tmp_path / "p"), frame)
    duck = str(tmp_path / "d" / "measurements.duckdb")
    tabular.write_database(frame, duck, "cell", if_exists="replace",
                           canonicalise=False)
    for source in (folder, duck):
        got = tabular._query_store(source, "SELECT COUNT(*) AS n FROM cell",
                                   report=None)
        assert int(got["n"].iloc[0]) == len(frame), source
        assert tabular.database_tables(source) == ("cell",)
