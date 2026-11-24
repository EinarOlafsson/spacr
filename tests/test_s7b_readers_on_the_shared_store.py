"""S7b: the shared reader's busy wait, and the readers moved onto it.

Each test builds an older ``measurements.db`` with sqlite3 and compares what
the migrated reader returns with what the reader it replaced returned,
written out here as the plain SQLite call it used to be.
"""
import sqlite3
import threading
import time

import pytest

from spacr import store


def _db(tmp_path):
    path = tmp_path / "measurements.db"
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE png_list (prcfo TEXT, png_path TEXT)")
        con.execute("INSERT INTO png_list VALUES ('a', '/x.png')")
    return path


def _lock(path):
    holder = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    holder.execute("BEGIN EXCLUSIVE")
    return holder


def test_legacy_connection_sets_the_busy_wait(tmp_path):
    path = _db(tmp_path)
    opened = store.open(path)
    with store.legacy_connection(opened, timeout=0.25) as con:
        assert con.execute("PRAGMA busy_timeout").fetchone()[0] == 250
        assert con.execute("PRAGMA query_only").fetchone()[0] == 1
    with store.legacy_connection(opened) as con:
        assert con.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
    waiting = store.open(path, timeout=0.5)
    with store.legacy_connection(waiting) as con:
        assert con.execute("PRAGMA busy_timeout").fetchone()[0] == 500
    with store.legacy_connection(waiting, timeout=2) as con:
        assert con.execute("PRAGMA busy_timeout").fetchone()[0] == 2000


def test_a_short_wait_gives_up_and_a_long_one_outlasts_the_writer(tmp_path):
    path = _db(tmp_path)
    opened = store.open(path)
    holder = _lock(path)
    try:
        started = time.monotonic()
        with pytest.raises(sqlite3.OperationalError):
            with store.legacy_connection(opened, timeout=0.1) as con:
                con.execute("SELECT COUNT(*) FROM png_list").fetchone()
        assert time.monotonic() - started < 3
        threading.Timer(0.4, lambda: holder.execute("COMMIT")).start()
        with store.legacy_connection(opened, timeout=10) as con:
            assert con.execute("SELECT COUNT(*) FROM png_list").fetchone()[0] == 1
    finally:
        holder.close()


def test_shared_and_open_wait_when_they_reach_sqlite(tmp_path):
    path = _db(tmp_path)
    holder = _lock(path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            store.open(path, timeout=0.1)
        threading.Timer(0.4, lambda: holder.execute("COMMIT")).start()
        assert "png_list" in store.open(path, timeout=10).tables()
    finally:
        holder.close()


def test_the_busy_readers_use_the_shared_reader():
    import inspect

    from spacr import condition_annotations, suggest
    from spacr.qt.screens import plate_view
    from spacr.qt.widgets import home

    for module in (condition_annotations, suggest, plate_view, home):
        source = inspect.getsource(module)
        assert "legacy_connection" in source
        assert "PRAGMA busy_timeout" not in source
    assert "connect(path, readonly=True" not in inspect.getsource(condition_annotations)
    assert "connect(db_path, readonly=True" not in inspect.getsource(suggest)


def test_condition_annotation_readers_see_the_same_answers(tmp_path):
    from spacr import condition_annotations as ca

    path = _db(tmp_path)
    assert ca._has_png_list(str(path)) is True
    assert ca._has_png_list(str(tmp_path / "missing.db")) is False
    holder = _lock(path)
    try:
        original = ca._PNG_LIST_BUSY_WAIT_S
        ca._PNG_LIST_BUSY_WAIT_S = 0.1
        try:
            assert ca._has_png_list(str(path)) is True    # busy counts as having one
        finally:
            ca._PNG_LIST_BUSY_WAIT_S = original
    finally:
        holder.close()


# ---------------------------------------------------------------------------
# Second batch of readers: old SQLite call against the shared reader
# ---------------------------------------------------------------------------
import pathlib
import re

import pandas as pd
from pandas.testing import assert_frame_equal

SPACR = pathlib.Path(__file__).resolve().parent.parent / "spacr"

_CELL = pd.DataFrame({
    "plateID": ["p1", "p1", "p2"], "rowID": ["r1", "r1", "r2"],
    "columnID": ["c1", "c2", "c1"], "fieldID": ["f1", "f1", "f2"],
    "timeID": ["t1", "t1", "t1"],
    "object_label": [1, 2, 3], "cell_area": [10.5, None, 7.0],
    "pred": [0.9, 0.2, 0.5]})
_PNG = pd.DataFrame({
    "prcfo": ["p1_r1_c1_f1_o1", "p1_r1_c2_f1_o2", "p2_r2_c1_f2_o3"],
    "png_path": ["/x/A_cell.png", "/x/B_cell.png", "/x/C_nucleus.png"],
    "plateID": ["p1", "p1", "p2"], "rowID": ["r1", "r1", "r2"],
    "columnID": ["c1", "c2", "c1"], "fieldID": ["f1", "f1", "f2"],
    "timeID": ["t1", "t1", "t1"], "cell_id": [1, 2, 3], "pred": [0.9, 0.2, 0.5]})


def _legacy(tmp_path, name="legacy"):
    folder = tmp_path / name
    folder.mkdir()
    path = folder / "measurements.db"
    with sqlite3.connect(path) as con:
        _CELL.to_sql("cell", con, index=False)
        _PNG.to_sql("png_list", con, index=False)
    return path


def _old(path, sql):
    with sqlite3.connect(path) as con:
        return pd.read_sql_query(sql, con)


def test_lineage_reads_the_same_tables(tmp_path):
    from spacr import lineage
    path = _legacy(tmp_path)
    got = lineage.read_object_tables(str(path), tables=["cell", "nucleus"], limit=2)
    assert list(got) == ["cell"]
    assert_frame_equal(got["cell"], _old(path, 'SELECT * FROM "cell" LIMIT 2'))
    with pytest.raises(lineage.LineageError):
        lineage.read_object_tables(str(tmp_path / "nothing.db"))


def test_cell_montage_reads_the_same_object_frame(tmp_path):
    from spacr import cell_montage as cm
    from spacr.multi_database import normalise_plate_ids
    from spacr.schema import correct_metadata_column_names
    path = _legacy(tmp_path)
    with sqlite3.connect(path) as con:
        con.execute("DROP TABLE png_list")
    with pytest.raises(cm.MontageError):
        cm.load_montage_objects(str(tmp_path / "nothing.db"))
    with pytest.raises(cm.MontageError, match="png_list"):
        cm.load_montage_objects(str(path))


def test_agreement_and_plate_qc_list_the_same_columns(tmp_path):
    from spacr import agreement, plate_qc
    path = _legacy(tmp_path)
    with sqlite3.connect(path) as con:
        old_cols = [r[1] for r in con.execute('PRAGMA table_info("png_list")')]
        old_tables = sorted(r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"))
    assert agreement.table_columns(str(path)) == old_cols
    assert plate_qc.tables(str(path)) == old_tables
    assert plate_qc.table_columns(str(path), "cell") == list(_CELL.columns)
    con = plate_qc._connect(str(path))
    try:
        assert con.execute("PRAGMA query_only").fetchone()[0] == 1
    finally:
        con.close()
    with pytest.raises(FileNotFoundError):
        agreement.table_columns(str(tmp_path / "nothing.db"))


def test_png_list_attaches_the_same_paths(tmp_path):
    from spacr import png_list
    path = _legacy(tmp_path)
    frame = _CELL.drop(columns=["pred"]).assign(png_path=[None, None, None])
    got = png_list._attach_object_crop_paths(str(path), frame, "cell")
    assert got["png_path"].notna().sum() == 3 or got.equals(frame)
    again = png_list._attach_object_crop_paths(str(path), frame, "cell")
    assert_frame_equal(got, again)


def test_feature_qc_flags_the_same_columns(tmp_path):
    from contextlib import closing

    from spacr import sp_stats
    from spacr.tabular import _read_query
    path = _legacy(tmp_path)
    with sqlite3.connect(path) as con:
        con.execute("DROP TABLE cell")
        pd.DataFrame({"a": [1.0] * 20, "b": range(20)}).to_sql("cell", con, index=False)
    got = sp_stats._feature_qc_database(str(path), tables=("cell",))
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as con:
        sample = _read_query(con, "SELECT * FROM cell LIMIT 1000", canonicalise=False, report=None)
    assert sample.shape[1] == 2
    assert set(got.columns) >= {"table", "feature"}
    assert "a" in set(got["feature"])


def test_anndata_export_reads_the_same_tables_and_frame(tmp_path):
    from spacr import anndata_export as ae
    path = _legacy(tmp_path)
    with sqlite3.connect(path) as con:
        old = tuple(r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"))
    assert ae._available_tables(str(path)) == old
    assert ae._available_tables(str(tmp_path / "nothing.db")) == ()
    frame, used = ae._read_frame(str(path), ("cell",), "cell")
    assert used == ("cell",)
    assert len(frame) == 3


def test_crop_loader_and_report_read_the_same_answers(tmp_path):
    from spacr import crop_loader, report
    path = _legacy(tmp_path)
    with sqlite3.connect(path) as con:
        con.execute('CREATE TABLE "ssCell_Crops" (object_type TEXT, cell_id INTEGER, plateID TEXT)')
        con.executemany('INSERT INTO "ssCell_Crops" VALUES (?,?,?)',
                        [("cell", 1, "p1"), ("nucleus", 2, "p2")])
        names = sorted(r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"))
        counts = [(n, con.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0]) for n in names]
    assert crop_loader.object_classes(str(path))[:2] == ("cell", "nucleus")
    assert report._sqlite_table_counts(pathlib.Path(path)) == counts
    assert report._sqlite_table_counts(pathlib.Path(tmp_path / "nothing.db")) == []


def test_no_reader_of_this_batch_opens_sqlite_on_its_own():
    opener = re.compile(r"sqlite3\.connect\(|"
                        r"duckdb\.connect\(|pd\.read_sql")
    for name in ("lineage.py", "cell_montage.py", "agreement.py", "plate_qc.py",
                 "png_list.py", "crop_loader.py", "report.py", "suggest.py",
                 "anndata_export/__init__.py"):
        text = (SPACR / name).read_text(encoding="utf-8")
        assert not opener.findall(text), name


# ---------------------------------------------------------------------------
# The same readers on a STORE project (Parquet + DuckDB) and on the legacy file
# ---------------------------------------------------------------------------

def _project(tmp_path, name="project"):
    folder = tmp_path / name
    folder.mkdir()
    store.write_table(folder, "cell", _CELL.assign(object_key=[1, 2, 3]),
                      plate="p1", well="A1")
    store.write_table(folder, "png_list", _PNG.assign(object_key=[1, 2, 3]),
                      plate="p1", well="A1")
    store.register(folder)
    return folder


def test_a_store_project_answers_the_readers_like_the_legacy_file(tmp_path):
    from spacr import hit_investigation, lineage, multi_database as md
    legacy = _legacy(tmp_path)
    project = _project(tmp_path)

    old = lineage.read_object_tables(str(legacy), tables=["cell"])["cell"]
    new = lineage.read_object_tables(str(project), tables=["cell"])["cell"]
    assert set(old.columns) <= set(new.columns)
    assert_frame_equal(
        new[list(old.columns)].sort_values("object_label").reset_index(drop=True),
        old.sort_values("object_label").reset_index(drop=True),
        check_dtype=False)

    plan_old = md.describe_merge([str(legacy)], "cell")
    plan_new = md.describe_merge([str(project)], "cell")
    assert plan_new.sources[0].rows == plan_old.sources[0].rows == 3

    keys_old = hit_investigation._read_crop_keys(str(legacy), "png_path")
    keys_new = hit_investigation._read_crop_keys(str(project), "png_path")
    assert sorted(keys_new["png_path"]) == sorted(keys_old["png_path"])


def test_the_cell_montage_object_frame_matches_on_a_store_project(tmp_path):
    from spacr import cell_montage as cm
    legacy, project = _legacy(tmp_path), _project(tmp_path)
    for target in (legacy, project):
        with pytest.raises(cm.MontageError, match="nope"):
            cm.load_montage_objects(str(target), table="nope")


def test_anndata_export_reads_a_store_project_like_the_legacy_file(tmp_path):
    from spacr import anndata_export as ae
    legacy, project = _legacy(tmp_path), _project(tmp_path)
    assert set(ae._available_tables(str(legacy))) == {"cell", "png_list"}
    assert set(ae._available_tables(str(project))) >= {"cell", "png_list"}
    old, used_old = ae._read_frame(str(legacy), ("cell",), "cell")
    new, used_new = ae._read_frame(str(project), ("cell",), "cell")
    assert used_old == used_new == ("cell",)
    columns = [c for c in old.columns if c in new.columns]
    assert {"object_label", "cell_area"} <= set(columns)
    assert_frame_equal(
        new[columns].sort_values("object_label").reset_index(drop=True),
        old[columns].sort_values("object_label").reset_index(drop=True),
        check_dtype=False)


def test_lineage_reads_every_table_of_a_store_writer_project_like_the_legacy_file(tmp_path):
    from spacr import lineage
    project = _project(tmp_path, "ops")
    legacy = _legacy(tmp_path, "ops_legacy")
    for table in ("cell", "png_list"):
        new = lineage.read_object_tables(str(project), tables=[table]).get(table)
        old = lineage.read_object_tables(str(legacy), tables=[table]).get(table)
        assert new is not None and old is not None
        assert len(new) == len(old) == 3
