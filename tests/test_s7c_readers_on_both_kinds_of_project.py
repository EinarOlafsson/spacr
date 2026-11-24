"""S7c: ``store.read_frame`` and the readers built on it, on both kinds of project.

Every test builds a spaCR store project (``store.write_table``) and the
equivalent older ``measurements.db``, runs one reader on each and checks that
the two answers are equal.
"""
import json
import os
import pathlib
import sqlite3
import threading
import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from spacr import store


@pytest.fixture(autouse=True)
def _fresh():
    store.release_shared()
    yield
    store.release_shared()


def _png_frame():
    n = 12
    return pd.DataFrame({
        "object_key": np.arange(n, dtype=np.int64),
        "plateID": ["p1"] * n,
        "rowID": ["r1"] * 6 + ["r2"] * 6,
        "columnID": ["c1", "c2", "c3"] * 4,
        "fieldID": ["f1"] * n,
        "cell_id": np.arange(1, n + 1, dtype=np.int64),
        "png_path": [f"/x/{i}.png" for i in range(n)],
        "ann_a": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        "ann_b": [1, 2, 2, 2, 1, 1, 1, 2, 1, 2, 2, 2],
        "ann_c": [None, 1, 1, 2, None, 2, 1, 1, 2, 2, 1, None],
        "area": np.linspace(1.0, 99.0, n),
        "area_flat": np.zeros(n),
    })


def _both(root, tables):
    """A store project and an older ``measurements.db`` holding ``tables``."""
    new = pathlib.Path(root) / "new" / "measurements"
    old = pathlib.Path(root) / "old" / "measurements"
    old.mkdir(parents=True)
    with sqlite3.connect(old / "measurements.db") as con:
        for name, frame in tables.items():
            frame.assign(plate="p1", well="A01").to_sql(name, con, index=False)
    for name, frame in tables.items():
        store.write_table(str(new), name, frame, plate="p1", well="A01",
                          run_id="t", sort_by=("object_key",),
                          key_columns=("object_key",))
    store.register(str(new))
    return str(new), str(old / "measurements.db")


@pytest.fixture
def both(tmp_path):
    return _both(tmp_path, {"png_list": _png_frame(), "cell": _png_frame()})


def _same(left, right):
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False)


def test_read_frame_gives_equal_frames_for_sql_and_table(both):
    new, old = both
    sql = "SELECT png_path, cell_id, area FROM png_list ORDER BY cell_id"
    _same(store.read_frame(new, sql), store.read_frame(old, sql))
    _same(store.read_frame(new, table="png_list", columns=["cell_id", "area"],
                           where={"cell_id": [2, 3, 5]}),
          store.read_frame(old, table="png_list", columns=["cell_id", "area"],
                           where={"cell_id": [2, 3, 5]}))
    for project in (new, old):
        got = store.read_frame(project, "SELECT cell_id FROM png_list "
                               "WHERE cell_id > ? ORDER BY cell_id", params=[10])
        assert got["cell_id"].tolist() == [11, 12]
        got = store.read_frame(project, table="png_list", columns=["cell_id"],
                               where="cell_id < ?", params=[3])
        assert sorted(got["cell_id"]) == [1, 2]


def test_read_frame_takes_an_open_store_and_refuses_bad_calls(both):
    new, old = both
    for project in (new, old):
        opened = store.shared(project)
        assert len(store.read_frame(opened, table="cell")) == 12
        with pytest.raises(store.StoreError):
            store.read_frame(opened)
        with pytest.raises(store.StoreError):
            store.read_frame(opened, "SELECT 1", table="cell")
        with pytest.raises(store.StoreError):
            store.read_frame(opened, table="cell", columns=["nope"])
        with pytest.raises(store.StoreError):
            store.read_frame(opened, table="nope")


def _lock(path):
    holder = sqlite3.connect(path, isolation_level=None,
                             check_same_thread=False)
    holder.execute("BEGIN EXCLUSIVE")
    return holder


def test_read_frame_passes_the_busy_timeout_to_an_older_file(both):
    _, old = both
    opened = store.open(old)
    holder = _lock(old)
    try:
        started = time.monotonic()
        with pytest.raises((sqlite3.OperationalError, pd.errors.DatabaseError)):
            store.read_frame(opened, "SELECT COUNT(*) FROM png_list", timeout=0.1)
        assert time.monotonic() - started < 3
        threading.Timer(0.4, lambda: holder.execute("COMMIT")).start()
        got = store.read_frame(opened, "SELECT COUNT(*) AS n FROM png_list",
                               timeout=10)
        assert int(got["n"][0]) == 12
    finally:
        holder.close()


def test_agreement_reads_both_kinds_of_project(both):
    from spacr import agreement

    new, old = both
    assert (agreement.table_columns(new, "png_list")
            == agreement.table_columns(old, "png_list"))
    assert (agreement.annotation_columns(new)
            == agreement.annotation_columns(old))
    assert "ann_a" in agreement.annotation_columns(new)
    columns = ["ann_a", "ann_b", "ann_c"]
    _same(agreement.load_annotations(new, columns),
          agreement.load_annotations(old, columns))
    with pytest.raises(ValueError):
        agreement.table_columns(new, "nope")
    with pytest.raises(FileNotFoundError):
        agreement.table_columns(os.path.join(new, "ghost"), "png_list")


def test_plate_qc_reads_both_kinds_of_project(both):
    from spacr import plate_qc

    new, old = both
    assert plate_qc.tables(new) == plate_qc.tables(old)
    assert plate_qc.table_columns(new, "cell") == plate_qc.table_columns(old, "cell")
    assert (plate_qc.numeric_columns(new, "cell")
            == plate_qc.numeric_columns(old, "cell"))
    assert "area" in plate_qc.numeric_columns(new, "cell")
    _same(plate_qc.load_plate_frame(new, "cell", "area"),
          plate_qc.load_plate_frame(old, "cell", "area"))
    _same(plate_qc.load_plate_frame(new, "cell", "area", limit=4),
          plate_qc.load_plate_frame(old, "cell", "area", limit=4))
    with pytest.raises(ValueError):
        plate_qc.table_columns(new, "nope")


def test_png_list_attaches_crop_paths_on_both_kinds_of_project(tmp_path):
    from spacr import png_list

    crops = _png_frame().drop(columns=["ann_a", "ann_b", "ann_c"])
    crops["cell_id"] = crops["cell_id"].astype(float)
    new, old = _both(tmp_path, {"png_list": crops})
    rows = pd.DataFrame({
        "plateID": ["p1"] * 3, "rowID": ["r1"] * 3,
        "columnID": ["c1", "c2", "c3"], "fieldID": ["f1"] * 3,
        "object_label": [1, 2, 3]})
    got_new = png_list._attach_object_crop_paths(new, rows.copy(), "cell")
    got_old = png_list._attach_object_crop_paths(old, rows.copy(), "cell")
    _same(got_new, got_old)
    assert got_new["png_path"].tolist() == ["/x/0.png", "/x/1.png", "/x/2.png"]


def test_sp_stats_feature_qc_reads_both_kinds_of_project(both):
    from spacr import sp_stats

    new, old = both
    got_new = sp_stats._feature_qc_database(new, tables=("cell", "missing"))
    got_old = sp_stats._feature_qc_database(old, tables=("cell", "missing"))
    _same(got_new, got_old)
    assert "area_flat" in set(got_new["feature"])


def test_crop_loader_reads_both_kinds_of_project(both):
    from spacr import crop_loader

    new, old = both
    assert (crop_loader.object_classes(new, "png_list")
            == crop_loader.object_classes(old, "png_list") == ("cell",))
    assert crop_loader.plates(new) == crop_loader.plates(old) == ("p1",)
    conn_new, conn_old = (crop_loader._connect(p) for p in (new, old))
    assert (crop_loader._table_columns(conn_new, "png_list")
            == crop_loader._table_columns(conn_old, "png_list"))
    assert crop_loader._table_columns(conn_new, "nope") == ()
    query = lambda path: crop_loader.CropQuery(  # noqa: E731
        source="database", path=path, object_type="cell", plate="p1", limit=5)
    plan_new = crop_loader.plan_from_database(query(new))
    plan_old = crop_loader.plan_from_database(query(old))
    assert plan_new.matched == plan_old.matched == 12
    assert plan_new.selected == plan_old.selected == 5
    with pytest.raises(crop_loader.CropLoadError):
        crop_loader.plan_from_database(crop_loader.CropQuery(
            source="database", path=new, object_type="cell",
            where="no_such_column = 1"))


def test_report_counts_the_tables_of_both_kinds_of_project(both):
    from spacr import report

    new, old = both
    assert (report._sqlite_table_counts(pathlib.Path(new))
            == report._sqlite_table_counts(pathlib.Path(old))
            == [("cell", 12), ("png_list", 12)])


def test_the_notebook_cell_reads_both_kinds_of_project(both, tmp_path):
    from spacr import notebook_export

    new, old = both
    cell = notebook_export._output_cell_for("measure", tmp_path)
    assert "store.read_frame(" in cell and "_sql_frame" not in cell
    for src in (os.path.dirname(new), os.path.dirname(os.path.dirname(old))):
        scope = {"SETTINGS": {"src": src}}
        exec(cell, scope)
        assert len(scope["cell_df"]) == 12


def test_read_frame_is_in_the_release_contract():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                        "release_contracts",
                        "s7c_read_frame_api_arrivals_2026-11-24.json")
    with open(path, encoding="utf-8") as handle:
        recorded = json.load(handle)
    assert recorded["spacr.store.read_frame"].startswith(
        "Rows of a project as a DataFrame")
