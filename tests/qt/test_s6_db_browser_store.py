"""The Database Browser opens spaCR store projects (unified-store step S6).

A store project (``spacr.duckdb`` plus registered Parquet tables) is read
through :class:`spacr.store_browse.StoreDb`, which pages with
:meth:`spacr.store.Store.page` (keyset), counts from the registry, sorts and
filters in DuckDB, reads only the visible columns of a wide table and
exports through :func:`spacr.store.export`. An older measurements.db opens
through the same handle, read-only. A 1M-row store answers its first page
in under 0.1 s.
"""
from __future__ import annotations

import os
import sqlite3
import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from spacr import power_ops, store
from spacr.qt.screens.db_browser import DbBrowserScreen, _export_format
from spacr.store_browse import StoreDb, bind_parameters

KEYS = ["plate", "well", "object_key", "timeID"]


def _screen_project(path, **shape):
    """A simulated OPS screen written into a store project and registered."""
    made = power_ops.write_ops_screen(power_ops.nominal_calibration(),
                                      str(path), {}, 3, **shape)
    store.register(str(path))
    return made


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """Two wells, 200 cells per table, about 20 features per table."""
    path = tmp_path_factory.mktemp("s6") / "measurements"
    _screen_project(path, plates=1, wells=2, fields=2, cells_per_field=50,
                    n_features=60)
    return str(path)


@pytest.fixture(scope="module")
def wide(tmp_path_factory):
    """One well, 100 features per table: wider than the Browser reads at once."""
    path = tmp_path_factory.mktemp("s6w") / "measurements"
    _screen_project(path, plates=1, wells=1, fields=1, cells_per_field=40,
                    n_features=300)
    return str(path)


@pytest.fixture
def legacy(tmp_path):
    """An older measurements.db with one cell table."""
    path = tmp_path / "run" / "measurements" / "measurements.db"
    path.parent.mkdir(parents=True)
    frame = pd.DataFrame({"prc": [f"p_{i}" for i in range(30)],
                          "cell_area": np.arange(30, dtype=float) + 0.5,
                          "gene": ["NT" if i % 3 else "g1" for i in range(30)]})
    con = sqlite3.connect(path)
    try:
        frame.to_sql("cell", con, index=False)
    finally:
        con.close()
    return str(path), frame


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, qt_theme_applied, tmp_path):
    """Preferences in a temp folder; no run journal; a fresh shared store."""
    from PySide6.QtCore import QSettings

    from spacr import run_journal

    for fmt in (QSettings.NativeFormat, QSettings.IniFormat):
        QSettings.setPath(fmt, QSettings.UserScope, str(tmp_path / "qs"))
    QSettings("spacr", "qt").clear()
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(run_journal, "runs_root", lambda: root)
    store.release_shared()
    yield
    store.release_shared()


@pytest.fixture
def screen(qtbot):
    """A synchronous Browser."""
    widget = DbBrowserScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


def _everything(project, table, columns):
    """The whole table in the Browser's unsorted order, straight from the store."""
    frame = store.open(project).query(table=table)
    return frame.sort_values(KEYS, kind="stable")[columns].reset_index(
        drop=True)


def _load_all(screen):
    """Fetch every chunk."""
    while screen.fetch_more():
        pass


def test_the_browser_opens_a_store_project_and_counts_from_the_registry(
        screen, project):
    assert screen.set_database(os.path.dirname(project)) is True
    assert screen.database_path() == project
    assert screen.tables() == ["cell", "cytoplasm", "nucleus"]
    assert screen.current_table() == "cell"
    assert screen.row_count() == 200
    assert screen.row_count_is_estimate() is False
    assert "200 rows" in screen.status_text()
    assert screen.set_database(os.path.join(project, store.CATALOG)) is True
    assert screen.database_path() == project


def test_paging_covers_every_row_once_in_key_order(screen, project):
    screen.set_database(project)
    screen._page_size_box.setValue(25)
    _load_all(screen)
    assert screen.is_fully_loaded()
    columns = screen.preview_columns()
    expected = _everything(project, "cell", columns)
    got = pd.DataFrame(screen.preview_rows(), columns=columns)
    assert len(got) == 200
    pd.testing.assert_frame_equal(got, expected, check_dtype=False)
    keys = [screen._model.row_key(i) for i in range(200)]
    assert len(set(keys)) == 200 and keys == sorted(keys)


def test_a_header_sort_orders_the_whole_table_in_duckdb(screen, project):
    screen.set_database(project)
    screen._page_size_box.setValue(30)
    section = screen.visible_columns().index("cell_area")
    screen._on_header_clicked(section)
    _load_all(screen)
    got = [row[section] for row in screen.preview_rows()]
    truth = store.open(project).query(table="cell")["cell_area"]
    assert got == sorted(truth.tolist(), reverse=True)
    assert "left out" in screen._sort_note.text()
    screen._on_header_clicked(section)
    _load_all(screen)
    assert [row[section] for row in screen.preview_rows()] == sorted(
        truth.tolist())


def test_filters_run_in_duckdb_and_count_exactly(screen, project):
    screen.set_database(project)
    frame = store.open(project).query(table="cell")
    assert screen.set_filter("gene", "=", "nontargeting") is True
    assert screen.row_count() == int((frame["gene"] == "nontargeting").sum())
    gene = screen.visible_columns().index("gene")
    assert {row[gene] for row in screen.preview_rows()} == {"nontargeting"}
    assert screen.set_filter("gene", "contains", "NONTARG") is True
    assert screen.row_count() == int((frame["gene"] == "nontargeting").sum())
    assert screen.set_raw_filter("cell_area > 1000") is True
    assert screen.row_count() == int((frame["cell_area"] > 1000).sum())
    assert screen.set_raw_filter(
        "cell_area > (SELECT count(*) FROM read_csv('/etc/hosts'))") is False
    assert "not allowed" in screen.status_text()


def test_distinct_values_come_from_the_key_table(project):
    db = StoreDb(project)
    truth = sorted(store.open(project).query(table="cell")["gene"].unique())
    assert db.distinct("cell", "gene") == truth
    assert db.distinct("cell", "well") == ["A01", "A02"]


def test_pages_are_cached_and_a_changed_project_is_read_again(project):
    db = StoreDb(project)
    first = db.chunk("cell", columns=["gene"], limit=10)
    assert db.chunk("cell", columns=["gene"], limit=10) == first
    assert db.last_sql.startswith("cache:")
    assert db.sync() is False


def test_bound_values_cannot_leave_their_literal():
    text = bind_parameters("gene = ? AND cell_area > ?", ("x' OR 1=1 --", 2.5))
    assert text == "gene = 'x'' OR 1=1 --' AND cell_area > 2.5"
    assert bind_parameters("a LIKE ? ESCAPE '\\'", ("%q%",)) == (
        "a ILIKE '%q%' ESCAPE '\\'")
    with pytest.raises(ValueError):
        bind_parameters("a = ?", ())


def test_a_wide_table_reads_only_the_visible_columns(screen, wide):
    screen.set_database(wide)
    db = screen._db
    total = len(db.columns("cell"))
    assert total > 60
    shown = screen.preview_columns()
    assert shown == db.default_columns("cell")
    assert f"of {total} columns" in screen._col_count_label.text()
    target = [c for c in db.columns("cell") if c not in shown][-1]
    assert target not in shown
    screen.set_column_filter(target)
    fetched = screen.preview_columns()
    assert target in fetched and len(fetched) < total
    assert screen.visible_columns() == [c for c in fetched if target in c]
    screen.set_column_filter("")
    assert screen.preview_columns() == db.default_columns("cell")


@pytest.mark.parametrize("suffix", [".csv", ".parquet", ".db"])
def test_an_export_round_trips(screen, project, tmp_path, suffix):
    screen.set_database(project)
    screen.set_filter("gene", "=", "nontargeting")
    columns = screen.visible_columns()
    target = tmp_path / f"out{suffix}"
    assert screen.export_rows(str(target), _export_format(str(target))) is True
    expected = _everything(project, "cell", columns)
    expected = expected[expected["gene"] == "nontargeting"].reset_index(
        drop=True)
    if suffix == ".csv":
        got = pd.read_csv(target)
    elif suffix == ".parquet":
        got = pd.read_parquet(target)
    else:
        with sqlite3.connect(target) as con:
            got = pd.read_sql_query('SELECT * FROM "cell"', con)
    assert list(got.columns) == columns
    for column in columns:
        left, right = got[column], expected[column]
        if pd.api.types.is_float_dtype(right):
            np.testing.assert_allclose(left.astype(float), right, rtol=1e-6)
        else:
            assert left.astype(str).tolist() == right.astype(str).tolist()
    assert "Exported" in screen.status_text()


def test_a_sqlite_export_splits_a_table_wider_than_sqlite_allows(tmp_path):
    project = str(tmp_path / "wide")
    frame = pd.DataFrame({"object_key": np.arange(5, dtype=np.int64)})
    for channel in ("dapi", "gfp"):
        for i in range(1050):
            frame[f"cell_{channel}_f{i}"] = np.float32(i)
    store.write_table(project, "cell", frame, plate="p1", well="A01",
                      run_id="t", sort_by=("object_key",),
                      key_columns=("object_key",))
    store.register(project)
    done = store.export(project, tmp_path / "x.db", fmt="sqlite",
                        tables="cell")
    assert done["rows"] == 5
    with sqlite3.connect(tmp_path / "x.db") as con:
        split = con.execute("SELECT table_name, columns FROM "
                            "spacr_export_tables ORDER BY 1").fetchall()
        assert [name for name, _n in split] == ["cell__dapi", "cell__gfp"]
        assert all(n < 2000 for _name, n in split)
        joined = pd.read_sql_query(
            'SELECT a.object_key, a."cell_dapi_f7", b."cell_gfp_f1049" FROM '
            'cell__dapi a JOIN cell__gfp b USING (object_key, plate, well)',
            con)
    assert joined["cell_dapi_f7"].tolist() == [7.0] * 5
    assert joined["cell_gfp_f1049"].tolist() == [1049.0] * 5


def test_an_older_measurements_db_opens_read_only(screen, legacy, tmp_path):
    path, frame = legacy
    with open(path, "rb") as handle:
        before = handle.read()
    assert screen.set_database(os.path.dirname(os.path.dirname(path))) is True
    assert screen.database_path() == path
    _load_all(screen)
    got = pd.DataFrame(screen.preview_rows(), columns=screen.preview_columns())
    pd.testing.assert_frame_equal(got, frame)
    assert screen.set_filter("gene", "contains", "nt") is True
    assert screen.row_count() == int((frame["gene"] == "NT").sum())
    assert screen._db.row_key("cell") == ("rowid", ["_rowid_"])
    screen.export_rows(str(tmp_path / "l.csv"), "csv")
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "l.csv"),
        frame[frame["gene"] == "NT"].reset_index(drop=True))
    with open(path, "rb") as handle:
        assert handle.read() == before


def _allow_editing():
    """Turn the Preferences opt-in on."""
    from spacr.qt.preferences import set_db_browser_editable

    set_db_browser_editable(True)


def test_an_older_file_cannot_be_armed_for_edits(screen, legacy):
    _allow_editing()
    path, _frame = legacy
    screen.confirm_edit_mode = lambda _m: pytest.fail("must not be asked")
    screen.set_database(path)
    assert screen.enable_edit_mode() is False
    assert "read-only" in screen.status_text()
    assert screen.edit_cell(0, "cell_area", "1") is False


def test_a_store_project_waits_for_the_edit_log(screen, project, monkeypatch):
    _allow_editing()
    monkeypatch.delattr(store, "edit", raising=False)
    screen.confirm_edit_mode = lambda _m: pytest.fail("must not be asked")
    screen.set_database(project)
    assert screen._db.can_edit() is False
    assert screen.enable_edit_mode() is False
    assert "edit log" in screen.status_text()


def test_store_edits_are_written_as_edit_rows(screen, project, monkeypatch):
    _allow_editing()
    written = []

    def _edit(target, rows, *, kind, author, source, session=None):
        """Record what the Browser asked the edit log to write."""
        written.append((target, rows, kind, source))
        return len(rows)

    monkeypatch.setattr(store, "edit", _edit, raising=False)
    screen.confirm_edit_mode = lambda message: "edit log" in message
    screen.set_database(project)
    assert screen.enable_edit_mode() is True, screen.status_text()
    column = screen.visible_columns().index("cell_area")
    key = screen._model.row_key(0)
    assert screen.edit_cell(0, "cell_area", "12.5") is True, \
        screen.status_text()
    assert screen.preview_rows()[0][column] == 12.5
    (target, rows, kind, source), = written
    assert target == project and kind == "browser" and source == "browser"
    assert rows == [dict(zip(KEYS, key), table="cell", column="cell_area",
                         value=12.5, op="set")]


def test_a_million_row_store_shows_its_first_page_in_under_a_tenth_of_a_second(
        tmp_path_factory):
    path = str(tmp_path_factory.mktemp("s6m") / "measurements")
    _screen_project(path, plates=1, wells=8, fields=10, cells_per_field=12500,
                    n_features=30)
    db = StoreDb(path)
    assert db.registry_count("cell") == 1_000_000
    columns = db.fetch_columns("cell", "") or db.columns("cell")
    db.chunk("nucleus", columns=None, limit=100)
    best = []
    for _attempt in range(3):
        db._pages.clear()
        started = time.perf_counter()
        _cols, rows, keys = db.chunk("cell", columns=columns, limit=100)
        best.append(time.perf_counter() - started)
    assert len(rows) == 100
    assert min(best) < 0.1, best
    started = time.perf_counter()
    for _page in range(20):
        _cols, rows, keys = db.chunk("cell", columns=columns, limit=100,
                                     after=keys[-1])
    assert (time.perf_counter() - started) / 20 < 0.1


def test_store_edits_go_through_the_real_edit_log(screen, tmp_path):
    """The Browser's edit hook against S5's real ``spacr.store.edit``.

    The stand-in above records what the Browser asks for; here the edit row
    lands in the project's edit log and the latest value reads back.
    """
    _allow_editing()
    path = str(tmp_path / "measurements")
    _screen_project(path, plates=1, wells=1, fields=1, cells_per_field=5,
                    n_features=60)
    screen.confirm_edit_mode = lambda message: "edit log" in message
    assert screen.set_database(path) is True, screen.status_text()
    assert screen._db.can_edit() is True
    assert screen.enable_edit_mode() is True, screen.status_text()
    column = screen.visible_columns().index("cell_area")
    key = dict(zip(KEYS, screen._model.row_key(0)))
    assert screen.edit_cell(0, "cell_area", "12.5") is True, \
        screen.status_text()
    assert screen.preview_rows()[0][column] == 12.5
    with store.open(path) as opened:
        log = opened.edits(kind="browser", latest=False)
    assert len(log) == 1
    row = log.iloc[0]
    assert row["table"] == "cell" and row["column"] == "cell_area"
    assert row["source"] == "browser"
    assert str(key["object_key"]) in str(row.to_dict())
