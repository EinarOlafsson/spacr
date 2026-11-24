"""S6 group 1: the Graph Builder, Image Scatter, Plate View watch and column
picker read measurement projects only through ``spacr.store``.

Each reader is checked against the frame the pre-S6 code produced from the
same fixture -- for an older measurements.db that is ``pandas.read_sql_query``
over a read-only SQLite connection, exactly what the readers ran before; for
a spaCR store project it is the frame that was written into the store.
"""
from __future__ import annotations

import ast
import json
import pathlib
import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr import store  # noqa: E402

MODULES = [
    "spacr/qt/widgets/column_picker.py",
    "spacr/qt/screens/graph_builder.py",
    "spacr/qt/screens/image_scatter.py",
    "spacr/qt/screens/plate_view.py",
]

ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _fresh_shared_stores():
    """No shared store outlives a test, so tmp paths never alias."""
    store.release_shared()
    yield
    store.release_shared()


def _cell_frame(n=12):
    """A cell table with every kind of value a measurement table holds."""
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "object_key": np.arange(n, dtype="int64"),
        "plate": "plate1",
        "well": ["A01" if i < n // 2 else "A02" for i in range(n)],
        "rowID": ["r1"] * n,
        "cell_area": rng.normal(100, 10, n).round(3),
        "cell_count": np.arange(n, dtype="int64") * 3,
        "label": [f"l{i}" for i in range(n)],
    })


@pytest.fixture
def legacy_db(tmp_path):
    """An older measurements.db, with an untyped mixed column and a view."""
    path = tmp_path / "run" / "measurements" / "measurements.db"
    path.parent.mkdir(parents=True)
    frame = _cell_frame()
    with sqlite3.connect(path) as con:
        frame.to_sql("cell", con, index=False)
        con.execute("CREATE TABLE png_list (png_path TEXT, annotate, score REAL)")
        con.executemany("INSERT INTO png_list VALUES (?, ?, ?)",
                        [("a.png", 1, 0.5), ("b.png", "x", None),
                         ("c.png", None, 2.0)])
        con.execute("CREATE TABLE zebra (a REAL)")
        con.execute("INSERT INTO zebra VALUES (1.5)")
        con.execute("CREATE VIEW big_cells AS SELECT * FROM cell "
                    "WHERE cell_area > 100")
    con.close()
    return path


def _old_read(path, sql):
    """What the pre-S6 readers ran: ``read_sql_query`` on a ``mode=ro`` file."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(sql, con)
    finally:
        con.close()


@pytest.fixture
def store_project(tmp_path):
    """A spaCR store project holding a cell table over two wells."""
    project = tmp_path / "proj"
    frame = _cell_frame()
    for well, part in frame.groupby("well"):
        store.write_table(project, "cell", part.reset_index(drop=True),
                          plate="plate1", well=well, run_id="t",
                          sort_by=("object_key",))
    store.register(project)
    return project, frame


def _same_rows(got, expected):
    """Compare a store read with the written frame, order-free on columns."""
    got = got.sort_values("object_key").reset_index(drop=True)
    expected = expected.sort_values("object_key").reset_index(drop=True)
    assert sorted(got.columns) == sorted(expected.columns)
    pd.testing.assert_frame_equal(got[list(expected.columns)], expected,
                                  check_dtype=False)


def test_the_migrated_modules_open_no_engine_themselves():
    """No sqlite3, duckdb, read_parquet or read_sql in the four modules."""
    for rel in MODULES:
        text = (ROOT / rel).read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(n.split(".")[0] in ("sqlite3", "duckdb")
                           for n in names), rel
        assert "read_parquet" not in text, rel
        assert "pd.read_sql" not in text, rel


def test_graph_builder_lists_tables_of_a_legacy_db_as_before(legacy_db):
    from spacr.qt.screens.graph_builder import table_names
    assert table_names(str(legacy_db)) == ["cell", "png_list", "zebra"]


def test_graph_builder_reads_a_legacy_table_exactly_as_before(legacy_db):
    from spacr.qt.screens.graph_builder import read_table
    for table in ("cell", "png_list", "zebra"):
        expected = _old_read(legacy_db, f'SELECT * FROM "{table}"')
        pd.testing.assert_frame_equal(read_table(str(legacy_db), table),
                                      expected)
    limited = read_table(str(legacy_db), "cell", limit=5)
    pd.testing.assert_frame_equal(
        limited, _old_read(legacy_db, 'SELECT * FROM "cell" LIMIT 5'))
    pd.testing.assert_frame_equal(
        read_table(str(legacy_db)), _old_read(legacy_db, 'SELECT * FROM "cell"'))


def test_graph_builder_reads_a_store_project(store_project):
    from spacr.qt.screens.graph_builder import read_table, table_names
    project, frame = store_project
    assert table_names(str(project)) == ["cell"]
    _same_rows(read_table(str(project), "cell"), frame)
    _same_rows(read_table(str(project)), frame)
    assert len(read_table(str(project), "cell", limit=4)) == 4
    _same_rows(read_table(str(project / "spacr.duckdb"), "cell"), frame)


def test_graph_builder_screen_loads_both_kinds(qtbot, legacy_db, store_project):
    from spacr.qt.screens.graph_builder import GraphBuilderScreen
    project, frame = store_project
    for path, expected, names in (
            (str(legacy_db), _old_read(legacy_db, 'SELECT * FROM "cell"'),
             ["cell", "png_list", "zebra"]),
            (str(project), frame, ["cell"])):
        screen = GraphBuilderScreen()
        qtbot.addWidget(screen)
        screen.load_path(path)
        qtbot.waitUntil(lambda s=screen: not s.is_busy(), timeout=20000)
        assert screen._frame is not None, screen._source.text()
        listed = [screen._table_picker.itemText(i)
                  for i in range(screen._table_picker.count())]
        assert listed == names
        _same_rows(screen._frame, expected)


def test_image_scatter_reads_a_legacy_db_as_before(legacy_db):
    from spacr.qt.screens.image_scatter import list_tables, load_scatter_frame
    from spacr.selection import with_object_type
    assert list_tables(str(legacy_db)) == ["cell", "png_list", "zebra"]
    expected = with_object_type(
        _old_read(legacy_db, 'SELECT * FROM "cell" LIMIT 7'), "cell")
    pd.testing.assert_frame_equal(
        load_scatter_frame(str(legacy_db), "cell", limit=7), expected)
    assert list_tables(str(legacy_db.parent / "missing.db")) == []
    with pytest.raises(FileNotFoundError, match="no measurements database"):
        load_scatter_frame(str(legacy_db.parent / "missing.db"), "cell")


def test_image_scatter_reads_a_store_project(store_project):
    from spacr.qt.screens.image_scatter import list_tables, load_scatter_frame
    from spacr.selection import with_object_type
    project, frame = store_project
    assert list_tables(str(project)) == ["cell"]
    got = load_scatter_frame(str(project), "cell")
    _same_rows(got, with_object_type(frame.copy(), "cell"))
    assert len(load_scatter_frame(str(project), "cell", limit=3)) == 3


def _watch(tmp_path, keys):
    """A watch folder whose ledger marks ``keys`` done."""
    work = tmp_path / "src" / "spacr_watch"
    work.mkdir(parents=True)
    (work / "watch_ledger.json").write_text(json.dumps(
        {"fields": {key: {"status": "done"} for key in keys}}),
        encoding="utf-8")
    return work


KEYS = ["plate1_A01_1", "plate1_A01_2", "plate1_B03_1"]


def _card(qtbot, tmp_path):
    from spacr.qt.screens import plate_view as pv
    card = pv._WatchLivePlate()
    qtbot.addWidget(card)
    card._source = str(tmp_path / "src")
    card._pipeline = "mask_measure"
    return card


def test_plate_view_watch_reads_a_legacy_db_as_before(qtbot, tmp_path):
    work = _watch(tmp_path, KEYS)
    database = work / "measurements" / "measurements.db"
    database.parent.mkdir()
    with sqlite3.connect(database) as con:
        con.execute("CREATE TABLE spacr_watch_fields "
                    "(field TEXT PRIMARY KEY, merged_at REAL)")
        con.executemany("INSERT INTO spacr_watch_fields VALUES (?, ?)",
                        [(KEYS[0], 1.0), (KEYS[2], 2.0)])
    con.close()
    card = _card(qtbot, tmp_path)
    card.refresh()
    assert dict(card._wells) == {("plate1", 1, 1): 1, ("plate1", 2, 3): 1}


def test_plate_view_watch_without_a_database_shows_nothing(qtbot, tmp_path):
    _watch(tmp_path, KEYS)
    card = _card(qtbot, tmp_path)
    card.refresh()
    assert dict(card._wells) == {}


def test_plate_view_watch_keeps_its_plate_when_the_db_cannot_be_read(
        qtbot, tmp_path):
    work = _watch(tmp_path, KEYS)
    database = work / "measurements" / "measurements.db"
    database.parent.mkdir()
    database.write_bytes(b"this is not a database at all, not even close....")
    card = _card(qtbot, tmp_path)
    card._wells = {"kept": 1}
    card.refresh()
    assert card._wells == {"kept": 1}


def test_plate_view_watch_reads_a_store_project(qtbot, tmp_path):
    from spacr.qt.screens import plate_view as pv
    work = _watch(tmp_path, KEYS)
    project = work / "measurements"
    fields = pd.DataFrame({"field": [KEYS[1], KEYS[2]],
                           "merged_at": [1.0, 2.0]})
    store.write_table(project, "spacr_watch_fields", fields, plate="plate1",
                      well="A01", run_id="t", sort_by=("field",))
    store.register(project)
    assert sorted(pv._committed_watch_fields(str(project))) == KEYS[1:]
    card = _card(qtbot, tmp_path)
    card.refresh()
    assert dict(card._wells) == {("plate1", 1, 1): 1, ("plate1", 2, 3): 1}


def test_column_picker_reads_a_legacy_schema_as_before(legacy_db):
    from spacr.qt.widgets.column_picker import read_schema
    payload = read_schema(str(legacy_db.parent.parent), preferred="png_list")
    assert payload["error"] == ""
    assert payload["tables"] == ["big_cells", "cell", "png_list", "zebra"]
    con = sqlite3.connect(f"file:{legacy_db}?mode=ro", uri=True)
    try:
        declared = [(r[1], r[2] or "") for r in
                    con.execute('PRAGMA table_info("png_list")')]
        estimate = con.execute('SELECT max(_rowid_) FROM "png_list"').fetchone()[0]
    finally:
        con.close()
    assert payload["columns"] == declared
    assert payload["rows"] == estimate == 3
    reader = payload["reader"]
    assert reader.count_non_null("png_list", "annotate") == 2
    assert len(reader.executed_sql()) == 5


def test_column_picker_reads_a_store_schema(qtbot, store_project):
    from spacr.qt.widgets.column_picker import ColumnPickerDialog, read_schema
    project, frame = store_project
    payload = read_schema(str(project), preferred="cell")
    assert payload["error"] == "" and payload["schema_error"] == ""
    assert payload["tables"] == ["cell"]
    expected = store.shared(project).columns("cell")
    assert payload["columns"] == list(zip(expected["name"],
                                          expected["stored_type"]))
    assert sorted(c for c, _t in payload["columns"]) == sorted(frame.columns)
    assert payload["rows"] == len(frame)
    reader = payload["reader"]
    assert reader.path == str(project)
    assert reader.count_non_null("cell", "label") == len(frame)
    joined = " ".join(reader.executed_sql()).upper()
    assert joined.count("SELECT COUNT(") == 1
    dialog = ColumnPickerDialog(db_path=str(project.parent / "proj"),
                                table="cell")
    qtbot.addWidget(dialog)
    assert dialog.table_names() == ["cell"]
    assert sorted(dialog.column_names()) == sorted(frame.columns)
    assert dialog.banner_text() == ""


def test_column_picker_reports_a_file_that_is_not_a_database(tmp_path):
    from spacr.qt.widgets.column_picker import open_reader
    bad = tmp_path / "measurements.db"
    bad.write_bytes(b"not a database, plainly not one, really not." * 4)
    reader, message = open_reader(str(bad))
    assert reader is None
    assert "is not a SQLite database" in message
