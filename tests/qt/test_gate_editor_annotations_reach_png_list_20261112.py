"""Gate Editor annotations are saved to png_list, where Classify reads them.

The follow-up to issue #186: Graph Builder saves each annotation column to
``png_list`` of each plate's ``measurements.db`` by ``prcfo`` as it is made,
and Gate Editor shares that save. Its annotations used to stay in memory,
behind a "Save annotated table" action that writes a table Classify never
reads.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMessageBox

from spacr import condition_annotations
from spacr.condition_annotations import new_definition
from spacr.qt import annotate_engine
from spacr.qt.screens.gate_editor import GateEditorScreen
from spacr.qt.widgets import condition_annotation_dialog
from spacr.qt.widgets.gate_spec import GateSet, ThresholdGate

COLUMN = "cc2d1a_recruitment"


@pytest.fixture(autouse=True)
def calm_preferences():
    from spacr.qt import preferences
    preferences.set_reduce_motion(True)
    preferences.set_ambient_enabled(False)


def _plate(root, plate, objects=4, cropped=3):
    """One plate folder: a ``cell`` table, and crops for all but one object."""
    database = root / plate / "measurements" / "measurements.db"
    database.parent.mkdir(parents=True)
    columns = [f"c{1 + index % 2}" for index in range(objects)]
    cells = pd.DataFrame({
        "plateID": plate, "rowID": "r1", "columnID": columns, "fieldID": "f1",
        "object_label": range(1, objects + 1),
        "prcf": [f"{plate}_r1_{column}_f1" for column in columns],
        "cell_area": [float(10 * index) for index in range(objects)]})
    keys = [f"{prcf}_o{label}" for prcf, label in zip(cells.prcf, cells.object_label)]
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE cell (plateID TEXT, rowID TEXT, columnID TEXT, fieldID TEXT, "
                   "object_label INTEGER, prcf TEXT, cell_area REAL)")
        db.executemany("INSERT INTO cell VALUES (?, ?, ?, ?, ?, ?, ?)",
                       cells.itertuples(index=False, name=None))
        db.execute("CREATE TABLE png_list (png_path TEXT, prcfo TEXT, plateID TEXT, annotate INTEGER)")
        db.executemany("INSERT INTO png_list VALUES (?, ?, ?, ?)",
                       [(f"/crops/{key}.png", key, plate, 1) for key in keys[:cropped]])
    return database, cells


def _project(tmp_path):
    """Two plates and the merged table of both, as Measurement scan writes it."""
    first, cells_1 = _plate(tmp_path, "plate1")
    second, cells_2 = _plate(tmp_path, "plate2")
    merged = pd.concat([cells_1.assign(source_database="plate1"),
                        cells_2.assign(source_database="plate2")], ignore_index=True)
    folder = tmp_path / "results"
    folder.mkdir()
    csv = folder / "merged_measurements.csv"
    merged.to_csv(csv, index=False)
    return [first, second], csv


def _annotate(screen, monkeypatch, column=COLUMN, metadata="columnID", include="c1"):
    """Annotate the active gate through the screen's own Annotate… path.

    The selection dialog is answered with a rule recipe, as a user pressing
    Apply would; everything after it is the screen's code.
    """
    screen.gates.set_gates(GateSet().add(ThresholdGate(name="all", column="cell_area", low=-1.0)))
    screen.gates.tree.select("all")
    assert screen.gates.tree.active_gate() == "all"
    definition = new_definition(screen._annotation_base_frame, screen._condition_source,
                                column=column)
    definition["conditions"] = [dict(name="recruited", metadata_column=metadata,
                                     include=include, exclude="", manual_rows=[])]
    monkeypatch.setattr(condition_annotation_dialog, "_annotate_selection",
                        lambda *args: definition)
    screen._annotate_selection()
    return definition


def _png_list(database):
    with sqlite3.connect(database) as db:
        names = [row[1] for row in db.execute("PRAGMA table_info(png_list)")]
        rows = db.execute("SELECT * FROM png_list").fetchall()
    return pd.DataFrame(rows, columns=names)


def _screen(qtbot):
    screen = GateEditorScreen(threaded=False)
    qtbot.addWidget(screen)
    return screen


def test_a_save_control_is_on_the_screen_and_says_why_it_is_off(qtbot):
    screen = _screen(qtbot)
    button = screen._save_annotations_button
    assert button.text() == "Save annotations"
    assert button.parent() is screen and not button.isHidden()
    assert not button.isEnabled()
    assert "Annotate…" in button.toolTip() and "gate" in button.toolTip()
    assert screen._annotation_state.isHidden()


def test_merged_plates_annotate_save_reopen_and_validate(qtbot, tmp_path, monkeypatch):
    databases, csv = _project(tmp_path)
    screen = _screen(qtbot)
    screen.load_path(str(csv))
    _annotate(screen, monkeypatch)

    assert not screen._unsaved
    for database in databases:
        stored = _png_list(database).set_index("prcfo")
        assert COLUMN in stored.columns
        plate = database.parent.parent.name
        assert stored.loc[f"{plate}_r1_c1_f1_o1", COLUMN] == "recruited"
        assert stored.loc[f"{plate}_r1_c1_f1_o3", COLUMN] == "recruited"
        assert pd.isna(stored.loc[f"{plate}_r1_c2_f1_o2", COLUMN])
        assert stored["annotate"].eq(1).all()
    note = screen._annotation_state.text()
    assert "6 objects in 2 database(s)" in note
    assert "2 objects have no png_list row" in note
    assert not screen._save_annotations_button.isEnabled()
    assert "saved to png_list" in screen._save_annotations_button.toolTip()
    screen.close()

    reopened = _screen(qtbot)
    reopened.load_path(str(csv))
    assert reopened._condition_definition is not None
    assert reopened._frame[COLUMN].fillna("").tolist()[:4] == ["recruited", "", "recruited", ""]
    assert not reopened._unsaved
    reopened.load_path(str(databases[0]), table="png_list")
    assert COLUMN in reopened._frame.columns

    from spacr.settings import set_default_classify
    from spacr.validate import ERROR, validate_settings

    settings = set_default_classify(settings={})
    settings.update(classifier_family="ml", dataset_mode="annotation", classes={},
                    annotation_column=COLUMN,
                    src=[str(database.parent.parent) for database in databases])
    about_png_list = [problem for problem in validate_settings(settings, "classify_merged")
                      if problem.severity == ERROR and "png_list" in problem.message
                      and problem.setting == "annotation_column"]
    assert not about_png_list
    settings["annotation_column"] = "never_saved"
    about_png_list = [problem for problem in validate_settings(settings, "classify_merged")
                      if problem.severity == ERROR and problem.setting == "annotation_column"
                      and "is not a column of png_list" in problem.message]
    assert len(about_png_list) == 1


def test_a_single_plate_source_reopens_with_its_columns(qtbot, tmp_path, monkeypatch):
    database, _cells = _plate(tmp_path, "plate1")
    screen = _screen(qtbot)
    screen.load_path(str(database), table="cell")
    _annotate(screen, monkeypatch)
    assert not screen._unsaved
    screen.close()

    reopened = _screen(qtbot)
    reopened.load_path(str(database), table="cell")
    assert reopened._condition_definition is not None
    assert reopened._frame[COLUMN].fillna("").tolist() == ["recruited", "", "recruited", ""]


def test_databases_loaded_together_each_keep_their_own_labels(qtbot, tmp_path, monkeypatch):
    first, _cells = _plate(tmp_path / "one", "screen")
    second, _cells = _plate(tmp_path / "two", "screen_b")
    screen = _screen(qtbot)
    screen.load_paths([str(first), str(second)], table="cell")
    assert len(screen._paths) == 2
    _annotate(screen, monkeypatch, include="c1")

    assert not screen._unsaved
    for database in (first, second):
        plate = database.parent.parent.name
        stored = _png_list(database).set_index("prcfo")
        assert stored.loc[f"{plate}_r1_c1_f1_o1", COLUMN] == "recruited"
        assert pd.isna(stored.loc[f"{plate}_r1_c2_f1_o2", COLUMN])
    assert "in 2 database(s)" in screen._annotation_state.text()
    screen.close()

    reopened = _screen(qtbot)
    reopened.load_paths([str(first), str(second)], table="cell")
    assert reopened._condition_definition is not None
    assert reopened._frame[COLUMN].fillna("").tolist().count("recruited") == 4


def test_a_busy_database_is_reported_and_saved_on_retry(qtbot, tmp_path, monkeypatch):
    database, _cells = _plate(tmp_path, "plate1")
    screen = _screen(qtbot)
    screen.load_path(str(database), table="cell")
    _annotate(screen, monkeypatch)
    monkeypatch.setattr(condition_annotations, "_PNG_LIST_BUSY_WAIT_S", 0.05)
    monkeypatch.setattr(annotate_engine, "_SCHEMA_WRITE_TIMEOUT_S", 0.05)
    holder = sqlite3.connect(database, isolation_level=None)
    try:
        holder.execute("BEGIN IMMEDIATE")
        holder.execute("UPDATE png_list SET annotate = 2")
        definition = _annotate(screen, monkeypatch, column="second_label")
        assert screen._unsaved
        assert "in use" in screen._annotation_state.text()
        assert "second_label" in screen._annotation_state.text()
        assert screen._save_annotations_button.isEnabled()
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    screen.save_annotations()
    assert not screen._unsaved
    assert "second_label" in _png_list(database).columns
    assert definition["column"] == "second_label"


def test_closing_with_unsaved_annotations_asks_first(qtbot, tmp_path, monkeypatch):
    database, _cells = _plate(tmp_path, "plate1")
    screen = _screen(qtbot)
    screen.show()
    qtbot.waitExposed(screen)
    screen.load_path(str(database), table="cell")
    monkeypatch.setattr(condition_annotations, "_PNG_LIST_BUSY_WAIT_S", 0.05)
    monkeypatch.setattr(annotate_engine, "_SCHEMA_WRITE_TIMEOUT_S", 0.05)
    holder = sqlite3.connect(database, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    holder.execute("UPDATE png_list SET annotate = 2")
    _annotate(screen, monkeypatch)
    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *args, **kwargs: asked.append(args[2]) or QMessageBox.Cancel)
    assert screen.close() is False
    assert screen.isVisible() and COLUMN in asked[0]
    holder.execute("ROLLBACK")
    holder.close()
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Save)
    assert screen.close() is True
    assert COLUMN in _png_list(database).columns


def test_an_annotate_column_is_never_overwritten(qtbot, tmp_path, monkeypatch):
    database, _cells = _plate(tmp_path, "plate1")
    screen = _screen(qtbot)
    screen.load_path(str(database), table="cell")
    _annotate(screen, monkeypatch, column="annotate")
    assert screen._unsaved
    assert "did not write" in screen._annotation_state.text()
    assert _png_list(database)["annotate"].eq(1).all()


def test_a_table_without_object_keys_says_why_it_cannot_be_saved(qtbot, tmp_path, monkeypatch):
    path = tmp_path / "wells.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE wells (wellID TEXT, cell_area REAL)")
        db.executemany("INSERT INTO wells VALUES (?, ?)", [("A1", 1.0), ("B1", 2.0)])
    screen = _screen(qtbot)
    screen.load_path(str(path), table="wells")
    _annotate(screen, monkeypatch, metadata="wellID", include="A")
    assert screen._unsaved
    assert not screen._save_annotations_button.isEnabled()
    assert "no prcfo column" in screen._save_annotations_button.toolTip()
