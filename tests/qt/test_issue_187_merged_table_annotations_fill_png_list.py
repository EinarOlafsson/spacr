"""Graph Builder annotations on a merged table fill each plate's png_list.

Issue #187: after #186 the annotation column appeared in every plate's
``png_list``, but every value was empty. The merged measurement table reads
plate ids through ``canonical_plate_id``, so a plate an older Measure run
stamped ``pplate1`` comes back as ``plate1``, while ``png_list`` keeps
``pplate1_r5_c1_f1_o1`` as written. The write-back matched the two keys as
text, so no row of such a plate matched, the column was added anyway and the
note still said the annotations were saved.

The merged table here is built by the Measurements tab's own merge,
:func:`merge_across_databases`, from databases with the real column schema.
"""
from __future__ import annotations

import logging
import sqlite3

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.condition_annotations import new_definition, save_annotations_to_png_list, source_context
from spacr.qt.screens.graph_builder import GraphBuilderScreen
from spacr.qt.widgets.measurement_scan_panel import merge_across_databases
from tests.real_schema_measurements_db import build_measurements_db

COLUMN = "cc2d1a_recruitment"


@pytest.fixture(autouse=True)
def calm_preferences():
    from spacr.qt import preferences
    preferences.set_reduce_motion(True)
    preferences.set_ambient_enabled(False)


def _merged_project(tmp_path, stamp=lambda plate: "p" + plate):
    """Two plate folders and their merged table as a CSV beside them.

    :param stamp: the plate id each database stores for its folder's plate.
    """
    databases = [build_measurements_db(tmp_path / plate, plate=stamp(plate), fields=("f1", "f2"),
                                       cells_per_field=(6, 5), seed=index)
                 for index, plate in enumerate(["plate1", "plate2"])]
    merged = merge_across_databases(databases, ["cell"])
    folder = tmp_path / "results"
    folder.mkdir()
    csv = folder / "merged_measurements.csv"
    merged.to_csv(csv, index=False)
    return databases, csv, merged


def _definition(frame, source):
    definition = new_definition(frame, source, column=COLUMN)
    definition["conditions"] = [dict(name="recruited", metadata_column="fieldID",
                                     include="f1", exclude="", manual_rows=[])]
    return definition


def _labels(database):
    with sqlite3.connect(database) as db:
        names = [row[1] for row in db.execute("PRAGMA table_info(png_list)")]
        if COLUMN not in names:
            return None
        return pd.DataFrame(db.execute(f"SELECT prcfo, {COLUMN} FROM png_list").fetchall(),
                            columns=["prcfo", COLUMN]).set_index("prcfo")[COLUMN]


def _screen(qtbot, csv):
    screen = GraphBuilderScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.load_path(str(csv))
    return screen


def test_a_merged_table_of_plates_stamped_pplate_fills_their_png_list(qtbot, tmp_path):
    databases, csv, merged = _merged_project(tmp_path)
    assert set(merged["plateID"]) == {"plate1", "plate2"}
    screen = _screen(qtbot, csv)
    screen.apply_condition_definition(_definition(screen._annotation_base_frame,
                                                  screen._condition_source))

    assert not screen._unsaved
    for database in databases:
        labels = _labels(database)
        plate = "p" + database.split("/")[-3]
        assert labels.loc[f"{plate}_r5_c1_f1_o1"] == "recruited"
        assert labels.loc[f"{plate}_r5_c1_f1_o6"] == "recruited"
        assert pd.isna(labels.loc[f"{plate}_r5_c1_f2_o1"])
        assert labels.notna().sum() == 6
    note = screen._annotation_state.text()
    assert "22 objects in 2 database(s)" in note
    assert "plate1: 11 png_list rows matched, 6 updated." in note
    assert "plate2: 11 png_list rows matched, 6 updated." in note
    assert "Warning" not in note


def test_each_plate_reports_matched_and_updated_png_list_rows(tmp_path, caplog):
    databases, csv, merged = _merged_project(tmp_path)
    source = source_context(str(csv))
    definition = _definition(merged, source)
    from spacr.condition_annotations import apply_conditions
    labelled = apply_conditions(merged, definition, source)
    caplog.set_level(logging.INFO, logger="spacr.condition_annotations")

    first = save_annotations_to_png_list(labelled, definition, source)
    again = save_annotations_to_png_list(labelled, definition, source)

    for database in databases:
        written = first["databases"][str(database)]
        assert (written["matched"], written["updated"], written["objects"]) == (11, 6, 11)
        assert written["unmatched"] == {"table": [], "png_list": []}
        assert (again["databases"][str(database)]["matched"],
                again["databases"][str(database)]["updated"]) == (11, 0)
    assert "11 rows matched the table, 6 updated" in caplog.text


def test_no_matching_png_list_row_warns_with_keys_from_both_sides_and_adds_no_column(
        qtbot, tmp_path, caplog):
    databases, csv, _merged = _merged_project(tmp_path, stamp=lambda plate: plate)
    for database in databases:
        with sqlite3.connect(database) as db:
            db.execute("UPDATE png_list SET prcfo = 'screenA_' || prcfo")
    caplog.set_level(logging.INFO, logger="spacr.condition_annotations")
    screen = _screen(qtbot, csv)
    screen.apply_condition_definition(_definition(screen._annotation_base_frame,
                                                  screen._condition_source))

    for database in databases:
        assert _labels(database) is None
    assert screen._unsaved
    note = screen._annotation_state.text()
    assert "Not saved" in note
    assert "Warning: no png_list row in plate1 matched this table by prcfo" in note
    assert "Table keys: plate1_r5_c1_f1_o1, plate1_r5_c1_f1_o2, plate1_r5_c1_f1_o3." in note
    assert "png_list keys: screenA_plate1_r5_c1_f1_o1" in note
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert "0 of 11 table objects matched" in warnings[0].getMessage()
    assert "screenA_plate1_r5_c1_f1_o1" in warnings[0].getMessage()
    assert screen._save_annotations_button.isEnabled()
