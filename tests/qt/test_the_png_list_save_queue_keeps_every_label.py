"""The png_list save queue never loses a label, whatever order things happen in.

Issue #186 made Graph Builder and Gate Editor save annotation columns to
``png_list`` as they are made, one save at a time. The end-to-end tests in
``test_issue_186_graph_builder_annotations_reach_png_list.py`` drive the
ordinary path. These drive the queue through the orders a real session
produces and a happy path never does: a retry with nothing to retry, a save
whose answer arrives after a newer annotation, a save job that dies, a
source that needs a project folder the user declines to give, a close while
labels are unsaved, and filter sets saved to and read from places that fail.
"""
from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMessageBox

from spacr import condition_annotations
from spacr.condition_annotations import new_definition, source_context
from spacr.derived_tables import default_definition, save_definition
from spacr.qt.screens import graph_builder
from tests.qt.test_issue_186_graph_builder_annotations_reach_png_list import (
    COLUMN, _annotate, _plate, _screen,
)


@pytest.fixture(autouse=True)
def calm_preferences():
    from spacr.qt import preferences
    preferences.set_reduce_motion(True)
    preferences.set_ambient_enabled(False)


@pytest.fixture
def dialogs(monkeypatch):
    """Every dialog the queue and the filter controls can raise, scripted."""
    script = {"folders": [], "save_paths": [], "open_paths": [],
              "answers": [], "warnings": [], "asked": []}

    def folder(*_args, **_kwargs):
        script["asked"].append("folder")
        return script["folders"].pop(0)

    monkeypatch.setattr(graph_builder.QFileDialog, "getExistingDirectory",
                        staticmethod(folder))
    monkeypatch.setattr(graph_builder.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (script["save_paths"].pop(0), "")))
    monkeypatch.setattr(graph_builder.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (script["open_paths"].pop(0), "")))
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: script["answers"].pop(0))
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda _parent, _title, text, *a, **k: script["warnings"].append(text))
    return script


def _unroutable(screen, tmp_path):
    """Queue labels for rows no plate database can take; the save needs a folder."""
    frame = pd.DataFrame({"plateID": ["elsewhere"], "rowID": ["r1"], "columnID": ["c1"],
                          "fieldID": ["f1"], "object_label": [1]})
    source = source_context(tmp_path / "loose.csv", None)
    definition = new_definition(frame, source, column=COLUMN)
    definition["conditions"] = [dict(name="x", metadata_column="columnID", include="c1",
                                     exclude="", manual_rows=[])]
    labelled = condition_annotations.apply_conditions(frame, definition, source)
    screen._queue_png_list_save(labelled, definition, source)
    return json.dumps(source, sort_keys=True)


def test_the_queue_survives_retries_stale_answers_and_a_dead_save_job(
        qtbot, tmp_path, dialogs):
    screen = _screen(qtbot)
    button, note = screen._save_annotations_button, screen._annotation_state

    screen.save_annotations()
    assert dialogs["asked"] == []

    key = _unroutable(screen, tmp_path)
    assert screen._save_notes[key][1] is True
    assert "choose the project folder" in note.text()

    dialogs["folders"] += ["", str(tmp_path), ""]
    screen.save_annotations()
    screen.save_annotations()
    assert dialogs["asked"] == ["folder", "folder"]
    assert screen._save_roots == [str(tmp_path)]
    assert key in screen._unsaved

    generation = screen._unsaved[key][3]
    screen._saving = ("someone else", 0)
    screen._save_queue.append(key)
    screen._retry_png_list_saves()
    assert screen._save_queue.count(key) == 1
    assert dialogs["asked"] == ["folder"] * 3
    screen._update_annotation_state()
    assert not button.isEnabled()
    assert button.toolTip() == "Saving the annotations to png_list…"
    screen._save_notes.clear()
    screen._update_annotation_state()
    assert note.text().endswith("Saving to png_list…")

    screen._saving = None
    screen._on_png_list_save_failed("nothing was running")
    assert screen._saving is None
    screen._saving = (key, generation)
    screen._save_queue.clear()
    screen._on_png_list_save_failed("the worker died")
    assert screen._saving is None
    assert screen._save_notes[key][0] == "Not saved: the worker died"

    screen._on_png_list_saved(key, generation - 1, ("failed", "an older save"))
    assert screen._save_notes[key][0] == "Not saved: the worker died"

    screen._save_queue[:] = ["ghost", key]
    screen._forget_unsaved(key)
    assert screen._save_queue == ["ghost"]
    screen._start_next_save()
    assert screen._save_queue == [] and screen._saving is None
    assert not screen._unsaved


def test_the_save_note_names_rows_without_keys_and_conflicting_labels(qtbot):
    screen = _screen(qtbot)
    report = {"databases": {"a.db": {"objects": 1200, "no_png_list_row": 0}},
              "no_key": 3, "conflicting": 2}

    assert screen._summarise_save(report) == (
        "Annotations saved to png_list: 1,200 objects in 1 database(s). "
        "3 rows have no object key. 2 objects are labelled differently on several "
        "rows; the first label was saved.")


def test_closing_with_unsaved_labels_lets_the_user_discard_or_stay(
        qtbot, tmp_path, dialogs):
    screen = _screen(qtbot)
    _unroutable(screen, tmp_path)

    dialogs["answers"] += [QMessageBox.Discard, QMessageBox.Save]
    assert screen._confirm_close_with_unsaved() is True
    assert screen._confirm_close_with_unsaved() is False
    assert len(dialogs["warnings"]) == 1
    assert "choose the project folder" in dialogs["warnings"][0]
    assert screen._unsaved


def test_a_saved_table_keeps_unsaved_labels_only_when_they_can_still_be_saved(
        qtbot, tmp_path):
    database, _cells = _plate(tmp_path, "plate1")
    screen = _screen(qtbot)
    keyable = pd.DataFrame({"prcfo": ["plate1_r1_c1_f1_o1"]})
    loose = pd.DataFrame({"wellID": ["A1"]})
    screen._unsaved = {"keyable": (keyable, {}, {}, 1, {}),
                       "loose": (loose, {}, {}, 1, {})}

    screen._on_annotated_table_saved(str(database), "keyable", "cell")
    screen._on_annotated_table_saved(str(database), "loose", "cell")

    assert list(screen._unsaved) == ["keyable"]
    assert screen._frame is not None and "cell_area" in screen._frame.columns


def test_filter_sets_are_saved_and_read_through_their_dialogs(
        qtbot, tmp_path, dialogs):
    database, _cells = _plate(tmp_path, "plate1")
    screen = _screen(qtbot)
    screen.load_path(str(database), table="cell")
    good = tmp_path / "filters.json"

    dialogs["save_paths"] += ["", str(tmp_path / "no" / "such" / "dir.json"), str(good)]
    screen._choose_save_filters()
    screen._choose_save_filters()
    assert screen._source.text().startswith("could not save that filter set: ")
    screen._choose_save_filters()
    assert json.loads(good.read_text(encoding="utf-8"))["saved_from"]["table"] == "cell"

    state = json.loads(good.read_text(encoding="utf-8"))
    state["filters"].append({"kind": "range", "column": "not_here", "enabled": True,
                             "low": 0.0, "high": 1.0})
    good.write_text(json.dumps(state), encoding="utf-8")
    dialogs["open_paths"] += ["", str(good)]
    screen._choose_load_filters()
    screen._choose_load_filters()
    assert "not_here" in screen._source.text()
    assert len(dialogs["warnings"]) == 1
    assert screen._load_filters(str(good), warn=False) == ["not_here"]
    assert len(dialogs["warnings"]) == 1

    before = screen._source.text()
    screen._say_dropped_filters([])
    assert screen._source.text() == before
    screen._say_dropped_filters(["area", "gene"])
    assert screen._source.text().endswith("filters removed, this table has no area, gene")


def test_a_capped_read_never_attaches_a_saved_annotation(qtbot, tmp_path):
    database, cells = _plate(tmp_path, "plate1")
    csv = tmp_path / "cells.csv"
    cells.to_csv(csv, index=False)
    save_definition(str(database), default_definition(str(database), ["cell"],
                                                      name="merged"))

    capped = graph_builder.read_table(str(csv), limit=2)
    derived = graph_builder.read_table(str(database), "merged", limit=3)

    assert len(capped) == 2 and "saved_condition_definition" not in capped.attrs
    assert len(derived) == 3 and "saved_condition_definition" not in derived.attrs


def test_a_damaged_save_receipt_reopens_the_source_without_its_labels(qtbot, tmp_path):
    database, _cells = _plate(tmp_path, "plate1")
    screen = _screen(qtbot)
    screen.load_path(str(database), table="cell")
    _annotate(screen)
    assert not screen._unsaved
    assert "saved_condition_definition" in graph_builder.read_table(
        str(database), "cell").attrs

    with sqlite3.connect(database) as db:
        db.execute(f'UPDATE "{condition_annotations.PROVENANCE_TABLE}" '
                   "SET payload_json = '{not json'")

    frame = graph_builder.read_table(str(database), "cell")
    assert "saved_condition_definition" not in frame.attrs
    assert len(frame) == 4
