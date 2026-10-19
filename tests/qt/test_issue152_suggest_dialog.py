"""Issue #152: Suggest asks which tables, feature classes and images to use.

The pages a user ticks are the pages as they were when the dialog opened:
the grid reflows when the window is resized, and a ticked "page 2" must
still mean the crops that page held.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("PySide6")


def _db(tmp_path: Path, n: int = 10) -> Path:
    """Ten cell crops and a cell and nucleus table with two channels."""
    db = tmp_path / "measurements.db"
    cells, nuclei, crops = [], [], []
    for obj in range(1, n + 1):
        cells.append({"plateID": "plate1", "rowID": "r1", "columnID": "c1",
                      "fieldID": "f1", "prcf": "plate1_r1_c1_f1",
                      "object_label": obj, "cell_area": float(obj),
                      "cell_channel_1_mean_intensity": float(obj)})
        nuclei.append({"plateID": "plate1", "rowID": "r1", "columnID": "c1",
                       "fieldID": "f1", "prcf": "plate1_r1_c1_f1",
                       "object_label": obj + 100, "cell_id": float(obj),
                       "nucleus_area": float(obj)})
        crops.append({"png_path": f"/c/{obj:02d}.png",
                      "prcfo": f"plate1_r1_c1_f1_o{obj}",
                      "cell_id": f"o{obj}", "annotate": None})
    with sqlite3.connect(db) as con:
        pd.DataFrame(cells).to_sql("cell", con, index=False)
        pd.DataFrame(nuclei).to_sql("nucleus", con, index=False)
        pd.DataFrame(crops).to_sql("png_list", con, index=False)
    return db


@pytest.fixture
def screen(qtbot, qt_theme_applied, tmp_path, monkeypatch):
    from spacr.qt.screens import annotate as mod

    widget = mod.AnnotateScreen()
    qtbot.addWidget(widget)
    widget._settings.db_path = str(_db(tmp_path))
    widget._settings.annotation_column = "annotate"
    widget._settings.grid_rows, widget._settings.grid_cols = 2, 2
    widget._total = 10
    widget._offset = 4
    started = []
    monkeypatch.setattr(
        widget, "_start_suggest",
        lambda **kwargs: started.append(kwargs))
    widget.started = started
    yield widget
    dialog = getattr(widget, "_suggest_dialog", None)
    if dialog is not None:
        try:
            dialog.close()
        except RuntimeError:
            pass
    if widget._worker is not None:
        widget._worker.stop(wait=True)


def test_the_dialog_defaults_to_the_annotated_object_s_table(screen):
    dialog = screen._open_suggest_dialog()

    assert dialog.objectName() == "AnnotateSuggestDialog"
    assert dialog.findChild(object, "AnnotateSuggestTable_cell").isChecked()
    assert not dialog.findChild(
        object, "AnnotateSuggestTable_nucleus").isChecked()
    assert dialog.selection() == {"feature_tables": ["cell"],
                                  "feature_channels": None,
                                  "feature_families": None}
    assert dialog.scope() == "screen"
    assert dialog.only_paths() == ["/c/05.png", "/c/06.png", "/c/07.png",
                                   "/c/08.png"]


def test_choosing_tables_and_channels_reaches_the_run(screen):
    dialog = screen._open_suggest_dialog()
    dialog.findChild(object, "AnnotateSuggestTable_nucleus").setChecked(True)
    dialog.findChild(object, "AnnotateSuggestChannel_channel_1").setChecked(
        False)
    dialog.findChild(object, "AnnotateSuggestScopeAll").setChecked(True)
    dialog.accept()

    assert screen.started == [{
        "this_page": False, "only_paths": None,
        "selection": {"feature_tables": ["cell", "nucleus"],
                      "feature_channels": ["none"],
                      "feature_families": None}}]


def test_the_pages_are_the_ones_shown_when_the_dialog_opened(screen):
    dialog = screen._open_suggest_dialog()
    pages = dialog.findChild(object, "AnnotateSuggestPages")
    assert pages.count() == 3

    screen._settings.grid_rows, screen._settings.grid_cols = 2, 3
    screen._offset = 0

    dialog.findChild(object, "AnnotateSuggestScopePages").setChecked(True)
    from PySide6.QtCore import Qt
    pages.item(1).setCheckState(Qt.Unchecked)
    pages.item(2).setCheckState(Qt.Checked)
    assert dialog.pages() == [2]
    assert dialog.only_paths() == ["/c/09.png", "/c/10.png"]
    dialog.accept()
    assert screen.started[-1]["only_paths"] == ["/c/09.png", "/c/10.png"]


def test_the_choice_is_remembered_for_the_database(screen):
    dialog = screen._open_suggest_dialog()
    dialog.findChild(object, "AnnotateSuggestTable_nucleus").setChecked(True)
    dialog.findChild(object, "AnnotateSuggestScopeAll").setChecked(True)
    dialog.accept()

    again = screen._open_suggest_dialog()
    assert again.findChild(object, "AnnotateSuggestTable_nucleus").isChecked()
    assert again.scope() == "all"
    assert screen._suggest_selection_options(None) == {
        "feature_tables": ["cell", "nucleus"]}


def test_no_page_ticked_cannot_be_accepted(screen):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDialogButtonBox

    dialog = screen._open_suggest_dialog()
    dialog.findChild(object, "AnnotateSuggestScopePages").setChecked(True)
    pages = dialog.findChild(object, "AnnotateSuggestPages")
    pages.item(1).setCheckState(Qt.Unchecked)
    ok = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok)
    assert not ok.isEnabled()


def test_the_worker_fits_on_the_chosen_tables_and_classes(tmp_path,
                                                         monkeypatch):
    from spacr import active_learning as al
    from spacr.qt.screens import annotate as mod

    db = _db(tmp_path)
    seen = {}

    def fake_round_features(*args, **kwargs):
        seen.update(kwargs)
        raise ValueError("stop after the features")

    monkeypatch.setattr(al, "round_features", fake_round_features)
    worker = mod._SuggestWorker(
        str(db), "annotate",
        {"round_index": 1, "feature_tables": ["nucleus"],
         "feature_channels": ["channel_1"],
         "feature_families": ["intensity"]})
    failed = []
    worker.failed.connect(failed.append)
    worker.run()

    assert failed and "stop after the features" in failed[0]
    assert seen["tables"] == ["nucleus"]
    assert seen["channels"] == ["channel_1"]
    assert seen["families"] == ["intensity"]
