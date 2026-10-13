"""Item 592: per-object settings are the only Mask generation layout.

The maintainer, 2026-09-28: "in mask generation, please make per object
settings the default and only mode (remove the preferences toggle.)". The
table also takes each object's "remove background" switch, the cell's
"adjust cells", and object filters as rows that can be added; and Output &
storage, Runtime & reliability and Segmentation robustness merge into one
"Quality Control" category.

THE PROPERTY THAT MAKES IT SAFE is unchanged: the grid edits the same widgets
the flat rows did, so `collect()` returns the same keys and an older settings
file loads into it unchanged.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


def _screen(qtbot, app_key="mask"):
    from PySide6.QtWidgets import QApplication

    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key)
    qtbot.addWidget(screen)
    QApplication.processEvents()
    return screen


def test_the_grid_is_always_there_and_the_rows_are_hidden(qtbot,
                                                          qt_theme_applied):
    screen = _screen(qtbot)

    assert screen._object_grid is not None
    assert screen.setting_row_is_visible("cell_diameter") is False
    assert "cell_diameter" in screen._settings_model._widgets


def test_there_is_no_preference_for_it_any_more(qtbot, qt_theme_applied):
    from spacr.qt import preferences as prefs

    for name in ("get_object_grid_enabled", "set_object_grid_enabled",
                 "DEFAULT_OBJECT_GRID",
                 "_tell_the_screens_the_object_grid_changed"):
        assert not hasattr(prefs, name), name
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    assert not [w for w in dialog.findChildren(object)
                if getattr(w, "objectName", lambda: "")()
                == "ObjectSettingsGrid"]


def test_other_modules_keep_their_flat_form(qtbot, qt_theme_applied):
    screen = _screen(qtbot, "measure")

    assert getattr(screen, "_object_grid", None) is None


def test_the_table_sits_where_the_segmentation_questions_start(
        qtbot, qt_theme_applied):
    screen = _screen(qtbot)
    layout = screen._settings_layout
    section = screen._object_grid.parent()
    while section is not None and not hasattr(section, "add_prose_row"):
        section = section.parent()
    index = layout.indexOf(section)
    assert index >= 0
    before = [str(layout.itemAt(i).widget().property(
        "settingsCategorySource") or "") for i in range(index)
        if layout.itemAt(i).widget() is not None]
    assert "Image Preprocessing" in before
    assert "Quality Control" not in before


def test_editing_a_cell_reaches_the_settings_the_run_reads(qtbot,
                                                           qt_theme_applied):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    screen = _screen(qtbot)
    model = screen._settings_model
    table = screen._object_grid._model
    row = list(table.table()).index("background")
    before = int(model.collect()["cell_background"])

    assert table.setData(table.index(row, 0), str(before + 7), Qt.EditRole)
    QApplication.processEvents()

    assert int(model.collect()["cell_background"]) == before + 7


def test_a_widget_reaches_the_cell_in_front_of_it(qtbot, qt_theme_applied):
    screen = _screen(qtbot)

    screen._settings_model.set_value_for_key("cell_diameter", 77)
    assert screen._object_grid.table()["diameter"]["cell"] == 77
    screen._object_grid.set_value("diameter", "cell", "42")
    assert screen._settings_model.collect()["cell_diameter"] == 42
    assert screen._object_grid_binding._busy is False


def test_the_grid_is_not_a_labelled_setting_row(qtbot, qt_theme_applied):
    screen = _screen(qtbot)
    section = next(s for s in screen._settings_sections
                   if screen._object_grid in s.findChildren(
                       type(screen._object_grid)))

    assert screen._object_grid not in [w for _l, w in section._row_widgets]


def test_remove_background_is_a_row_for_every_object(qtbot,
                                                     qt_theme_applied):
    from PySide6.QtCore import Qt

    screen = _screen(qtbot)
    # 2026-09-29 (item 592, "hide unset objects"): a column is drawn only
    # for an object whose channel is set, so the nucleus and pathogen are
    # switched on first.
    for number, obj in enumerate(("nucleus", "pathogen"), start=1):
        assert screen._settings_model.set_value_for_key(
            f"{obj}_channel", number)
    grid = screen._object_grid
    row = grid.table()["remove_background"]
    assert {"cell", "nucleus", "pathogen"} <= set(row)
    assert "cytoplasm" not in row
    for key in ("remove_background_cell", "remove_background_nucleus",
                "remove_background_pathogen"):
        assert screen.setting_row_is_visible(key) is False

    model = grid._model
    r = list(model.table()).index("remove_background")
    c = list(model.objects()).index("nucleus")
    index = model.index(r, c)
    assert model.flags(index) & Qt.ItemIsUserCheckable
    assert model.setData(index, Qt.Checked, Qt.CheckStateRole)
    assert screen._settings_model.collect()[
        "remove_background_nucleus"] is True
    assert model.data(index, Qt.CheckStateRole) == Qt.Checked


def test_adjust_cells_is_a_row_for_the_cell_only(qtbot, qt_theme_applied):
    screen = _screen(qtbot)
    grid = screen._object_grid
    row = grid.table()["adjust_cells"]
    assert list(row) == ["cell"]
    assert screen.setting_row_is_visible("adjust_cells") is False
    was = screen._settings_model.collect()["adjust_cells"]
    grid.set_value("adjust_cells", "cell", str(not was))
    assert screen._settings_model.collect()["adjust_cells"] is (not was)
    assert grid.set_value("adjust_cells", "nucleus", "True") is False


def test_filters_are_rows_that_can_be_added_for_several_objects(
        qtbot, qt_theme_applied):
    screen = _screen(qtbot)
    # 2026-09-29 (item 592, "hide unset objects"): a column is drawn only
    # for an object whose channel is set, so the nucleus and pathogen are
    # switched on first.
    for number, obj in enumerate(("nucleus", "pathogen"), start=1):
        assert screen._settings_model.set_value_for_key(
            f"{obj}_channel", number)
    grid = screen._object_grid
    model = screen._settings_model
    assert screen.setting_row_is_visible("object_filters") is False

    assert grid.add_filter("area")
    assert grid.add_filter("mean_intensity")
    assert grid.filter_properties() == ("area", "intensity_mean")
    assert grid.add_filter("area") is False
    assert grid.add_filter("not_a_property") is False

    assert grid.set_value("filter:area", "cell", "200 - 5000")
    assert grid.set_value("filter:area", "nucleus", "50")
    assert grid.set_value("filter:intensity_mean", "pathogen", "– 900")
    assert grid.set_value("filter:area", "pathogen", "9 - 3") is False

    filters = model.collect()["object_filters"]
    assert filters["cell"] == [{"property": "area", "min": 200.0,
                                "max": 5000.0}]
    assert filters["nucleus"] == [{"property": "area", "min": 50.0,
                                   "max": None}]
    assert filters["pathogen"] == [{"property": "intensity_mean",
                                    "min": None, "max": 900.0}]
    assert "cytoplasm" not in grid.table()["filter:area"]

    assert grid.set_value("filter:area", "nucleus", "")
    assert "nucleus" not in model.collect()["object_filters"]


def test_a_filter_from_a_settings_file_shows_as_a_row(qtbot,
                                                      qt_theme_applied):
    screen = _screen(qtbot)
    model = screen._settings_model
    model.set_value_for_key(
        "object_filters",
        {"cell": [{"property": "solidity", "min": 0.9, "max": None}]})

    grid = screen._object_grid
    assert grid.table()["filter:solidity"]["cell"] == "0.9 –"


def test_an_old_settings_file_still_loads(qtbot, qt_theme_applied):
    """Retired per-object bounds still become filters, now shown as rows."""
    from spacr.settings import _fold_object_bounds

    old = {"cell_min_area": 150, "remove_background_cell": True,
           "adjust_cells": False}
    folded = _fold_object_bounds(dict(old), quiet=True)
    screen = _screen(qtbot)
    model = screen._settings_model
    for key, value in folded.items():
        assert model.set_value_for_key(key, value), key
    grid = screen._object_grid
    assert grid.table()["filter:area"]["cell"] == "150 –"
    assert grid.table()["remove_background"]["cell"] is True
    assert grid.table()["adjust_cells"]["cell"] is False
    collected = model.collect()
    assert collected["object_filters"]["cell"][0]["min"] == 150.0
    assert collected["remove_background_cell"] is True


def test_quality_control_holds_output_runtime_and_robustness():
    from spacr.settings import categories
    from spacr.qt.screens.settings_model import (_category_parents,
                                                 categories_for_app)

    sections = categories_for_app("mask", categories)
    for gone in ("Output & Storage", "Runtime & Reliability"):
        assert gone not in sections
    qc = sections["Quality Control"]
    for key in ("seg_qc", "save", "keep_npz", "n_jobs", "strict_errors",
                "mask_parallel"):
        assert key in qc, key
    assert _category_parents("mask")["Segmentation Robustness α"] \
        == "Quality Control"
