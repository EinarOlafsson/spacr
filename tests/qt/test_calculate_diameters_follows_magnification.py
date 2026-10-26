"""Item 696, 2026-10-26: "Calculate diameters" on the Make Masks screen.

On (the default), a blank cell, nucleus or pathogen diameter is calculated
from Magnification, greyed, and moves live with it -- in Per-object settings
and in what ``collect()`` hands the run. A diameter the user sets is PINNED:
drawn bold, kept across magnification changes, until its cell is cleared or
the switch is turned off and on again, which recalculates all three.

Off, turning it off writes the calculated numbers into the blank diameters,
and magnification changes no longer touch any diameter.

A settings file is applied one widget at a time, so loading never pins,
freezes or resets anything halfway.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402

from spacr.object_settings_table import _resolved_diameter  # noqa: E402

_OBJECTS = ("cell", "nucleus", "pathogen")


def _screen(qtbot):
    from PySide6.QtWidgets import QApplication

    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    screen.apply_settings_dict({"cell_channel": 0, "nucleus_channel": 1,
                                "pathogen_channel": 2, "magnification": 20,
                                "cell_diameter": None,
                                "nucleus_diameter": None,
                                "pathogen_diameter": None})
    QApplication.processEvents()
    return screen


def _set_magnification(screen, value):
    model = screen._settings_model
    assert model.set_value_for_key("magnification", value)


def _switch(screen):
    return screen._settings_model._widgets["calculate_diameters"]


def _shown(screen, obj, role=Qt.DisplayRole):
    grid = screen._object_grid
    model = grid._model
    row = grid.questions().index("diameter")
    column = grid.objects().index(obj)
    return model.data(model.index(row, column), role)


def _run_diameters(screen):
    settings = screen._settings_model.collect()
    return [_resolved_diameter(settings[f"{obj}_diameter"],
                               settings["magnification"], obj)
            for obj in _OBJECTS]


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    return _screen(qtbot)


def test_the_switch_is_on_and_directly_under_magnification(screen):
    model = screen._settings_model
    assert model.collect()["calculate_diameters"] is True
    assert _switch(screen).isChecked()
    keys = list(model._widgets)
    assert keys.index("calculate_diameters") == keys.index("magnification") + 1
    assert screen.setting_row_is_visible("calculate_diameters")


def test_on_a_magnification_change_recalculates_the_shown_and_run_values(
        screen):
    assert [_shown(screen, obj) for obj in _OBJECTS] == ["120", "60", "20"]
    for mag, expected in ((40, ["160", "75", "40"]),
                          (60, ["200", "90", "60"]),
                          (100, ["280", "120", "100"]),
                          (10, ["100", "52", "10"])):
        _set_magnification(screen, mag)
        assert [_shown(screen, obj) for obj in _OBJECTS] == expected
        assert _run_diameters(screen) == [float(v) for v in expected]


def test_a_typed_diameter_is_pinned_bold_and_kept(screen):
    grid = screen._object_grid
    assert grid.set_value("diameter", "nucleus", "70")
    assert screen._settings_model.collect()["nucleus_diameter"] == 70
    assert _shown(screen, "nucleus", Qt.FontRole).bold()
    assert "Pinned" in _shown(screen, "nucleus", Qt.ToolTipRole)
    assert not _shown(screen, "cell", Qt.FontRole).bold()
    _set_magnification(screen, 60)
    assert [_shown(screen, obj) for obj in _OBJECTS] == ["200", "70", "60"]
    assert _run_diameters(screen) == [200.0, 70.0, 60.0]


def test_clearing_a_pinned_cell_recalculates_it(screen):
    grid = screen._object_grid
    grid.set_value("diameter", "cell", "150")
    _set_magnification(screen, 40)
    assert _shown(screen, "cell") == "150"
    grid.set_value("diameter", "cell", "")
    assert screen._settings_model.collect()["cell_diameter"] is None
    assert _shown(screen, "cell") == "160"


def test_turning_it_off_holds_every_diameter_still(screen):
    grid = screen._object_grid
    grid.set_value("diameter", "pathogen", "auto")
    _switch(screen).click()
    settings = screen._settings_model.collect()
    assert settings["calculate_diameters"] is False
    assert (settings["cell_diameter"], settings["nucleus_diameter"],
            settings["pathogen_diameter"]) == (120, 60, "auto")
    font = _shown(screen, "cell", Qt.FontRole)
    assert font is None or not font.bold()
    assert "is off" in _shown(screen, "cell", Qt.ToolTipRole)
    _set_magnification(screen, 60)
    assert [_shown(screen, obj) for obj in _OBJECTS] == ["120", "60", "auto"]
    assert _run_diameters(screen) == [120.0, 60.0, "auto"]


def test_off_a_cleared_cell_is_refilled_at_the_current_magnification(screen):
    _switch(screen).click()
    _set_magnification(screen, 40)
    screen._object_grid.set_value("diameter", "cell", "")
    assert screen._settings_model.collect()["cell_diameter"] == 160
    assert _shown(screen, "cell") == "160"


def test_turning_it_back_on_recalculates_all_three(screen):
    grid = screen._object_grid
    _switch(screen).click()
    grid.set_value("diameter", "nucleus", "33")
    _set_magnification(screen, 40)
    _switch(screen).click()
    settings = screen._settings_model.collect()
    assert settings["calculate_diameters"] is True
    assert [settings[f"{obj}_diameter"] for obj in _OBJECTS] == [None] * 3
    assert [_shown(screen, obj) for obj in _OBJECTS] == ["160", "75", "40"]
    _set_magnification(screen, 20)
    assert _run_diameters(screen) == [120.0, 60.0, 20.0]


def test_loading_settings_neither_freezes_nor_resets(screen):
    screen._object_grid.set_value("diameter", "cell", "90")
    screen.apply_settings_dict({"calculate_diameters": False,
                                "magnification": 60,
                                "nucleus_diameter": None})
    settings = screen._settings_model.collect()
    assert settings["calculate_diameters"] is False
    assert settings["nucleus_diameter"] is None
    assert settings["cell_diameter"] == 90
    screen.apply_settings_dict({"calculate_diameters": True,
                                "cell_diameter": 95})
    settings = screen._settings_model.collect()
    assert settings["calculate_diameters"] is True
    assert settings["cell_diameter"] == 95


def test_saved_settings_reload_into_a_fresh_screen(qtbot, qt_theme_applied,
                                                   tmp_path):
    from spacr.cli import load_settings_file, write_settings_file

    first = _screen(qtbot)
    first._object_grid.set_value("diameter", "nucleus", "70")
    _switch(first).click()
    _set_magnification(first, 40)
    saved = first._settings_model.collect()
    paths = write_settings_file(saved, tmp_path / "mask_settings.csv")
    loaded = load_settings_file(paths[0])
    assert loaded["calculate_diameters"] is False

    second = _screen(qtbot)
    second.apply_settings_dict(loaded)
    again = second._settings_model.collect()
    assert again["calculate_diameters"] is False
    assert not _switch(second).isChecked()
    assert [again[f"{obj}_diameter"] for obj in _OBJECTS] == [120, 70, 20]
    assert [_shown(second, obj) for obj in _OBJECTS] == ["120", "70", "20"]
    _set_magnification(second, 60)
    assert _run_diameters(second) == [120.0, 70.0, 20.0]
