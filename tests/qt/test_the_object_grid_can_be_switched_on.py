"""The per-object grid replaces the flat rows without changing the file.

78 of Mask's 201 settings are the same twenty-odd questions asked once per
object type, so the form asks 203 questions before anything is segmented. The
grid asks each question once, with a column per object.

IT IS OFF UNLESS CHOSEN. This is the most-used screen in the application, so
the grid arrives as an offer rather than as a change to what everyone already
knows -- and the first test here is that a user who has chosen nothing sees
exactly the form they saw yesterday.

THE PROPERTY THAT MAKES IT SAFE, and what every test below is about: the grid
edits the same widgets the flat rows do. `collect()` is unchanged, so a
settings file written with the grid on is the same file written with it off,
and nothing downstream of the panel learns the grid exists.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def grid_preference():
    """Restore whatever the machine had, whatever a test does to it."""
    from spacr.qt import preferences as prefs

    was = prefs.get_object_grid_enabled()
    yield prefs
    prefs.set_object_grid_enabled(was)


def _screen(qtbot, prefs, on):
    from PySide6.QtWidgets import QApplication
    from spacr.qt.screens.app_screen import AppScreen

    prefs.set_object_grid_enabled(on)
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    QApplication.processEvents()
    return screen


def test_the_default_is_the_form_that_was_always_there(qtbot, grid_preference):
    """Nothing changes for a user who has not asked for the grid."""
    from spacr.qt import preferences as prefs

    assert prefs.DEFAULT_OBJECT_GRID is False

    screen = _screen(qtbot, grid_preference, False)

    assert not hasattr(screen, "_object_grid"), "the grid mounted uninvited"
    assert screen.setting_row_is_visible("cell_diameter") is True


def test_switching_it_on_puts_the_grid_where_the_rows_were(qtbot,
                                                           grid_preference):
    """The rows go, the grid arrives, the widgets stay."""
    screen = _screen(qtbot, grid_preference, True)

    assert hasattr(screen, "_object_grid"), "the grid did not mount"
    assert screen.setting_row_is_visible("cell_diameter") is False
    # HIDDEN, NOT DROPPED. The widget is what `collect()` reads and what the
    # grid writes through to, and the settings search still indexes it.
    assert "cell_diameter" in screen._settings_model._widgets


def test_the_settings_are_the_same_either_way(qtbot, grid_preference):
    """The whole safety argument, asked directly."""
    off = _screen(qtbot, grid_preference, False)._settings_model.collect()
    on = _screen(qtbot, grid_preference, True)._settings_model.collect()

    assert set(on) == set(off), "the grid changed which settings exist"
    for key, value in off.items():
        assert on[key] == value, f"{key} differs with the grid on"


def test_editing_a_cell_reaches_the_settings_the_run_reads(qtbot,
                                                           grid_preference):
    """Through `setData`, which is what typing in a cell calls."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    screen = _screen(qtbot, grid_preference, True)
    model = screen._settings_model
    table = screen._object_grid._model
    row = list(table.table()).index("min_area")
    before = int(model.collect()["cell_min_area"])

    assert table.setData(table.index(row, 0), str(before + 7), Qt.EditRole)
    QApplication.processEvents()

    assert int(model.collect()["cell_min_area"]) == before + 7


def test_a_row_the_grid_speaks_for_stays_hidden_across_a_refresh(
        qtbot, grid_preference):
    """`refresh_object_visibility` recomputes its hidden set from scratch.

    So the grid's keys are kept in a set of their own and unioned in. Putting
    them into `_hidden_by_the_run` would show every one of them again the
    first time a channel was typed.
    """
    screen = _screen(qtbot, grid_preference, True)

    screen._settings_model.refresh_object_visibility()

    assert screen.setting_row_is_visible("cell_diameter") is False


def test_the_grid_is_not_a_labelled_setting_row(qtbot, grid_preference):
    """It goes in through `add_prose_row`, and that matters.

    Every entry in `Section._row_widgets` is taken to BE a labelled setting
    by the module smoke test, which asserts each field carries a `settingKey`
    and that its label holds linked API help. A grid is neither.
    """
    screen = _screen(qtbot, grid_preference, True)
    section = next(s for s in screen._settings_sections
                   if screen._object_grid in s.findChildren(type(screen._object_grid)))

    assert screen._object_grid not in [w for _l, w in section._row_widgets]
