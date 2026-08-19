"""Instruction 169 B: the figures container's size is the reader's to set.

Reported: "cant control the height of the containers in the figures
container in the measurements tab. i need to be able to make each taller".

`FigureGridView.set_target_cell_width` had existed all along and NOTHING
CALLED IT -- a setter with no caller is a control that does not exist, which
is why the complaint survived the method being written. These tests pin the
caller, not the method.

Width and not height, because the tiles keep each figure's own aspect ratio:
setting the width IS setting the height, and a separate height control would
either fight the aspect or distort the figure.
"""
import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """QSettings of our own -- these tests must not resize the user's grid."""
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences

    path = tmp_path / "spacr.ini"
    monkeypatch.setattr(
        preferences, "_settings",
        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


@pytest.fixture()
def screen(qtbot, store):
    from spacr.qt.screens.app_screen import AppScreen

    widget = AppScreen("regression")
    qtbot.addWidget(widget)
    return widget


def test_there_is_a_size_control_at_all(screen):
    assert screen._figure_size is not None
    assert screen._figure_size.isEnabled()


def test_it_spans_the_widths_the_grid_accepts(screen):
    from spacr.qt.widgets.figure_grid_view import MAX_CELL_PX, MIN_CELL_PX

    assert screen._figure_size.minimum() == MIN_CELL_PX
    assert screen._figure_size.maximum() == MAX_CELL_PX


def test_moving_it_resizes_the_tiles(screen):
    """The control has to reach the grid, which is the half that was missing."""
    screen._figure_size.setValue(400)
    assert screen._figure_grid._target == 400
    screen._figure_size.setValue(240)
    assert screen._figure_grid._target == 240


def test_a_value_past_the_end_is_clamped_not_refused(screen):
    from spacr.qt.widgets.figure_grid_view import MAX_CELL_PX

    screen._figure_grid.set_target_cell_width(10_000)
    assert screen._figure_grid._target == MAX_CELL_PX


def test_the_size_is_remembered(screen, store):
    """A READING preference: big figures stay big on the next run."""
    screen._figure_size.setValue(380)
    assert store.get_figure_grid_size() == 380


def test_a_remembered_size_is_applied_on_open(qtbot, store):
    from spacr.qt.screens.app_screen import AppScreen

    store.set_figure_grid_size(500)
    later = AppScreen("regression")
    qtbot.addWidget(later)
    assert later._figure_size.value() == 500
    assert later._figure_grid._target == 500


def test_a_nonsense_stored_size_falls_back(store):
    """A settings file edited by hand must not leave the grid unopenable."""
    from spacr.qt.widgets.figure_grid_view import TARGET_CELL_PX

    store._settings().setValue(store._KEY_FIGURE_GRID_SIZE, "enormous")
    assert store.get_figure_grid_size() == TARGET_CELL_PX


def test_the_grid_is_still_the_first_page(screen):
    """Wrapping the grid in a page must not move it out from under index 0.

    `_show_figure_grid` sets index 0, and every "back to all figures" button
    goes through it.
    """
    screen._show_figure_grid()
    assert screen._figures_stack.currentIndex() == 0
    page = screen._figures_stack.widget(0)
    assert screen._figure_grid in page.findChildren(type(screen._figure_grid))
