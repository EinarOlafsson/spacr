"""How many cells a montage row holds must not depend on the window size.

Reported as: "the cell tab shows 3 cells per well and then more if i change
the size of the container... change this".

Both the tab and the well page computed their column count as
``viewport_width // cell_px``. So the montage changed shape whenever the
window did, and two wells looked at one after the other -- which is the
entire point of the panel -- were not laid out the same way.

The count is a preference now and the THUMBNAILS take up the slack: a wider
panel draws the same cells larger, up to their natural size, rather than
fitting more in. How many ROWS fit is still measured, because that is what
decides a page.

A narrow panel still has to do something, and what it does is shrink the
pictures to a floor and then let the page scroll -- never silently show
fewer cells, which is the behaviour being replaced.
"""
from __future__ import annotations

import pytest

from spacr.qt.preferences import (DEFAULT_MONTAGE_COLUMNS,
                                  MONTAGE_COLUMNS_RANGE, get_montage_columns,
                                  set_montage_columns)
from spacr.qt.widgets.cell_montage_view import (MIN_THUMBNAIL_PX,
                                                THUMBNAIL_PX, _WellTab)


WIDTHS = (400, 700, 1200, 1800, 2400)


@pytest.fixture
def well_tab(qapp):
    tab = _WellTab(("plate1", "r5", "c1"), "r5/c1")
    try:
        yield tab
    finally:
        tab.deleteLater()
        qapp.processEvents()


@pytest.fixture(autouse=True)
def _restore_preference():
    before = get_montage_columns()
    yield
    set_montage_columns(before)


def test_the_column_count_does_not_move_with_the_window(well_tab):
    seen = set()
    for width in WIDTHS:
        well_tab._scroll.viewport().resize(width, 600)
        seen.add(well_tab.geometry_page()[0])

    assert len(seen) == 1, (
        f"the montage showed {sorted(seen)} columns at widths {WIDTHS}; the "
        f"row length must be the same at every size")
    assert seen.pop() == DEFAULT_MONTAGE_COLUMNS


def test_a_wider_panel_draws_bigger_cells_not_more_of_them(well_tab):
    """The slack goes into the thumbnails, which is the point of fixing this."""
    well_tab._scroll.viewport().resize(400, 600)
    narrow_columns, _ = well_tab.geometry_page()
    narrow_thumb = well_tab._thumb_px

    well_tab._scroll.viewport().resize(1200, 600)
    wide_columns, _ = well_tab.geometry_page()
    wide_thumb = well_tab._thumb_px

    assert narrow_columns == wide_columns
    assert wide_thumb > narrow_thumb


def test_a_thumbnail_never_grows_past_its_natural_size(well_tab):
    """Past THUMBNAIL_PX a crop is an interpolated blur."""
    well_tab._scroll.viewport().resize(4000, 600)
    well_tab.geometry_page()

    assert well_tab._thumb_px <= THUMBNAIL_PX


def test_a_very_narrow_panel_shrinks_to_a_floor_and_stops(well_tab):
    """Below this a thumbnail stops being a picture of a cell."""
    well_tab._scroll.viewport().resize(60, 600)
    columns, _ = well_tab.geometry_page()

    assert columns == DEFAULT_MONTAGE_COLUMNS, "it dropped columns instead"
    assert well_tab._thumb_px >= MIN_THUMBNAIL_PX


@pytest.mark.parametrize("columns", [1, 3, 4, 8, 12])
def test_the_preference_is_what_decides(well_tab, columns):
    set_montage_columns(columns)
    well_tab._scroll.viewport().resize(1200, 600)

    assert well_tab.geometry_page()[0] == columns


def test_the_preference_is_clamped_to_something_usable():
    low, high = MONTAGE_COLUMNS_RANGE
    assert set_montage_columns(0) == low
    assert set_montage_columns(9999) == high
    assert set_montage_columns("not a number") == DEFAULT_MONTAGE_COLUMNS


def test_the_rows_are_still_measured(well_tab):
    """Only the columns are decided; a taller panel still holds more."""
    well_tab._scroll.viewport().resize(1200, 300)
    _columns, short_page = well_tab.geometry_page()
    well_tab._scroll.viewport().resize(1200, 1200)
    _columns, tall_page = well_tab.geometry_page()

    assert tall_page > short_page


def test_the_preferences_dialog_offers_it(qapp):
    """A preference with no control is one the user cannot reach."""
    from PySide6.QtWidgets import QDialogButtonBox, QSpinBox

    from spacr.qt.preferences import PreferencesDialog

    dialog = PreferencesDialog()
    try:
        spins = [s for s in dialog.findChildren(QSpinBox)
                 if (s.minimum(), s.maximum()) == MONTAGE_COLUMNS_RANGE]
        assert spins, "no control for the montage row length"
        spins[0].setValue(5)
        dialog.findChildren(QDialogButtonBox)[0].accepted.emit()
        assert get_montage_columns() == 5
    finally:
        dialog.deleteLater()
        qapp.processEvents()
