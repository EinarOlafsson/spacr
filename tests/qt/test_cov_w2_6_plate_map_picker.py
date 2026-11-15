"""Press, drag, release -- the gesture, driven through the real widgets.

`select_region` existed for a long time with nothing calling it from the
mouse: the rectangle a user expects from press-move-release was implemented
and unreachable. So the wiring tests here send real Qt mouse events at real
well buttons rather than calling the drag methods directly, because the
route between the two is the thing that goes missing.
"""
from __future__ import annotations

import pytest

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QInputDialog, QWidget

from spacr.qt.widgets.plate_map_picker import CHOSEN, EMPTY, PlateMapPicker, _Well


@pytest.fixture
def picker(qapp):
    dialog = PlateMapPicker(layout=96)
    dialog.show()
    qapp.processEvents()
    yield dialog
    dialog.close()
    dialog.deleteLater()


@pytest.fixture
def wired_well(picker):
    """One well whose parent chain reaches the picker, as ``_picker``
    expects: a holder that is a direct child of the dialog."""
    holder = QWidget(picker)
    well = _Well(3, 4, holder)
    yield well
    well.deleteLater()
    holder.deleteLater()


def _press(well, modifiers=Qt.NoModifier):
    centre = well.rect().center()
    QApplication.sendEvent(well, QMouseEvent(
        QMouseEvent.MouseButtonPress, centre, well.mapToGlobal(centre),
        Qt.LeftButton, Qt.LeftButton, modifiers))


def _move_to(source, target):
    """A move whose GLOBAL position is over ``target``, delivered to
    ``source`` -- which is what Qt does, because the press grabbed it."""
    centre = target.rect().center()
    QApplication.sendEvent(source, QMouseEvent(
        QMouseEvent.MouseMove,
        source.mapFromGlobal(target.mapToGlobal(centre)),
        target.mapToGlobal(centre), Qt.NoButton, Qt.LeftButton,
        Qt.NoModifier))


def _release(well):
    centre = well.rect().center()
    event = QMouseEvent(QMouseEvent.MouseButtonRelease, centre,
                        well.mapToGlobal(centre), Qt.LeftButton,
                        Qt.NoButton, Qt.NoModifier)
    QApplication.sendEvent(well, event)
    return event


# --------------------------------------------------------------------------
# a well has to be able to find its picker
# --------------------------------------------------------------------------

def test_a_well_finds_the_picker_it_belongs_to(picker):
    """The scroll area reparents the holder into its own VIEWPORT, so a
    walk of a fixed number of parents lands on that viewport instead of the
    dialog: `_picker` searches its ancestors for the PlateMapPicker."""
    assert picker._wells[(1, 1)]._picker() is picker


def test_a_well_with_no_holder_has_no_picker(qapp):
    orphan = _Well(1, 1)
    try:
        assert orphan._picker() is None
    finally:
        orphan.deleteLater()


def test_a_well_reaches_the_picker_when_its_holder_is_the_dialogs_child(
        wired_well, picker):
    assert wired_well._picker() is picker


def test_a_well_with_no_picker_is_still_an_ordinary_button(qapp):
    """A well outside a picker has nobody to run a drag for it, so press,
    move and release fall straight through to the button: it toggles, and
    nothing goes looking for a dialog that is not there."""
    orphan = _Well(1, 1)
    orphan.resize(22, 22)
    try:
        _press(orphan)
        _move_to(orphan, orphan)
        _release(orphan)

        assert orphan.isChecked()
        assert CHOSEN in orphan.styleSheet()
    finally:
        orphan.deleteLater()


# --------------------------------------------------------------------------
# the drag, through real mouse events on a well that can reach its picker
# --------------------------------------------------------------------------

def test_a_press_anchors_the_drag_on_the_well_it_landed_on(wired_well,
                                                           picker):
    _press(wired_well)
    assert picker._anchor == (3, 4)
    assert picker._adding is False


def test_a_release_after_a_drag_is_swallowed_by_the_well(wired_well, picker):
    """The rectangle is already painted; letting the click through would
    toggle the anchor a second time."""
    _press(wired_well)
    _move_to(wired_well, picker._wells[(5, 6)])
    assert _release(wired_well).isAccepted()
    assert picker._anchor is None


def test_a_release_with_no_drag_behind_it_is_still_an_ordinary_click(
        wired_well, picker):
    _press(wired_well)
    _release(wired_well)
    assert wired_well.isChecked()


# --------------------------------------------------------------------------
# the drag API, driven directly
# --------------------------------------------------------------------------


def test_growing_then_shrinking_a_drag_leaves_nothing_behind(picker):
    """Each preview redraws from the state at the PRESS, not from the last
    frame."""
    picker.begin_drag(3, 3)
    picker.drag_to(picker._wells[(6, 8)].mapToGlobal(QPoint(2, 2)))
    picker.drag_to(picker._wells[(4, 4)].mapToGlobal(QPoint(2, 2)))
    picker.finish_drag()
    assert picker.selection() == {(3, 3), (3, 4), (4, 3), (4, 4)}


def test_a_finished_drag_reports_that_one_happened(picker):
    picker.begin_drag(3, 3)
    picker.drag_to(picker._wells[(3, 5)].mapToGlobal(QPoint(2, 2)))
    assert picker.finish_drag() is True
    assert "3 well(s) chosen" in picker._caption.text()


def test_returning_to_the_anchor_is_still_a_drag(picker):
    """A gesture that has already left its well stays a drag, so the release
    is swallowed rather than clicking the anchor back off."""
    picker.begin_drag(2, 2)
    picker.drag_to(picker._wells[(4, 4)].mapToGlobal(QPoint(2, 2)))

    picker.drag_to(picker._wells[(2, 2)].mapToGlobal(QPoint(2, 2)))

    assert picker.selection() == {(2, 2)}
    assert picker.finish_drag() is True


def test_the_well_under_a_point_is_found_by_asking_the_grid(picker):
    """The pressed well keeps the mouse grab, so a sibling's `enterEvent`
    never fires and the well under the pointer has to be looked up."""
    target = picker._wells[(4, 7)]
    assert picker.well_at(target.mapToGlobal(
        target.rect().center())) == (4, 7)


def test_no_well_is_under_a_point_off_the_plate(picker):
    assert picker.well_at(picker.mapToGlobal(QPoint(-999, -999))) is None


def test_a_press_anchors_the_drag_where_it_started(picker):
    picker._begin(4, 7)
    assert picker._anchor == (4, 7)


def test_a_release_with_no_drag_behind_it_reports_none(picker):
    picker.begin_drag(1, 1)
    assert picker.finish_drag() is False
    assert picker._anchor is None


# --------------------------------------------------------------------------
# select_region
# --------------------------------------------------------------------------


def test_a_region_can_be_cleared_as_well_as_chosen(picker):
    picker.select_region((1, 1), (3, 3), choosing=True)
    picker.select_region((2, 2), (3, 3), choosing=False)
    assert picker.selection() == {(1, 1), (1, 2), (1, 3), (2, 1), (3, 1)}


# --------------------------------------------------------------------------
# the layout
# --------------------------------------------------------------------------


def test_the_plate_button_opens_the_layout_dialog(picker, monkeypatch, qapp):
    monkeypatch.setattr(QInputDialog, "getItem",
                        staticmethod(lambda *a, **k: ("6", True)))
    picker.plate_button.click()
    qapp.processEvents()
    assert picker._layout_size == 6


# --------------------------------------------------------------------------
# the value the field gets back
# --------------------------------------------------------------------------

def test_an_unparsable_starting_value_opens_the_picker_empty(qapp):
    """The picker is how a user fixes a value they typed wrong, so it must
    open rather than refuse."""
    dialog = PlateMapPicker(value="not a well spec", layout=96)
    try:
        assert dialog.selection() == set()
        assert dialog.value() == ""
    finally:
        dialog.deleteLater()


def test_a_starting_value_is_shown_as_the_selection(qapp):
    dialog = PlateMapPicker(value="A1,A2,A3", layout=96)
    try:
        assert dialog.selection() == {(1, 1), (1, 2), (1, 3)}
        assert "3 well(s) chosen" in dialog._caption.text()
        assert dialog.value() == "A01,A02,A03"
    finally:
        dialog.deleteLater()


# --------------------------------------------------------------------------
# what a chosen well looks like
# --------------------------------------------------------------------------

def test_clicking_a_well_fills_it_and_clicking_again_empties_it(picker, qapp):
    well = picker._wells[(1, 1)]
    assert EMPTY in well.styleSheet()
    well.click()
    qapp.processEvents()
    assert CHOSEN in well.styleSheet()
    well.click()
    qapp.processEvents()
    assert EMPTY in well.styleSheet()
