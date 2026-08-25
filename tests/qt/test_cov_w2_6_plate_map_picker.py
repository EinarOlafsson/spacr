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


# --------------------------------------------------------------------------
# the drag, through real mouse events on a well that can reach its picker
# --------------------------------------------------------------------------

def test_a_press_anchors_the_drag_on_the_well_it_landed_on(wired_well,
                                                           picker):
    _press(wired_well)
    assert picker._anchor == (3, 4)
    assert picker._adding is False


def test_holding_ctrl_at_the_press_makes_the_drag_add(wired_well, picker):
    picker.set_selection({(8, 11)})
    _press(wired_well, modifiers=Qt.ControlModifier)
    assert picker._adding is True
    assert picker._before == {(8, 11)}


def test_a_move_while_pressed_previews_the_rectangle(wired_well, picker):
    """A selection you cannot see until you let go is one you have to undo
    to correct."""
    _press(wired_well)
    _move_to(wired_well, picker._wells[(5, 6)])
    assert picker.selection() == {(r, c) for r in (3, 4, 5)
                                  for c in (4, 5, 6)}


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


def test_dragging_across_the_plate_selects_the_rectangle(picker, qapp):
    """The gesture through a REAL well of the grid, whose holder sits in the
    scroll area's viewport -- the route the fixed-step parent walk broke."""
    start, corner = picker._wells[(2, 2)], picker._wells[(4, 5)]
    _press(start)
    _move_to(start, corner)
    _release(start)
    qapp.processEvents()
    assert picker.selection() == {(r, c) for r in (2, 3, 4)
                                  for c in (2, 3, 4, 5)}


def test_a_press_and_release_with_no_movement_is_still_a_click(picker, qapp):
    well = picker._wells[(6, 6)]
    assert not well.isChecked()
    _press(well)
    _release(well)
    qapp.processEvents()
    assert well.isChecked()
    assert picker.selection() == {(6, 6)}


# --------------------------------------------------------------------------
# the drag API, driven directly
# --------------------------------------------------------------------------

def test_a_drag_replaces_the_selection_unless_ctrl_is_held(picker):
    picker.set_selection({(8, 11)})
    picker.begin_drag(1, 1)
    picker.drag_to(picker._wells[(2, 2)].mapToGlobal(QPoint(2, 2)))
    picker.finish_drag()
    assert picker.selection() == {(1, 1), (1, 2), (2, 1), (2, 2)}


def test_ctrl_dragging_adds_to_what_is_already_chosen(picker):
    picker.set_selection({(8, 11)})
    picker.begin_drag(1, 1, Qt.ControlModifier)
    picker.drag_to(picker._wells[(2, 2)].mapToGlobal(QPoint(2, 2)))
    picker.finish_drag()
    assert (8, 11) in picker.selection()
    assert {(1, 1), (1, 2), (2, 1), (2, 2)} <= picker.selection()


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


def test_a_move_over_the_same_well_does_not_redraw(picker):
    picker.begin_drag(2, 2)
    point = picker._wells[(2, 4)].mapToGlobal(QPoint(2, 2))
    picker.drag_to(point)
    before = picker.selection()
    picker.set_selection(set())
    picker.drag_to(point)
    assert picker.selection() == set() != before


def test_a_move_outside_the_grid_leaves_the_preview_alone(picker):
    picker.begin_drag(2, 2)
    picker.drag_to(picker._wells[(2, 4)].mapToGlobal(QPoint(2, 2)))
    before = picker.selection()
    picker.drag_to(picker.mapToGlobal(QPoint(-500, -500)))
    assert picker.selection() == before


def test_a_move_with_no_press_behind_it_does_nothing(picker):
    picker.set_selection({(1, 1)})
    picker._anchor = None
    picker.drag_to(picker._wells[(3, 3)].mapToGlobal(QPoint(2, 2)))
    assert picker.selection() == {(1, 1)}


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

def test_a_region_is_selected_whichever_corner_it_is_given_from(picker):
    picker.select_region((5, 6), (3, 4), choosing=True)
    assert picker.selection() == {(r, c) for r in (3, 4, 5) for c in (4, 5, 6)}


def test_a_region_can_be_cleared_as_well_as_chosen(picker):
    picker.select_region((1, 1), (3, 3), choosing=True)
    picker.select_region((2, 2), (3, 3), choosing=False)
    assert picker.selection() == {(1, 1), (1, 2), (1, 3), (2, 1), (3, 1)}


def test_a_region_with_no_verdict_inverts_the_well_it_started_on(picker):
    """Spreadsheet-style: dragging from a chosen well clears, dragging from
    an unchosen one selects."""
    picker.select_region((1, 1), (2, 2))
    assert picker.selection() == {(1, 1), (1, 2), (2, 1), (2, 2)}
    picker.select_region((1, 1), (2, 2))
    assert picker.selection() == set()


def test_a_region_that_runs_off_the_plate_selects_what_is_there(picker):
    picker.select_region((7, 11), (20, 40), choosing=True)
    assert picker.selection() == {(7, 11), (7, 12), (8, 11), (8, 12)}


# --------------------------------------------------------------------------
# the layout
# --------------------------------------------------------------------------

def test_the_plate_button_offers_every_supported_layout(picker, monkeypatch):
    from spacr.well_spec import LAYOUTS

    seen = {}

    def _ask(parent, title, label, items, current, editable):
        seen["items"] = list(items)
        seen["current"] = current
        return "384", True

    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(_ask))
    assert picker.ask_for_layout() == 384
    assert seen["items"] == [str(size) for size in sorted(LAYOUTS)]
    assert seen["items"][seen["current"]] == "96"


def test_cancelling_the_plate_dialog_keeps_the_layout(picker, monkeypatch):
    monkeypatch.setattr(QInputDialog, "getItem",
                        staticmethod(lambda *a, **k: ("384", False)))
    assert picker.ask_for_layout() == 96
    assert picker._layout_size == 96


def test_an_unknown_current_layout_falls_back_to_the_default(picker,
                                                             monkeypatch):
    """The dialog still has to open on something, and the default is the
    only size that is always in the list."""
    from spacr.well_spec import DEFAULT_LAYOUT, LAYOUTS

    seen = {}

    def _ask(parent, title, label, items, current, editable):
        seen["current"] = current
        return items[0], True

    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(_ask))
    picker._layout_size = 999
    picker.ask_for_layout()
    assert seen["current"] == sorted(LAYOUTS).index(DEFAULT_LAYOUT)


def test_the_plate_button_opens_the_layout_dialog(picker, monkeypatch, qapp):
    monkeypatch.setattr(QInputDialog, "getItem",
                        staticmethod(lambda *a, **k: ("6", True)))
    picker.plate_button.click()
    qapp.processEvents()
    assert picker._layout_size == 6


def test_a_layout_change_says_how_many_wells_it_dropped(picker):
    """A layout-induced selection change has to be visible; wells that
    silently vanish are wells the user still believes are chosen."""
    picker.set_selection({(1, 1), (8, 12)})
    picker.ask_for_layout(6)
    assert picker.selection() == {(1, 1)}
    assert "1 well(s) from the previous selection" in picker._caption.text()
    assert "6-well plate (2 x 3)" in picker._caption.text()


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


def test_a_well_chosen_without_a_click_still_looks_chosen(picker):
    """`set_selection` blocks the wells' signals so the count is said once
    rather than once per well, which also silences `toggled -> _paint`: it
    repaints itself, or a chosen well is drawn empty."""
    picker.set_selection({(1, 1)})
    assert picker._wells[(1, 1)].isChecked()
    assert CHOSEN in picker._wells[(1, 1)].styleSheet()
