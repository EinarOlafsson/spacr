"""The plate viewer is ONE grid: locked, square, and dragged across.

    "all elements of the plate locked to each other (wells and column and
    row axis labels) ... the wells should be squares not rectangles ... the
    user should be able to click and drag to select not only click."

MEASURED AT TWO WINDOW SIZES, because the fault this guards is one that
only appears when the layout has spare space to hand out. Wells pinned to a
square side ask for a fixed amount; a grid with nowhere to put the rest
shares it between the cells, and the column numbers and row letters slide
away from the wells they name as the window grows. The numbers below are
read back off the widgets rather than looked at: a label's centre against
its well's centre, a well's width against its own height.

The drag is driven with real Qt mouse events on real wells of the grid,
never by calling the drag methods, because the route from the button to the
dialog is the part that goes missing: a scroll area reparents the widget it
is given into its own viewport, so a well that looks for its picker a fixed
number of parents up finds that viewport and every drag guard reads False.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QScrollArea

pytestmark = pytest.mark.qt

from spacr.qt.widgets.plate_map_picker import (CHOSEN, EMPTY, WELL_SIDE,
                                               PlateMapPicker)

#: The two window widths every geometry claim is checked at.
NARROW, WIDE = 700, 1400


@pytest.fixture
def picker(qtbot):
    widget = PlateMapPicker(layout=96)
    qtbot.addWidget(widget)
    widget.resize(NARROW, 600)
    widget.show()
    QApplication.processEvents()
    return widget


def _sized(picker, width):
    picker.resize(width, 600)
    QApplication.processEvents()
    return picker


def _centre_x(picker, widget):
    return widget.mapTo(picker, widget.rect().center()).x()


def _centre_y(picker, widget):
    return widget.mapTo(picker, widget.rect().center()).y()


def _column_label(picker, column):
    return picker._grid.itemAtPosition(0, column).widget()


def _row_label(picker, row):
    return picker._grid.itemAtPosition(row, 0).widget()


def _press(well, modifiers=Qt.NoModifier):
    centre = well.rect().center()
    QApplication.sendEvent(well, QMouseEvent(
        QMouseEvent.MouseButtonPress, centre, well.mapToGlobal(centre),
        Qt.LeftButton, Qt.LeftButton, modifiers))


def _move_over(source, target):
    """A move whose GLOBAL position is over ``target``, delivered to the
    pressed widget -- which is where Qt sends it, because the press grabbed
    the mouse and the well under the pointer never hears about it."""
    centre = target.rect().center()
    QApplication.sendEvent(source, QMouseEvent(
        QMouseEvent.MouseMove,
        source.mapFromGlobal(target.mapToGlobal(centre)),
        target.mapToGlobal(centre), Qt.NoButton, Qt.LeftButton,
        Qt.NoModifier))


def _release(well):
    centre = well.rect().center()
    QApplication.sendEvent(well, QMouseEvent(
        QMouseEvent.MouseButtonRelease, centre, well.mapToGlobal(centre),
        Qt.LeftButton, Qt.NoButton, Qt.NoModifier))


# --------------------------------------------------------------------------
# 1: every part of the plate is locked to every other part
# --------------------------------------------------------------------------

class TestTheLabelsAreLockedToTheWells:

    @pytest.mark.parametrize("width", [NARROW, WIDE])
    def test_every_column_number_sits_over_its_column(self, picker, width):
        _sized(picker, width)

        adrift = {column: (_centre_x(picker, _column_label(picker, column)),
                           _centre_x(picker, picker._wells[(1, column)]))
                  for column in range(1, 13)
                  if _centre_x(picker, _column_label(picker, column))
                  != _centre_x(picker, picker._wells[(1, column)])}

        assert not adrift, f"labels off their columns at {width} px: {adrift}"

    @pytest.mark.parametrize("width", [NARROW, WIDE])
    def test_every_row_letter_sits_beside_its_row(self, picker, width):
        _sized(picker, width)

        adrift = {row: (_centre_y(picker, _row_label(picker, row)),
                        _centre_y(picker, picker._wells[(row, 1)]))
                  for row in range(1, 9)
                  if _centre_y(picker, _row_label(picker, row))
                  != _centre_y(picker, picker._wells[(row, 1)])}

        assert not adrift, f"letters off their rows at {width} px: {adrift}"

    def test_a_label_stays_one_cell_from_the_plate_at_any_width(self, picker):
        """The failure this is aimed at: the labels keep their register with
        the wells while being pushed a window's width away from them, which
        is three things near each other rather than one grid."""
        step = WELL_SIDE + picker._grid.horizontalSpacing()

        def gap():
            return (_centre_x(picker, picker._wells[(1, 1)])
                    - _centre_x(picker, _row_label(picker, 1)))

        _sized(picker, NARROW)
        assert gap() == step

        _sized(picker, WIDE)
        assert gap() == step

    def test_the_numbers_stay_one_cell_above_the_plate_too(self, picker):
        step = WELL_SIDE + picker._grid.verticalSpacing()

        def gap():
            return (_centre_y(picker, picker._wells[(1, 1)])
                    - _centre_y(picker, _column_label(picker, 1)))

        _sized(picker, NARROW)
        assert gap() == step

        _sized(picker, WIDE)
        assert gap() == step

    def test_no_well_moves_relative_to_any_other(self, picker):
        """700 px of extra window must not put one pixel between two
        wells."""
        def step():
            a, right, down = (picker._wells[(1, 1)], picker._wells[(1, 2)],
                              picker._wells[(2, 1)])
            return (_centre_x(picker, right) - _centre_x(picker, a),
                    _centre_y(picker, down) - _centre_y(picker, a))

        _sized(picker, NARROW)
        before = step()

        _sized(picker, WIDE)

        assert step() == before
        assert before == (WELL_SIDE + picker._grid.horizontalSpacing(),
                          WELL_SIDE + picker._grid.verticalSpacing())

    def test_the_labels_and_the_wells_are_carried_by_one_widget(self, picker):
        """So they scroll together: one holder in the scroll area, not a
        label strip beside a plate."""
        holder = picker._wells[(1, 1)].parentWidget()

        assert _column_label(picker, 1).parentWidget() is holder
        assert _row_label(picker, 1).parentWidget() is holder
        assert picker.findChild(QScrollArea).widget() is holder

    def test_the_lock_survives_a_layout_change(self, picker):
        """A rebuilt grid keeps no stretch from the one before it, or the
        spare space is left sitting in the middle of the new plate."""
        picker.ask_for_layout(1536)
        picker.ask_for_layout(6)
        _sized(picker, WIDE)

        step = WELL_SIDE + picker._grid.horizontalSpacing()
        assert (_centre_x(picker, picker._wells[(1, 2)])
                - _centre_x(picker, picker._wells[(1, 1)])) == step
        assert (_centre_x(picker, picker._wells[(1, 1)])
                - _centre_x(picker, _row_label(picker, 1))) == step


# --------------------------------------------------------------------------
# 2: a well is a square
# --------------------------------------------------------------------------

class TestAWellIsASquare:

    @pytest.mark.parametrize("width", [NARROW, WIDE])
    def test_every_well_is_as_wide_as_it_is_tall(self, picker, width):
        _sized(picker, width)

        oblong = {cell: (well.width(), well.height())
                  for cell, well in picker._wells.items()
                  if well.width() != well.height()}

        assert not oblong, f"rectangles at {width} px: {oblong}"

    @pytest.mark.parametrize("width", [NARROW, WIDE])
    def test_a_well_is_the_same_square_at_either_size(self, picker, width):
        """The grid takes the space it needs and leaves the rest, rather
        than dividing whatever it is given."""
        _sized(picker, width)
        well = picker._wells[(4, 7)]

        assert (well.width(), well.height()) == (WELL_SIDE, WELL_SIDE)

    @pytest.mark.parametrize("layout", [6, 96, 1536])
    def test_every_layout_is_squares(self, qtbot, layout):
        widget = PlateMapPicker(layout=layout)
        qtbot.addWidget(widget)
        widget.resize(WIDE, 900)
        widget.show()
        QApplication.processEvents()

        assert all(well.width() == well.height() == WELL_SIDE
                   for well in widget._wells.values())

    def test_the_plate_can_be_scrolled_to_when_it_outgrows_the_window(
            self, qtbot):
        """Keeping the square costs width, and the wells that no longer fit
        have to be reachable rather than squeezed."""
        widget = PlateMapPicker(layout=1536)
        qtbot.addWidget(widget)
        widget.resize(NARROW, 600)
        widget.show()
        QApplication.processEvents()

        area = widget.findChild(QScrollArea)
        assert area.horizontalScrollBar().isVisible()
        assert widget._wells[(1, 48)].width() == WELL_SIDE


# --------------------------------------------------------------------------
# 3: click and drag selects a rectangle
# --------------------------------------------------------------------------

class TestPressMoveReleaseSelectsTheBlock:

    @pytest.mark.parametrize("width", [NARROW, WIDE])
    def test_dragging_from_b2_to_d5_selects_exactly_that_rectangle(
            self, picker, width):
        _sized(picker, width)
        start, corner = picker._wells[(2, 2)], picker._wells[(4, 5)]

        _press(start)
        _move_over(start, corner)
        _release(start)
        QApplication.processEvents()

        block = {(r, c) for r in (2, 3, 4) for c in (2, 3, 4, 5)}
        assert len(block) == 12
        assert picker.selection() == block

    def test_the_wells_outside_the_rectangle_are_left_alone(self, picker):
        start, corner = picker._wells[(2, 2)], picker._wells[(4, 5)]

        _press(start)
        _move_over(start, corner)
        _release(start)
        QApplication.processEvents()

        assert not picker._wells[(2, 1)].isChecked()
        assert not picker._wells[(5, 5)].isChecked()
        assert not picker._wells[(1, 1)].isChecked()

    def test_a_swept_well_is_painted_as_it_is_swept(self, picker):
        """A selection you cannot see until you let go is one you have to
        undo to correct."""
        start = picker._wells[(2, 2)]

        _press(start)
        _move_over(start, picker._wells[(4, 5)])

        assert CHOSEN in picker._wells[(3, 4)].styleSheet()

    def test_a_press_with_no_move_is_still_an_ordinary_click(self, picker):
        well = picker._wells[(6, 6)]

        _press(well)
        _release(well)
        QApplication.processEvents()

        assert picker.selection() == {(6, 6)}

    def test_the_value_the_field_gets_back_is_the_rectangle(self, picker):
        """What the picker writes, `well_spec` reads."""
        from spacr.well_spec import parse

        start = picker._wells[(2, 2)]
        _press(start)
        _move_over(start, picker._wells[(4, 5)])
        _release(start)
        QApplication.processEvents()

        assert parse(picker.value(), 96) == picker.selection()

    def test_a_well_in_the_grid_can_reach_the_picker_that_owns_it(
            self, picker):
        """The one line the whole gesture hung on."""
        assert picker._wells[(2, 2)]._picker() is picker


# --------------------------------------------------------------------------
# 4: a well chosen without a click looks chosen
# --------------------------------------------------------------------------

class TestAChosenWellIsPainted:

    def test_set_selection_paints_what_it_chose(self, picker):
        picker.set_selection({(1, 1)})

        assert picker._wells[(1, 1)].isChecked()
        assert CHOSEN in picker._wells[(1, 1)].styleSheet()

    def test_it_takes_the_well_names_the_field_is_written_in(self, picker):
        """`A01` is the vocabulary the setting uses, so the picker answers
        to it as readily as to a row and column."""
        picker.set_selection(["A01"])

        assert picker.selection() == {(1, 1)}
        assert CHOSEN in picker._wells[(1, 1)].styleSheet()

    def test_what_it_did_not_choose_is_painted_empty(self, picker):
        picker.set_selection({(1, 1)})
        picker.set_selection({(2, 2)})

        assert EMPTY in picker._wells[(1, 1)].styleSheet()
        assert CHOSEN in picker._wells[(2, 2)].styleSheet()

    def test_the_starting_value_opens_painted(self, qtbot):
        widget = PlateMapPicker(value="A01", layout=96)
        qtbot.addWidget(widget)

        assert CHOSEN in widget._wells[(1, 1)].styleSheet()

    def test_a_selection_kept_across_a_layout_change_stays_painted(
            self, picker):
        picker.set_selection({(1, 1)})

        picker.ask_for_layout(384)

        assert CHOSEN in picker._wells[(1, 1)].styleSheet()
