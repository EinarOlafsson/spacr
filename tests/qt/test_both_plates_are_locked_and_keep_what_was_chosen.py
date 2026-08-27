"""Both plate viewers, measured together under one hostile stylesheet.

TWO PLATES ARE DRAWN IN THIS APPLICATION and they have already drifted apart
once. ``spacr.qt.widgets.plate_map_picker`` picks wells for a setting;
``spacr.qt.screens.experiment_design`` draws the plate being designed. The
first lost its square to the theme -- ``QPushButton { min-height: 22px;
padding: 8px 12px }`` is not merely drawn by ``QStyleSheetStyle``, it is
polished into a real ``setMinimumHeight`` that OVERWRITES the minimum
``setFixedSize`` wrote, leaving a well whose minimum outranks its maximum --
and the second was saved only by the absence of a blanket ``QLabel`` rule
for the same machinery to polish in.

So every geometry claim here is made with the application themed AND a
hostile blanket rule appended, ``QLabel { min-height: 60px; padding: 9px }``,
and both plates are built inside the same rule. That is the mechanism rather
than today's numbers: 60 plus twice 9 is 78, and a cell that answers to it is
78 px tall in a grid pitched for 22.

The guards each plate already carries measure ONE of them. This file exists
to measure them against the same sheet at the same time, because "it is not
broken today" and "it cannot break" are different claims and only the second
one is worth having.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

pytestmark = pytest.mark.qt

from spacr.qt.screens.experiment_design import (WELL_SIDE,
                                                ExperimentDesignScreen)
from spacr.qt.widgets.plate_map_picker import PlateMapPicker

#: The two window widths every geometry claim is checked at. The fault this
#: guards only appears once the layout has spare space to hand out.
NARROW, WIDE = 900, 1900

#: A blanket rule of exactly the shape that took the first plate's square
#: away, aimed at the type the second plate's cells are.
HOSTILE = "QLabel { min-height: 60px; padding: 9px; }"


@pytest.fixture
def hostile_theme(qt_theme_applied):
    """The real application stylesheet with the blanket rule appended.

    APPENDED, NOT SUBSTITUTED. A guard that replaces the theme with its own
    hostile rule has measured a stylesheet nobody runs, and the interaction
    between the theme's own geometry rules and the blanket one is the half
    that decides the answer.
    """
    app = QApplication.instance()
    was = app.styleSheet()
    app.setStyleSheet(was + "\n" + HOSTILE)
    try:
        yield app
    finally:
        app.setStyleSheet(was)


def _design(qtbot, width=WIDE, well_count=96):
    widget = ExperimentDesignScreen(threaded=False)
    qtbot.addWidget(widget)
    if well_count != 96:
        widget._format.setCurrentIndex(widget._format.findData(well_count))
    widget.resize(width, 800)
    widget.show()
    QApplication.processEvents()
    return widget


def _picker(qtbot, width=WIDE, layout=96):
    widget = PlateMapPicker(layout=layout)
    qtbot.addWidget(widget)
    widget.resize(width, 800)
    widget.show()
    QApplication.processEvents()
    return widget


# -- reading the cells back off whichever plate it is ----------------------
#
# The two plates hold their wells differently -- a list of labels against a
# dict of buttons -- and the point of this file is to make one claim about
# both, so each is read into the same shape here.

def _design_cells(screen) -> dict:
    cells = {("well", well.row, well.column): well
             for well in screen._well_labels}
    rows = max(row for _, row, _ in cells)
    columns = max(column for _, _, column in cells)
    cells.update({("column header", 0, c):
                  screen._plate_grid.itemAtPosition(0, c).widget()
                  for c in range(1, columns + 1)})
    cells.update({("row header", r, 0):
                  screen._plate_grid.itemAtPosition(r, 0).widget()
                  for r in range(1, rows + 1)})
    return cells


def _picker_cells(picker) -> dict:
    cells = {("well", row, column): well
             for (row, column), well in picker._wells.items()}
    rows = max(row for _, row, _ in cells)
    columns = max(column for _, _, column in cells)
    cells.update({("column header", 0, c):
                  picker._grid.itemAtPosition(0, c).widget()
                  for c in range(1, columns + 1)})
    cells.update({("row header", r, 0):
                  picker._grid.itemAtPosition(r, 0).widget()
                  for r in range(1, rows + 1)})
    return cells


def _oblong(cells) -> dict:
    """Every cell that is not ``WELL_SIDE`` square, with what it measured."""
    return {name: (cell.width(), cell.height())
            for name, cell in cells.items()
            if (cell.width(), cell.height()) != (WELL_SIDE, WELL_SIDE)}


# --------------------------------------------------------------------------
# 1: neither plate can be unlocked by a blanket rule
# --------------------------------------------------------------------------

class TestNeitherPlateCanBeUnlocked:

    @pytest.mark.parametrize("width", [NARROW, WIDE])
    def test_every_cell_of_the_design_plate_is_square(self, qtbot,
                                                      hostile_theme, width):
        # HELD IN A NAME. `qtbot.addWidget` keeps only a weak reference, so
        # a plate read straight out of the call that built it is collected
        # mid-measurement and every cell answers "already deleted".
        plate = _design(qtbot, width=width)
        cells = _design_cells(plate)

        assert len(cells) == 96 + 8 + 12
        assert not _oblong(cells), \
            f"the blanket rule unlocked the design plate: {_oblong(cells)}"

    @pytest.mark.parametrize("width", [NARROW, WIDE])
    def test_every_cell_of_the_picker_plate_is_square(self, qtbot,
                                                      hostile_theme, width):
        plate = _picker(qtbot, width=width)
        cells = _picker_cells(plate)

        assert len(cells) == 96 + 8 + 12
        assert not _oblong(cells), \
            f"the blanket rule unlocked the picker plate: {_oblong(cells)}"

    def test_both_header_strips_on_both_plates_hold_it_too(self, qtbot,
                                                           hostile_theme):
        """A strip is a row of CELLS, not a caption beside the plate: one
        header inflated to 78 px tall carries every number 56 px off the
        column it names while the wells below it are still square."""
        strips = {}
        plates = ((_design(qtbot), _design_cells),
                  (_picker(qtbot), _picker_cells))
        for plate, read in plates:
            cells = read(plate)
            strips[type(plate).__name__] = {
                "column": {cell.height() for name, cell in cells.items()
                           if name[0] == "column header"},
                "row": {cell.width() for name, cell in cells.items()
                        if name[0] == "row header"},
            }

        assert strips == {
            "ExperimentDesignScreen": {"column": {WELL_SIDE},
                                       "row": {WELL_SIDE}},
            "PlateMapPicker": {"column": {WELL_SIDE}, "row": {WELL_SIDE}},
        }

    @pytest.mark.parametrize("well_count", [6, 96, 384, 1536])
    def test_every_plate_format_holds_on_both(self, qtbot, hostile_theme,
                                              well_count):
        screen, dialog = (_design(qtbot, well_count=well_count),
                          _picker(qtbot, layout=well_count))
        design, picker = _design_cells(screen), _picker_cells(dialog)

        assert not _oblong(design), f"design at {well_count}"
        assert not _oblong(picker), f"picker at {well_count}"

    def test_the_lock_survives_the_user_changing_the_format(self, qtbot,
                                                            hostile_theme):
        """The plate is rebuilt from scratch on every format change, so the
        square has to be restated rather than merely set up once."""
        screen = _design(qtbot)
        for well_count in (384, 6, 1536, 96):
            screen._format.setCurrentIndex(
                screen._format.findData(well_count))
            QApplication.processEvents()

            assert not _oblong(_design_cells(screen)), f"after {well_count}"


# --------------------------------------------------------------------------
# 2: what the user chose survives the redraw
# --------------------------------------------------------------------------
#
# Every well of the design plate is destroyed and rebuilt on every `refresh`,
# and `refresh` is wired to `textChanged` on the plate name, `valueChanged`
# on the seed and `itemChanged` on the condition table. A selection that
# lived only in the widgets was therefore wiped by typing.

class TestTheSelectionOutlivesTheRedraw:

    def test_typing_in_the_plate_name_does_not_wipe_the_selection(self,
                                                                  qtbot):
        screen = _design(qtbot)
        screen.begin_well_drag(2, 2, Qt.NoModifier)
        screen.drag_wells_to(_centre_of(screen, 4, 5))
        screen.finish_well_drag()
        chosen = screen.selected_wells()
        assert len(chosen) == 12

        screen._plate_id.setText("plate1x")
        QApplication.processEvents()

        assert screen.selected_wells() == chosen

    def test_nor_does_nudging_the_seed(self, qtbot):
        screen = _design(qtbot)
        screen.begin_well_drag(3, 3, Qt.NoModifier)
        screen.finish_well_drag()

        screen._seed.setValue(7)
        QApplication.processEvents()

        assert screen.selected_wells() == {(3, 3)}

    def test_nor_does_editing_the_conditions(self, qtbot):
        screen = _design(qtbot)
        screen.begin_well_drag(3, 3, Qt.NoModifier)
        screen.finish_well_drag()

        screen._add_row()
        QApplication.processEvents()

        assert screen.selected_wells() == {(3, 3)}

    def test_the_kept_wells_are_drawn_chosen_and_still_square(self, qtbot,
                                                              hostile_theme):
        """Kept as coordinates and put back on the widgets as a property the
        sheet selects on -- and the selection is drawn as a rim, which is
        part of the widget's size."""
        screen = _design(qtbot)
        screen.begin_well_drag(3, 3, Qt.NoModifier)
        screen.finish_well_drag()

        screen._plate_id.setText("plate1x")
        QApplication.processEvents()

        drawn = {(well.row, well.column) for well in screen._well_labels
                 if well.property("spacrWellChosen") == "true"}
        assert drawn == {(3, 3)}
        assert not _oblong(_design_cells(screen))

    def test_a_smaller_plate_forgets_the_wells_it_no_longer_has(self, qtbot):
        """H13 is a well on a 384 plate and nothing at all on a 96, so it is
        dropped rather than carried as a coordinate naming no widget."""
        screen = _design(qtbot, well_count=384)
        screen.begin_well_drag(9, 20, Qt.NoModifier)
        screen.finish_well_drag()
        assert screen.selected_wells() == {(9, 20)}

        screen._format.setCurrentIndex(screen._format.findData(96))
        QApplication.processEvents()

        assert screen.selected_wells() == set()

    def test_and_going_back_up_does_not_resurrect_it(self, qtbot):
        screen = _design(qtbot, well_count=384)
        screen.begin_well_drag(9, 20, Qt.NoModifier)
        screen.finish_well_drag()

        screen._format.setCurrentIndex(screen._format.findData(96))
        QApplication.processEvents()
        screen._format.setCurrentIndex(screen._format.findData(384))
        QApplication.processEvents()

        assert screen.selected_wells() == set()

    def test_nor_can_a_later_ctrl_drag_bring_it_back(self, qtbot):
        """A Ctrl drag adds its rectangle to the state at the press, and
        that state is remembered separately -- so it has to forget the
        dropped wells too, or the union puts them back."""
        screen = _design(qtbot, well_count=384)
        screen.begin_well_drag(9, 20, Qt.NoModifier)
        screen.finish_well_drag()

        screen._format.setCurrentIndex(screen._format.findData(96))
        QApplication.processEvents()
        screen.begin_well_drag(1, 1, Qt.ControlModifier)
        screen.finish_well_drag()

        assert screen.selected_wells() == {(1, 1)}


def _centre_of(screen, row, column):
    well = next(well for well in screen._well_labels
                if (well.row, well.column) == (row, column))
    return well.mapToGlobal(well.rect().center())


# --------------------------------------------------------------------------
# 3: the gesture lands the same well on both plates
# --------------------------------------------------------------------------
#
# Driven with real Qt mouse events on real cells, never by calling the drag
# methods: hit-testing is where a geometry change shows up, because the well
# under the pointer is found by asking every cell whether a GLOBAL point is
# inside it, and a cell that is not the size the grid pitched it at answers
# for its neighbour's pixels.

def _press(cell, modifiers=Qt.NoModifier):
    centre = cell.rect().center()
    QApplication.sendEvent(cell, QMouseEvent(
        QMouseEvent.MouseButtonPress, centre, cell.mapToGlobal(centre),
        Qt.LeftButton, Qt.LeftButton, modifiers))


def _wobble(cell, dx=1, dy=1):
    """The pixel or two of pointer travel every real click carries."""
    local = cell.rect().center() + QPoint(dx, dy)
    QApplication.sendEvent(cell, QMouseEvent(
        QMouseEvent.MouseMove, local, cell.mapToGlobal(local),
        Qt.NoButton, Qt.LeftButton, Qt.NoModifier))


def _release(cell):
    centre = cell.rect().center()
    QApplication.sendEvent(cell, QMouseEvent(
        QMouseEvent.MouseButtonRelease, centre, cell.mapToGlobal(centre),
        Qt.LeftButton, Qt.NoButton, Qt.NoModifier))


class TestAPressWithNoMovementLandsOneWell:

    def test_on_the_design_plate_under_the_blanket_rule(self, qtbot,
                                                        hostile_theme):
        screen = _design(qtbot)
        well = next(well for well in screen._well_labels
                    if (well.row, well.column) == (6, 6))

        _press(well)
        _release(well)
        QApplication.processEvents()

        assert screen.selected_wells() == {(6, 6)}

    def test_on_the_picker_plate_under_the_blanket_rule(self, qtbot,
                                                        hostile_theme):
        picker = _picker(qtbot)

        _press(picker._wells[(6, 6)])
        _release(picker._wells[(6, 6)])
        QApplication.processEvents()

        assert picker.selection() == {(6, 6)}


class TestAWobbleChangesNothingAClickWouldNot:
    """No hand holds still, and a move that never leaves the well pressed on
    is not a drag. Read as one it draws a one-well rectangle, which on the
    picker also hands the release to the click and toggles the well straight
    back off -- so a wobbled click lands nothing AND wipes the plate the user
    had built up a well at a time."""

    def test_the_picker_keeps_the_wells_already_chosen(self, qtbot,
                                                       hostile_theme):
        picker = _picker(qtbot)
        picker.set_selection({(1, 1), (1, 2), (8, 12)})

        well = picker._wells[(6, 6)]
        _press(well)
        _wobble(well, dx=-1, dy=2)
        _release(well)
        QApplication.processEvents()

        assert picker.selection() == {(1, 1), (1, 2), (6, 6), (8, 12)}

    def test_the_design_plate_keeps_them_under_the_key_that_adds(
            self, qtbot, hostile_theme):
        """This plate REPLACES on a plain press, which is the gesture its
        own drag uses; Ctrl is what adds. The claim that matters is the same
        one either way: the wobble did not change the answer."""
        screen = _design(qtbot)
        wells = {(well.row, well.column): well
                 for well in screen._well_labels}
        _press(wells[(1, 1)], Qt.ControlModifier)
        _release(wells[(1, 1)])

        _press(wells[(6, 6)], Qt.ControlModifier)
        _wobble(wells[(6, 6)], dx=-1, dy=2)
        _release(wells[(6, 6)])
        QApplication.processEvents()

        assert screen.selected_wells() == {(1, 1), (6, 6)}

    def test_and_a_wobbled_press_answers_exactly_as_a_still_one(self, qtbot,
                                                                hostile_theme):
        """Measured rather than reasoned about: the same gesture with and
        without the wobble, on the same plate, compared."""
        still = _design(qtbot)
        wells = {(w.row, w.column): w for w in still._well_labels}
        _press(wells[(6, 6)])
        _release(wells[(6, 6)])
        QApplication.processEvents()

        wobbled = _design(qtbot)
        wells = {(w.row, w.column): w for w in wobbled._well_labels}
        _press(wells[(6, 6)])
        _wobble(wells[(6, 6)])
        _release(wells[(6, 6)])
        QApplication.processEvents()

        assert wobbled.selected_wells() == still.selected_wells() == {(6, 6)}


# --------------------------------------------------------------------------
# 4: what the two plates share, they share by importing
# --------------------------------------------------------------------------

class TestTheSharedPartsAreOneCopy:
    """The two drifted apart once already: the same square, the same
    constant and the same header cell written twice, and only one of the
    copies repaired when the fault was found."""

    def test_one_well_side(self):
        from spacr.qt.screens import experiment_design
        from spacr.qt.widgets import plate_map_picker

        assert (experiment_design.WELL_SIDE
                is plate_map_picker.WELL_SIDE)

    def test_one_locked_square(self):
        from spacr.qt.screens import experiment_design
        from spacr.qt.widgets import plate_map_picker

        assert (experiment_design._locked_square
                is plate_map_picker._locked_square)

    def test_one_header_cell(self):
        from spacr.qt.screens import experiment_design
        from spacr.qt.widgets import plate_map_picker

        assert experiment_design._Header is plate_map_picker._Header

    def test_the_square_is_stated_in_content_box_pixels(self):
        """The shared statement is what the lock is, so it is read here
        rather than trusted: Qt adds the border, the padding and the margin
        back on to whatever ``min-`` and ``max-`` ask for, so a cell that
        asked for ``WELL_SIDE`` while drawing a rim would come out wider
        than the grid is pitched for by twice that rim."""
        from spacr.qt.widgets.plate_map_picker import _locked_square

        assert f"min-width: {WELL_SIDE}px" in _locked_square(0)
        assert f"max-height: {WELL_SIDE}px" in _locked_square(0)
        assert f"min-width: {WELL_SIDE - 4}px" in _locked_square(2)
        assert "padding: 0px" in _locked_square(2)
        assert "margin: 0px" in _locked_square(2)


#: A rule the padding-and-height guard above does not answer. Qt adds the
#: BORDER back on top of padding, margin and the min/max box alike, so a
#: sheet that only states a border is a separate way to take the square.
HOSTILE_RIM = "QLabel { border: 5px solid red; }"


@pytest.fixture
def hostile_rim(qt_theme_applied):
    app = QApplication.instance()
    was = app.styleSheet()
    app.setStyleSheet(was + "\n" + HOSTILE_RIM)
    try:
        yield app
    finally:
        app.setStyleSheet(was)


class TestABorderIsTakenBackOffToo:
    """The picker's headers pinned everything except the border width.

    `experiment_design._plate_header` already stated `border-width: 0px` for
    exactly this reason; `plate_map_picker._Header` did not, so under a
    border-only sheet its 22 px squares became 32 px and every row letter
    drifted off the row it names.
    """

    def test_the_pickers_headers_keep_their_square(self, qtbot, hostile_rim):
        from spacr.qt.widgets.plate_map_picker import _Header

        header = _Header("A")
        qtbot.addWidget(header)
        header.show()
        QApplication.processEvents()
        header.ensurePolished()
        assert header.size().height() == WELL_SIDE
        assert header.size().width() == WELL_SIDE

    def test_the_wells_keep_theirs(self, qtbot, hostile_rim):
        """The wells are buttons, so the label rule should never have reached
        them -- asserted so the header fix is not credited for it."""
        from spacr.qt.widgets.plate_map_picker import _Well

        well = _Well(0, 0)
        qtbot.addWidget(well)
        well.show()
        QApplication.processEvents()
        well.ensurePolished()
        assert well.size().height() == WELL_SIDE

    def test_a_whole_picker_stays_pitched(self, qtbot, hostile_rim):
        picker = PlateMapPicker()
        qtbot.addWidget(picker)
        picker.show()
        QApplication.processEvents()
        for child in picker.findChildren(type(picker)):
            pass
        from spacr.qt.widgets.plate_map_picker import _Header
        heads = picker.findChildren(_Header)
        assert heads, "the picker drew no headers to measure"
        sizes = {(h.size().width(), h.size().height()) for h in heads}
        assert sizes == {(WELL_SIDE, WELL_SIDE)}, sizes


class TestTheTwoPlatesShareOneDefinition:
    """They drifted once. These pin what must stay one copy.

    NOT the cell class, and that is measured rather than assumed. Building
    1,536 cells takes 0.472 s as a QLabel and 0.751 s as a QPushButton --
    59% more, on a workstation, for a plate that draws every well. The
    picker's wells are CHECKABLE, so their state IS the selection and a
    button is the right shape; the design map is a picture that reports
    clicks, and 1,536 checkable buttons is a heavier answer to a lighter
    question. Merging them would contradict instruction 268, which is about
    exactly this cost on a slow machine.
    """

    def test_one_well_side(self):
        from spacr.qt.screens import experiment_design
        from spacr.qt.widgets import plate_map_picker

        assert experiment_design.WELL_SIDE is plate_map_picker.WELL_SIDE

    def test_the_design_map_declares_no_well_side_of_its_own(self):
        import inspect

        from spacr.qt.screens import experiment_design

        source = inspect.getsource(experiment_design)
        assert "\nWELL_SIDE =" not in source, (
            "the second plate declared its own again; they drifted once")

    def test_one_locked_square_helper(self):
        from spacr.qt.screens import experiment_design
        from spacr.qt.widgets import plate_map_picker

        assert (experiment_design._locked_square
                is plate_map_picker._locked_square)

    def test_one_header_class(self):
        from spacr.qt.screens import experiment_design
        from spacr.qt.widgets import plate_map_picker

        assert experiment_design._Header is plate_map_picker._Header

    def test_the_design_well_states_its_own_square(self):
        """The whole point of the item: locked by construction rather than
        by the absence of a blanket QLabel rule."""
        import inspect

        from spacr.qt.screens import experiment_design

        sheet = inspect.getsource(experiment_design._well_sheet)
        assert "_locked_square" in sheet
        assert "border-width" in sheet
