"""194: the plate map is squares you drag across, and it does not drift.

    "the plate map should be made up of squares that are clickable nad the
    user should be able to drag and select. and the elements should be fixed
    to each other now if i change the size of the window the elements drift
    appart"

THE DRIFT WAS MEASURABLE AND IS THE POINT. The design plate map's wells were
`QLabel`s with `setMinimumSize(22, 18)` and no maximum, so every one stretched
with the window: at 900, 1500 and 1900 px wide, a single well went 65 -> 111
-> 141 px across while staying 18 tall. A plate map is a picture of a physical
object, and the point of the picture is that its proportions are the object's.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot):
    from spacr.qt.screens.experiment_design import ExperimentDesignScreen

    widget = ExperimentDesignScreen()
    qtbot.addWidget(widget)
    widget.resize(1200, 800)
    widget.show()
    QApplication.processEvents()
    return widget


def _wells(screen) -> dict:
    return {(w.row, w.column): w for w in screen._well_labels}


def _centre(screen, cell):
    well = _wells(screen)[cell]
    return well.mapToGlobal(well.rect().center())


def _step(screen) -> tuple:
    """The origin-to-origin distance between neighbouring wells."""
    wells = _wells(screen)
    a, right, down = wells[(1, 1)], wells[(1, 2)], wells[(2, 1)]
    origin = a.mapTo(screen, a.rect().topLeft())
    return (right.mapTo(screen, right.rect().topLeft()).x() - origin.x(),
            down.mapTo(screen, down.rect().topLeft()).y() - origin.y())


# --------------------------------------------------------------------------- #
#  B: the elements are fixed to each other
# --------------------------------------------------------------------------- #

class TestItDoesNotDriftApart:

    def test_a_well_is_square(self, screen):
        well = _wells(screen)[(1, 1)]

        assert well.width() == well.height()

    @pytest.mark.parametrize("size", [(900, 600), (1500, 900), (1900, 1100)])
    def test_and_stays_square_at_every_window_size(self, screen, size):
        screen.resize(*size)
        QApplication.processEvents()

        well = _wells(screen)[(1, 1)]
        assert well.width() == well.height()

    def test_no_well_moves_relative_to_any_other(self, screen):
        """The reported fault, measured rather than looked at: 400 px of
        window must not put one pixel between two wells."""
        screen.resize(900, 600)
        QApplication.processEvents()
        before = _step(screen)

        screen.resize(1900, 1100)
        QApplication.processEvents()

        assert _step(screen) == before

    def test_the_step_is_the_well_plus_the_spacing(self, screen):
        """Not the well plus whatever the layout had left over."""
        from spacr.qt.screens.experiment_design import WELL_SIDE

        dx, dy = _step(screen)
        assert dx == WELL_SIDE + screen._plate_grid.horizontalSpacing()
        assert dy == WELL_SIDE + screen._plate_grid.verticalSpacing()

    def test_the_headers_stay_over_their_columns(self, screen):
        """A header that drifts away from the column it labels is worse than
        no header: it is still readable and now wrong."""
        def offset():
            head = screen._plate_grid.itemAtPosition(0, 1).widget()
            well = _wells(screen)[(1, 1)]
            return (head.mapTo(screen, head.rect().center()).x()
                    - well.mapTo(screen, well.rect().center()).x())

        screen.resize(900, 600)
        QApplication.processEvents()
        before = offset()

        screen.resize(1900, 1100)
        QApplication.processEvents()

        assert offset() == before


# --------------------------------------------------------------------------- #
#  A: clickable squares, and a drag selects a block
# --------------------------------------------------------------------------- #

class TestAPressSelectsAWell:

    def test_pressing_one_selects_exactly_it(self, screen):
        screen.begin_well_drag(2, 3, Qt.NoModifier)
        screen.finish_well_drag()

        assert screen.selected_wells() == {(2, 3)}

    def test_every_well_knows_its_name_assigned_or_not(self, screen):
        """A name is a COORDINATE. It was set only on assigned wells, so a
        selection over empty ones could not say which wells it held."""
        screen.begin_well_drag(2, 3, Qt.NoModifier)
        screen.finish_well_drag()

        assert screen.selected_well_names() == ["B03"]

    def test_a_second_press_replaces_the_first(self, screen):
        screen.begin_well_drag(2, 3, Qt.NoModifier)
        screen.finish_well_drag()
        screen.begin_well_drag(1, 1, Qt.NoModifier)
        screen.finish_well_drag()

        assert screen.selected_wells() == {(1, 1)}


class TestADragSelectsTheBlock:

    def test_press_move_release_selects_the_rectangle(self, screen):
        screen.begin_well_drag(2, 3, Qt.NoModifier)
        screen.drag_wells_to(_centre(screen, (4, 5)))
        screen.finish_well_drag()

        assert screen.selected_wells() == {
            (r, c) for r in (2, 3, 4) for c in (3, 4, 5)}

    def test_the_names_come_back_in_reading_order(self, screen):
        screen.begin_well_drag(2, 3, Qt.NoModifier)
        screen.drag_wells_to(_centre(screen, (3, 4)))
        screen.finish_well_drag()

        assert screen.selected_well_names() == ["B03", "B04", "C03", "C04"]

    def test_dragging_backwards_selects_the_same_block(self, screen):
        screen.begin_well_drag(4, 5, Qt.NoModifier)
        screen.drag_wells_to(_centre(screen, (2, 3)))
        screen.finish_well_drag()

        assert screen.selected_wells() == {
            (r, c) for r in (2, 3, 4) for c in (3, 4, 5)}

    def test_shrinking_the_rectangle_leaves_nothing_behind(self, screen):
        """Redrawn from the state at the PRESS rather than from the last
        frame, or growing and then shrinking keeps what it passed over."""
        screen.begin_well_drag(2, 2, Qt.NoModifier)
        screen.drag_wells_to(_centre(screen, (5, 5)))
        screen.drag_wells_to(_centre(screen, (3, 3)))
        screen.finish_well_drag()

        assert screen.selected_wells() == {
            (r, c) for r in (2, 3) for c in (2, 3)}

    def test_the_preview_is_visible_during_the_drag(self, screen):
        """A selection you cannot see until you let go is one you have to
        undo to correct."""
        screen.begin_well_drag(2, 3, Qt.NoModifier)
        screen.drag_wells_to(_centre(screen, (4, 5)))

        assert len(screen.selected_wells()) == 9, "nothing showed mid-drag"

    def test_a_drag_off_the_plate_changes_nothing(self, screen):
        from PySide6.QtCore import QPoint

        screen.begin_well_drag(2, 3, Qt.NoModifier)
        screen.drag_wells_to(QPoint(-500, -500))
        screen.finish_well_drag()

        assert screen.selected_wells() == {(2, 3)}


class TestCtrlAdds:

    def test_a_second_block_is_added(self, screen):
        screen.begin_well_drag(1, 1, Qt.NoModifier)
        screen.drag_wells_to(_centre(screen, (1, 2)))
        screen.finish_well_drag()

        screen.begin_well_drag(5, 5, Qt.ControlModifier)
        screen.drag_wells_to(_centre(screen, (5, 6)))
        screen.finish_well_drag()

        assert screen.selected_wells() == {(1, 1), (1, 2), (5, 5), (5, 6)}

    def test_and_a_plain_drag_still_replaces(self, screen):
        screen.begin_well_drag(1, 1, Qt.ControlModifier)
        screen.finish_well_drag()
        screen.begin_well_drag(5, 5, Qt.NoModifier)
        screen.finish_well_drag()

        assert screen.selected_wells() == {(5, 5)}


class TestTheSelectionIsMarkedWithoutHidingTheRole:
    """The fill says what the well IS -- control, treatment, blank -- and
    replacing it would trade the information the map exists for against the
    one thing the user can already see."""

    def test_a_chosen_well_carries_the_property_the_sheet_reads(self, screen):
        screen.begin_well_drag(2, 3, Qt.NoModifier)
        screen.finish_well_drag()

        assert _wells(screen)[(2, 3)].property("spacrWellChosen") == "true"

    def test_an_unchosen_one_does_not(self, screen):
        screen.begin_well_drag(2, 3, Qt.NoModifier)
        screen.finish_well_drag()

        assert _wells(screen)[(1, 1)].property("spacrWellChosen") == "false"

    def test_the_role_survives_being_chosen(self, screen):
        before = _wells(screen)[(2, 3)].property("spacrWellRole")

        screen.begin_well_drag(2, 3, Qt.NoModifier)
        screen.finish_well_drag()

        assert _wells(screen)[(2, 3)].property("spacrWellRole") == before

    def test_the_sheet_styles_the_selection_last(self):
        """So it wins over the role and edge rules rather than being
        overwritten by them.

        Read off the APPLICATION's sheet, which is where the screen's block
        actually lands -- `theme` collects it from the module rather than
        the widget setting its own.
        """
        from spacr.qt.theme import stylesheet

        sheet = stylesheet()
        assert "spacrWellChosen" in sheet, \
            "the selection has no style, so a drag shows nothing"
        assert sheet.index("spacrWellChosen") > sheet.index("spacrWellEdge")

    def test_the_selection_outline_is_the_accent(self):
        """An outline, not a fill: the fill is what the well IS."""
        from spacr.qt.theme import stylesheet

        sheet = stylesheet()
        block = sheet[sheet.index("spacrWellChosen"):][:200]
        assert "border" in block
        assert "background" not in block.split("}")[0]


# --------------------------------------------------------------------------- #
#  The OTHER plate map: 185's picker, whose select_region nothing called
# --------------------------------------------------------------------------- #

class TestThePickerDragsToo:
    """`select_region` has been in `PlateMapPicker` since 185 and NOTHING
    CALLED IT FROM THE MOUSE. It was written as a method so a test could
    select a rectangle without a human, and that is all it was ever used for
    -- the gesture a user expects was implemented and unreachable, which is
    the same fault 185's own button had.
    """

    @pytest.fixture
    def picker(self, qtbot):
        from spacr.qt.widgets.plate_map_picker import PlateMapPicker

        widget = PlateMapPicker(layout=96)
        qtbot.addWidget(widget)
        widget.resize(900, 600)
        widget.show()
        QApplication.processEvents()
        return widget

    def _centre(self, picker, cell):
        well = picker._wells[cell]
        return well.mapToGlobal(well.rect().center())

    def test_a_drag_selects_the_rectangle(self, picker):
        picker.begin_drag(2, 3, Qt.NoModifier)
        picker.drag_to(self._centre(picker, (4, 5)))
        picker.finish_drag()

        assert picker.selection() == {
            (r, c) for r in (2, 3, 4) for c in (3, 4, 5)}

    def test_and_it_writes_the_value_the_parser_reads_back(self, picker):
        """The whole point of the picker: what it writes, `well_spec` reads."""
        from spacr.well_spec import parse

        picker.begin_drag(2, 3, Qt.NoModifier)
        picker.drag_to(self._centre(picker, (3, 4)))
        picker.finish_drag()

        assert parse(picker.value(), 96) == picker.selection()

    def test_the_preview_shows_while_dragging(self, picker):
        picker.begin_drag(2, 3, Qt.NoModifier)
        picker.drag_to(self._centre(picker, (4, 5)))

        assert len(picker.selection()) == 9

    def test_shrinking_leaves_nothing_behind(self, picker):
        picker.begin_drag(2, 2, Qt.NoModifier)
        picker.drag_to(self._centre(picker, (5, 5)))
        picker.drag_to(self._centre(picker, (3, 3)))
        picker.finish_drag()

        assert picker.selection() == {(r, c) for r in (2, 3) for c in (2, 3)}

    def test_ctrl_adds_a_second_block(self, picker):
        picker.begin_drag(1, 1, Qt.NoModifier)
        picker.drag_to(self._centre(picker, (1, 2)))
        picker.finish_drag()

        picker.begin_drag(5, 5, Qt.ControlModifier)
        picker.drag_to(self._centre(picker, (5, 6)))
        picker.finish_drag()

        assert picker.selection() == {(1, 1), (1, 2), (5, 5), (5, 6)}

    def test_a_press_with_no_move_is_not_a_drag(self, picker):
        """`finish_drag` returns whether one happened, and a plain click has
        to fall through to the button's own toggle rather than being
        swallowed."""
        picker.begin_drag(2, 3, Qt.NoModifier)

        assert picker.finish_drag() is False

    def test_the_picker_does_not_drift_either(self, picker):
        def step():
            a, b = picker._wells[(1, 1)], picker._wells[(1, 2)]
            return (b.mapTo(picker, b.rect().topLeft()).x()
                    - a.mapTo(picker, a.rect().topLeft()).x())

        before = step()
        picker.resize(1600, 1000)
        QApplication.processEvents()

        assert step() == before
