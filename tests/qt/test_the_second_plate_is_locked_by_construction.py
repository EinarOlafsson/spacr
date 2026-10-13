"""The design screen's plate holds its square because it SAYS so.

Two plates are drawn in this application, and the first one lost its square
to the theme: `spacr/qt/theme.py` carries
``QPushButton { min-height: 22px; padding: 8px 12px }``, and
``QStyleSheetStyle`` does not merely draw that rule -- it polishes it into a
real ``setMinimumHeight`` on every button, 22 plus twice the padding plus
twice the rim, which OVERWRITES the minimum ``setFixedSize`` wrote. The
maximum survives, so the well is left holding a minimum taller than its
maximum, and Qt resolves that in favour of the minimum: 22 wide by 40 tall.

THIS plate was never broken, and that was luck rather than design. Its well
is a ``QLabel``, and every ``QLabel`` rule in the generated stylesheet is
object-name scoped -- there was simply no blanket geometry rule to polish
in. The day anyone adds one, it unlocks exactly as the first did, silently.

So every measurement below is taken with a HOSTILE blanket rule installed,
``QLabel { min-height: 60px; padding: 9px }``, which is the mechanism rather
than today's numbers: 60 plus twice 9 is 78, and each well came out 78, 80
or 82 tall depending on the rim its state draws, with the column numbers
riding 56 px above the columns they name. A guard that measures this plate
against a bare ``QApplication`` cannot see this class of defect at all.
"""
from __future__ import annotations

import ast
import io

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

pytestmark = pytest.mark.qt

# well_side() IS A BASE NOW, NOT THE ANSWER. It was read as a raw pixel
# constant, so at the 200% font scale a two-digit column header wanted 30 px
# inside a 22 px cell and columns 10 to 24 had their numbers cut in half.
# `well_side()` routes it through `scaled_px`, which is the house mechanism
# for a widget size set from Python. The invariant these tests exist for is
# unchanged and still checked below: ONE geometry, shared by both plates --
# it is now one function rather than one integer.
from spacr.qt.screens.experiment_design import (ExperimentDesignScreen,
                                                well_side)
from spacr.qt.widgets.plate_layout import (PLATE_FORMATS, ROLE_BLANK,
                                           ROLE_NEGATIVE, ROLE_POSITIVE,
                                           ROLE_TREATMENT, Condition)

#: The two window widths every geometry claim is checked at. The fault this
#: guards only shows when the layout has spare space to hand out.
NARROW, WIDE = 900, 1900

#: A blanket rule of the shape that took the other plate's square away.
HOSTILE = "QLabel { min-height: 60px; padding: 9px; }"


def _screen(qtbot, width=NARROW, well_count=96):
    widget = ExperimentDesignScreen(threaded=False)
    qtbot.addWidget(widget)
    if well_count != 96:
        widget._format.setCurrentIndex(widget._format.findData(well_count))
    widget.resize(width, 800)
    widget.show()
    QApplication.processEvents()
    return widget


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    """The plate under the application stylesheet, which is what runs."""
    return _screen(qtbot)


@pytest.fixture
def hostile(qtbot, qt_theme_applied):
    """A factory for the plate under theme + blanket ``QLabel`` geometry."""
    app = QApplication.instance()
    was = app.styleSheet()
    app.setStyleSheet(was + "\n" + HOSTILE)
    try:
        yield lambda **kwargs: _screen(qtbot, **kwargs)
    finally:
        app.setStyleSheet(was)


def _wells(screen) -> dict:
    return {(well.row, well.column): well for well in screen._well_labels}


def _column_header(screen, column):
    return screen._plate_grid.itemAtPosition(0, column).widget()


def _row_header(screen, row):
    return screen._plate_grid.itemAtPosition(row, 0).widget()


def _oblong(screen) -> dict:
    """Every cell of the plate that is not ``WELL_SIDE`` square."""
    square = (well_side(), well_side())
    cells = {(well.row, well.column): well
             for well in screen._well_labels}
    rows = max(row for row, _ in cells)
    columns = max(column for _, column in cells)
    cells.update({("column header", c): _column_header(screen, c)
                  for c in range(1, columns + 1)})
    cells.update({("row header", r): _row_header(screen, r)
                  for r in range(1, rows + 1)})
    return {name: (cell.width(), cell.height())
            for name, cell in cells.items()
            if (cell.width(), cell.height()) != square}


# --------------------------------------------------------------------------
# 1: a blanket rule cannot unlock the plate
# --------------------------------------------------------------------------

class TestABlanketRuleCannotUnlockThePlate:

    @pytest.mark.parametrize("width", [NARROW, WIDE])
    def test_every_well_keeps_its_square(self, hostile, width):
        plate = hostile(width=width)

        assert not _oblong(plate), \
            f"a blanket QLabel rule unlocked the plate: {_oblong(plate)}"

    def test_both_header_strips_keep_it_too(self, hostile):
        """The strips are cells of the grid, not captions beside it: a
        header inflated to 78 px tall carries the numbers off their columns
        while every well below it is still square."""
        plate = hostile()

        assert (_column_header(plate, 1).height(),
                _row_header(plate, 1).width()) == (well_side(), well_side())

    def test_a_number_still_sits_over_the_column_it_names(self, hostile):
        plate = hostile(width=WIDE)

        adrift = {column for column in range(1, 13)
                  if _column_header(plate, column).mapTo(
                      plate, _column_header(plate, column).rect().center()).x()
                  != _wells(plate)[(1, column)].mapTo(
                      plate, _wells(plate)[(1, column)].rect().center()).x()}

        assert not adrift, f"numbers off their columns: {sorted(adrift)}"

    @pytest.mark.parametrize("well_count", [6, 96, 384])
    def test_every_plate_format_holds(self, hostile, well_count):
        plate = hostile(well_count=well_count)

        rows, columns = PLATE_FORMATS[well_count]
        assert len(plate._well_labels) == rows * columns
        assert not _oblong(plate)

    def test_and_every_role_a_well_can_hold(self, hostile):
        """The rim a well draws depends on its state, and the state is what
        the sheet selects on: a fill for a role, an outline for a well that
        holds nothing, a mark for the edge and for the selection."""
        plate = hostile()
        plate._set_conditions([Condition("neg", 4, ROLE_NEGATIVE),
                               Condition("pos", 4, ROLE_POSITIVE),
                               Condition("treat", 4, ROLE_TREATMENT),
                               Condition("blank", 4, ROLE_BLANK)])
        plate.refresh()
        QApplication.processEvents()

        drawn = {well.property("spacrWellRole") for well in plate._well_labels}
        assert {"negative_control", "positive_control", "treatment", "blank",
                "empty"} <= drawn, f"not every role was on the plate: {drawn}"
        assert not _oblong(plate)


# --------------------------------------------------------------------------
# 2: the square does not depend on what the well is doing
# --------------------------------------------------------------------------

class TestChoosingAWellDoesNotResizeIt:
    """The selection is drawn as a RIM, and a rim is part of the widget's
    height once the square is stated in content-box pixels -- Qt adds the
    border, the padding and the margin back on. A well marked 2 px where it
    was outlined at 1 grows by two pixels and shoves the rest of its row
    along, which is the cost of reading the rim widths in two places."""

    def test_the_chosen_well_is_the_same_size_as_its_neighbour(self, screen):
        chosen, beside = _wells(screen)[(3, 3)], _wells(screen)[(3, 4)]
        before = chosen.size()

        screen.begin_well_drag(3, 3, Qt.NoModifier)
        screen.finish_well_drag()
        QApplication.processEvents()

        assert chosen.size() == before
        assert chosen.size() == beside.size()

    def test_an_edge_well_is_the_same_size_as_an_inner_one(self, screen):
        edge = [well for well in screen._well_labels
                if well.property("spacrWellEdge") == "true"]
        assert edge, "no edge well was marked, so nothing was measured"

        assert all(well.size() == _wells(screen)[(2, 2)].size()
                   for well in edge)

    def test_the_sheet_draws_the_rim_the_square_allowed_for(self):
        """Read off the sheet the application actually builds. The widths
        are what the well subtracts from its side, so a border widened here
        and nowhere else is a well bigger than its neighbours."""
        from spacr.qt.screens.experiment_design import (MARK_RIM,
                                                        OUTLINE_RIM)
        from spacr.qt.theme import stylesheet

        sheet = stylesheet()
        chosen = sheet[sheet.index("spacrWellChosen"):][:200]
        edge = sheet[sheet.index("spacrWellEdge"):][:200]
        empty = sheet[sheet.index('spacrWellRole="empty"'):][:250]

        assert f"border: {MARK_RIM}px" in chosen
        assert f"border: {MARK_RIM}px" in edge
        assert f"border: {OUTLINE_RIM}px" in empty

        # And the states the table says draw no rim draw none: a border put
        # here alone is a border the well overrides back to nothing, so the
        # outline you typed is not the one you get.
        for role in ("negative_control", "positive_control", "treatment"):
            block = sheet[sheet.index(f'spacrWellRole="{role}"'):]
            block = block[:block.index("}")]
            assert "border:" not in block and "border-width" not in block, \
                f"{role} draws a rim the well made no room for: {block}"


# --------------------------------------------------------------------------
# 3: one implementation, not two that look alike
# --------------------------------------------------------------------------

class TestBothPlatesShareOneImplementation:
    """They had already drifted once: the same square, the same constant and
    the same header written twice, and only one of the copies fixed."""

    def test_the_header_is_the_picker_s_header(self):
        from spacr.qt.screens import experiment_design
        from spacr.qt.widgets import plate_map_picker

        assert experiment_design._Header is plate_map_picker._Header

    def test_the_square_is_stated_by_the_picker_s_function(self):
        from spacr.qt.screens import experiment_design
        from spacr.qt.widgets import plate_map_picker

        assert (experiment_design._locked_square
                is plate_map_picker._locked_square)

    def test_the_screen_declares_no_well_side_of_its_own(self):
        """An identity check on the value cannot say this -- 22 is 22
        whoever wrote it -- so the module is read instead."""
        from spacr.qt.screens import experiment_design

        tree = ast.parse(io.open(experiment_design.__file__,
                                 encoding="utf-8").read())
        assigned = {target.id for node in tree.body
                    if isinstance(node, ast.Assign)
                    for target in node.targets
                    if isinstance(target, ast.Name)}
        assigned |= {node.target.id for node in tree.body
                     if isinstance(node, ast.AnnAssign)
                     and isinstance(node.target, ast.Name)}
        imported = {alias.name for node in tree.body
                    if isinstance(node, ast.ImportFrom)
                    and (node.module or "").endswith("plate_map_picker")
                    for alias in node.names}

        assert "WELL_SIDE" not in assigned, \
            "the design screen declares a second WELL_SIDE"
        # IMPORTED AS A FUNCTION. The side follows the font scale now, so a
        # constant read at import would freeze it at whatever the scale was
        # when this module loaded -- which is the drift this class guards
        # against, wearing a different hat.
        assert "well_side" in imported

    def test_the_cells_on_the_plate_are_that_header(self, screen):
        """What the drift looked like: a bare `QLabel` made on the spot,
        centred, and left to size itself."""
        from spacr.qt.widgets.plate_map_picker import _Header

        assert isinstance(_column_header(screen, 1), _Header)
        assert isinstance(_row_header(screen, 1), _Header)


# --------------------------------------------------------------------------
# 4: the gesture still lands on the well the pointer is over
# --------------------------------------------------------------------------
#
# Driven with real Qt mouse events on real cells of the grid rather than by
# calling the drag methods, because hit-testing is where a geometry change
# shows up: `well_at` asks every label whether a GLOBAL point is inside it,
# so a well that is not the size the grid pitched it at answers for its
# neighbour's pixels.

def _press(well, modifiers=Qt.NoModifier):
    centre = well.rect().center()
    QApplication.sendEvent(well, QMouseEvent(
        QMouseEvent.MouseButtonPress, centre, well.mapToGlobal(centre),
        Qt.LeftButton, Qt.LeftButton, modifiers))


def _wobble(well, dx=1, dy=1):
    """The pixel or two of pointer travel every real click carries, which no
    hand can leave out."""
    local = well.rect().center() + QPoint(dx, dy)
    QApplication.sendEvent(well, QMouseEvent(
        QMouseEvent.MouseMove, local, well.mapToGlobal(local),
        Qt.NoButton, Qt.LeftButton, Qt.NoModifier))


def _move_over(source, target):
    """A move whose GLOBAL position is over ``target``, delivered to the
    pressed widget -- which is where Qt sends it, because the press grabbed
    the mouse."""
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


class TestTheGestureLandsWhereThePointerIs:

    def test_a_press_with_no_movement_selects_exactly_one_well(self, screen):
        well = _wells(screen)[(6, 6)]

        _press(well)
        _release(well)
        QApplication.processEvents()

        assert screen.selected_wells() == {(6, 6)}

    def test_a_one_pixel_wobble_still_lands_that_well(self, screen):
        well = _wells(screen)[(6, 6)]

        _press(well)
        _wobble(well)
        _release(well)
        QApplication.processEvents()

        assert screen.selected_wells() == {(6, 6)}

    def test_a_wobbled_click_keeps_the_wells_already_chosen(self, screen):
        """Ctrl adds a well to a selection built up one at a time, and a
        wobble read as a drag off the anchor would take the rest with it."""
        first, second = _wells(screen)[(1, 1)], _wells(screen)[(1, 2)]
        _press(first)
        _move_over(first, second)
        _release(first)

        well = _wells(screen)[(6, 6)]
        _press(well, Qt.ControlModifier)
        _wobble(well, dx=-1, dy=2)
        _release(well)
        QApplication.processEvents()

        assert screen.selected_wells() == {(1, 1), (1, 2), (6, 6)}

    def test_a_drag_selects_the_rectangle_it_swept(self, screen):
        start, corner = _wells(screen)[(2, 2)], _wells(screen)[(4, 5)]

        _press(start)
        _move_over(start, corner)
        _release(start)
        QApplication.processEvents()

        assert screen.selected_wells() == {
            (r, c) for r in (2, 3, 4) for c in (2, 3, 4, 5)}

    def test_the_press_lands_on_the_right_well_under_a_blanket_rule(
            self, hostile):
        """The lock is not only about the picture. A well 78 px tall covers
        three rows of the grid, and the well the pointer is over is found by
        asking every label whether the point is inside it."""
        plate = hostile()
        well = _wells(plate)[(5, 7)]

        _press(well)
        _release(well)
        QApplication.processEvents()

        assert plate.selected_wells() == {(5, 7)}

    def test_and_a_drag_under_one_still_sweeps_the_rectangle(self, hostile):
        plate = hostile()
        start, corner = _wells(plate)[(2, 2)], _wells(plate)[(4, 5)]

        _press(start)
        _move_over(start, corner)
        _release(start)
        QApplication.processEvents()

        assert plate.selected_wells() == {
            (r, c) for r in (2, 3, 4) for c in (2, 3, 4, 5)}
