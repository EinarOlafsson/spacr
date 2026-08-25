"""The figure grid under teardown, bad state, and a shape it cannot fit.

The grid is rebuilt every time a figure lands and is torn down when the screen
closes, so most of what is checked here is the same question asked in five
places: what happens when the widget a layout pass is holding has already gone.
The answer has to be "nothing visible" -- a fold gesture that raises takes the
run panel down, and a saved arrangement that will not parse must leave the grid
usable rather than empty.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent, QPixmap  # noqa: E402

from spacr.qt.widgets import figure_grid_view as fgv  # noqa: E402

pytestmark = pytest.mark.qt


def _pixmap(width, height):
    pixmap = QPixmap(width, height)
    pixmap.fill()
    return pixmap


@pytest.fixture()
def grid(qtbot):
    widget = fgv.FigureGridView()
    qtbot.addWidget(widget)
    widget.resize(1400, 900)
    return widget


# ---------------------------------------------------------------------------
# the heading ink
# ---------------------------------------------------------------------------

def test_a_heading_still_gets_ink_when_the_palette_will_not_load(monkeypatch):
    """A heading with no colour is a heading nobody can read.

    The accent is resolved at draw time so a runtime theme change reaches it.
    That makes the palette a live dependency of every relayout, and a bare or
    part-torn-down process has to fall back to a real colour rather than
    emitting a stylesheet with an empty ``color:``.
    """
    from spacr.qt import theme

    def refuse():
        raise RuntimeError("no palette in this process")

    monkeypatch.setattr(theme, "active_palette", refuse)

    style = fgv._heading_style()

    assert fgv._HEADING_FALLBACK in style
    assert style.startswith(fgv.HEADING_STYLE)


def test_a_heading_uses_the_live_accent_when_the_palette_is_there(qapp):
    """The fallback must not be what a working install gets."""
    from spacr.qt.theme import active_palette

    style = fgv._heading_style()

    assert active_palette()["accent"] in style


# ---------------------------------------------------------------------------
# the keyboard on a section heading
# ---------------------------------------------------------------------------

def test_a_key_that_is_not_the_fold_key_is_passed_on(qtbot, grid):
    """Only Return, Enter and Space fold; the rest belong to Qt.

    The heading takes strong focus so it can be reached by keyboard. Swallowing
    every key there would break Tab out of the heading and type-ahead in the
    panel behind it.
    """
    grid.set_figures([_pixmap(400, 400)], sections=[("run one", 0, 1)])
    header = grid._headers[0]

    header.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_A,
                                   Qt.NoModifier, "a"))
    assert grid.is_section_collapsed("run one", 0) is False

    header.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Space,
                                   Qt.NoModifier, " "))
    assert grid.is_section_collapsed("run one", 0) is True


# ---------------------------------------------------------------------------
# a cell with no picture in it
# ---------------------------------------------------------------------------

def test_a_cell_with_no_picture_claims_a_square_and_scales_to_nothing(qtbot):
    """An empty tile must not divide by a zero height or paint a stretched blank.

    A live tile is built before its pyqtgraph panel has been photographed, so
    a null pixmap reaches the cell in the ordinary course of a run.
    """
    cell = fgv._FigureCell(0, QPixmap(), "", None)
    qtbot.addWidget(cell)

    assert cell.aspect() == 1.0

    cell.fit_to(200)
    assert cell._image.pixmap().isNull()


def test_a_cell_asked_to_fit_no_width_is_left_alone(qtbot):
    """A zero-width viewport happens before the first show.

    Scaling into it would set a zero-height pixmap on the label and the figure
    would be gone until something else forced a relayout.
    """
    cell = fgv._FigureCell(0, _pixmap(400, 200), "", None)
    qtbot.addWidget(cell)
    cell.fit_to(300)
    before = cell._image.pixmap().size()

    cell.fit_to(0)

    assert cell._image.pixmap().size() == before


# ---------------------------------------------------------------------------
# discarding a widget Qt has already taken
# ---------------------------------------------------------------------------

def test_discarding_a_tile_qt_already_deleted_is_a_no_op(qtbot, grid):
    """The screen can close during the layout pass that is discarding tiles.

    Unparenting is the point -- a tile still parented to the body goes on
    painting itself -- but a tile whose C++ half is gone raises on the way,
    and taking the relayout down with it leaves the grid half-rebuilt.
    """
    reached: list = []

    class _AlreadyGone:
        def setParent(self, parent):
            reached.append("setParent")
            raise RuntimeError("Internal C++ object already deleted.")

        def deleteLater(self):
            reached.append("deleteLater")

    grid._discard(_AlreadyGone())
    assert reached == ["setParent"], "deletion was attempted on a dead wrapper"

    grid._discard(None)
    assert reached == ["setParent"], "nothing to discard is not an error"

    # The relayout that was discarding tiles still finishes.
    grid.set_figures([_pixmap(400, 400)])
    assert len(grid._cells) == 1


# ---------------------------------------------------------------------------
# putting a saved arrangement back
# ---------------------------------------------------------------------------

def test_an_arrangement_that_is_not_a_mapping_applies_nothing(grid):
    """A workspace file from another version can hold anything at all."""
    assert grid.apply_workspace_state(None) is False
    assert grid.apply_workspace_state([1, 2, 3]) is False
    assert grid.apply_workspace_state("cell_width=400") is False


def test_a_saved_cell_width_that_is_not_a_number_leaves_the_width_alone(grid):
    """One unreadable key must not cost the rest of the arrangement.

    The folded sections are the expensive half to lose: a sweep of sixty
    trials is unusable if every run unfolds because the tile size was written
    as text.
    """
    grid.set_target_cell_width(400)
    before = grid._target

    applied = grid.apply_workspace_state({"cell_width": "wide",
                                          "collapsed": [["run one", 0]]})

    assert applied is True
    assert grid._target == before
    assert grid.is_section_collapsed("run one", 0) is True


def test_a_saved_cell_width_that_is_a_number_goes_through_the_setter(grid):
    """The other half of the same branch: a good width still applies."""
    applied = grid.apply_workspace_state({"cell_width": 512})

    assert applied is True
    assert grid._target == 512


# ---------------------------------------------------------------------------
# folding and unfolding
# ---------------------------------------------------------------------------

def test_a_section_can_be_brought_back_after_it_was_folded_away(grid):
    """Unfolding has to forget the key, not merely stop adding it."""
    grid.set_figures([_pixmap(400, 400)], sections=[("run one", 0, 1)])

    grid.set_section_collapsed("run one", 0, True)
    assert grid.is_section_collapsed("run one", 0) is True

    assert grid._cells[0].isHidden() is True

    grid.set_section_collapsed("run one", 0, False)
    assert grid.is_section_collapsed("run one", 0) is False
    assert grid._cells[0].isHidden() is False


def test_a_heading_that_names_no_section_is_left_expanded(grid):
    """A header with no ``section_key`` cannot identify anything to fold.

    Returning "expanded" is the safe answer: the alternative folds away
    figures the user can then not get back, because nothing knows which key
    to discard.
    """
    class _Nameless:
        section_key = None

    assert grid.toggle_section(_Nameless()) is True
    assert grid._collapsed == set()


def test_a_header_rebuilt_between_click_and_query_is_not_raised(grid):
    """The grid rebuilds whenever a figure lands, including mid-gesture.

    ``_is_raised`` decides whether the click means "take me there" or "fold
    this away". A header whose C++ half went in between is not at the top of
    anything, and raising here would cost the click and the panel.
    """
    class _Gone:
        section_key = ("run one", 0)

        def mapTo(self, *args):
            raise RuntimeError("Internal C++ object already deleted.")

        def rect(self):
            raise RuntimeError("Internal C++ object already deleted.")

    assert grid._is_raised(_Gone()) is False


def test_scrolling_to_a_section_that_is_no_longer_there_reports_failure(grid):
    """The scroll runs an event-loop turn after the toggle.

    By then the run can have been cleared, so the header the gesture was
    about is simply not in the list any more. Reporting that is how the caller
    knows the view did not move.
    """
    grid.set_figures([_pixmap(400, 400)], sections=[("run one", 0, 1)])

    assert grid._scroll_section_to_top(("run two", 7)) is False


def test_scrolling_to_a_header_that_died_reports_failure_rather_than_raising(
        grid):
    """The header is in the list and its wrapper is dead. Same answer."""
    class _Gone:
        section_key = ("run one", 0)

        def mapTo(self, *args):
            raise RuntimeError("Internal C++ object already deleted.")

        def rect(self):
            raise RuntimeError("Internal C++ object already deleted.")

    grid._headers = [_Gone()]

    assert grid._scroll_section_to_top(("run one", 0)) is False


# ---------------------------------------------------------------------------
# a figure too wide for the room left in its row
# ---------------------------------------------------------------------------

def test_a_figure_wider_than_the_room_left_starts_the_next_row(qtbot,
                                                               monkeypatch):
    """A wide figure wraps rather than being squeezed into the gap.

    Every figure takes one slot today, so nothing in an ordinary run reaches
    the wrap. The rule is what makes a multi-slot span safe to reintroduce,
    and a span that overflows its row silently overlaps the cell beside it.
    """
    monkeypatch.setattr(fgv, "cells_across", lambda *args, **kwargs: 3)
    monkeypatch.setattr(fgv, "cell_span", lambda aspect: 2)

    grid = fgv.FigureGridView()
    qtbot.addWidget(grid)
    grid.resize(1400, 900)
    grid.set_figures([_pixmap(1200, 400), _pixmap(1200, 400)])

    positions = [grid._grid.getItemPosition(index)
                 for index in range(grid._grid.count())]

    assert len(positions) == 2
    # Two slots are taken of three; the second two-wide figure does not fit
    # in the one that is left, so it starts a row rather than hanging off
    # the edge of the one it is on.
    assert [(row, column) for row, column, _rs, _cs in positions] == [(0, 0),
                                                                      (1, 0)]
    assert [span for _r, _c, _rs, span in positions] == [2, 2]
