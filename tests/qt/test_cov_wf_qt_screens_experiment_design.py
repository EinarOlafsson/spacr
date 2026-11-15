"""The design screen's last unlit corners: no style, and a stray layout item.

Five branches of :mod:`spacr.qt.screens.experiment_design` are the ones that
only run when the world is slightly wrong -- a widget whose ``style()`` hands
back nothing, a layout holding something that is not a widget, an assignment
failure that carries no message. None of them is a scenario the screen chooses;
all of them are scenarios it has to survive, because the alternative is a
plate designer that raises ``AttributeError`` while the user is typing a plate
name and loses the layout they were halfway through.

Each test drives the ordinary path in the same breath as the odd one, so the
"nothing broke" half is measured against a working half rather than against
nothing at all.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QSizePolicy, QSpacerItem

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot):
    """A design screen with the default three conditions on a 96-well plate."""
    from spacr.qt.screens.experiment_design import ExperimentDesignScreen

    widget = ExperimentDesignScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


def _grid_items(layout):
    """Every item currently in ``layout``, without taking any of them out."""
    return [layout.itemAt(index) for index in range(layout.count())]


# --------------------------------------------------------------------------- #
#  A widget whose style is gone
# --------------------------------------------------------------------------- #

def test_a_well_with_no_style_keeps_its_square_instead_of_crashing(screen):
    """A well re-states its square even when Qt hands back no style object.

    ``_Well.lock_square`` runs on every selection change and on every one of
    the 96 wells of a redraw, and its second half asks the widget for its
    style so it can re-polish it. A widget that has been reparented out of a
    window can answer that with nothing. If the well went ahead and called
    ``unpolish`` on it, dragging a rectangle across the plate would die with
    an ``AttributeError`` mid-gesture and take the user's whole layout with
    it -- so the styleless well must simply keep the size and the sheet it
    already had.
    """
    from spacr.qt.screens.experiment_design import (
        MARK_RIM,
        OUTLINE_RIM,
    )
    # THE SIDE COMES FROM THE PICKER, AND THROUGH ITS ACCESSOR. The plate map
    # moved to `plate_map_picker`, and there `WELL_SIDE` is the unscaled BASE
    # -- reading it raw is the defect that module's own docstring records:
    # "THE GLYPHS GREW AND THE BOX DID NOT", because the font scale applies to
    # the glyph and not to a hard-coded box. `well_side()` is the scaled
    # value, so the assertion below asks for the size the well actually has at
    # whatever scale the test is running under.
    from spacr.qt.widgets.plate_map_picker import well_side

    well = next(label for label in screen._well_labels
                if (label.row, label.column) == (3, 4))
    # An unassigned well is outlined, which is the rim the sheet below states.
    assert well.property("spacrWellRole") == "empty"
    assert well._rim == OUTLINE_RIM
    sheet_before = well.styleSheet()

    # The ordinary half: a real style, nothing about the well's state changed,
    # so the widget is re-polished and left at exactly the same size.
    well.lock_square()
    assert well._rim == OUTLINE_RIM
    assert well.styleSheet() == sheet_before
    assert well.size() == QSize(well_side(), well_side())

    # The odd half: the style is gone. Same outcome, no exception.
    well.style = lambda: None
    try:
        well.lock_square()
    finally:
        del well.style
    assert well._rim == OUTLINE_RIM
    assert well.styleSheet() == sheet_before
    assert well.size() == QSize(well_side(), well_side())

    # And the widget is still live afterwards: choosing it thickens the rim
    # from an outline to a mark and re-states the sheet at the new width.
    well.setProperty("spacrWellChosen", "true")
    well.lock_square()
    assert well._rim == MARK_RIM
    assert well.styleSheet() != sheet_before
    assert f"border-width: {MARK_RIM}px" in well.styleSheet()
    assert well.size() == QSize(well_side(), well_side())


# --------------------------------------------------------------------------- #
#  A layout holding something that is not a widget
# --------------------------------------------------------------------------- #

def test_a_spacer_in_the_plate_grid_does_not_stop_the_redraw(screen):
    """The plate redraw empties its grid whatever the grid is holding.

    Every ``refresh`` -- one per keystroke in the plate name -- tears the
    whole plate down and builds it again. It drains the grid by taking item
    zero until the grid is empty, and a spacer item answers ``widget()`` with
    nothing. A redraw that assumed every item were a widget would raise on
    the first stretch anyone adds to that panel and leave the user looking at
    a half-demolished plate.
    """
    spacer = QSpacerItem(4, 4, QSizePolicy.Policy.Fixed,
                         QSizePolicy.Policy.Fixed)
    screen._plate_grid.addItem(spacer, 150, 150)
    assert screen._plate_grid.count() == 117

    screen.refresh()

    # 96 wells + 12 column headers + 8 row headers, and no spacer left over.
    assert len(screen._well_labels) == 96
    assert screen._plate_grid.count() == 116
    assert all(item.widget() is not None
               for item in _grid_items(screen._plate_grid))
    names = {label.property("wellName") for label in screen._well_labels}
    assert "A01" in names and "H12" in names


def test_a_stretch_in_the_findings_panel_does_not_stop_the_redraw(screen):
    """The findings list rebuilds itself past a non-widget layout item.

    The findings are the reason this screen exists: they name the things that
    cannot be repaired after the plate is run. They are rebuilt from scratch
    on every refresh by the same drain-the-layout loop, so a stretch or a
    spacer sitting in that panel must be dropped rather than crashed on --
    otherwise a stale warning would stay on screen after the user fixed it,
    or no warning would appear at all.
    """
    screen._findings_layout.addStretch(1)
    before = screen._findings_layout.count()
    assert before == len(screen._findings_labels) + 1

    screen.refresh()

    assert screen._findings_labels, "the default plate has findings to show"
    assert screen._findings_layout.count() == len(screen._findings_labels)
    assert all(item.widget() is not None
               for item in _grid_items(screen._findings_layout))
    text = screen.findings_text()
    assert "control wells are on the plate edge" in text
    assert text.startswith(("STOP ", "! ", "- "))


# --------------------------------------------------------------------------- #
#  An assignment failure that carries no message
# --------------------------------------------------------------------------- #
