"""FlowView's box, and the two lines that change its shape.

THE BLACK BOX, ONE LAYER AT A TIME. FlowView draws on a page that shows the
theme behind it, and every opaque layer between the two reads as a black
rectangle sitting on top of that page. Four were found and cleared before
this file existed -- the graphics scene's brush, the view widget, its
viewport, and the inspector -- and a fifth was still there: the SECTION in
`spacr.qt.screens.classify` that hosts the panel painted `surface`, which is
opaque. So the layers below it were transparent onto a slab.

The rim and the corner radius stay. What was asked for is a box around BOTH
the graph and the text under it, which is what a rim with nothing behind it
is; what was there was a slab behind them, which is not the same thing and
does not look like one.

AND TWO LINES YOU CAN PULL. The splitter between the graph and the inspector
already existed and was already draggable -- but a QSplitter draws no handle
on this style, so it was six pixels of nothing that only somebody who knew
it was there could find. The second line, under the inspector, is new: the
splitter divides the panel's own height between its two halves, and this one
changes how much height the panel has at all.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

if importlib.util.find_spec("PySide6") is None:
    pytest.skip("PySide6 is not installed", allow_module_level=True)

from spacr.flowview.panel import FlowViewPanel, _PanelHeightGrip  # noqa: E402
from spacr.flowview.trace import get_collector  # noqa: E402


@pytest.fixture
def panel(qtbot):
    """An embedded panel, which is how Classify hosts it."""
    made = FlowViewPanel(get_collector(), embedded=True)
    qtbot.addWidget(made)
    return made


# ---------------------------------------------------------------------------
# The box
# ---------------------------------------------------------------------------

def test_the_section_around_it_is_not_filled():
    """The last opaque layer, and the one that was still a black box.

    Checked as the stylesheet the screen registers rather than by grabbing a
    pixel, because the colour that WOULD be painted is exactly the question:
    a screenshot of a panel over a dark theme cannot tell "transparent onto
    dark" from "painted dark".
    """
    from spacr.qt.screens.classify import FLOWVIEW_SECTION_NAME, _flowview_section_qss

    qss = _flowview_section_qss({"surface": "#123456",
                                 "border_soft": "#654321"})
    section = qss.split(f"QWidget#{FLOWVIEW_SECTION_NAME}")[1]

    assert "background: transparent" in section
    assert "#123456" not in section, "the section is still painting a slab"
    # The box itself survives: a rim, and corners that are not square.
    assert "#654321" in section
    assert "border-radius" in section


def test_nothing_inside_the_panel_paints_over_the_page(panel):
    """The four layers cleared before this file existed, kept cleared."""
    assert "background: transparent" in panel.view.styleSheet()
    assert panel.scene.backgroundBrush().color().alpha() == 0
    assert panel.view.viewport().autoFillBackground() is False
    assert "background: transparent" in panel.styleSheet()


# ---------------------------------------------------------------------------
# The two lines
# ---------------------------------------------------------------------------

def test_the_divider_between_graph_and_text_is_visible(panel):
    """Draggable was never the problem; findable was.

    A handle with no width and no paint is a feature only its author knows
    about, so both are asserted -- and the width has to be a real number of
    pixels, not merely non-zero, or the line is there and unhittable.
    """
    assert panel._splitter.handleWidth() >= _PanelHeightGrip.HEIGHT
    assert "border-bottom" in panel._splitter.styleSheet()


def test_there_is_a_second_line_under_the_text_box(panel):
    """The one that resizes the panel rather than dividing it."""
    assert isinstance(panel._bottom_grip, _PanelHeightGrip)
    assert panel._bottom_grip.height() == _PanelHeightGrip.HEIGHT
    assert "border-bottom" in panel._bottom_grip.styleSheet()


def test_the_lines_are_written_so_qt_reads_them_as_alpha(panel):
    """`#RRGGBBAA` is a CSS spelling, and a Qt stylesheet is not CSS.

    Qt parses an eight-digit hex as `#AARRGGBB`, so `#FFFFFF1A` -- white at
    10% in a browser -- becomes opaque rgb(255, 255, 26). That is the bright
    yellow rim that was reported around the inspector on 2026-09-01, and the
    same literal is CORRECT in `export.py`, which really is CSS. Anything
    translucent in a Qt stylesheet has to say `rgba()`.
    """
    from spacr.flowview import panel as module

    for sheet in (panel._splitter.styleSheet(),
                  panel._bottom_grip.styleSheet()):
        assert "rgba(" in sheet
    assert module._SPLIT_LINE.startswith("rgba(")
    assert module._SPLIT_HOVER.startswith("rgba(")


# ---------------------------------------------------------------------------
# What pulling the second line does
# ---------------------------------------------------------------------------

def test_dragging_the_lower_line_sets_the_height(panel):
    """The whole point of the affordance."""
    panel.set_user_height(400)

    assert panel.height() == 400


def test_it_cannot_be_dragged_shut(panel):
    """A drag that runs off the top must leave an edge to drag back.

    Squeezing the panel to nothing would take the grip with it, and the only
    way back would be to reopen the module.
    """
    panel.set_user_height(1)

    assert panel.height() >= _PanelHeightGrip.HEIGHT
    assert panel.height() >= panel.INSPECTOR_MIN_HEIGHT


def test_double_clicking_gives_the_height_back(panel):
    """The way out of a drag, and the same gesture the table's grip uses."""
    panel.set_user_height(400)
    panel.reset_user_height()

    assert panel.maximumHeight() > 400
    assert panel.minimumHeight() == 0
