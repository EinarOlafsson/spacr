"""FlowView and the parameter sweep show the page, not a black rectangle.

Reported 2026-09-04:

    "flow view still has the black background that needs to be fixed to a
     transparent box with rounded edges"
    "regression modual paramiter sweep has black box background which should
     be transparen t background"

BOTH ARE THE BLANKET RULE, the same one that produced the dock's box. The
application sheet carries ``QWidget { background-color: bg }``; any container
not tagged out of it paints the WINDOW colour -- which is not a surface, so no
value of the page-opacity preference can reach it, and the panel sits as a
slab over the animated backdrop.

MEASURED, each widget rendered over magenta so anything it fills is obvious:

    FlowViewPanel                     #ff00ff   (already transparent)
      its QSplitter                   #000000   <- the FlowView box
    ParameterSweepScreen              #000000   <- the sweep box
      its QSplitter                   #000000

A ``QSplitter`` is a plain ``QWidget``, so it takes the blanket rule however
transparent its parent is, and neither screen's QSS ever named it. The
tagging is done in the Qt layer at the point each panel is built --
``spacr/flowview/panel.py`` is shared and imports no Qt theme.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QSplitter

MARKER = "#ff00ff"


def _fills(widget) -> str:
    """What ``widget`` leaves on a magenta ground, at its top-left."""
    shot = QPixmap(max(1, widget.width()), max(1, widget.height()))
    shot.fill(QColor(MARKER))
    widget.render(shot)
    return shot.toImage().pixelColor(6, 6).name().lower()


def test_the_parameter_sweep_is_not_a_slab(qtbot, qt_theme_applied):
    """The screen and its splitter both show what is behind them."""
    from spacr.qt.screens.parameter_sweep import _make_screen

    screen = _make_screen()
    qtbot.addWidget(screen)
    screen.resize(900, 600)

    assert _fills(screen) == MARKER, (
        "the sweep screen paints a slab over the backdrop")
    splitter = screen.findChild(QSplitter)
    assert splitter is not None, "the sweep no longer has a splitter to check"
    assert _fills(splitter) == MARKER, (
        "the sweep's splitter paints a black rectangle")


def test_the_flowview_splitter_is_not_a_slab(qtbot, qt_theme_applied):
    """FlowView's panel was always transparent; its splitter was not."""
    from spacr.flowview.panel import FlowViewPanel
    from spacr.flowview.trace import get_collector
    from spacr.qt.theme import clear_container_surfaces

    panel = FlowViewPanel(get_collector(), None, auto_start=False,
                          embedded=True)
    qtbot.addWidget(panel)
    panel.resize(500, 380)

    splitter = panel.findChild(QSplitter)
    assert splitter is not None, "FlowView no longer has a splitter to check"
    assert _fills(splitter) == "#000000", (
        "this test's premise is gone -- the splitter no longer fills, so it "
        "is no longer proving that clearing it is what fixes the box")

    # What `ClassifyScreen` does when it embeds the panel.
    clear_container_surfaces(panel)
    assert _fills(splitter) == MARKER, (
        "clearing the container surfaces did not stop the splitter filling")


def test_classify_clears_the_flowview_it_embeds():
    """The fix has to be AT the embed site, or it never runs.

    `spacr/flowview/panel.py` is shared and imports no Qt theme, so the
    tagging cannot live there; asserted on the source so a refactor that
    drops the call is caught.
    """
    import inspect

    from spacr.qt.screens import classify

    source = inspect.getsource(classify)
    assert "clear_container_surfaces(panel)" in source, (
        "Classify stopped clearing the FlowView panel it embeds")
