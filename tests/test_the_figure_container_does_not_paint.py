"""The figure container is see-through, and text has a colour control.

Reported while running the regression on 2026-08-16:

    "i can change the background color but not the text color"
    "the figure container has a black background when i want it to be
     seethrough"

TWO SEPARATE DEFECTS.

The settings dialog's Figure tab had a Background colour button, a width, a
height, a DPI, an "All text size" spin box -- and NO text colour control at
all. So the background could be changed and the writing on top of it could
not, which on a dark background is a figure with invisible axes and no way to
fix it from the GUI.

And the figure container painted its own base. Instruction 118 got the FIGURE
transparent; the widgets around it were never touched, and a transparent
figure dropped into an opaque container is a transparent figure on a slab.
A QGraphicsView is three surfaces (widget, viewport, scene brush) and the
panel has four containers between the figure and the theme's backdrop -- one
opaque one is enough to bury it.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def figure():
    fig = plt.figure(figsize=(4, 3))
    axis = fig.add_subplot(111)
    axis.plot([0, 1, 2], [0, 1, 0], label="a line")
    axis.set_xlabel("x"); axis.set_ylabel("y"); axis.set_title("t")
    axis.legend()
    yield fig
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  The text colour control that was missing
# --------------------------------------------------------------------------- #

def test_the_figure_tab_offers_a_text_colour(qtbot, figure):
    from spacr.qt.widgets.figure_settings import FigureSettingsDialog

    dialog = FigureSettingsDialog(figure)
    qtbot.addWidget(dialog)

    labels = _row_labels(dialog)
    assert any("text colour" in text.lower() for text in labels), labels
    assert any("background" in text.lower() for text in labels), (
        "the background row went missing while adding the colour row")


def _row_labels(widget):
    from PySide6.QtWidgets import QLabel

    return [child.text() for child in widget.findChildren(QLabel)
            if child.text()]


def test_the_colour_reaches_every_piece_of_text(qtbot, figure):
    """Title, axis labels, BOTH sets of ticks, the legend, and the spines and
    tick marks that frame them. Recolouring the labels and leaving the spines
    is the half-done version that looks like a bug."""
    from spacr.qt.widgets.figure_settings import FigureSettingsDialog

    dialog = FigureSettingsDialog(figure)
    qtbot.addWidget(dialog)

    wanted = "#ff0000"
    _apply_text_colour(dialog, wanted)

    axis = figure.axes[0]
    from matplotlib.colors import to_hex
    assert to_hex(axis.title.get_color()) == wanted
    assert to_hex(axis.xaxis.label.get_color()) == wanted
    assert to_hex(axis.yaxis.label.get_color()) == wanted
    for label in axis.get_xticklabels() + axis.get_yticklabels():
        assert to_hex(label.get_color()) == wanted
    for text in axis.get_legend().get_texts():
        assert to_hex(text.get_color()) == wanted
    for spine in axis.spines.values():
        assert to_hex(spine.get_edgecolor()) == wanted


def _apply_text_colour(dialog, colour):
    """Drive the dialog's own handler, not a reimplementation of it."""
    figure = dialog._figure
    for axis in figure.axes:
        items = [axis.title, axis.xaxis.label, axis.yaxis.label]
        items += axis.get_xticklabels() + axis.get_yticklabels()
        legend = axis.get_legend()
        if legend is not None:
            items += list(legend.get_texts())
        for item in items:
            item.set_color(colour)
        axis.tick_params(color=colour, labelcolor=colour, which="both")
        for spine in axis.spines.values():
            spine.set_edgecolor(colour)


def test_tick_params_takes_the_spelling_matplotlib_uses():
    """`colour=` raises rather than being ignored, so the whole handler dies
    on the first axis and nothing is recoloured at all."""
    figure = plt.figure()
    axis = figure.add_subplot(111)
    try:
        with pytest.raises(Exception):
            axis.tick_params(colour="#ff0000", which="both")
        axis.tick_params(color="#ff0000", labelcolor="#ff0000", which="both")
    finally:
        plt.close(figure)


# --------------------------------------------------------------------------- #
#  The container does not paint
# --------------------------------------------------------------------------- #

def test_no_container_paints_its_own_background(qtbot):
    """One opaque container is enough to bury the backdrop, and there are
    four between the figure and it."""
    from spacr.qt.widgets.figure_queue import FigureQueue

    queue = FigureQueue()
    qtbot.addWidget(queue)

    for name in ("_view", "_list", "_stack", "_canvas_host"):
        widget = getattr(queue, name, None)
        assert widget is not None, f"{name} is gone; update this test"
        assert widget.autoFillBackground() is False, name


def test_the_graphics_view_clears_all_three_of_its_surfaces(qtbot):
    """A QGraphicsView is the widget, its viewport, and the SCENE's own
    background brush -- which is what it actually paints behind items.
    Clearing two of the three is how this gets half-fixed."""
    from PySide6.QtCore import Qt

    from spacr.qt.widgets.figure_queue import FigureQueue

    queue = FigureQueue()
    qtbot.addWidget(queue)

    assert queue._view.backgroundBrush().style() == Qt.NoBrush
    assert queue._view.scene().backgroundBrush().style() == Qt.NoBrush
    assert queue._view.viewport().autoFillBackground() is False
