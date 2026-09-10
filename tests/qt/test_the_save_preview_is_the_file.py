"""The save dialog offers the file's own settings, and previews the file.

Two asks, both about the gap between what a figure looks like on screen
and what it should look like once it has left the application:

* "i need more settings when saving like the aspect ratio, line width,
  test size, axis titles ... more than the settings when rigt clicking
  the graph". A file is read at a size and on a page the screen never
  had, so the things worth changing for it are not the things worth
  changing on screen.
* "the preview in the popup window should be representative of what gets
  saved including the background". It was not: the raster exporter fills
  the page behind the scene when the file is written, and the preview
  asked for a transparent one -- so a figure saved onto white previewed
  onto nothing.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from spacr.qt.widgets.fast_plots import FastPlot
from spacr.qt.widgets.save_figure_dialog import SaveFigureDialog


@pytest.fixture
def plot(qtbot, qt_theme_applied):
    made = FastPlot()
    qtbot.addWidget(made)
    rng = np.random.default_rng(0)
    made.add_scatter(np.arange(30.0), rng.normal(size=30))
    return made


def _corner(pixmap):
    return pixmap.toImage().pixelColor(2, 2)


def test_the_preview_is_drawn_on_the_page_the_file_gets(plot):
    """White background chosen, white page previewed."""
    on_white = plot.styled_snapshot(240, background="#ffffff")

    assert on_white is not None
    corner = _corner(on_white)
    assert corner.alpha() == 255, "the preview page is still transparent"
    assert corner.red() > 240 and corner.blue() > 240


def test_a_tile_is_still_transparent(plot):
    """The page belongs to the FILE. A thumbnail composited onto the
    figure grid must not gain an opaque slab, which is the report this
    behaviour came from in the first place."""
    tile = plot.snapshot(200)

    assert tile is not None
    assert _corner(tile).alpha() == 0


def test_the_dialog_offers_the_settings_a_file_needs(plot, qtbot):
    dialog = SaveFigureDialog(plot)
    qtbot.addWidget(dialog)

    for name in ("background", "line_colour", "line_width", "ink",
                 "graph_shape", "format", "dpi", "width", "height"):
        assert hasattr(dialog, name), f"no {name} control"
    # Plot-owned styling lives on the plot's right-click menu, where one
    # choice reaches the screen and every export. The file dialog must not
    # reintroduce a second source of truth for it.
    for name in ("aspect", "text_px", "x_title", "y_title"):
        assert not hasattr(dialog, name), f"duplicate {name} control"


def test_an_untouched_control_changes_nothing(plot, qtbot):
    """Every one defaults to "as drawn", so saving is still one click."""
    dialog = SaveFigureDialog(plot)
    qtbot.addWidget(dialog)

    assert dialog._extra_styling() == {}


def test_the_values_reach_the_render(plot, qtbot):
    dialog = SaveFigureDialog(plot)
    qtbot.addWidget(dialog)
    dialog.graph_shape.setCurrentIndex(dialog.graph_shape.findData("square"))
    dialog.line_width.setValue(3.0)

    assert dialog._extra_styling() == {
        "canvas_shape": "square", "line_width": 3.0}


def test_the_screen_keeps_what_it_had(plot):
    """Styling is for the file; the plot behind the dialog is untouched."""
    before_font = plot.font_size()
    plot.plot.setLabel("bottom", "on screen")

    plot.styled_snapshot(240, background="#ffffff", font_size=22,
                         line_width=5.0, aspect=1.0, x_title="in the file")

    assert plot.font_size() == before_font
    assert plot.plot.getAxis("bottom").labelText == "on screen"
    assert not plot.plot.getViewBox().state.get("aspectLocked")
