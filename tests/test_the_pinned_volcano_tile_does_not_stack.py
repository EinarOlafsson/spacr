"""The volcano thumbnail is ONE volcano, not every volcano so far.

Reported 2026-08-17 (instruction 128 P, item 2): "the thumbnail iage of the
volcano plot looks like several volcano plot itterations pasted on top of each
other".

The first diagnosis -- that the LIVE pyqtgraph scene accumulates data items --
was measured and RULED OUT the same day: across five consecutive redraws
through the baseline, level and threshold paths `plot.listDataItems()` went
1, 1, 0, 0, 0, so `_reset_scene` is reached and the scene is clean.

It is the GRID that accumulates, and specifically `FigureGridView.set_pinned`.
The live volcano is pinned as a tile by `_pin_regression_graph`, which runs on
every grid refresh -- once per figure burst while a run streams, plus every
time the user returns to the grid or right-clicks the tile. `set_pinned` used
to rebind `_pinned` and relayout, and `_relayout`'s `takeAt` removes a widget
from the LAYOUT while leaving it a visible child of the body AT ITS OLD
GEOMETRY. `clear()` could not collect the strays either, because it walks the
layout and a stray is no longer in it. So every pin left the previous tile
painted underneath the new one at identical coordinates.

That is invisible with opaque pictures and glaring with these: `FastPlot
.snapshot` returns a TRANSPARENT pixmap on purpose, so every buried volcano
showed straight through the ones in front of it. Twelve runs on one screen
(ols_11, ols_12, guide_permutation, guide_permutation_1) is a dozen volcanoes
painted at once -- exactly what was reported.

Measured on the real widget before the fix: five `set_pinned` calls left five
visible cells all at (6, 6, 229, 171) and all five pictures painted.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _block(colour, width=200, height=140):
    """An opaque picture -- stands in for a snapshot with a background."""
    from PySide6.QtGui import QColor, QPixmap

    pixmap = QPixmap(width, height)
    pixmap.fill(QColor(colour))
    return pixmap


def _one_dot(x, colour, width=200, height=140):
    """A TRANSPARENT page with one opaque dot -- one volcano iteration.

    Shaped like what `FastPlot.snapshot` actually returns: the exporter is
    given a transparent background so the tile does not sit on a slab. Each
    iteration puts its dot somewhere else, so counting the dots visible at
    once counts the iterations painted at once.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QPainter, QPixmap

    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(colour))
    painter.drawEllipse(x, 50, 30, 30)
    painter.end()
    return pixmap


def _tiles(grid):
    """Every figure tile still parented to the grid's body.

    The BODY, not the layout: the whole bug is a widget that has left the
    layout and is still a visible child painting itself, so a count taken off
    the layout would have reported one tile throughout.
    """
    from spacr.qt.widgets.figure_grid_view import _FigureCell

    return [child for child in grid._body.children()
            if isinstance(child, _FigureCell)]


@pytest.fixture()
def grid(qtbot):
    from spacr.qt.widgets.figure_grid_view import FigureGridView

    view = FigureGridView()
    qtbot.addWidget(view)
    view.resize(700, 500)
    return view


# --------------------------------------------------------------------------- #
#  The tile itself
# --------------------------------------------------------------------------- #

def test_pinning_the_volcano_twelve_times_leaves_one_tile(grid):
    """`_pin_regression_graph` runs on every grid refresh, so a screen that
    has done twelve runs has pinned a dozen times. Before the fix that was a
    dozen live tiles; the user saw all of them."""
    for step in range(12):
        assert grid.set_pinned(_block("#3355ff"), "regression — interactive")

    assert len(_tiles(grid)) == 1


def test_the_buried_tiles_were_at_the_very_same_coordinates(grid):
    """Why it read as "pasted on top of each other" rather than as a mess:
    every pin lands in slot 0, so the strays are pixel-aligned with the tile
    in front of them."""
    for _ in range(4):
        grid.set_pinned(_block("#3355ff"), "regression — interactive")

    geometries = {(tile.geometry().x(), tile.geometry().y())
                  for tile in _tiles(grid)}
    assert len(geometries) == 1, geometries
    assert len(_tiles(grid)) == 1


def test_only_the_newest_volcano_is_painted(grid, qtbot):
    """The visible symptom, driven end to end.

    Three transparent snapshots, each with its dot somewhere else. If the
    tiles stack, all three dots are on screen at once -- which is the
    complaint. Only the newest should be.

    The event loop is turned BETWEEN the pins, and that is not decoration: a
    stray tile only paints once it has been through a show and a layout pass,
    which is exactly what a real screen gives it -- `_pin_regression_graph`
    fires on a 250 ms debounce, seconds apart. Pinning three times inside one
    turn hides the bug this test exists to catch.
    """
    grid.show()
    for x, colour in ((10, "#ff0000"), (80, "#00ff00"), (150, "#0000ff")):
        grid.set_pinned(_one_dot(x, colour), "regression — interactive")
        qtbot.wait(10)

    image = grid._body.grab().toImage()
    painted = set()
    for y in range(image.height()):
        for x in range(image.width()):
            colour = image.pixelColor(x, y)
            red, green, blue = colour.red(), colour.green(), colour.blue()
            if red > 200 and green < 60 and blue < 60:
                painted.add("first")
            elif green > 200 and red < 60 and blue < 60:
                painted.add("second")
            elif blue > 200 and red < 60 and green < 60:
                painted.add("third")
    assert painted == {"third"}, painted


def test_removing_the_tile_takes_it_off_the_body_too(grid):
    """A run that has fitted nothing gets no tile, and "no tile" has to mean
    no widget -- not an unparented one still painting the last screen's
    volcano over an empty grid."""
    grid.set_pinned(_block("#3355ff"), "regression — interactive")
    assert len(_tiles(grid)) == 1

    assert grid.set_pinned(None) is False

    assert _tiles(grid) == []


def test_pinning_still_works_after_all_that(grid):
    """The tile has to survive being fixed: it is the only route from the
    grid back to the interactive volcano."""
    grid.set_pinned(_block("#3355ff"), "regression — interactive")
    grid.set_pinned(None)

    assert grid.set_pinned(_block("#33ff55"), "regression — interactive")
    tiles = _tiles(grid)
    assert len(tiles) == 1
    assert tiles[0].index == -1


def test_the_tile_survives_the_runs_figures_being_replaced(grid):
    """`clear()` deliberately spares the pinned tile -- a run streaming new
    figures must not make the interactive graph disappear. Destroying the
    strays must not have taken that with it."""
    grid.set_pinned(_block("#3355ff"), "regression — interactive")

    grid.set_figures([_block("#888888"), _block("#999999")], ["a", "b"])

    assert grid._pinned is not None
    assert len(_tiles(grid)) == 3       # the pinned one plus the two figures


# --------------------------------------------------------------------------- #
#  The maintainer's own path
# --------------------------------------------------------------------------- #

def test_the_regression_screen_pins_one_volcano_however_often_it_refreshes(
        qtbot):
    """The bug as it actually reaches a user, through the real screen.

    `_pin_regression_graph` is called by `_refresh_figure_grid`, which runs on
    a 250 ms debounce for every burst of figures a run streams, plus every
    return to the grid and every right-click on the tile. Nothing here is
    stubbed: a real coefficient table, the real interactive volcano, the real
    `FastPlot.snapshot`, the real grid.

    Measured through this exact path: five refreshes left FIVE tiles before
    the fix and one after.
    """
    pytest.importorskip("pyqtgraph")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import numpy as np
    import pandas as pd

    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    assert screen._results_panel is not None

    rng = np.random.default_rng(0)
    rows = 200
    frame = pd.DataFrame({
        "feature": [f"fraction:grna[{i}_1]" for i in range(rows)],
        "coefficient": rng.normal(0, .5, rows),
        "p_value": rng.uniform(size=rows),
        "q_value": rng.uniform(size=rows),
        "condition": list(rng.choice(["nc", "other"], rows, p=[.1, .9])),
        "multiple_testing_method": "fdr_bh"})
    assert screen._results_panel.set_frame(frame)
    # No tile at all would make the count below pass for the wrong reason.
    assert screen._results_panel.volcano.snapshot() is not None

    for _ in range(5):
        screen._refresh_figure_grid()
        qtbot.wait(5)

    pinned = [tile for tile in _tiles(screen._figure_grid)
              if tile.index == -1]
    assert len(pinned) == 1


# --------------------------------------------------------------------------- #
#  The same leak, on a folded run's cells
# --------------------------------------------------------------------------- #

def test_a_folded_runs_cells_are_destroyed_when_the_grid_is_rebuilt(grid):
    """The other half of "the layout is not the whole grid".

    `_relayout` deliberately leaves a folded run's cells OUT of the layout, so
    the next run flows up under the folded heading instead of into a hole.
    `clear()` walked the layout, so those cells were never destroyed: `_cells`
    was emptied out from under them while they stayed children of the body.
    Nothing on screen, but a sweep that folds its runs away leaked one widget
    per figure, and this is the accumulation that ate the volcano tile.
    """
    figures = [_block("#444444") for _ in range(4)]
    grid.set_figures(figures, ["a", "b", "c", "d"],
                     sections=[("ols_11", 0, 4)])
    grid.set_section_collapsed("ols_11", 0, True)
    assert grid.is_section_collapsed("ols_11", 0)
    assert len(_tiles(grid)) == 4

    grid.set_figures([_block("#555555")], ["e"], sections=[("ols_12", 0, 1)])

    assert len(_tiles(grid)) == 1
