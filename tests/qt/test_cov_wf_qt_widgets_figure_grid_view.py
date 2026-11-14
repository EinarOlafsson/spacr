"""The gestures on the figure grid that nothing had yet driven.

Four of them are the half of a branch that means "and now do nothing": a
heading that never found the grid above it, a right button released on that
heading, a tile asked to redraw before it was ever drawn, and a saved
arrangement that carries no tile width. Each of those is what a user meets
when the grid is mid-rebuild or the workspace file came from another version,
and each has to be inert rather than loud -- an exception raised on the way
out of a Qt event handler takes the run panel down with it.

The fifth is the opposite: the right-click menu on the interactive regression
tile, the one live tile that IS a real figure, whose "restyle this graph"
gesture reaches a signal a screen is already wired to.

Every test here drives the branch BOTH ways, because an assertion that
nothing happened passes just as well against a test that exercised nothing.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QPixmap  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QWidget  # noqa: E402

from spacr.qt.hidpi import logical_size  # noqa: E402
from spacr.qt.widgets import figure_grid_view as fgv  # noqa: E402

pytestmark = pytest.mark.qt


def _pixmap(width, height):
    """A drawable figure image of a known shape."""
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.white)
    return pixmap


@pytest.fixture()
def grid(qtbot):
    """A grid wide enough to lay several cells across."""
    widget = fgv.FigureGridView()
    qtbot.addWidget(widget)
    widget.resize(1200, 800)
    return widget


class _RecordingView(QWidget):
    """An ancestor that knows how to toggle a section, and says when asked.

    ``_SectionHeader`` finds its view by walking up the parent chain for
    anything answering ``toggle_section`` -- the header is built inside a
    layout pass, so it cannot be handed the view. Recording the call is how
    the two halves of "did the gesture arrive" can be told apart.
    """

    def __init__(self):
        super().__init__()
        self.toggled = []

    def toggle_section(self, header):           # noqa: D102 - see class doc
        self.toggled.append(getattr(header, "section_key", None))
        return True


# ---------------------------------------------------------------------------
# a heading with no grid above it
# ---------------------------------------------------------------------------

def test_a_heading_that_lost_its_grid_folds_nothing_and_stays_usable(qtbot):
    """A heading outlives the view it was built in, and must not raise.

    Every relayout destroys the previous headings and builds new ones, and a
    heading is unparented from the body the instant the grid is rebuilt --
    which happens whenever a figure lands, including while the user's finger
    is on the key. If the fold reached through a ``None`` view, that keypress
    would raise out of ``keyPressEvent`` and take the run panel with it.
    Losing the fold is the acceptable outcome; losing the window is not.
    """
    view = _RecordingView()
    qtbot.addWidget(view)
    header = fgv._SectionHeader("run one", ("run one", 0), view)
    header.resize(200, 24)

    QTest.keyClick(header, Qt.Key_Return)

    assert view.toggled == [("run one", 0)]

    # Now the same widget, with the grid taken out from under it: the ONLY
    # difference between the two halves is whether an ancestor answers
    # `toggle_section`.
    header.setParent(None)
    qtbot.addWidget(header)

    QTest.keyClick(header, Qt.Key_Space)
    QTest.keyClick(header, Qt.Key_Enter)

    assert view.toggled == [("run one", 0)]
    assert header.is_expanded() is True
    assert header.text() == "run one"


# ---------------------------------------------------------------------------
# redrawing a cell for a new pixel density
# ---------------------------------------------------------------------------

def test_a_cell_only_redraws_itself_once_it_has_been_given_a_width(qtbot):
    """Dragging the window to a denser screen redraws figures, not blanks.

    A grid moved to a second monitor keeps its cell widths, so no relayout
    arrives to refit the pictures -- the device-ratio callback is what does.
    It can fire before the cell has ever been laid out (a tile built and
    reparented in the same turn), and refitting to a width of zero would
    scale the figure to nothing and leave an empty slot on the grid.
    """
    cell = fgv._FigureCell(0, _pixmap(200, 100), "a plate heatmap")
    qtbot.addWidget(cell)

    cell._refit()

    assert cell._fit_width == 0
    assert cell._image.pixmap().isNull() is True

    cell.fit_to(120)
    drawn = logical_size(cell._image.pixmap())

    assert cell._fit_width == 120
    assert drawn.width() == 120
    assert drawn.height() == 60          # the 2:1 shape is preserved

    # Blank it and let the ratio change re-fire: the SAME width comes back.
    cell._image.setPixmap(QPixmap())
    assert cell._image.pixmap().isNull() is True

    cell._refit()

    assert logical_size(cell._image.pixmap()).width() == 120
    assert cell._image.height() == drawn.height()


# ---------------------------------------------------------------------------
# the right-click menu on a live tile
# ---------------------------------------------------------------------------

def test_right_clicking_the_regression_tile_reaches_the_pinned_menu(grid):
    """The interactive volcano is the one tile that is a live figure.

    "all gigures should be editable by right clicking" -- and the regression
    screen listens on ``pinned_menu_requested``, not on the general
    per-key signal, because it was wired before there were other panels. A
    right-click that reached only the general signal would open no menu at
    all on the one tile where restyling does the most.
    """
    grid.set_live_tiles([("qq", _pixmap(300, 200), "Q-Q")])
    assert grid.set_pinned(_pixmap(400, 300), "volcano") is True

    keys = []
    pinned_positions = []
    grid.live_tile_menu_requested.connect(
        lambda key, position: keys.append(key))
    grid.pinned_menu_requested.connect(pinned_positions.append)

    pinned = grid._pinned
    pinned.customContextMenuRequested.emit(QPoint(4, 5))

    assert keys == [fgv.PINNED_KEY]
    assert pinned_positions == [pinned.mapToGlobal(QPoint(4, 5))]

    # The other half of the same branch: a panel that is NOT the regression
    # graph reaches the general signal only, or every panel's restyle menu
    # would be built from the volcano's figure.
    other = [cell for cell in grid._live if cell.live_key == "qq"][0]
    other.customContextMenuRequested.emit(QPoint(1, 2))

    assert keys == [fgv.PINNED_KEY, "qq"]
    assert pinned_positions == [pinned.mapToGlobal(QPoint(4, 5))]


def test_pressing_the_regression_tile_reaches_the_pinned_activation(grid):
    """The press half of the same split, for the same reason as the menu.

    A screen that has not learned about the other panels still opens the
    volcano; a screen that has, gets a key and can raise the right graph.
    """
    grid.set_live_tiles([("residuals", _pixmap(300, 200), "Residuals")])
    grid.set_pinned(_pixmap(400, 300), "volcano")

    keys = []
    opened_pinned = []
    grid.live_tile_activated.connect(keys.append)
    grid.pinned_activated.connect(lambda: opened_pinned.append(True))

    grid._pinned.clicked.emit(-1)

    assert keys == [fgv.PINNED_KEY]
    assert opened_pinned == [True]

    other = [cell for cell in grid._live if cell.live_key == "residuals"][0]
    other.clicked.emit(-1)

    assert keys == [fgv.PINNED_KEY, "residuals"]
    assert opened_pinned == [True]


# ---------------------------------------------------------------------------
# putting a saved arrangement back
# ---------------------------------------------------------------------------


def test_a_restored_fold_actually_hides_that_runs_figures(grid):
    """Restoring the arrangement has to reach the layout, not just a set.

    The saved state is applied while the grid already holds a run's figures
    -- reopening a workspace rebuilds the queue first -- so a collapsed key
    that only landed in the set would leave every figure on screen under a
    heading drawn as folded, and the run the user put away would be back.
    """
    grid.set_figures([_pixmap(400, 300), _pixmap(400, 300)],
                     sections=[("run one", 0, 2)])
    original_width = grid._target

    assert grid.apply_workspace_state(
        {"collapsed": [["run one", 0]]}) is True

    assert grid._target == original_width
    assert grid.is_section_collapsed("run one", 0) is True
    assert [cell.isHidden() for cell in grid._cells] == [True, True]

    # Bringing it back is the same document with the key gone: the figures
    # have to reappear, or a fold is a one-way door.
    assert grid.apply_workspace_state(
        {"collapsed": [], "cell_width": 260}) is True

    assert grid.is_section_collapsed("run one", 0) is False
    assert [cell.isHidden() for cell in grid._cells] == [False, False]
