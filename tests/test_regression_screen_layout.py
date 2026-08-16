"""The regression screen: results on the LEFT, pressable figure tiles right.

Instruction 119 section A, in the maintainer's words:

    "figures should pop up in a grid above the console"
    "if a figure is clicked it should fill the container"
    "the results for the regression shoild pop up in a container to the left
     of the figures"
    "when i say figure grid im thinking of tiles that can be pressed"

They were three pages of one tab stack, which satisfies none of it: picking a
row changes ONE POINT in the volcano, and nobody can see that happen if the
table and the figure are never on screen together. The tabs are gone.

Deliberately different from the parameter search (116), which IS tabs -- its
runs are navigation between runs, and picking one replaces the whole grid.
Anyone who unifies the two surfaces has removed the reason for both.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt


@pytest.fixture()
def screen(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    widget = AppScreen("regression")
    qtbot.addWidget(widget)
    return widget


def _figure(seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(4 + seed, 3))
    figure.add_subplot(111).plot([0, 1], [seed, 1])
    return figure


# --------------------------------------------------------------------------- #
#  The shape
# --------------------------------------------------------------------------- #

def test_the_results_are_beside_the_figures_not_behind_a_tab(screen):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QSplitter

    split = screen._figures_split
    assert isinstance(split, QSplitter)
    assert split.orientation() == Qt.Horizontal, (
        "the results are stacked above the figures, not beside them")
    assert split.widget(0) is screen._results_panel, (
        "the results panel is not the LEFT half")
    assert split.widget(1) is screen._figures_stack


def test_both_halves_are_visible_at_once(screen, qtbot):
    """The whole point. A tab stack shows one at a time and that is what
    made clicking a row useless."""
    screen._figures_card.show()
    qtbot.waitExposed(screen._figures_card)

    assert screen._results_panel.isVisibleTo(screen._figures_card)
    assert screen._figure_grid.isVisibleTo(screen._figures_card)


def test_the_old_tab_stack_is_gone(screen):
    """Named so that re-adding it fails here rather than quietly undoing the
    request."""
    assert getattr(screen, "_figures_tabs", None) is None


def test_a_module_without_results_is_untouched(qtbot):
    """Only the regression screen gets this. Every other module still opens
    into the one-at-a-time queue."""
    from spacr.qt.screens.app_screen import AppScreen

    other = AppScreen("mask")
    qtbot.addWidget(other)
    assert other._results_panel is None
    assert other._figures_stack is None


# --------------------------------------------------------------------------- #
#  Tiles that can be pressed
# --------------------------------------------------------------------------- #

def test_the_grid_is_what_shows_first(screen):
    assert screen._figures_stack.currentIndex() == 0
    assert screen._figures_stack.currentWidget() is screen._figure_grid


def test_pressing_a_tile_fills_the_container(screen):
    """"if a figure is clicked it should fill the container"."""
    for i in range(3):
        screen._on_figure_ready(_figure(i))
    screen._refresh_figure_grid()
    assert screen._figure_grid._cells, "no tiles were built"

    screen._figure_grid._cells[1].clicked.emit(1)

    assert screen._figures_stack.currentWidget() is screen._figure_detail
    assert screen._figure_queue.isVisibleTo(screen._figure_detail)


def test_a_tile_is_pressable_by_mouse_not_only_by_signal(screen, qtbot):
    """Driving the signal proves the wiring; driving the mouse proves the
    tile is actually a button-shaped thing the user can hit."""
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QMouseEvent

    screen._on_figure_ready(_figure(0))
    screen._refresh_figure_grid()
    cell = screen._figure_grid._cells[0]

    seen = []
    screen._figure_grid.figure_activated.connect(seen.append)
    cell.mousePressEvent(QMouseEvent(
        QMouseEvent.MouseButtonPress, QPoint(4, 4), Qt.LeftButton,
        Qt.LeftButton, Qt.NoModifier))

    assert seen == [0]
    assert cell.cursor().shape() == Qt.PointingHandCursor, (
        "a pressable tile should say so under the pointer")


def test_there_is_a_way_back_to_the_grid(screen):
    """A view you can enter and not leave is a trap."""
    screen._on_figure_ready(_figure(0))
    screen._refresh_figure_grid()
    screen._open_figure_from_grid(0)
    assert screen._figures_stack.currentWidget() is screen._figure_detail

    screen._show_figure_grid()

    assert screen._figures_stack.currentWidget() is screen._figure_grid


# --------------------------------------------------------------------------- #
#  The grid keeps up without being rebuilt seventeen times
# --------------------------------------------------------------------------- #

def test_a_burst_of_figures_is_one_rebuild(screen, qtbot, monkeypatch):
    """A run streams seventeen figures in and the grid is on screen for all
    of them. One relayout per arrival is sixteen wasted."""
    rebuilds = []
    monkeypatch.setattr(screen, "_refresh_figure_grid",
                        lambda: rebuilds.append(1))
    screen._grid_refresh.timeout.disconnect()
    screen._grid_refresh.timeout.connect(screen._refresh_figure_grid)

    for i in range(17):
        screen._on_figure_ready(_figure(i % 3))
    assert rebuilds == [], "rebuilt before the burst finished"

    qtbot.wait(400)
    assert rebuilds == [1], f"{len(rebuilds)} rebuilds for one burst"


def test_the_grid_holds_every_figure_the_run_made(screen):
    for i in range(5):
        screen._on_figure_ready(_figure(i % 3))
    screen._refresh_figure_grid()

    assert len(screen._figure_grid._cells) == 5


def test_showing_a_figure_does_not_draw_it_over_the_grid(screen):
    """_on_figure_ready calls _figure_queue.show(). If the queue were the
    stack's own page that would paint it on top of the grid, which is why it
    lives inside a wrapper."""
    screen._on_figure_ready(_figure(0))

    assert screen._figures_stack.currentWidget() is screen._figure_grid
    assert screen._figure_queue.parent() is screen._figure_detail
