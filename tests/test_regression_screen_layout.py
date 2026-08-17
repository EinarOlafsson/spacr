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
    assert split.widget(0) is screen._results_tabs, (
        "the results are not the LEFT half")
    # WHICH TAB IS FIRST CHANGED ON 2026-08-17; what this test is about did
    # not. Its subject is the SPLITTER -- results BESIDE the figures rather
    # than behind them -- and the tab order was only ever a passenger.
    # Instruction 128 J moved Runs in front of Results by name ("the run tab
    # should be before the results tab and results should be shown for the
    # chosen run"), so this now asserts the property the old line was reaching
    # for through position: Results is what the screen opens on.
    # `test_results_is_what_opens_first` in test_the_sweep_runs_are_a_tab.py
    # owns it; it is asserted here too because a layout test that quietly
    # stopped checking anything about the tabs is how it would be lost.
    assert screen._results_tabs.currentWidget() is screen._results_panel, (
        "Results is not what the screen opens on -- it is what a finished "
        "run opens into")
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


# --------------------------------------------------------------------------- #
#  The regression graph is the LIVE one, and it gets the big half
# --------------------------------------------------------------------------- #

def _real_results():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(11)
    n = 300
    return pd.DataFrame({
        "feature": ["Intercept"] + [f"fraction:grna[{i}_1]" for i in range(n)],
        "coefficient": np.concatenate([[0.19], rng.normal(size=n)]),
        "p_value": np.concatenate([[3e-46], rng.uniform(size=n)]),
    })


def test_the_volcano_is_not_squeezed_into_the_results_panel(screen):
    """"that is the slowest graph and the one i want to be interactive." A
    thumbnail above its own table is not that."""
    assert screen._results_panel.external_volcano is True
    tabs = screen._results_panel.tabs
    assert tabs.tabText(0) == "Coefficients", (
        "the results panel still owns the volcano")
    assert screen._results_panel.volcano.parent() is not None
    assert screen._figures_stack.indexOf(screen._volcano_page) == 2


def test_the_pinned_tile_is_a_readable_picture_not_a_blank_box(screen):
    """The tile was right; grabbing the widget was wrong.

    "the regression plot isnt shown in all figures (i want it also shown
    there)". It was tried once, drew as an empty box with a caption under it,
    and was deleted rather than fixed. The cause was never the tile: the
    volcano sits on a stacked page nobody has opened, inside a splitter
    collapsed to nothing, so it is 100x9 pixels and `grab()` returns 100x9 of
    one colour. Rendering the pyqtgraph SCENE ignores the widget's geometry
    entirely, which is the whole reason it works.
    """
    screen._results_panel.set_frame(_real_results())
    screen._refresh_figure_grid()

    pinned = screen._figure_grid._pinned
    assert pinned is not None, "the live regression graph is not on the grid"
    pixmap = pinned._pixmap
    assert pixmap.width() >= 400 and pixmap.height() >= 200, (
        f"the tile is {pixmap.width()}x{pixmap.height()}, which is the "
        f"collapsed widget's geometry rather than a picture of the plot")
    image = pixmap.toImage()
    colours = {image.pixel(x, y)
               for x in range(0, image.width(), 5)
               for y in range(0, image.height(), 5)}
    assert len(colours) > 20, (
        f"the tile has {len(colours)} distinct colours, so it is the blank "
        f"box again rather than a volcano")


def test_pressing_the_pinned_tile_opens_the_live_graph_not_a_picture(screen):
    """The route back that did not exist.

    Every other tile opens a saved figure. Before this, the ONLY way back to
    the interactive volcano was to select a row in the coefficient table --
    so a user who pressed "← All figures" had left the live graph and could
    not return to it deliberately.
    """
    screen._results_panel.set_frame(_real_results())
    screen._refresh_figure_grid()
    screen._show_figure_grid()
    assert screen._figures_stack.currentWidget() is screen._figure_grid

    screen._figure_grid._pinned.clicked.emit(-1)

    assert screen._figures_stack.currentWidget() is screen._volcano_page


def test_the_pinned_tile_never_shifts_which_figure_a_tile_opens(screen):
    """The trap this whole mechanism exists to avoid.

    `_FigureCell.index` is the position in the pixmap list, and
    `figure_activated` hands it straight to `FigureQueue.show_index`. A live
    tile INSERTED into that list would shift every figure after it, so every
    tile would open its neighbour's figure -- silently, because a figure does
    open. The pinned cell carries -1, is never one of `_cells`, and has its
    own signal.
    """
    screen._results_panel.set_frame(_real_results())
    for i in range(4):
        screen._on_figure_ready(_figure(i))
    screen._refresh_figure_grid()
    assert screen._figure_grid._pinned is not None, "no pinned tile to test"

    opened = []
    screen._figure_grid.figure_activated.connect(opened.append)
    for cell in screen._figure_grid._cells:
        cell.clicked.emit(cell.index)

    assert [cell.index for cell in screen._figure_grid._cells] == [0, 1, 2, 3]
    assert opened == [0, 1, 2, 3], (
        f"a tile opened the wrong figure: {opened}")
    assert screen._figure_grid._pinned.index == -1


def test_the_pinned_tile_takes_no_panel_letter(screen):
    """Publication lettering belongs to the RUN's figures.

    A letter on the live tile would make the run's own panel A into panel B,
    and the legend the run printed names A.
    """
    screen._results_panel.set_frame(_real_results())
    for i in range(3):
        screen._on_figure_ready(_figure(i))
    screen._refresh_figure_grid()

    assert screen._figure_grid._pinned.letter == ""
    assert [cell.letter for cell in screen._figure_grid._cells] == \
        ["A", "B", "C"]


def test_right_clicking_the_live_tile_offers_the_graphs_own_menu(screen):
    """"all gigures should be editable by right clicking".

    The queue's menu is built from a matplotlib figure at an index, and this
    tile is neither. Its own menu is the one that can restyle it.
    """
    screen._results_panel.set_frame(_real_results())
    screen._refresh_figure_grid()

    seen = []
    screen._figure_grid.pinned_menu_requested.connect(seen.append)
    screen._figure_grid._pinned.menu_requested.emit(-1, None)

    assert len(seen) == 1, "the live tile's right-click reached nobody"
    menu = screen._results_panel.volcano.build_style_menu()
    assert "Point size…" in [a.text() for a in menu.actions()]


def test_the_grid_keeps_a_pinned_tile_apart_from_the_runs_figures(qtbot):
    """A pinned tile keeps its own signal so it can never be mistaken for one
    of the run's figures -- a sentinel index down the shared signal would be
    the wrong figure waiting to be opened."""
    from PySide6.QtGui import QPixmap

    from spacr.qt.widgets.figure_grid_view import FigureGridView

    grid = FigureGridView()
    qtbot.addWidget(grid)
    pixmap = QPixmap(80, 60)
    pixmap.fill()
    assert grid.set_pinned(pixmap, "live") is True

    opened, pinned = [], []
    grid.figure_activated.connect(opened.append)
    grid.pinned_activated.connect(lambda: pinned.append(1))
    grid._pinned.clicked.emit(-1)

    assert pinned == [1] and opened == []


def test_a_pinned_tile_survives_a_new_run(qtbot):
    """clear() replaces the run's figures; a pinned tile is not one of them."""
    from PySide6.QtGui import QPixmap

    from spacr.qt.widgets.figure_grid_view import FigureGridView

    grid = FigureGridView()
    qtbot.addWidget(grid)
    pixmap = QPixmap(80, 60)
    pixmap.fill()
    grid.set_pinned(pixmap, "live")

    grid.set_figures([], [])

    assert grid._pinned is not None


def test_picking_a_guide_raises_the_graph_it_was_rung_on(screen):
    """A ring drawn on a view nobody is looking at is not a highlight."""
    frame = _real_results()
    screen._results_panel.set_frame(frame)
    screen._show_figure_grid()
    assert screen._figures_stack.currentWidget() is screen._figure_grid

    screen._results_panel.table.table.selectRow(4)

    assert screen._figures_stack.currentWidget() is screen._volcano_page
    assert screen._results_panel.volcano._selected_key is not None


def test_no_pinned_tile_before_anything_is_fitted(screen):
    """An empty plot tile invites a click that shows an empty plot."""
    screen._refresh_figure_grid()
    assert screen._figure_grid._pinned is None


def test_a_tile_with_no_picture_is_removed_rather_than_left_blank(qtbot):
    """The whole reason the live tile was deleted last time was a tile with no
    picture in it. When there is nothing to photograph the answer is no tile,
    not an empty one -- an empty tile invites a click that opens an empty
    plot."""
    from PySide6.QtGui import QPixmap

    from spacr.qt.widgets.figure_grid_view import FigureGridView

    grid = FigureGridView()
    qtbot.addWidget(grid)
    pixmap = QPixmap(80, 60)
    pixmap.fill()
    grid.set_pinned(pixmap, "live")
    assert grid._pinned is not None

    assert grid.set_pinned(None) is False
    assert grid._pinned is None
    assert grid.set_pinned(QPixmap()) is False, "a null pixmap became a tile"


# --------------------------------------------------------------------------- #
#  The gene tile appears WITH the graph
# --------------------------------------------------------------------------- #

def test_the_gene_tile_is_beside_the_graph_not_behind_a_tab(screen):
    """"when a gene is clicked a tile should appear with all the information
    on that gene" -- appear, beside the point that was clicked. A tile the
    user has to go and find is a tile they will not look at."""
    split = screen._gene_split
    assert split.widget(0) is screen._results_panel.volcano
    assert split.widget(1) is screen._results_panel.gene

    tabs = screen._results_panel.tabs
    assert "Gene" not in [tabs.tabText(i) for i in range(tabs.count())], (
        "the tile is in two places at once")


def test_an_unclicked_screen_is_all_graph(screen):
    """Nothing has been clicked, so there is nothing to say about a gene."""
    assert screen._gene_split.sizes()[1] == 0


def test_clicking_a_guide_opens_the_tile_and_fills_it(screen):
    screen._results_panel.set_frame(_real_results())
    screen._results_panel.table.table.selectRow(4)

    assert screen._gene_split.sizes()[1] > 0, "the tile stayed shut"
    assert screen._results_panel.gene.tile is not None, "the tile is empty"
    assert screen._results_panel.gene.feature == \
        screen._results_panel.volcano._selected_key, (
            "the tile is describing a different guide than the one rung")


def test_the_divider_is_the_users_after_the_first_click(screen):
    """Reasserting a size on every click would fight anyone who dragged the
    tile bigger to read it, or shut to see the whole plot."""
    screen._results_panel.set_frame(_real_results())
    screen._results_panel.table.table.selectRow(4)

    screen._gene_split.setSizes([900, 0])          # the user shuts it
    screen._results_panel.table.table.selectRow(9)

    assert screen._gene_split.sizes()[1] == 0, (
        "clicking again forced the tile back open")


# --------------------------------------------------------------------------- #
#  The publication figure
# --------------------------------------------------------------------------- #

def test_there_is_no_publication_button(screen):
    """It was the wrong answer and it is gone.

    "there is a non functional publication button that i did not ask for i
    asked you to make the all figures pannel publication style (with each
    panel having an uppercase letter) and be on a grid."

    The GRID is the publication panel. A button beside it that draws a
    different figure is a second surface answering the same question.
    """
    from PySide6.QtWidgets import QPushButton

    buttons = [b.text() for b in screen._figure_detail.findChildren(QPushButton)]
    assert not any("Publication" in text for text in buttons), buttons


def test_it_draws_the_sheet_into_the_ordinary_queue(screen):
    """Not a bespoke viewer: the sheet restyles, exports and saves through
    the same path as every other figure, so it inherits those fixes instead
    of collecting its own copies of the bugs."""
    screen._results_panel.set_frame(_real_results())
    before = screen._figure_queue.count()

    screen._show_publication_sheet()

    assert screen._figure_queue.count() == before + 1
    assert screen._figures_stack.currentWidget() is screen._figure_detail


def test_with_no_table_it_says_so_instead_of_drawing_nothing(screen):
    screen._show_publication_sheet()

    text = screen._console.copy_all() if hasattr(screen._console, "copy_all") \
        else ""
    assert "nothing to draw" in text.lower()
    assert screen._figure_queue.count() == 0


def test_the_legend_reaches_the_console(screen):
    """A journal figure without its legend is half a figure, and the legend
    is generated from the panels so it cannot describe a different one."""
    screen._results_panel.set_frame(_real_results())
    screen._show_publication_sheet()

    text = screen._console.copy_all() if hasattr(screen._console, "copy_all") \
        else ""
    assert "(A)" in text and "tested coefficients" in text
