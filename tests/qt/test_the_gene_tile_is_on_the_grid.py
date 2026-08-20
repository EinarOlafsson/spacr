"""121's last owed item, and the broken window found under it.

"the tile in the FIGURE GRID rather than only in a tab."

THE CONTRACT ALREADY ALLOWED IT, which the instruction did not know. It
recorded that `_FigureCell.index` is the position in the pixmap list and that
inserting a tile would shift every figure after it -- true of a FIGURE tile.
But the grid already carries tiles that are not the run's figures: a live
tile is built with index ``-1`` and a `live_key`, "precisely so none of them
can ever be mistaken for a figure". The gene tile is that same shape and
costs no change to the index contract at all.

AND UNDER IT: seven tests asserting `currentWidget() is <widget>` had been
failing for a while, because `_figures_stack` holds a `grid_page` built
around the grid and `_results_tabs` holds a splitter around the results
panel. The widget is not the page. Those assertions read as if they were
about layout, which is why the failure sat unclaimed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def screen(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    widget = AppScreen("regression")
    qtbot.addWidget(widget)
    return widget


class TestTheScreenSaysWhatItIsShowing:

    def test_the_grid_is_recognised_inside_its_page(self, screen):
        screen._show_figure_grid()

        assert screen.showing_the_figure_grid()
        assert not screen.showing_the_live_graph()

    def test_the_live_graph_is_recognised_too(self, screen):
        screen._show_regression_graph()

        assert screen.showing_the_live_graph()
        assert not screen.showing_the_figure_grid()

    def test_a_missing_container_is_not_showing_anything(self, screen):
        assert screen._is_the_page(None, None) is False

    def test_a_widget_that_is_the_page_itself_still_counts(self, screen,
                                                           qtbot):
        """The accessor must not REQUIRE the wrapping -- a container that
        holds the widget directly is the simpler case and still true."""
        from PySide6.QtWidgets import QStackedWidget, QWidget

        stack = QStackedWidget()
        qtbot.addWidget(stack)
        page = QWidget()
        stack.addWidget(page)
        stack.setCurrentWidget(page)

        assert screen._is_the_page(stack, page)


class TestTheGeneTileIsAGridTile:

    def test_no_gene_means_no_tile(self, screen):
        """A tile saying "nothing selected" is a second way of saying what
        the empty Gene tab already says, and it would take a cell from the
        figures every run.

        `to_pixmap` CANNOT BE THE TEST for that: it renders a placeholder
        when nothing is selected, so a null-pixmap check never fired and the
        grid grew an empty tile on every run. `feature()` is what the panel
        is showing.
        """
        panel = screen._results_panel

        assert screen._gene_tile_entry(panel) == []

    def test_a_panel_without_a_gene_tile_is_not_an_error(self, screen):
        class Bare:
            pass

        assert screen._gene_tile_entry(Bare()) == []

    def test_the_entry_is_the_shape_set_live_tiles_takes(self, screen,
                                                         monkeypatch):
        from PySide6.QtGui import QPixmap

        class _Gene:
            def feature(self):
                return "TGGT1_231640"

            def to_pixmap(self, width=0):
                return QPixmap(120, 80)

        class _Panel:
            gene = _Gene()

        entry = screen._gene_tile_entry(_Panel())

        assert len(entry) == 1
        key, pixmap, title = entry[0]
        assert key == screen.GENE_TILE_KEY
        assert not pixmap.isNull()
        assert title == "Gene"

    def test_a_gene_tile_that_will_not_render_is_skipped(self, screen):
        class _Gene:
            def feature(self):
                return "TGGT1_231640"

            def to_pixmap(self, width=0):
                raise RuntimeError("no gene data")

        class _Panel:
            gene = _Gene()

        assert screen._gene_tile_entry(_Panel()) == [], (
            "a tile is a courtesy; it must not be why the grid does not draw")

    def test_it_carries_no_figure_index(self, qtbot):
        """The whole reason it costs nothing: a live-shaped tile is -1, so it
        cannot shift which figure any other tile opens."""
        from PySide6.QtGui import QPixmap

        from spacr.qt.widgets.figure_grid_view import FigureGridView

        grid = FigureGridView()
        qtbot.addWidget(grid)
        grid.set_live_tiles([("gene", QPixmap(120, 80), "Gene")])

        assert all(cell.index == -1 for cell in grid._live)
        assert grid.live_tile_keys() == ["gene"]
