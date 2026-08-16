"""A run's figures on a grid, each at its own aspect ratio.

The panel showed one figure at a time, stretched into whatever shape it
happened to be. A plate heatmap squashed into a square is no longer a heatmap
of a plate: the wells stop being square and positional artefacts, the reason
to look at one, become invisible.
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _pixmap(width, height):
    from PySide6.QtGui import QPixmap

    pixmap = QPixmap(width, height)
    pixmap.fill()
    return pixmap


class TestTheLayout:

    def test_a_wider_panel_shows_more_figures_not_bigger_ones(self):
        """The opposite of what stretch-to-fit does."""
        from spacr.qt.widgets.figure_grid_view import cells_across

        assert cells_across(1400) > cells_across(700) > cells_across(320)
        assert cells_across(320) == 1
        assert cells_across(0) == 1

    def test_a_wide_figure_gets_a_wide_cell(self):
        """A plate is 24x16 wells; a volcano is square. They are not the
        same shape and must not get the same box."""
        from spacr.qt.widgets.figure_grid_view import cell_span

        assert cell_span(1.0) == 1          # square plot
        assert cell_span(1.5) == 2          # plate heatmap
        assert cell_span(2.4) == 2          # very wide

    def test_the_number_of_columns_is_bounded(self):
        """One figure must not eat the panel, and thirty must not be
        unreadable."""
        from spacr.qt.widgets.figure_grid_view import cells_across

        assert 1 <= cells_across(100) <= 6
        assert 1 <= cells_across(10000) <= 6


class TestTheGrid:

    @pytest.fixture()
    def grid(self, qtbot):
        from spacr.qt.widgets.figure_grid_view import FigureGridView

        widget = FigureGridView()
        qtbot.addWidget(widget)
        widget.resize(1400, 900)
        return widget

    def test_every_figure_keeps_the_aspect_it_was_drawn_at(self, grid):
        shapes = [(800, 800), (1200, 800), (1600, 700)]
        grid.set_figures([_pixmap(w, h) for w, h in shapes])
        seen = [round(cell.aspect(), 2) for cell in grid._cells]
        assert seen == [round(w / h, 2) for w, h in shapes]

    def test_a_run_worth_of_figures_all_appear(self, grid):
        """Seventeen is what a regression run produces; all of them scroll."""
        assert grid.set_figures([_pixmap(800, 800) for _ in range(17)]) == 17

    def test_clicking_a_cell_reports_which_one(self, grid):
        grid.set_figures([_pixmap(800, 800) for _ in range(4)])
        seen = []
        grid.figure_activated.connect(seen.append)
        grid._cells[2].clicked.emit(2)
        assert seen == [2]

    def test_null_figures_are_skipped_rather_than_shown_blank(self, grid):
        from PySide6.QtGui import QPixmap

        assert grid.set_figures([_pixmap(800, 800), QPixmap(), None]) == 1

    def test_clearing_removes_every_cell(self, grid):
        grid.set_figures([_pixmap(800, 800) for _ in range(5)])
        grid.clear()
        assert grid._cells == []
        assert grid._grid.count() == 0

    def test_the_image_is_never_set_to_scale_its_contents(self, grid):
        """setScaledContents(True) IS the stretch this replaces."""
        grid.set_figures([_pixmap(1600, 700)])
        assert not grid._cells[0]._image.hasScaledContents()
