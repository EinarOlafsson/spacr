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

    def test_every_figure_gets_exactly_one_slot(self):
        """A plate is 24x16 wells; a volcano is square. They are not the
        same shape and must not get the same box."""
        from spacr.qt.widgets.figure_grid_view import cell_span

        # ONE SLOT PER FIGURE, whatever its shape. A wide figure used to take
        # a DOUBLE cell, so four plate heatmaps took eight slots and wrapped
        # onto two rows: "they should take 1 slot per plate so in my case 4
        # slots". The aspect ratio is preserved INSIDE the cell, so a wide
        # figure simply sits shorter in its slot.
        assert cell_span(1.0) == 1          # square plot
        assert cell_span(1.5) == 1          # plate heatmap
        assert cell_span(3.0) == 1          # anything at all

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


class TestThePublicationGrid:
    """"i asked you to make the all figures pannel publication style (with
    each panel having an uppercase letter) and be on a grid"."""

    def _pixmaps(self, n, wide=False):
        from PySide6.QtGui import QPixmap

        out = []
        for _ in range(n):
            pixmap = QPixmap(300 if not wide else 450, 200)
            pixmap.fill()
            out.append(pixmap)
        return out

    def test_every_tile_carries_an_upper_case_letter(self, qtbot):
        from spacr.qt.widgets.figure_grid_view import FigureGridView

        grid = FigureGridView()
        qtbot.addWidget(grid)
        grid.set_figures(self._pixmaps(4), ["a", "b", "c", "d"])

        assert [c.letter for c in grid._cells] == ["A", "B", "C", "D"]

    def test_the_lettering_carries_past_z(self, qtbot):
        from spacr.qt.widgets.figure_grid_view import _letter_for

        assert _letter_for(0) == "A"
        assert _letter_for(25) == "Z"
        assert _letter_for(26) == "AA"

    def test_the_letter_is_actually_on_the_tile(self, qtbot):
        """Storing it is not showing it."""
        from PySide6.QtWidgets import QLabel

        from spacr.qt.widgets.figure_grid_view import FigureGridView

        grid = FigureGridView()
        qtbot.addWidget(grid)
        grid.set_figures(self._pixmaps(2), ["a", "b"])

        texts = [w.text() for w in grid._cells[1].findChildren(QLabel)]
        assert "B" in texts, texts

    def test_four_wide_figures_take_four_slots(self, qtbot):
        """Not eight. The exact complaint: four plate heatmaps wrapped onto
        two rows because each took a double-width cell."""
        from spacr.qt.widgets.figure_grid_view import FigureGridView

        grid = FigureGridView()
        qtbot.addWidget(grid)
        grid.resize(1400, 600)
        grid.set_figures(self._pixmaps(4, wide=True), ["p1", "p2", "p3", "p4"])

        spans = [grid._grid.getItemPosition(i)[3]
                 for i in range(grid._grid.count())]
        assert spans == [1, 1, 1, 1], spans

    def test_a_tile_does_not_paint_its_own_ground(self, qtbot):
        """"on the grid (all figures) the graphs still have a black
        background" -- the figures are transparent; the tile behind them was
        not, so every one sat on a slab."""
        from spacr.qt.widgets.figure_grid_view import FigureGridView

        grid = FigureGridView()
        qtbot.addWidget(grid)
        grid.set_figures(self._pixmaps(1), ["only"])

        cell = grid._cells[0]
        assert cell.autoFillBackground() is False
        assert "transparent" in cell.styleSheet()
        assert cell._image.autoFillBackground() is False
