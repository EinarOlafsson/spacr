"""Clearing the figure grid discards every figure tile but keeps the live tiles."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QPixmap

from spacr.qt.widgets import figure_grid_view as fg

pytestmark = pytest.mark.qt


def _pix(colour="#336699"):
    pixmap = QPixmap(24, 24)
    pixmap.fill(QColor(colour))
    return pixmap


@pytest.fixture
def grid(qtbot):
    view = fg.FigureGridView()
    qtbot.addWidget(view)
    view.resize(400, 300)
    return view


def test_clear_discards_figure_tiles_but_never_the_live_ones(grid, monkeypatch):
    grid.set_live_tiles([("regression", _pix()), ("qq", _pix("#aa5533"))])
    grid.set_figures([_pix(), _pix(), _pix()], titles=["a", "b", "c"])
    live = list(grid._live)
    figures = list(grid._cells)
    assert len(live) == 2 and len(figures) == 3
    discarded = []
    monkeypatch.setattr(fg.FigureGridView, "_discard",
                        staticmethod(discarded.append))
    grid.clear()
    assert all(any(d is cell for d in discarded) for cell in figures)
    assert not any(d is cell for d in discarded for cell in live)
    assert len(discarded) == len({id(d) for d in discarded}), (
        "a tile seen in the grid and in the cell list is discarded once")
    assert grid._cells == [] and grid._live == live


def test_clear_also_discards_a_figure_tile_the_grid_no_longer_holds(grid, monkeypatch):
    grid.set_figures([_pix(), _pix()], titles=["a", "b"])
    figures = list(grid._cells)
    while grid._grid.count():
        grid._grid.takeAt(0)
    discarded = []
    monkeypatch.setattr(fg.FigureGridView, "_discard",
                        staticmethod(discarded.append))
    grid.clear()
    assert [id(d) for d in discarded] == [id(c) for c in figures]
