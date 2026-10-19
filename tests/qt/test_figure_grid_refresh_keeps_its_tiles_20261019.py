"""A figure arriving during a run adds one tile; it does not rebuild the grid.

Lesson 13's batch run streams figures into the all-figures grid, and every
debounced refresh used to destroy every tile and smooth-scale every figure
again: 93-318 ms of GUI thread per arrival while the run was on screen. The
grid now keeps the tiles whose figure, caption and letter are unchanged, and
a tile already drawn at its width is not scaled again. Counted, not timed.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QPixmap  # noqa: E402

from spacr.qt.widgets import figure_grid_view as fgv  # noqa: E402

pytestmark = pytest.mark.qt


def _pixmap(width, height, colour=Qt.white):
    pixmap = QPixmap(width, height)
    pixmap.fill(colour)
    return pixmap


@pytest.fixture()
def grid(qtbot):
    widget = fgv.FigureGridView()
    qtbot.addWidget(widget)
    widget.resize(1200, 800)
    return widget


@pytest.fixture()
def scalings(monkeypatch):
    calls = []
    real = fgv.scaled_for

    def counting(source, *args, **kwargs):
        calls.append(source.cacheKey())
        return real(source, *args, **kwargs)

    monkeypatch.setattr(fgv, "scaled_for", counting)
    return calls


def test_a_new_figure_adds_one_tile_and_scales_only_that_figure(grid, scalings):
    figures = [_pixmap(400, 300), _pixmap(300, 300), _pixmap(500, 250)]
    titles = ["one", "two", "three"]
    assert grid.set_figures(figures, titles) == 3
    first = list(grid._cells)
    scalings.clear()

    figures.append(_pixmap(320, 240))
    titles.append("four")
    assert grid.set_figures(figures, titles) == 4

    assert grid._cells[:3] == first
    assert scalings == [figures[3].cacheKey()]


def test_an_unchanged_refresh_scales_nothing(grid, scalings):
    figures = [_pixmap(400, 300), _pixmap(300, 300)]
    grid.set_figures(figures, ["a", "b"])
    before = list(grid._cells)
    scalings.clear()

    grid.set_figures(figures, ["a", "b"])

    assert grid._cells == before
    assert scalings == []


def test_a_changed_figure_caption_or_letter_gets_a_new_tile(grid):
    figures = [_pixmap(400, 300), _pixmap(300, 300)]
    grid.set_figures(figures, ["a", "b"])
    old_first, old_second = grid._cells

    replaced = _pixmap(400, 300, Qt.black)
    grid.set_figures([replaced, figures[1]], ["a", "renamed"])

    assert grid._cells[0] is not old_first
    assert grid._cells[1] is not old_second
    assert grid._cells[0]._pixmap.cacheKey() == replaced.cacheKey()
    assert old_first.parent() is None and old_second.parent() is None

    kept_first, lettered_b = grid._cells
    grid.set_figures([replaced, figures[1]], ["a", "renamed"],
                     sections=[("run 1", 0, 1), ("run 2", 1, 1)])
    assert [cell.letter.lower() for cell in grid._cells] == ["a", "a"]
    assert grid._cells[0] is kept_first
    assert grid._cells[1] is not lettered_b


def test_removed_figures_leave_the_grid(grid):
    figures = [_pixmap(400, 300), _pixmap(300, 300), _pixmap(200, 300)]
    grid.set_figures(figures, ["a", "b", "c"])
    dropped = grid._cells[2]

    assert grid.set_figures(figures[:2], ["a", "b"]) == 2

    assert dropped not in grid._cells
    assert dropped.parent() is None


def test_a_new_width_or_density_still_redraws_a_kept_tile(grid, scalings):
    figures = [_pixmap(400, 300)]
    grid.set_figures(figures, ["a"])
    cell = grid._cells[0]
    scalings.clear()

    cell.fit_to(cell._fit_width + 40)
    assert scalings == [figures[0].cacheKey()]

    scalings.clear()
    cell._refit()
    assert scalings == [figures[0].cacheKey()]
