"""The search figure grid reflows its columns when its viewport is resized.

Pinned behaviour of :class:`spacr.qt.widgets.figure_grid.SearchFigureGrid`:
widening the grid on screen re-lays the figures out in more columns once
the resize debounce has run, without any call from the caller.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.figure_grid import SearchFigureGrid  # noqa: E402

pytestmark = pytest.mark.qt


def test_widening_the_grid_adds_columns(qtbot, tmp_path):
    grid = SearchFigureGrid()
    qtbot.addWidget(grid)
    grid.resize(260, 900)
    grid.show()
    qtbot.waitExposed(grid)
    for n in range(8):
        grid.add_figure(str(tmp_path / f"trial_{n}.png"), {"trial": n})
    narrow = grid.columns()
    assert narrow >= 1

    grid.resize(1600, 900)

    qtbot.waitUntil(lambda: grid.columns() > narrow, timeout=3000)
    assert grid.count() == 8
