"""199 B: a figure that cannot be drawn explains itself.

    "for the matplot lib graphs, i can not see #2 when i click on it this
     run."

The tile had a picture on it -- the grid skips a null pixmap -- so the
raster existed when the grid was built and the full-size view showed
nothing when it was clicked.

WORSE THAN EMPTY, and this is what the fix is really about: `show_index`
only called `set_pixmap` when it HAD a pixmap, so a figure it could not
produce left the previous figure on screen. Clicking figure 2 showed figure
1, and nothing said so.

THE THIRD CANDIDATE IS RULED OUT HERE, not left open: `set_figures`
enumerates over the queue's own list and skips the holes with `continue`,
so a missing figure does not shift the indices of the ones after it. The
test below pins that, because it is the kind of thing a later "tidy up"
rewrites into a filter.
"""
from __future__ import annotations

import os
import tempfile

import matplotlib
import pytest

matplotlib.use("Agg")

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


@pytest.fixture
def queue(qtbot):
    import matplotlib.pyplot as plt

    from spacr.qt.widgets.figure_queue import FigureQueue

    widget = FigureQueue()
    qtbot.addWidget(widget)
    for i in range(3):
        figure = plt.figure()
        figure.add_subplot(111).plot([0, 1], [i, 1])
        widget.add_figure(figure)
    return widget


def _strand(queue, index):
    """Leave figure ``index`` with no picture anywhere."""
    queue._ram.pop(index, None)
    queue._figures.pop(index, None)
    queue._png_paths.pop(index, None)


class TestItSaysWhichOfTheThree:
    """Told apart rather than merged into one apology: a spill that will not
    restore is a disk problem, a missing path is a figure that was never
    saved, and an unreadable file is a corrupt one."""

    def test_never_saved(self, queue):
        _strand(queue, 1)
        said = queue._why_not_shown(1)
        assert "Figure 2" in said
        assert "no saved image" in said

    def test_saved_then_lost(self, queue):
        _strand(queue, 1)
        queue._png_paths[1] = "/nowhere/figure2.png"
        said = queue._why_not_shown(1)
        assert "gone" in said
        assert "/nowhere/figure2.png" in said     # the path, so it can be checked

    def test_present_but_unreadable(self, queue):
        _strand(queue, 1)
        handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        handle.write(b"not a png at all")
        handle.close()
        try:
            queue._png_paths[1] = handle.name
            said = queue._why_not_shown(1)
            assert "could not be read" in said
            assert handle.name in said
        finally:
            os.unlink(handle.name)

    def test_the_three_are_different_sentences(self, queue):
        _strand(queue, 1)
        never = queue._why_not_shown(1)
        queue._png_paths[1] = "/nowhere/figure2.png"
        lost = queue._why_not_shown(1)
        assert never != lost


class TestItNeverLeavesThePreviousFigureUp:

    def test_a_missing_figure_replaces_what_was_there(self, queue):
        """The bug behind the report: showing figure 1 and calling it 2."""
        queue.show_index(0)
        before = queue._view._pixmap_item

        _strand(queue, 1)
        queue.show_index(1)

        assert queue._view._pixmap_item is not None
        assert queue._view._pixmap_item is not before

    def test_the_explanation_is_a_real_picture(self, queue):
        _strand(queue, 1)
        pixmap = queue._explanation_pixmap(queue._why_not_shown(1))

        assert not pixmap.isNull()
        assert pixmap.width() > 0 and pixmap.height() > 0

    def test_it_is_painted_from_the_palette_not_a_literal(self, queue):
        """178's rule and 198's: a hex typed in reads on one theme and
        vanishes on the other, and the author sees only the one they use."""
        import inspect

        body = inspect.getsource(queue._explanation_pixmap.__func__)
        assert "active_palette" in body
        for literal in ("#fff", "#FFF", "#000", "Qt.white", "Qt.black"):
            assert literal not in body, literal

    def test_a_figure_that_is_fine_still_draws_itself(self, queue):
        """The explanation must not have become the answer for everything.

        Driven down the RASTER path deliberately. A healthy figure prefers
        its live canvas and returns before the raster view is touched, so
        asserting on `_pixmap_item` for an untouched figure tests the live
        path and calls it the raster one."""
        queue._figures.pop(0, None)          # no live canvas: raster it is
        assert not queue.has_live_figure(0)

        pixmap = queue._pixmap_for(0)
        assert pixmap is not None, "the PNG is still on disk"

        queue.show_index(0)
        assert queue._view._pixmap_item is not None
        # The real picture, not a manufactured explanation of its absence.
        assert queue._view._pixmap_item.pixmap().size() == pixmap.size()


class TestTheIndicesDoNotShift:
    """The candidate 199 B lists third, ruled out and kept ruled out."""

    def test_a_hole_does_not_renumber_the_tiles_after_it(self, qtbot):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPixmap

        from spacr.qt.widgets.figure_grid_view import FigureGridView

        grid = FigureGridView()
        qtbot.addWidget(grid)
        good = QPixmap(40, 30)
        good.fill(Qt.red)

        # Figure 1 (index 1) could not be read: a hole in the middle.
        grid.set_figures([good, None, good, good])

        assert [cell.index for cell in grid._cells] == [0, 2, 3]

    def test_clicking_the_tile_after_a_hole_opens_that_figure(self, qtbot):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPixmap

        from spacr.qt.widgets.figure_grid_view import FigureGridView

        grid = FigureGridView()
        qtbot.addWidget(grid)
        good = QPixmap(40, 30)
        good.fill(Qt.red)
        grid.set_figures([good, None, good])

        seen = []
        grid.figure_activated.connect(seen.append)
        last = grid._cells[-1]
        last.resize(60, 40)
        qtbot.mouseClick(last, Qt.LeftButton)

        # The third figure is index 2 even though it is the second TILE.
        assert seen == [2]
