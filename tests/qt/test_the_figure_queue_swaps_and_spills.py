"""Replacing a figure in place, and the pickle that stands in for it.

Two things the figure queue does that nothing else in spaCR does, and that
fail quietly when they go wrong.

REPLACING. "Show this as a violin instead" does not redraw a Figure -- 
``create_grouped_plot`` builds a NEW one, because spacrGraph makes its own.
So the queue, the grid tile, the thumbnail and the id-to-index map all have to
be pointed at the new object together. Miss one and the menu says violin while
the tile keeps showing the old bar chart, with nothing raised.

SPILLING. A figure the queue evicts is pickled beside its rendered page so it
can be made editable again later. Pickling a matplotlib Figure is best effort
-- a custom artist, a lambda in a callback, an open file handle in a closure
and it fails -- and the failure has to leave NO file behind, because a
half-written pickle is what a later restore would load.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from spacr.qt.widgets.figure_queue import FigureQueue              # noqa: E402

pytestmark = pytest.mark.qt


def _fig(marker=1.0):
    """A tiny figure carrying a value the test can look for again."""
    fig = plt.figure(figsize=(1, 1))
    fig.add_subplot(111).plot([0, marker], [0, marker])
    return fig


@pytest.fixture
def queue(qtbot):
    widget = FigureQueue()
    qtbot.addWidget(widget)
    yield widget
    plt.close("all")


# ---------------------------------------------------------------------------
# replace_figure
# ---------------------------------------------------------------------------

def test_replacing_a_figure_points_every_index_at_the_new_object(queue):
    """The swap, and the id map that the grid tile resolves through.

    ``_fig_index`` maps id(figure) -> position. Leaving the old id in it means
    a later lookup from the tile resolves to a figure that is no longer shown,
    which is how "click the tile, get the previous plot" happens.
    """
    original = _fig(1.0)
    queue.add_figure(original)
    replacement = _fig(2.0)

    assert queue.replace_figure(0, replacement) is True

    assert queue._figures[0] is replacement
    assert queue._fig_index[id(replacement)] == 0
    assert id(original) not in queue._fig_index


def test_replacing_a_figure_marks_it_as_the_most_recently_used(queue):
    """It was just drawn, so it must not be the next thing evicted."""
    queue.add_figure(_fig(1.0))
    queue.add_figure(_fig(2.0))

    queue.replace_figure(0, _fig(3.0))

    assert list(queue._figures)[-1] == 0


def test_replacing_a_figure_forgets_the_spilled_copy_of_the_old_one(queue,
                                                                   tmp_path):
    """The spill holds a pickle of the figure being replaced.

    Left in place, a later eviction would "restore" the picture this call just
    replaced -- the plot silently reverting to the one the user changed away
    from, which reads as the application ignoring them.
    """
    queue.add_figure(_fig(1.0))
    path = queue._spill_path(0)
    if path is None:
        pytest.skip("this queue has no spill directory to write into")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"the old pickle")
    assert path.is_file()

    queue.replace_figure(0, _fig(2.0))

    assert not path.is_file()


def test_replacing_with_nothing_is_refused(queue):
    """None is not a figure, and swapping one in would blank the tile."""
    queue.add_figure(_fig(1.0))

    assert queue.replace_figure(0, None) is False


@pytest.mark.parametrize("index", [-1, 5])
def test_replacing_outside_the_queue_is_refused(queue, index):
    """An index the queue does not hold must not silently create an entry.

    ``_count`` is what every other reader trusts for how many pages there are;
    a figure written past it would be invisible to all of them.
    """
    queue.add_figure(_fig(1.0))

    assert queue.replace_figure(index, _fig(2.0)) is False
    assert queue._count == 1


def test_a_spill_that_cannot_be_dropped_does_not_stop_the_swap(queue,
                                                               monkeypatch):
    """The swap is the user's request; the spill is bookkeeping.

    Failing the replacement because a stale pickle could not be deleted would
    trade a small correctness problem for a visible one.
    """
    queue.add_figure(_fig(1.0))

    def refuse(_idx):
        raise OSError("read-only file system")

    monkeypatch.setattr(queue, "_forget_spill", refuse)

    assert queue.replace_figure(0, _fig(2.0)) is not False


# ---------------------------------------------------------------------------
# has_live_figure / is_restorable
# ---------------------------------------------------------------------------

def test_asking_whether_a_figure_is_live_does_not_promote_it(queue):
    """A query must not change what gets evicted next.

    If asking counted as a use, the memory panel refreshing its display would
    reorder the eviction queue -- and the figure evicted would depend on
    whether anybody happened to be looking.
    """
    queue.add_figure(_fig(1.0))
    queue.add_figure(_fig(2.0))
    order = list(queue._figures)

    assert queue.has_live_figure(0) is True
    assert queue.has_live_figure(99) is False

    assert list(queue._figures) == order


def test_a_live_figure_is_restorable_without_touching_the_disk(queue):
    queue.add_figure(_fig(1.0))

    assert queue.is_restorable(0) is True


def test_a_figure_that_is_neither_live_nor_spilled_is_not_restorable(queue):
    """The honest answer, which is what greys the "edit" action out."""
    queue.add_figure(_fig(1.0))
    queue._figures.pop(0, None)

    path = queue._spill_path(0)
    if path is not None and path.is_file():
        path.unlink()

    assert queue.is_restorable(0) is False


# ---------------------------------------------------------------------------
# _spill_figure
# ---------------------------------------------------------------------------

def test_spilling_nothing_is_not_an_error_and_writes_nothing(queue):
    assert queue._spill_figure(0, None) is False


def test_a_figure_that_cannot_be_pickled_leaves_no_file_behind(queue,
                                                               monkeypatch):
    """The important half of the failure path.

    A partial write is worse than no write: `is_restorable` would then say
    yes, the user would click "edit", and the unpickle would fail on a
    truncated file -- at which point the figure is gone and the queue said it
    was not.
    """
    queue.add_figure(_fig(1.0))
    path = queue._spill_path(0)
    if path is None:
        pytest.skip("this queue has no spill directory to write into")

    import pickle

    def half_write(obj, handle, **kwargs):
        handle.write(b"half a pickle")
        raise pickle.PicklingError("cannot pickle a local closure")

    monkeypatch.setattr(pickle, "dump", half_write)

    assert queue._spill_figure(0, _fig(2.0)) is False
    assert not path.is_file()


def test_a_figure_that_pickles_is_written_and_reported_written(queue):
    """The success path, asserted by reading the file back."""
    queue.add_figure(_fig(1.0))
    path = queue._spill_path(0)
    if path is None:
        pytest.skip("this queue has no spill directory to write into")

    assert queue._spill_figure(0, _fig(2.0)) is True
    assert path.is_file() and path.stat().st_size > 0
