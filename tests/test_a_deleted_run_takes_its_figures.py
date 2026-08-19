"""What leaves the Runs list leaves the figures with it. Instruction 146.

`FigureQueue` had `mark_run` and `run_sections` but no way to drop ONE run's
tiles -- `clear()` is all-or-nothing, so removing a run meant losing every other
run's figures too, and the Runs tab could not offer it.

THE HARD PART IS THE RENUMBERING. Every figure is keyed by a dense integer
index across six maps, and `_runs` records each section's start as one of those
integers, so removing a section from the middle has to shift everything above it
down in all six and move every later section's start with them.
"""

import spacr

assert "/codex/repo/spacr/" in spacr.__file__, spacr.__file__

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _queue(qtbot):
    from spacr.qt.widgets.figure_queue import FigureQueue

    queue = FigureQueue()
    qtbot.addWidget(queue)
    return queue


def _figure(n):
    # CLOSED AFTER HANDING IT OVER. pyplot keeps a reference to every figure
    # until it is closed, and a test that opens two dozen trips matplotlib's
    # own "More than 20 figures" warning -- which is the leak HANDOFF 3d says
    # makes the Qt suite decelerate.
    fig = plt.figure()
    fig.add_subplot(111).plot([0, 1], [0, n])
    return fig


def _fill(queue, plan):
    """plan: [(label, how_many)] -> the queue, filled run by run."""
    for label, count in plan:
        queue.mark_run(label)
        for i in range(count):
            fig = _figure(i)
            queue.add_figure(fig)
            plt.close(fig)
    return queue


def test_the_middle_run_goes_and_the_others_renumber(qtbot):
    queue = _fill(_queue(qtbot), [("A", 2), ("B", 3), ("C", 2)])
    assert [(n, c) for n, _s, c in queue.run_sections()] == [
        ("A", 2), ("B", 3), ("C", 2)]

    assert queue.forget_run("B") == 3

    sections = queue.run_sections()
    assert [(n, c) for n, _s, c in sections] == [("A", 2), ("C", 2)]
    # C MOVED DOWN. Leaving its start where it was would point it into the
    # hole B left, and every figure it names would be the wrong one.
    assert dict((n, s) for n, s, _c in sections)["C"] == 2
    assert queue._count == 4


def test_the_first_run_can_go(qtbot):
    queue = _fill(_queue(qtbot), [("A", 2), ("B", 2)])
    assert queue.forget_run("A") == 2
    sections = queue.run_sections()
    assert [(n, s, c) for n, s, c in sections] == [("B", 0, 2)]


def test_the_last_run_can_go(qtbot):
    queue = _fill(_queue(qtbot), [("A", 2), ("B", 2)])
    assert queue.forget_run("B") == 2
    assert [(n, c) for n, _s, c in queue.run_sections()] == [("A", 2)]
    assert queue._count == 2


def test_every_per_index_map_is_renumbered(qtbot):
    """Six maps are keyed by the index. A miss leaves a figure keyed to a
    slot that now belongs to a different run, which is worse than losing it.
    """
    queue = _fill(_queue(qtbot), [("A", 2), ("B", 2), ("C", 2)])
    queue.forget_run("B")

    for name in ("_figures", "_titles", "_png_paths", "_ram", "_pdf_state"):
        mapping = getattr(queue, name)
        assert all(0 <= key < queue._count for key in mapping), (
            f"{name} still holds an index outside the queue")
    assert all(0 <= value < queue._count
               for value in queue._fig_index.values())


def test_an_unknown_label_is_not_a_failure(qtbot):
    """A run that drew nothing is a run with nothing to forget."""
    queue = _fill(_queue(qtbot), [("A", 2)])
    assert queue.forget_run("nobody") == 0
    assert queue._count == 2


def test_a_run_that_drew_nothing_loses_its_mark(qtbot):
    queue = _queue(qtbot)
    queue.mark_run("A")
    queue.add_figure(_figure(1))
    queue.mark_run("empty")
    assert queue.forget_run("empty") == 0
    assert "empty" not in [n for n, _s, _c in queue.run_sections()]


def test_the_current_index_follows(qtbot):
    """A current index inside the forgotten span names a figure that is gone.

    Left alone the queue shows the wrong figure, which is the same class of
    bug as a mark pointing at a run that is not on screen.
    """
    queue = _fill(_queue(qtbot), [("A", 2), ("B", 2), ("C", 2)])
    queue._current = 5                      # in C
    queue.forget_run("B")
    assert queue._current == 3

    queue2 = _fill(_queue(qtbot), [("A", 2), ("B", 2)])
    queue2._current = 3                     # in B, which is about to go
    queue2.forget_run("B")
    assert 0 <= queue2._current < queue2._count
