"""A second run starts its lettering at A, in its own section.

Reported 2026-08-16: "i did a second run and the figure letters just keep
climing i want each run seperated into sections or tabs (tabs like entries
are devided up in the console.)"

The queue accumulates figures across runs, which is the point of it -- an
earlier run stays reachable. But the grid letters its cells, and A PANEL
LETTER BELONGS TO A FIGURE: a figure is one run's worth of panels, and
lettering across runs says nothing at all. A second run continuing at L is
not a label, it is a queue index wearing a letter.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

pytestmark = pytest.mark.qt


def _figure():
    figure = plt.figure(figsize=(3, 2))
    figure.add_subplot(111).plot([0, 1], [0, 1])
    return figure


@pytest.fixture
def queue(qtbot):
    from spacr.qt.widgets.figure_queue import FigureQueue

    widget = FigureQueue()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def grid(qtbot):
    from spacr.qt.widgets.figure_grid_view import FigureGridView

    widget = FigureGridView()
    qtbot.addWidget(widget)
    return widget


def _run(queue, label, count):
    queue.mark_run(label)
    for _ in range(count):
        figure = _figure()
        queue.add_figure(figure)
        plt.close(figure)


# --------------------------------------------------------------------------- #
#  The lettering
# --------------------------------------------------------------------------- #

def test_a_second_run_restarts_at_a(queue, grid):
    _run(queue, "first", 3)
    _run(queue, "second", 3)

    grid.set_figures(queue.all_pixmaps(), queue.figure_titles(),
                     sections=queue.run_sections())

    assert [cell.letter for cell in grid._cells] == \
        ["A", "B", "C", "A", "B", "C"]


def test_without_sections_the_lettering_is_continuous(queue, grid):
    """The old behaviour, kept for a caller with no runs to divide."""
    _run(queue, "only", 4)

    grid.set_figures(queue.all_pixmaps(), queue.figure_titles())

    assert [cell.letter for cell in grid._cells] == ["A", "B", "C", "D"]


def test_the_sections_say_where_each_run_starts(queue):
    _run(queue, "first", 2)
    _run(queue, "second", 3)

    assert queue.run_sections() == [("first", 0, 2), ("second", 2, 3)]


# --------------------------------------------------------------------------- #
#  The edges a real session hits
# --------------------------------------------------------------------------- #

def test_a_run_that_drew_nothing_does_not_leave_an_empty_section(queue):
    """Marked at the START of a run, so a run that produced no figures is
    two marks with nothing between them."""
    queue.mark_run("empty")
    _run(queue, "productive", 2)

    assert queue.run_sections() == [("productive", 0, 2)]


def test_figures_that_arrived_before_any_run_are_not_dropped(queue):
    """A figure loaded from disk, or a queue used outside a pipeline."""
    figure = _figure()
    queue.add_figure(figure)
    plt.close(figure)
    _run(queue, "then a run", 2)

    sections = queue.run_sections()
    assert sections[0][1] == 0 and sections[0][2] == 1
    assert sum(count for _label, _start, count in sections) == queue.count()


def test_an_empty_queue_has_no_sections(queue):
    assert queue.run_sections() == []


def test_every_figure_belongs_to_exactly_one_section(queue):
    _run(queue, "a", 2)
    _run(queue, "b", 1)
    _run(queue, "c", 3)

    sections = queue.run_sections()
    covered = sum(count for _label, _start, count in sections)
    assert covered == queue.count() == 6
    starts = [start for _label, start, _count in sections]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)


# --------------------------------------------------------------------------- #
#  It is visible, not merely computed
# --------------------------------------------------------------------------- #

def test_the_grid_draws_a_heading_for_each_run(qtbot, queue, grid):
    """Restarting the lettering is only legible if the reader can see where
    one run ends -- otherwise two panels called A look like a bug."""
    from PySide6.QtWidgets import QLabel

    _run(queue, "first run", 2)
    _run(queue, "second run", 2)
    grid.resize(900, 600)
    grid.set_figures(queue.all_pixmaps(), queue.figure_titles(),
                     sections=queue.run_sections())

    texts = [w.text() for w in grid._body.findChildren(QLabel)]
    assert "first run" in texts, texts
    assert "second run" in texts, texts


def test_one_run_gets_no_heading(qtbot, queue, grid):
    """A heading over the only section is furniture."""
    from PySide6.QtWidgets import QLabel

    _run(queue, "the only run", 3)
    grid.set_figures(queue.all_pixmaps(), queue.figure_titles(),
                     sections=queue.run_sections())

    texts = [w.text() for w in grid._body.findChildren(QLabel)]
    assert "the only run" not in texts


def test_the_screen_marks_a_run_when_it_starts(qtbot):
    """Named so that moving the mark to the first figure fails here: a run
    that draws nothing would then vanish from the grid entirely."""
    import inspect

    from spacr.qt.screens.app_screen import AppScreen

    source = inspect.getsource(AppScreen._on_run)
    assert "mark_run" in source, (
        "the run is not marked at its start; the grid cannot section it")
