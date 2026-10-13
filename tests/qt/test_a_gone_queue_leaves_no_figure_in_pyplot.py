"""A Figure the queue showed is released from pyplot when the queue lets go.

pyplot keeps every ``plt.figure()`` in its registry until ``plt.close``. A
queue that was closed, deleted, or told to forget a run used to leave its
Figures there, so a long session -- or a serial test run -- kept every plot
it had ever drawn.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pytest
from PySide6.QtCore import QCoreApplication, QEvent

from spacr.qt.widgets.figure_queue import FigureQueue

pytestmark = pytest.mark.qt


def _figures(queue, n):
    made = []
    for i in range(n):
        figure = plt.figure()
        figure.add_subplot(111).plot([0, 1], [i, 1])
        queue.add_figure(figure)
        made.append(figure)
    return made


def _registered(figures):
    numbers = set(plt.get_fignums())
    return [f for f in figures if f.number in numbers]


def test_closing_the_queue_releases_its_figures(qtbot):
    queue = FigureQueue()
    qtbot.addWidget(queue)
    made = _figures(queue, 3)
    assert _registered(made) == made
    queue.close()
    assert _registered(made) == []


def test_deleting_the_queue_releases_its_figures(qtbot):
    queue = FigureQueue()
    made = _figures(queue, 2)
    queue.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert _registered(made) == []


def test_forgetting_a_run_releases_only_that_runs_figures(qtbot):
    queue = FigureQueue()
    qtbot.addWidget(queue)
    queue.mark_run("first")
    first = _figures(queue, 2)
    queue.mark_run("second")
    second = _figures(queue, 2)
    assert queue.forget_run("first") == 2
    assert _registered(first) == []
    assert _registered(second) == second
    queue.clear()
    assert _registered(second) == []
