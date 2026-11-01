"""A new Figure that inherits a collected Figure's id() is still queued.

The queue recognises a re-emitted figure by ``id(fig)``. Python reuses the id
of a collected object, so once an evicted Figure is gone the next Figure can
carry the same number. That newcomer has to be queued as a new figure, not
taken for the old one and silently dropped.

Whether CPython hands out the same address is up to its allocator, so the
reuse is made certain here: the module's ``id`` gives the newcomer the number
the collected Figure had.
"""
from __future__ import annotations

import builtins
import gc
import weakref

import matplotlib

matplotlib.use("Agg")
import pytest
from matplotlib.figure import Figure

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402

from spacr.qt.widgets import figure_queue as fq  # noqa: E402
from spacr.qt.widgets.figure_queue import FigureQueue  # noqa: E402

pytestmark = pytest.mark.qt


def _fig():
    fig = Figure(figsize=(1, 1))
    fig.add_subplot(111).plot([0, 1], [0, 1])
    return fig


@pytest.fixture
def queue(qtbot):
    widget = FigureQueue()
    qtbot.addWidget(widget)
    yield widget
    widget.clear()


def _turn_the_event_loop():
    loop = QEventLoop()
    QTimer.singleShot(50, loop.quit)
    loop.exec()


def test_a_newcomer_that_reuses_an_evicted_figures_id_is_queued(
        queue, monkeypatch):
    old = _fig()
    old_idx = queue.add_figure(old)
    queue.add_figure(_fig())
    assert queue._evict_live_figure(old_idx) is True
    old_id = id(old)
    gone = weakref.ref(old)
    del old
    _turn_the_event_loop()
    gc.collect()
    assert gone() is None

    newcomer = _fig()
    newcomer_id = builtins.id(newcomer)
    monkeypatch.setattr(
        fq, "id",
        lambda obj: old_id if builtins.id(obj) == newcomer_id
        else builtins.id(obj),
        raising=False)
    before = queue._count

    idx = queue.add_figure(newcomer)

    assert idx == before
    assert queue._count == before + 1
    assert queue._figures[idx] is newcomer
    assert queue._fig_index[old_id] == idx


def test_the_same_object_after_eviction_is_reselected_not_requeued(queue):
    fig = _fig()
    idx = queue.add_figure(fig)
    queue.add_figure(_fig())
    assert queue._evict_live_figure(idx) is True
    count = queue._count

    assert queue.add_figure(fig) == idx
    assert queue._count == count
