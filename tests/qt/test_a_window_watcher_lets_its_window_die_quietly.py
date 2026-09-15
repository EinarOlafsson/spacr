"""An overlay or drawer that filters its window lets that window go quietly.

The shape CI hit in ``AmbientWidget`` and ``DnaRainWidget`` (see
``test_a_backdrop_outlives_nothing_it_watches.py``), found elsewhere by
sweeping ``spacr/qt`` for event filters that keep a strong reference to the
window they watch: ``_TourOverlay._window``, ``ShortcutOverlay._window`` and
``EdgeDrawer._host``. Each is a child of that window, so the two form a
reference cycle. The cycle collector frees it by clearing the wrapper's
``__dict__`` first; the window's destructor then reached the filter, which
raised ``AttributeError`` inside the Qt event loop. A probe that ran the real
collector raised exactly that for all three.

Their references stay strong, because these windows live as long as the app and
other methods use them. What changed is that the filters pass an event on when
the reference is gone. As in the backdrop tests, ``__dict__`` is cleared by hand
instead of calling ``gc.collect()``, which conftest warns can segfault mid-run.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QMainWindow, QWidget  # noqa: E402


def _tour_overlay():
    from spacr.qt import first_run as fr

    window = QMainWindow()
    window.resize(600, 400)
    # on_finish, so nothing here can retire the app-wide first-run flag.
    return window, fr._TourOverlay(window, fr.DEFAULT_TOUR[:1],
                                   on_finish=lambda: None)


def _shortcut_overlay():
    from spacr.qt import shortcuts as sc

    window = QMainWindow()
    window.resize(600, 400)
    return window, sc.ShortcutOverlay(window)


def _edge_drawer():
    from spacr.qt.widgets.drawer import EdgeDrawer

    host = QWidget()
    host.resize(600, 400)
    panel = QWidget()
    panel.resize(200, 400)
    return host, EdgeDrawer(host, panel, width=200)


WATCHERS = pytest.mark.parametrize(
    "make", [_tour_overlay, _shortcut_overlay, _edge_drawer],
    ids=["tour_overlay", "shortcut_overlay", "edge_drawer"])


def _raised_by(qtbot, step):
    """What ``step`` raised, as text, whether PySide reported it through the
    event loop or let it out of the call."""
    problems = []
    with qtbot.capture_exceptions() as raised:
        try:
            step()
        except AttributeError as exc:
            problems.append(f"AttributeError: {exc}")
    problems.extend(f"{kind.__name__}: {value}" for kind, value, _tb in raised)
    raised.clear()
    return problems


@WATCHERS
def test_emptied_by_the_collector_it_lets_its_window_die_quietly(qtbot,
                                                                  make):
    import shiboken6

    window, watcher = make()
    window.show()
    qtbot.waitExposed(window)

    watcher.__dict__.clear()
    problems = _raised_by(qtbot, lambda: shiboken6.delete(window))
    assert not problems, problems


@WATCHERS
def test_it_still_follows_the_size_of_the_window_it_watches(qtbot, make):
    """The normal path: the window's resize still reaches the filter."""
    window, watcher = make()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)

    window.resize(640, 520)
    qtbot.waitUntil(lambda: window.height() == 520)
    qtbot.waitUntil(lambda: watcher.height() == window.height())
