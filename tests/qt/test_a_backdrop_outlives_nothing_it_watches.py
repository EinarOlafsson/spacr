"""A backdrop that watches its window lets that window go without raising.

CI run 34989909231 on 2a84d1d60 failed twice on one shape, both times inside
the Qt event loop: ``'AmbientWidget' object has no attribute '_watched'`` from
``AmbientWidget.eventFilter``, and ``'DnaRainWidget' object has no attribute
'_watched'`` / ``'_timer'`` from ``DnaRainWidget.eventFilter`` and
``hideEvent -> stop``.

NOT CONSTRUCTION ORDER. Neither ``__init__`` installs a filter or shows
anything; both store ``_watched`` and ``_timer`` before returning, and the
filter is installed later, by ``showEvent``. The cause was a REFERENCE CYCLE:
``showEvent`` kept the window in ``_watched``, and the window owns the widget's
wrapper (or IS the widget, when it is its own window). Such a pair is freed only
by Python's cycle collector, which clears each wrapper's ``__dict__`` BEFORE the
C++ objects are destroyed. The window's destructor then hides its children and
notifies its event filters, and both land on a widget with no attributes left.
When the collector happens to run decides which test fails, which is why the
failures moved between tests.

Two repairs, tested apart:

* the watch is a weak reference, so no cycle forms and ordinary reference
  counting frees the window while every attribute is still in place;
* the two handlers that teardown reaches pass the event on when the attributes
  are gone, because a cycle made anywhere else reopens the same path.

Neither test calls ``gc.collect()``: conftest records that collecting a live Qt
heap in the middle of a run can segfault. The first test proves the cycle is
gone with the collector switched off. The second clears ``__dict__`` by hand,
which is exactly what the collector does to a wrapper, and then destroys the
window synchronously, as the collector's last step does.
"""
from __future__ import annotations

import gc
import weakref

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, Qt  # noqa: E402
from PySide6.QtWidgets import QWidget  # noqa: E402


def _ambient(host):
    from spacr.qt.widgets.ambient import AmbientWidget

    widget = AmbientWidget(host, seed=1)
    if host is not None:
        widget.follow_parent()
    return widget


def _dna_rain(host):
    from spacr.qt.widgets.dna_rain import DnaRainWidget

    widget = DnaRainWidget(host, seed=1)
    if host is not None:
        widget.follow_parent()
    return widget


BACKDROPS = pytest.mark.parametrize(
    "make", [_ambient, _dna_rain], ids=["ambient", "dna_rain"])


def _raised_by(qtbot, step):
    """What ``step`` raised, as text, whether PySide reported it through the
    event loop or let it out of the call.

    Text rather than the exception objects, so a failure report holds no
    traceback into a half-destroyed widget.
    """
    problems = []
    with qtbot.capture_exceptions() as raised:
        try:
            step()
        except AttributeError as exc:
            problems.append(f"AttributeError: {exc}")
    problems.extend(f"{kind.__name__}: {value}" for kind, value, _tb in raised)
    raised.clear()
    return problems


@BACKDROPS
@pytest.mark.parametrize("hosted", [True, False],
                         ids=["inside_a_window", "its_own_window"])
def test_watching_a_window_does_not_keep_the_backdrop_alive(qapp, make,
                                                             hosted):
    """With the collector OFF, dropping the last name must free the window.

    A window that survives this is held by a cycle, and a cycle is what lets
    the collector empty the widget before the window's teardown reaches it.
    """
    import shiboken6

    def build():
        host = QWidget() if hosted else None
        widget = make(host)
        window = host if hosted else widget
        window.resize(320, 240)
        window.show()
        qapp.processEvents()
        # The watch was really made: showEvent ran and started the timer.
        assert widget.is_running()
        return weakref.ref(window)

    was_enabled = gc.isenabled()
    gc.disable()
    try:
        ref = build()
        survivor = ref()
        alive = survivor is not None
        if alive:
            # Destroy it now, attributes intact, rather than leave a cycle for
            # the collector to empty during some later test.
            shiboken6.delete(survivor)
        del survivor
    finally:
        if was_enabled:
            gc.enable()
    assert not alive, (
        "the window outlived its last reference, so the backdrop and the "
        "window it watches form a reference cycle")


@BACKDROPS
def test_a_backdrop_emptied_by_the_collector_lets_its_window_die_quietly(
        qtbot, make):
    """The collector's order of events, reproduced without the collector."""
    import shiboken6

    host = QWidget()
    widget = make(host)
    host.resize(320, 240)
    host.show()
    qtbot.waitExposed(host)
    assert widget.is_running()

    widget.__dict__.clear()
    problems = _raised_by(qtbot, lambda: shiboken6.delete(host))
    assert not problems, problems


@BACKDROPS
def test_a_filter_event_with_no_watch_is_passed_on(qtbot, make):
    """Only the watch is gone: the filter says so instead of raising."""
    host = QWidget()
    qtbot.addWidget(host)
    widget = make(host)
    host.resize(320, 240)
    host.show()
    qtbot.waitExposed(host)

    del widget._watched
    answers = []

    def deliver():
        for kind in (QEvent.Hide, QEvent.Show, QEvent.WindowStateChange):
            answers.append(widget.eventFilter(host, QEvent(kind)))

    problems = _raised_by(qtbot, deliver)
    assert not problems, problems
    assert answers == [False, False, False]


@BACKDROPS
def test_the_window_is_still_watched_and_a_hide_still_stops(qtbot, make):
    """The normal path: the window's events arrive through the installed
    filter, a minimised window pauses the backdrop, and a hide stops it."""
    host = QWidget()
    qtbot.addWidget(host)
    widget = make(host)
    host.resize(320, 240)
    host.show()
    qtbot.waitExposed(host)
    assert widget._watched() is host
    assert widget.is_running()

    host.setWindowState(Qt.WindowMinimized)
    QCoreApplication.sendEvent(host, QEvent(QEvent.WindowStateChange))
    assert not widget.is_running(), "a minimised window must pause it"

    host.setWindowState(Qt.WindowNoState)
    QCoreApplication.sendEvent(host, QEvent(QEvent.WindowStateChange))
    assert widget.is_running(), "a restored window must resume it"

    host.hide()
    assert not widget.is_running(), "hiding must stop the timer"
