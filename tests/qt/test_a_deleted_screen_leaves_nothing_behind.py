"""A screen Qt has deleted must not stay alive on the Python side.

Found by a serial ``pytest tests/qt`` (features/new/47) that climbed about a
megabyte of RSS per test until the suite's own memory guard stopped it. Two
shapes were behind the regression screen's share of that, and both keep a
screen's Python wrappers alive after Qt has destroyed every widget:

* a ``QWidget`` subclass defined per call, whose methods close over the host.
  PySide6 never releases a class it has had to wrap, so the closure pinned
  every ``AppScreen`` the sweep card was built for -- six MB and ~5,000
  objects a screen;
* a slot closing over ``self`` connected for good to the process-wide
  ``path_probe.probes.answered``. Every file list ever built stayed alive and
  one more receiver ran on every probe answer.

Each test builds the thing, lets Qt delete it the way a closed screen is
deleted, and asks whether anything is left.
"""
from __future__ import annotations

import gc
import os
import weakref

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, SIGNAL                         # noqa: E402
from PySide6.QtWidgets import (QApplication, QVBoxLayout,          # noqa: E402
                               QWidget)

pytestmark = pytest.mark.qt


def _let_qt_delete(widget) -> None:
    """Close, schedule and deliver the deletion, then collect cycles."""
    widget.close()
    widget.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    gc.collect()


def test_the_sweep_card_does_not_keep_its_screen_alive(qapp):
    """The deferred sweep panel holds its host, and lets go with it.

    Built the way ``AppScreen`` builds it: the host keeps the panel in an
    attribute and the panel sits in a card's layout, so host and panel form
    a cycle only the collector can free -- which the old per-call class,
    pinned by PySide6, kept it from doing.
    """
    from spacr.qt.screens.parameter_sweep import _lazy_sweep_panel

    host = QWidget()
    card = QWidget(host)
    host.sweep = _lazy_sweep_panel(host)
    QVBoxLayout(card).addWidget(host.sweep)
    del card
    gone = weakref.ref(host)

    _let_qt_delete(host)
    del host
    gc.collect()

    assert gone() is None, (
        "the screen a sweep card was built for is still alive after Qt "
        "deleted it; something outside Python's view holds it")


def test_every_sweep_card_shares_one_class(qapp):
    """One class per process, not one per card: PySide6 keeps them all."""
    from spacr.qt.screens.parameter_sweep import _lazy_sweep_panel

    first = _lazy_sweep_panel(None)
    second = _lazy_sweep_panel(None)
    try:
        assert type(first) is type(second), (
            "each call defined a new widget class, and PySide6 never frees "
            "one; a long session would pile them up")
    finally:
        _let_qt_delete(first)
        _let_qt_delete(second)


def test_a_deleted_file_list_stops_listening_to_the_path_probe(qapp):
    """The probe connection ends with the widget, and does not pin it."""
    from spacr.qt import path_probe
    from spacr.qt.widgets.file_list import FilePathListWidget

    signal = SIGNAL("answered(QString,bool)")
    before = path_probe.probes.receivers(signal)

    widget = FilePathListWidget()
    during = path_probe.probes.receivers(signal)
    gone = weakref.ref(widget)

    _let_qt_delete(widget)
    del widget
    gc.collect()

    assert during == before + 1, \
        "a file list follows the path probe while it is alive"
    assert path_probe.probes.receivers(signal) == before, (
        "the file list is gone but its slot is still connected to the "
        "process-wide probe signal")
    assert gone() is None, \
        "the probe connection kept a deleted file list alive"


def test_a_merge_panel_deleted_with_its_screen_lets_go_of_it(qapp):
    """Qt deletes the panel with its parent; it never sees a closeEvent."""
    from spacr.qt import path_probe
    from spacr.qt.widgets.measurement_scan_panel import DatabaseMergePanel

    signal = SIGNAL("answered(QString,bool)")
    before = path_probe.probes.receivers(signal)

    host = QWidget()
    QVBoxLayout(host).addWidget(DatabaseMergePanel(threaded=False))
    during = path_probe.probes.receivers(signal)
    gone = weakref.ref(host)

    _let_qt_delete(host)
    del host
    gc.collect()

    assert during == before + 1, \
        "a merge panel follows the path probe while it is alive"
    assert path_probe.probes.receivers(signal) == before, (
        "the panel went with its screen but its slot is still connected "
        "to the process-wide probe signal")
    assert gone() is None, \
        "the merge panel's probe slot kept the screen around it alive"


def test_a_read_still_out_does_not_own_the_merge_panel(qapp):
    """A reader thread must never be the panel's last owner.

    Once the probe slot stopped pinning every merge panel, this is what the
    pin had been hiding: the reader thread's target was a bound method of
    the panel, so a read that outlived the panel dropped the last reference
    ON THE READER THREAD, and PySide6 destroyed the widget there -- a
    segfault in the next event the GUI thread delivered, reproduced by
    ``test_the_measurements_tab_never_waits_on_a_database.py``. The panel
    has to be freed here, on the GUI thread, while its read is still out.
    """
    import threading

    from spacr.qt.widgets.measurement_scan_panel import DatabaseMergePanel

    release = threading.Event()
    finished = threading.Event()

    def parked():
        """A read on a mount that has not woken up yet."""
        release.wait(30)
        finished.set()
        return ()

    panel = DatabaseMergePanel(threaded=False)
    with panel._read_budget(budget=0):
        panel._read_off_thread(("parked",), parked)
    gone = weakref.ref(panel)

    try:
        _let_qt_delete(panel)
        del panel
        gc.collect()
        still_there = gone() is not None
    finally:
        release.set()
    finished.wait(30)

    assert finished.is_set(), "the parked read never ran"
    assert not still_there, (
        "a read that was still out kept the merge panel alive; when it "
        "lands it would destroy the widget on the reader thread")
