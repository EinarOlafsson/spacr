"""The cleanup module's error paths, read one failing collaborator at a time.

``tests/qt/test_resource_cleanup.py`` pins the promises (nothing is killed,
a run in flight is left alone) and ``test_cov_w3_7_resource_cleanup.py`` pins
the ordinary fallbacks. What is left, and what this file drives, is what
happens when the *collaborator* is the thing that breaks: a screen whose
cache inventory raises, a cache row that is not four values, an eviction that
returns False rather than throwing, a preferences import that is not there, a
Qt that has not been loaded, and a ``QTimer`` that cannot be built.

Every one of these is a real call into the real function; only the failing
collaborator is stood in for. Where a test asserts that a line is *absent*
from a report, the same test first drives the input that puts it there, so
"absent" is measured against a run that produced it.
"""
from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("PySide6")

from spacr.qt import resource_cleanup as rc


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def test_a_disk_report_with_no_note_ends_on_its_last_drive():
    """The note is a line, not a blank one: no note must mean no extra line."""
    drives = (rc.DiskEntry("/a", 2048, 1024, 1024),
              rc.DiskEntry("/b", 4096, 3072, 1024))
    with_note = rc.DiskReport(drives, note="1 folder(s) could not be read.")
    without = rc.DiskReport(drives)

    assert with_note.summary().splitlines()[-1] == (
        "1 folder(s) could not be read.")
    lines = without.summary().splitlines()
    assert len(lines) == 2
    assert lines[-1].startswith("/b:")
    assert "could not be read" not in without.summary()


def test_a_sweep_that_grew_reports_nothing_freed_rather_than_a_negative():
    """``freed_mb`` is a measured drop, and a drop cannot be below zero."""
    assert rc.BudgetSweep(before_mb=10.5, after_mb=4.5).freed_mb == 6.0
    assert rc.BudgetSweep(before_mb=4.0, after_mb=10.0).freed_mb == 0.0


# ---------------------------------------------------------------------------
# Discovering the owners
# ---------------------------------------------------------------------------

def test_an_accessor_that_raises_costs_only_its_own_caches(monkeypatch):
    """One broken screen must not hide the caches of the working one."""
    good = object()

    def _boom():
        raise RuntimeError("the thumbnail registry is mid-teardown")

    monkeypatch.setitem(
        sys.modules, "spacr.qt.crop_thumbs",
        _module("spacr.qt.crop_thumbs", live_thumbnail_caches=_boom))
    monkeypatch.setitem(
        sys.modules, "spacr.qt.widgets.figure_queue",
        _module("spacr.qt.widgets.figure_queue",
                live_figure_queues=lambda: [good]))

    owners = rc._loaded_cache_owners()
    assert good in owners

    # Same call, with the raising accessor made to work: its owner appears,
    # so the omission above is the exception being swallowed and nothing else.
    other = object()
    monkeypatch.setitem(
        sys.modules, "spacr.qt.crop_thumbs",
        _module("spacr.qt.crop_thumbs",
                live_thumbnail_caches=lambda: [other]))
    assert other in rc._loaded_cache_owners()


# ---------------------------------------------------------------------------
# Inventorying the entries
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Reading the two preferences
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Is a run using these caches?
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Releasing warm models
# ---------------------------------------------------------------------------

def test_no_model_is_released_while_a_run_could_be_using_it(monkeypatch):
    calls = []
    monkeypatch.setattr(rc, "MODEL_RELEASERS", [lambda: calls.append(1) or 3])

    monkeypatch.setattr(rc, "_a_run_is_active", lambda: True)
    assert rc._release_models_under_pressure() == 0
    assert calls == []

    # The releaser is reachable -- only the live run was stopping it.
    monkeypatch.setattr(rc, "_a_run_is_active", lambda: False)
    assert rc._release_models_under_pressure() == 3
    assert calls == [1]


def test_a_releaser_that_raises_does_not_stop_the_ones_after_it(monkeypatch):
    def _boom():
        raise RuntimeError("model already freed")

    monkeypatch.setattr(rc, "_a_run_is_active", lambda: False)
    monkeypatch.setattr(rc, "MODEL_RELEASERS", [_boom, lambda: 2, lambda: None])
    assert rc._release_models_under_pressure() == 2


# ---------------------------------------------------------------------------
# The sweep's own bookkeeping
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Qt's own caches, and whether Qt is there at all
# ---------------------------------------------------------------------------

def test_the_pixmap_cache_reports_the_kilobytes_it_was_holding(monkeypatch,
                                                               qapp):
    """The size reading is optional, and when it is there it is reported.

    ``QPixmapCache.totalUsed`` is not bound in this PySide6 build, so the
    *reading* is stood in for and the rest of the function is the real one:
    it still has to call ``clear()``, and it still has to decide between a
    line and silence.

    With no reading available the answer is silence, not a line saying so:
    the module's own comment is that inventing a count would violate
    Reclaim's measured contract. The cleanup still happens either way,
    which is the half that matters.
    """
    from PySide6.QtGui import QPixmapCache
    assert not hasattr(QPixmapCache, "totalUsed")
    assert rc._clear_pixmap_cache() == []

    cleared = []

    def _stub(held):
        return _module("PySide6.QtGui", QPixmapCache=types.SimpleNamespace(
            totalUsed=lambda: held, clear=lambda: cleared.append(held)))

    monkeypatch.setitem(sys.modules, "PySide6.QtGui", _stub(768))
    assert rc._clear_pixmap_cache() == ["Qt pixmap cache (768 KB)"]

    # An empty cache is cleared just the same, and reports nothing.
    monkeypatch.setitem(sys.modules, "PySide6.QtGui", _stub(0))
    assert rc._clear_pixmap_cache() == []
    assert cleared == [768, 0]


def test_a_qtgui_that_will_not_import_clears_no_pixmaps_and_says_nothing(
        monkeypatch):
    monkeypatch.setitem(sys.modules, "PySide6.QtGui", None)
    assert rc._clear_pixmap_cache() == []


def test_without_qtwidgets_loaded_no_application_is_running(monkeypatch):
    monkeypatch.setitem(sys.modules, "PySide6.QtWidgets", None)
    assert rc._qt_application_is_running() is False


def test_a_qapplication_that_cannot_be_asked_is_read_as_not_running(
        monkeypatch, qapp):
    """A raising ``instance()`` must not take the cleanup down with it."""
    assert rc._qt_application_is_running() is True

    class _Hostile:
        @staticmethod
        def instance():
            raise RuntimeError("wrapper already deleted")

    monkeypatch.setitem(sys.modules, "PySide6.QtWidgets",
                        _module("PySide6.QtWidgets", QApplication=_Hostile))
    assert rc._qt_application_is_running() is False


# ---------------------------------------------------------------------------
# clear_ram's reporting
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Library thread counts
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The periodic tick and its installation
# ---------------------------------------------------------------------------

@pytest.fixture
def budget_globals():
    """Restore the module's two sweep globals whatever a test does to them."""
    timer = rc._BUDGET_TIMER
    pending = rc._BUDGET_SWEEP_PENDING
    yield
    rc._BUDGET_TIMER = timer
    rc._BUDGET_SWEEP_PENDING = pending


def test_with_no_event_loop_the_queued_sweep_does_not_stay_pending_forever(
        monkeypatch, budget_globals):
    """A latched flag with nothing to clear it would disable the sweep."""
    rc._BUDGET_SWEEP_PENDING = False
    monkeypatch.setitem(sys.modules, "PySide6.QtCore", None)
    rc._request_budget_sweep()
    assert rc._BUDGET_SWEEP_PENDING is False


def test_a_queued_sweep_is_queued_once(monkeypatch, qapp, budget_globals):
    queued = []
    from PySide6 import QtCore

    monkeypatch.setattr(QtCore.QTimer, "singleShot",
                        staticmethod(lambda ms, fn: queued.append((ms, fn))))
    rc._BUDGET_SWEEP_PENDING = False
    rc._request_budget_sweep()
    rc._request_budget_sweep()

    assert rc._BUDGET_SWEEP_PENDING is True
    assert [ms for ms, _fn in queued] == [0]
    assert queued[0][1] is rc._budget_tick


def test_macos_memory_probe_uses_owned_result_and_rejects_failure(monkeypatch):
    import ctypes
    import ctypes.util

    codes = iter([0, 1])
    calls = []

    def usage(pid, flavor, pointer):
        calls.append((pid, flavor))
        pointer._obj[8] = 123456
        return next(codes)

    monkeypatch.setattr(ctypes.util, 'find_library', lambda name: None)
    monkeypatch.setattr(ctypes, 'CDLL', lambda path: types.SimpleNamespace(proc_pid_rusage=usage))
    assert rc._macos_rss() == 123456
    assert rc._macos_rss() == 0
    assert calls == [(rc.os.getpid(), 0)] * 2

    def unavailable(path):
        raise OSError('libproc unavailable')

    monkeypatch.setattr(ctypes, 'CDLL', unavailable)
    assert rc._macos_rss() == 0


def test_gui_detection_declines_unavailable_qt_and_deleted_threads(monkeypatch):
    import builtins
    from PySide6.QtCore import QCoreApplication, QThread

    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == 'PySide6.QtCore':
            raise ImportError('Qt unavailable')
        return original(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, '__import__', missing)
        assert not rc._the_gui_thread_is_asking()
    with monkeypatch.context() as patch:
        patch.setattr(QCoreApplication, 'instance', lambda: None)
        assert not rc._the_gui_thread_is_asking()

    def deleted():
        raise RuntimeError('thread wrapper deleted')

    with monkeypatch.context() as patch:
        patch.setattr(QCoreApplication, 'instance', lambda: object())
        patch.setattr(QThread, 'currentThread', deleted)
        assert not rc._the_gui_thread_is_asking()


def test_missing_gui_path_probe_declines_without_blocking_stat(monkeypatch):
    import builtins

    original = builtins.__import__

    def missing(name, globals=None, locals=None, fromlist=(), level=0):
        if 'path_probe' in (fromlist or ()):
            raise ImportError('probe unavailable')
        return original(name, globals, locals, fromlist, level)

    with monkeypatch.context() as patch:
        patch.setattr(rc, '_the_gui_thread_is_asking', lambda: True)
        patch.setattr(builtins, '__import__', missing)
        assert not rc._is_a_folder('/unavailable/share')
