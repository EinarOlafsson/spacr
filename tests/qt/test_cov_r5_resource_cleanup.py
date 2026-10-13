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

import logging
import sys
import types

import pytest

pytest.importorskip("PySide6")

from spacr.qt import resource_cleanup as rc


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

class _Owner:
    """A cache owner with the two-method protocol the sweep discovers."""

    def __init__(self, rows, *, dropped=True, inventory_error=None):
        self.__name__ = "fake.owner"
        self._rows = rows
        self._dropped = dropped
        self._inventory_error = inventory_error
        self.drop_calls = []

    def cache_budget_entries(self):
        if self._inventory_error is not None:
            raise self._inventory_error
        return list(self._rows)

    def drop_cache_budget_entry(self, key):
        self.drop_calls.append(key)
        if isinstance(self._dropped, Exception):
            raise self._dropped
        return self._dropped


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


def test_the_largest_unit_is_terabytes_however_big_the_number_is():
    """Every size returns from inside the unit loop, TB included.

    This pins the guarantee that makes ``human_bytes``'s trailing
    ``return f"{value:.1f} TB"`` (line 285) unreachable: the loop's own
    condition is ``value < 1024 or unit == "TB"``, so the final iteration
    returns whatever the value is. The loop can never fall through, and the
    statement after it is dead code kept as a belt-and-braces default.
    """
    assert rc.human_bytes(512) == "512 B"
    assert rc.human_bytes(1536) == "1.5 KB"
    # 8 EB -- eight million TB, and still returned by the TB iteration.
    huge = rc.human_bytes(1024 ** 6 * 8)
    assert huge.endswith(" TB")
    assert float(huge.split()[0]) > 8_000_000


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

def test_an_owner_whose_inventory_raises_is_named_in_the_errors():
    broken = _Owner([], inventory_error=ValueError("db is closed"))
    working = _Owner([("k", 1024, 100.0, False)])

    records, errors = rc._collect_budget_entries([broken, working])

    assert [row.label for row in records] == ["fake.owner['k']"]
    assert len(errors) == 1
    assert "inventory failed" in errors[0]
    assert "db is closed" in errors[0]


def test_a_malformed_cache_row_is_reported_by_ordinal_not_dropped_silently():
    """A three-value row cannot be evicted, so it has to be named instead."""
    owner = _Owner([("good", 2 * 1024 * 1024, 100.0, False),
                    ("short",),
                    ("also-good", 1024, 50.0, True)])

    records, errors = rc._collect_budget_entries([owner])

    assert [row.label for row in records] == [
        "fake.owner['good']", "fake.owner['also-good']"]
    assert records[0].megabytes == pytest.approx(2.0)
    assert len(errors) == 1
    assert errors[0].startswith("fake.owner entry 1: invalid")


def test_a_very_long_cache_key_is_elided_rather_than_printed_whole():
    owner = _Owner([("k" * 500, 1024, 1.0, False)])
    records, errors = rc._collect_budget_entries([owner])
    assert errors == []
    assert records[0].label.endswith("...]")
    assert "..." in records[0].label
    assert len(records[0].label) < 200


# ---------------------------------------------------------------------------
# Reading the two preferences
# ---------------------------------------------------------------------------

def test_an_explicit_value_is_never_overwritten_by_the_preference(monkeypatch):
    """Each of the two settings is read only when it was not passed in."""
    import spacr.qt.preferences as prefs

    monkeypatch.setattr(prefs, "get_idle_minutes", lambda: 41.0)
    monkeypatch.setattr(prefs, "get_cache_ceiling_mb", lambda: 4242)

    # ceiling supplied, idle read from the preference
    assert rc._budget_values(None, 512) == (41.0, 512.0)
    # idle supplied, ceiling read from the preference
    assert rc._budget_values(3, None) == (3.0, 4242.0)


def test_unreadable_preferences_fall_back_to_the_documented_defaults(
        monkeypatch):
    from spacr.qt.memory_budget import (
        DEFAULT_CACHE_CEILING_MB,
        DEFAULT_IDLE_MINUTES,
    )

    # ``None`` in sys.modules makes the function-local import raise.
    monkeypatch.setitem(sys.modules, "spacr.qt.preferences", None)

    assert rc._budget_values(None, None) == (
        float(DEFAULT_IDLE_MINUTES), float(DEFAULT_CACHE_CEILING_MB))
    # The half that WAS supplied still survives the fallback.
    assert rc._budget_values(9, None) == (
        9.0, float(DEFAULT_CACHE_CEILING_MB))
    assert rc._budget_values(None, 77) == (
        float(DEFAULT_IDLE_MINUTES), 77.0)


def test_a_negative_setting_is_clamped_to_zero_not_passed_on():
    assert rc._budget_values(-5, -9) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Is a run using these caches?
# ---------------------------------------------------------------------------

def test_with_no_bridge_loaded_no_run_is_active(monkeypatch):
    """The registry is read, never imported: absent means idle."""
    monkeypatch.delitem(sys.modules, "spacr.qt.bridge", raising=False)
    assert rc._a_run_is_active() is False

    # Same function, with a bridge that reports a running job.
    live = _module("spacr.qt.bridge",
                   registry=lambda: types.SimpleNamespace(
                       active=lambda: ["a-run"]))
    monkeypatch.setitem(sys.modules, "spacr.qt.bridge", live)
    assert rc._a_run_is_active() is True


def test_a_bridge_without_a_callable_registry_is_treated_as_idle(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "spacr.qt.bridge",
        _module("spacr.qt.bridge", registry="not-callable"))
    assert rc._a_run_is_active() is False


def test_a_registry_that_raises_is_read_as_busy_not_as_idle(monkeypatch):
    """The unsafe answer is "idle", so an unreadable registry says busy."""
    def _boom():
        raise RuntimeError("registry mutating")

    monkeypatch.setitem(sys.modules, "spacr.qt.bridge",
                        _module("spacr.qt.bridge", registry=_boom))
    assert rc._a_run_is_active() is True


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

def test_an_eviction_that_throws_is_an_error_line_not_a_freed_megabyte():
    owner = _Owner([("hot", 8 * 1024 * 1024, 0.0, False)],
                   dropped=RuntimeError("still mapped"))

    sweep = rc.sweep_memory_budget(now=10_000.0, idle_minutes=1,
                                   ceiling_mb=4096, headroom_short=False,
                                   owners=[owner])

    assert owner.drop_calls == ["hot"]
    assert sweep.dropped == ()
    assert sweep.freed_mb == 0.0
    assert len(sweep.errors) == 1
    assert "eviction failed" in sweep.errors[0]
    assert "still mapped" in sweep.errors[0]


def test_an_owner_that_declines_to_drop_is_attempted_but_not_counted():
    """``drop_cache_budget_entry`` returning False is a refusal, not a bug."""
    refuses = _Owner([("pinned-elsewhere", 4 * 1024 * 1024, 0.0, False)],
                     dropped=False)
    sweep = rc.sweep_memory_budget(now=10_000.0, idle_minutes=1,
                                   ceiling_mb=4096, headroom_short=False,
                                   owners=[refuses])

    assert refuses.drop_calls == ["pinned-elsewhere"]
    assert sweep.dropped == ()
    assert sweep.errors == ()
    assert sweep.after_mb == sweep.before_mb

    # The same entry, from an owner that agrees, IS counted -- so the empty
    # tuple above is the refusal and not a sweep that never ran.
    agrees = _Owner([("pinned-elsewhere", 4 * 1024 * 1024, 0.0, False)])
    agreed = rc.sweep_memory_budget(now=10_000.0, idle_minutes=1,
                                    ceiling_mb=4096, headroom_short=False,
                                    owners=[agrees])
    assert agreed.dropped == ("fake.owner['pinned-elsewhere']",)
    assert agreed.freed_mb == pytest.approx(4.0)


def test_one_pass_stops_at_max_entries_and_says_it_is_incomplete():
    """A bounded pass leaves the rest for the next tick rather than freezing."""
    rows = [(f"k{index}", 1024 * 1024, float(index)) for index in range(4)]
    owner = _Owner([(key, size, used, False) for key, size, used in rows])

    sweep = rc.sweep_memory_budget(now=10_000.0, idle_minutes=1,
                                   ceiling_mb=4096, headroom_short=False,
                                   max_entries=2, owners=[owner])

    assert len(owner.drop_calls) == 2
    assert len(sweep.dropped) == 2
    assert sweep.complete is False

    # Unbounded, the same four go in one pass.
    owner2 = _Owner([(key, size, used, False) for key, size, used in rows])
    full = rc.sweep_memory_budget(now=10_000.0, idle_minutes=1,
                                  ceiling_mb=4096, headroom_short=False,
                                  owners=[owner2])
    assert len(full.dropped) == 4
    assert full.complete is True


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

def test_a_collection_that_frees_nothing_adds_no_line_to_the_report(
        monkeypatch):
    monkeypatch.setitem(sys.modules, "PySide6.QtWidgets", None)
    monkeypatch.setattr(rc, "process_rss", lambda: 4096)

    monkeypatch.setattr(rc.gc, "collect", lambda: 0)
    quiet = rc.clear_ram()
    assert not any("unreachable" in line for line in quiet.details)

    monkeypatch.setattr(rc.gc, "collect", lambda: 7)
    loud = rc.clear_ram()
    assert "7 unreachable objects collected" in loud.details


def test_a_process_that_really_shrank_gets_no_allocator_apology(monkeypatch):
    """The "allocator kept the pages" note is only for when it kept them."""
    import spacr.qt.widgets.data_filter_panel as panel

    readings = iter([200 * 1024 * 1024, 120 * 1024 * 1024,
                     120 * 1024 * 1024, 200 * 1024 * 1024])
    monkeypatch.setattr(rc, "process_rss", lambda: next(readings))
    monkeypatch.setitem(sys.modules, "PySide6.QtWidgets", None)
    monkeypatch.setattr(rc.gc, "collect", lambda: 0)

    monkeypatch.setattr(panel, "_KINDS_CACHE", {"a": 1, "b": 2})
    shrank = rc.clear_ram()
    assert shrank.freed == 80 * 1024 * 1024
    assert "allocator" not in shrank.note
    assert any("_KINDS_CACHE (2 entries)" in line for line in shrank.details)

    # Same call, same details, a process that did not shrink: the note is back.
    monkeypatch.setattr(panel, "_KINDS_CACHE", {"a": 1, "b": 2})
    grew = rc.clear_ram()
    assert grew.freed == 0
    assert "allocator" in grew.note


# ---------------------------------------------------------------------------
# Library thread counts
# ---------------------------------------------------------------------------

def test_libraries_that_are_not_loaded_are_not_imported_to_be_lowered(
        monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "cv2", raising=False)
    assert rc._lower_library_threads() == []
    assert "torch" not in sys.modules
    assert "cv2" not in sys.modules

    # Loaded, the very same call lowers both -- so "[]" above is the absence
    # of the modules and not a function that does nothing.
    torch_set = []
    cv_set = []
    monkeypatch.setitem(sys.modules, "torch", _module(
        "torch", get_num_threads=lambda: 16,
        set_num_threads=torch_set.append))
    monkeypatch.setitem(sys.modules, "cv2", _module(
        "cv2", getNumThreads=lambda: 12, setNumThreads=cv_set.append))

    done = rc._lower_library_threads(target=2)
    assert done == ["torch threads 16 → 2", "OpenCV threads 12 → 2"]
    assert torch_set == [2] and cv_set == [2]


def test_only_the_library_that_is_loaded_is_touched(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setitem(sys.modules, "cv2", _module(
        "cv2", getNumThreads=lambda: 9, setNumThreads=lambda value: None))
    # A target below the floor is raised to it: one thread is not "lowered".
    assert rc._lower_library_threads(target=1) == [
        f"OpenCV threads 9 → {rc.MIN_LIBRARY_THREADS}"]


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


def test_a_tick_that_freed_something_logs_what_it_freed(monkeypatch, caplog,
                                                        budget_globals):
    monkeypatch.setattr(rc, "sweep_memory_budget", lambda: rc.BudgetSweep(
        before_mb=40.0, after_mb=15.0, dropped=("a", "b"),
        models_released=2, vram_freed=3 * 1024 * 1024))
    rc._BUDGET_SWEEP_PENDING = True

    # DEBUG, NOT INFO, and it says "host RSS" now. Both were deliberate: the
    # line fired on every module open for housekeeping nobody asked for, and
    # it read as nonsense -- before_mb/after_mb are host RSS while vram_freed
    # is device memory, so "0.0 -> 0.0 MiB ... and 2.6 GB VRAM released" was
    # correct and unreadable at once. The test kept capturing at INFO and
    # asserting the old wording, so it saw an empty log.
    with caplog.at_level(logging.DEBUG, logger=rc.LOG.name):
        rc._budget_tick()

    assert rc._BUDGET_SWEEP_PENDING is False
    message = caplog.text
    assert "memory budget: host RSS 40.0 -> 15.0 MiB" in message
    assert "2 cache entries" in message
    assert "VRAM released" in message


def test_a_tick_that_freed_nothing_reports_only_its_errors(monkeypatch,
                                                           caplog,
                                                           budget_globals):
    monkeypatch.setattr(rc, "sweep_memory_budget", lambda: rc.BudgetSweep(
        errors=("screen: inventory failed (db is closed)",)))

    with caplog.at_level(logging.DEBUG, logger=rc.LOG.name):
        rc._budget_tick()

    assert "memory budget sweep: screen: inventory failed" in caplog.text
    assert "MiB" not in caplog.text


def test_a_sweep_that_throws_leaves_the_tick_standing(monkeypatch, caplog,
                                                      budget_globals):
    def _boom():
        raise RuntimeError("owner list mutated")

    monkeypatch.setattr(rc, "sweep_memory_budget", _boom)
    rc._BUDGET_SWEEP_PENDING = True

    with caplog.at_level(logging.DEBUG, logger=rc.LOG.name):
        rc._budget_tick()

    assert rc._BUDGET_SWEEP_PENDING is False
    assert "the live-cache budget sweep failed" in caplog.text


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


def test_a_dead_timer_wrapper_is_replaced_rather_than_asked_twice(
        monkeypatch, qapp, budget_globals):
    class _Deleted:
        @staticmethod
        def isActive():
            raise RuntimeError("Internal C++ object already deleted.")

    rc._BUDGET_TIMER = _Deleted()
    try:
        assert rc.install_budget_sweep() is True
        assert rc._BUDGET_TIMER is not None
        assert not isinstance(rc._BUDGET_TIMER, _Deleted)
        assert rc._BUDGET_TIMER.isActive()
        assert rc._BUDGET_TIMER.interval() == rc.BUDGET_SWEEP_INTERVAL_MS
    finally:
        if rc._BUDGET_TIMER is not None and not isinstance(
                rc._BUDGET_TIMER, _Deleted):
            rc._BUDGET_TIMER.stop()
            rc._BUDGET_TIMER.deleteLater()


def test_a_stopped_timer_is_restarted_rather_than_reported_installed(
        monkeypatch, qapp, budget_globals):
    stopped = types.SimpleNamespace(isActive=lambda: False)
    rc._BUDGET_TIMER = stopped
    try:
        assert rc.install_budget_sweep() is True
        assert rc._BUDGET_TIMER is not stopped
        assert rc._BUDGET_TIMER.isActive()
    finally:
        if rc._BUDGET_TIMER is not stopped:
            rc._BUDGET_TIMER.stop()
            rc._BUDGET_TIMER.deleteLater()


def test_a_running_timer_is_left_exactly_as_it_was(qapp, budget_globals):
    running = types.SimpleNamespace(isActive=lambda: True)
    rc._BUDGET_TIMER = running
    assert rc.install_budget_sweep() is True
    assert rc._BUDGET_TIMER is running


def test_an_uninstallable_sweep_says_so_instead_of_raising(monkeypatch,
                                                           budget_globals):
    rc._BUDGET_TIMER = None
    monkeypatch.setitem(sys.modules, "PySide6.QtCore", None)
    assert rc.install_budget_sweep() is False
    assert rc._BUDGET_TIMER is None
