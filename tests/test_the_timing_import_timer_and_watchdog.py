"""The two recorders in ``spacr.qt.timing`` that reach outside the module.

The import timer wraps ``SourceFileLoader.exec_module`` for the whole
interpreter, and the stall watchdog is a live ``QTimer``. Both are global, so
both are the parts that go untested by default and both are the parts whose
failure is most expensive: a leaked loader wrapper slows every import in the
process, and a watchdog that miscounts is the freeze measurement lying.

Everything here restores what it patched, checked in a ``finally``.
"""
from __future__ import annotations

import importlib
import importlib.machinery
import os
import sys
import time

import pytest


@pytest.fixture
def timing(monkeypatch):
    """``spacr.qt.timing`` re-imported with timing and import attribution on."""
    previous_environment = {
        name: os.environ.get(name)
        for name in ("SPACR_TIMING", "SPACR_TIMING_IMPORTS")
    }
    monkeypatch.setenv("SPACR_TIMING", "1")
    monkeypatch.setenv("SPACR_TIMING_IMPORTS", "1")
    saved = sys.modules.get("spacr.qt.timing")
    module = importlib.reload(importlib.import_module("spacr.qt.timing"))
    assert module.ENABLED and module.IMPORT_TIMING_ENABLED
    try:
        yield module
    finally:
        for name, value in previous_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if saved is not None:
            sys.modules["spacr.qt.timing"] = saved
        importlib.reload(saved or module)


@pytest.fixture
def loader_restored():
    """Put ``exec_module`` back however the test ends.

    Without this a failing test leaves a timing wrapper on every source
    import for the remainder of the session -- which is both a slowdown and a
    recorder writing into a module the next test has reloaded away.
    """
    source = importlib.machinery.SourceFileLoader.exec_module
    extension = importlib.machinery.ExtensionFileLoader.exec_module
    try:
        yield
    finally:
        importlib.machinery.SourceFileLoader.exec_module = source
        importlib.machinery.ExtensionFileLoader.exec_module = extension


# ---------------------------------------------------------------------------
# the import timer
# ---------------------------------------------------------------------------

def test_installing_the_import_timer_wraps_the_loader(timing, loader_restored):
    """It really is the loader that gets wrapped, not a shadow of it."""
    before = importlib.machinery.SourceFileLoader.exec_module

    timing._install_import_timer()

    assert importlib.machinery.SourceFileLoader.exec_module is not before


def test_installing_it_twice_does_not_wrap_the_wrapper(timing,
                                                       loader_restored):
    """The latch matters: each install adds a layer to every future import.

    Two launches in one process -- a test session, an embedded run -- would
    otherwise stack loaders, and the recorded time for a module would be the
    time of the wrapper below it.
    """
    timing._install_import_timer()
    once = importlib.machinery.SourceFileLoader.exec_module

    timing._install_import_timer()

    assert importlib.machinery.SourceFileLoader.exec_module is once


def test_an_import_slower_than_the_floor_is_recorded_with_its_name(
        timing, loader_restored, tmp_path, monkeypatch):
    """End to end: install, import a genuinely slow module, read it back."""
    monkeypatch.setattr(timing, "IMPORT_FLOOR_MS", 1.0)
    module = tmp_path / "a_deliberately_slow_module.py"
    module.write_text("import time\ntime.sleep(0.02)\nVALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    timing._install_import_timer()
    timing._IMPORTS.clear()
    importlib.import_module("a_deliberately_slow_module")

    recorded = [i for i in timing._IMPORTS
                if i["name"] == "a_deliberately_slow_module"]
    assert len(recorded) == 1
    assert recorded[0]["took"] >= 0.019
    assert recorded[0]["thread"] == "MainThread"


def test_an_import_under_the_floor_is_not_recorded(timing, loader_restored,
                                                   tmp_path, monkeypatch):
    """There are thousands of them; a report listing all of them is unreadable."""
    monkeypatch.setattr(timing, "IMPORT_FLOOR_MS", 10_000.0)
    module = tmp_path / "an_instant_module.py"
    module.write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    timing._install_import_timer()
    timing._IMPORTS.clear()
    importlib.import_module("an_instant_module")

    assert not [i for i in timing._IMPORTS if i["name"] == "an_instant_module"]


def test_the_finder_declines_every_import_it_is_asked_about(timing):
    """It measures; it must never decide where a module comes from.

    Returning a spec here would make the timer part of import resolution,
    and a diagnostic that changes which module is loaded is a bug generator.
    """
    finder = timing._ImportTimer()

    assert finder.find_spec("json") is None
    assert finder.find_spec("a_module_that_does_not_exist") is None
    assert finder.find_module("json") is None


def test_the_finder_notes_who_asked_for_a_module_it_has_not_seen(timing):
    """The attribution is the useful half: "3 s of torch, asked by qt/app.py"."""
    finder = timing._ImportTimer()

    finder.find_spec("a_module_nobody_has_imported_yet")

    name, started, caller = finder._pending
    assert name == "a_module_nobody_has_imported_yet"
    assert isinstance(started, float)
    assert caller == "" or ".py:" in caller


def test_a_module_already_imported_is_not_timed_again(timing):
    """``sys.modules`` hits cost nothing and would drown the real ones."""
    finder = timing._ImportTimer()
    finder._pending = ("sentinel", 0.0, "")

    assert finder.find_spec("sys") is None
    assert finder._pending[0] == "sentinel", "it recorded a cache hit"


# ---------------------------------------------------------------------------
# the stall watchdog
# ---------------------------------------------------------------------------

@pytest.mark.qt
def test_the_watchdog_records_a_beat_that_arrived_late(timing, qtbot,
                                                       monkeypatch):
    """The freeze measurement itself, driven by firing the timer by hand.

    Sleeping for a real stall would make the test slow AND flaky; what is
    being checked is the arithmetic and the threshold, so the timer's own
    signal is emitted after the floor is lowered.
    """
    monkeypatch.setattr(timing, "STALL_FLOOR_MS", 1.0)
    timing._STALLS.clear()

    watchdog = timing.watch_the_gui_thread()
    assert watchdog is not None
    try:
        time.sleep(0.01)
        watchdog.timeout.emit()

        assert len(timing._STALLS) == 1
        entry = timing._STALLS[0]
        assert entry["late_ms"] >= 9.0
        assert entry["source"] == "event-loop watchdog"
        assert entry["started_at"] < entry["at"]
    finally:
        watchdog.stop()
        watchdog.deleteLater()


@pytest.mark.qt
def test_a_beat_that_arrived_on_time_is_not_a_stall(timing, qtbot):
    """Sixteen milliseconds of lateness is the normal case, not a freeze.

    The floor is what keeps the report about the interface having actually
    frozen; without it every beat is an entry and the section is noise.
    """
    timing._STALLS.clear()

    watchdog = timing.watch_the_gui_thread()
    try:
        watchdog.timeout.emit()

        assert timing._STALLS == []
    finally:
        watchdog.stop()
        watchdog.deleteLater()


@pytest.mark.qt
def test_the_watchdog_updates_the_last_beat_it_reports(timing, qtbot):
    """``last_gui_beat_at`` is how another screen asks "is the loop alive?"."""
    watchdog = timing.watch_the_gui_thread()
    try:
        first = timing.last_gui_beat_at()
        time.sleep(0.005)
        watchdog.timeout.emit()

        assert timing.last_gui_beat_at() > first
    finally:
        watchdog.stop()
        watchdog.deleteLater()


# ---------------------------------------------------------------------------
# readiness subscribers
# ---------------------------------------------------------------------------

def test_a_readiness_subscriber_is_registered_once_and_removable(timing):
    """Double-subscribing would report one screen's readiness twice."""
    def listener(_entry):
        pass

    timing.subscribe_readiness(listener)
    timing.subscribe_readiness(listener)
    assert timing._READY_CALLBACKS.count(listener) == 1

    timing.unsubscribe_readiness(listener)
    assert listener not in timing._READY_CALLBACKS


def test_unsubscribing_something_never_subscribed_is_not_an_error(timing):
    """Teardown runs on paths where setup did not, and must not raise there."""
    before = list(timing._READY_CALLBACKS)
    timing.unsubscribe_readiness(lambda _entry: None)
    assert timing._READY_CALLBACKS == before


def test_the_event_loop_start_is_recorded_once(timing):
    """A second call must not move the origin every screen is measured from."""
    timing._EVENT_LOOP_STARTED_AT = None
    timing._MARKS.clear()

    timing.event_loop_started()
    first = timing._EVENT_LOOP_STARTED_AT
    timing.event_loop_started()

    assert timing._EVENT_LOOP_STARTED_AT == first
    assert len([m for m in timing._MARKS
                if m["name"] == "event loop began"]) == 1


def test_cancelling_interactive_probes_retires_only_the_matching_ones(timing):
    """A screen that closed before it painted must not keep a probe alive.

    Matching on name and detail is what lets one screen give up without
    retiring the readiness measurement another screen is still waiting on.
    """
    class _Probe:
        def __init__(self, name, detail):
            self.report_name = name
            self.report_detail = detail
            self.retired = False

        def _retire(self):
            self.retired = True

    wanted = _Probe("regression", "first paint")
    other = _Probe("home", "first paint")
    timing._ACTIVE_PROBES.clear()
    timing._ACTIVE_PROBES.extend([wanted, other])
    try:
        retired = timing.cancel_interactive(name="regression")

        assert retired == 1
        assert wanted.retired and not other.retired
    finally:
        timing._ACTIVE_PROBES.clear()


def test_a_probe_whose_widget_is_gone_is_dropped_rather_than_raised_through(
        timing):
    """A deleted Qt object raises RuntimeError from Python, mid-teardown.

    Letting that out of ``cancel_interactive`` would take down the screen
    change that was cancelling it.
    """
    class _Dead:
        report_name = "gone"
        report_detail = ""

        def _retire(self):
            raise RuntimeError("wrapped C/C++ object has been deleted")

    timing._ACTIVE_PROBES.clear()
    timing._ACTIVE_PROBES.append(_Dead())
    try:
        assert timing.cancel_interactive(name="gone") == 1
        assert timing._ACTIVE_PROBES == []
    finally:
        timing._ACTIVE_PROBES.clear()
