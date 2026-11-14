"""The failure paths of ``spacr.qt.timing``: the ones only a broken run takes.

Everything here is instrumentation, so every one of these paths exists to
keep a diagnostic from becoming the fault. Pinned in this file:

* the import-attribution frame walks -- the ``/spacr/`` frame they look for,
  the top of a stack they can run off, and an interpreter that refuses to
  hand out a frame at all;
* an interpreter with no ``ExtensionFileLoader`` to wrap;
* readiness probes whose widget, subscriber or event filter dies underneath
  them, which is the ordinary case when a screen is navigated away from
  while it is still being measured;
* the report and snapshot fields that only appear when a source of facts is
  missing -- no budget, no Qt, no ``resource``, no display, no preferences;
* ``begin`` refusing a spawn timestamp it cannot use, and refusing to run
  twice.

Nothing here leaves a probe in the register: the ``timing_on`` fixture
retires any left in teardown.
"""
from __future__ import annotations

import sys
import time
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from spacr.qt import timing


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def timing_on(monkeypatch):
    """One empty, enabled instrumentation session with no process residue.

    Patching the module globals rather than reloading keeps every other
    reference to ``spacr.qt.timing`` in the session pointing at this module.
    """
    monkeypatch.setattr(timing, "ENABLED", True)
    monkeypatch.setattr(timing, "_START", time.perf_counter())
    monkeypatch.setattr(timing, "_EVENT_LOOP_STARTED_AT", None)
    monkeypatch.setattr(timing, "_SPANS", [])
    monkeypatch.setattr(timing, "_IMPORTS", [])
    monkeypatch.setattr(timing, "_STALLS", [])
    monkeypatch.setattr(timing, "_MARKS", [])
    monkeypatch.setattr(timing, "_READINESS", [])
    monkeypatch.setattr(timing, "_READY_CALLBACKS", [])
    monkeypatch.setattr(timing, "_ACTIVE_PROBES", [])
    yield
    for probe in list(timing._ACTIVE_PROBES):
        try:
            probe._retire()
        except (AttributeError, RuntimeError):
            pass




def _page():
    root = QWidget()
    layout = QVBoxLayout(root)
    button = QPushButton("usable")
    layout.addWidget(button)
    return root, button




# ---------------------------------------------------------------------------
# _ImportTimer.find_spec -- who asked for this module?
# ---------------------------------------------------------------------------


def test_the_finders_note_hook_leaves_the_pending_attribution_alone(timing_on):
    """``note`` is the seam the loader wrapper took over; it must stay inert.

    A ``note`` that cleared or rewrote ``_pending`` would silently drop the
    attribution for whichever import was in flight.
    """
    finder = timing._ImportTimer()
    finder.find_spec("a_module_awaiting_its_attribution")
    pending = finder._pending

    assert finder.note() is None
    assert finder._pending == pending


# ---------------------------------------------------------------------------
# _install_import_timer
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# the exec_module wrapper -- what triggered this import?
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# probes that die under the event loop
# ---------------------------------------------------------------------------

def test_a_probe_that_dies_when_the_loop_starts_is_dropped_not_raised_through(
        timing_on):
    """``event_loop_started`` runs once, on the path to the first frame.

    A deleted widget behind one probe must not stop the loop from telling the
    others -- and a probe that has already dropped its own registration must
    not make the drop fail either.
    """
    class _Dead:
        def event_loop_started(self):
            raise RuntimeError("Internal C++ object already deleted.")

    class _DeadAndAlreadyGone:
        def event_loop_started(self):
            timing._ACTIVE_PROBES.remove(self)
            raise RuntimeError("Internal C++ object already deleted.")

    class _Healthy:
        def __init__(self):
            self.told = 0

        def event_loop_started(self):
            self.told += 1

    dead, vanished, healthy = _Dead(), _DeadAndAlreadyGone(), _Healthy()
    timing._ACTIVE_PROBES.extend([dead, vanished, healthy])

    timing.event_loop_started()

    assert healthy.told == 1
    assert timing._ACTIVE_PROBES == [healthy]
    assert timing._EVENT_LOOP_STARTED_AT is not None


def test_cancelling_by_detail_leaves_the_same_screen_under_another_detail(
        timing_on):
    """Two modules share a report name and differ only by their detail.

    Matching on detail is what lets one of them give up without retiring the
    readiness measurement the other is still waiting on.
    """
    class _Probe:
        def __init__(self, detail):
            self.report_name = "interactive module"
            self.report_detail = detail
            self.retired = False

        def _retire(self):
            self.retired = True

    wanted = _Probe("classify")
    other = _Probe("measure")
    timing._ACTIVE_PROBES.extend([wanted, other])

    retired = timing.cancel_interactive(detail="classify")

    assert retired == 1
    assert wanted.retired is True
    assert other.retired is False


def test_cancelling_a_probe_that_died_and_deregistered_itself_still_counts(
        timing_on):
    """Cancel races the probe's own retirement; both may reach the register.

    A probe destroyed with its widget raises from ``_retire``. If it had
    already dropped its registration on the way down, the tidy-up that
    follows must not raise in turn -- and a probe still registered must
    genuinely be dropped, or the next screen would inherit it.
    """
    class _DeadButRegistered:
        report_name = "gone"
        report_detail = ""

        def _retire(self):
            raise RuntimeError("Internal C++ object already deleted.")

    class _DeadAndDeregistered:
        report_name = "gone"
        report_detail = ""

        def _retire(self):
            timing._ACTIVE_PROBES.remove(self)
            raise RuntimeError("Internal C++ object already deleted.")

    class _Live:
        report_name = "still here"
        report_detail = ""

        def __init__(self):
            self.retired = False

        def _retire(self):
            self.retired = True

    registered = _DeadButRegistered()
    deregistered = _DeadAndDeregistered()
    live = _Live()
    timing._ACTIVE_PROBES.extend([registered, deregistered, live])

    assert timing.cancel_interactive(name="gone") == 2
    assert timing._ACTIVE_PROBES == [live]
    assert live.retired is False


# ---------------------------------------------------------------------------
# watch_interactive
# ---------------------------------------------------------------------------


def test_retiring_a_probe_twice_does_no_work_the_second_time(qtbot, timing_on):
    """Cancel and settle can both retire the same probe.

    Without the latch the second pass would re-run the teardown -- and would
    remove whatever now stands where the probe used to be in the register.
    """
    root, _button = _page()
    qtbot.addWidget(root)
    probe = timing.watch_interactive(root, "interactive module", "twice")

    probe._retire()
    assert probe.done is True
    assert probe not in timing._ACTIVE_PROBES

    # A stale re-registration stands in for "something else is in the
    # register now": the latch must return before touching it.
    timing._ACTIVE_PROBES.append(probe)
    probe._retire()

    assert timing._ACTIVE_PROBES == [probe]


def test_a_probe_whose_registration_is_already_gone_still_retires(
        qtbot, timing_on):
    """A new timing session empties the register under a live probe."""
    root, _button = _page()
    qtbot.addWidget(root)
    probe = timing.watch_interactive(root, "interactive module", "orphan")
    assert probe in timing._ACTIVE_PROBES

    timing._ACTIVE_PROBES.remove(probe)
    probe._retire()

    assert probe.done is True
    assert timing._ACTIVE_PROBES == []


# ---------------------------------------------------------------------------
# report and snapshot with a source of facts missing
# ---------------------------------------------------------------------------

def test_a_readiness_entry_with_no_budget_gets_no_verdict(timing_on):
    """Only measurements taken against a budget may be graded.

    A diagnostic ``watch_interactive`` call passes no budget; printing "OK"
    for it would invent a release contract that was never asked for.
    """
    timing._READINESS.extend([
        {"at": 1.0, "duration_s": 0.5, "name": "diagnostic page",
         "detail": "no budget", "budget_s": None, "within_budget": None,
         "painted_usable_controls": 3},
        {"at": 2.0, "duration_s": 1.0, "name": "interactive module",
         "detail": "probe", "budget_s": 10.0, "within_budget": True,
         "painted_usable_controls": 2},
    ])

    lines = timing.report().splitlines()

    ungraded = [line for line in lines if "diagnostic page" in line][0]
    graded = [line for line in lines if "interactive module" in line][0]
    assert ungraded.endswith("3 painted control(s)")
    assert graded.endswith("OK")


def test_peak_memory_prefers_the_windows_peak_working_set(monkeypatch):
    """``peak_wset`` is a peak; ``rss`` is only the current value.

    Reporting the current RSS as a peak on a platform that has the real one
    would understate every measurement taken after a spike.
    """
    monkeypatch.setitem(sys.modules, "resource", None)
    fake = ModuleType("psutil")
    info = SimpleNamespace(peak_wset=512 * 1024 * 1024, rss=64 * 1024 * 1024)
    fake.Process = lambda: SimpleNamespace(memory_info=lambda: info)
    monkeypatch.setitem(sys.modules, "psutil", fake)

    assert timing._peak_rss_mb() == 512.0

    del info.peak_wset
    assert timing._peak_rss_mb() == 64.0


def test_a_snapshot_taken_without_qt_loaded_carries_no_qt_version(monkeypatch):
    """``snapshot`` never imports Qt; it reports what the process already has.

    A benchmark that failed before Qt was imported must still produce a
    readable artifact, and one whose Qt cannot answer must not claim a
    version it did not get.
    """
    assert isinstance(timing.snapshot()["environment"]["qt"], str)

    monkeypatch.delitem(sys.modules, "PySide6.QtCore")
    assert timing.snapshot()["environment"]["qt"] is None

    core = ModuleType("PySide6.QtCore")

    def _refuses():
        raise RuntimeError("Qt was unloaded")

    core.qVersion = _refuses
    monkeypatch.setitem(sys.modules, "PySide6.QtCore", core)

    assert timing.snapshot()["environment"]["qt"] is None


# ---------------------------------------------------------------------------
# begin
# ---------------------------------------------------------------------------
