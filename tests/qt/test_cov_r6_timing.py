"""The timing paths taken when a fact is simply not there to record.

``test_cov_r5_timing.py`` pins the sources of fact that *raise*. What is left,
and what this file drives, is the source that is absent or older than the
reader: a watchdog row written before the canonical timestamps existed, a
``watch_interactive`` call with timing off or with no widget, a CUDA allocator
that was never initialised, a hardware profile taken before the QApplication
is built, a preferences module with no performance level to give, and
``begin`` in an ordinary un-instrumented run.

Every assertion that a field is absent is paired, in the same test, with the
input that fills it in, so the absence is measured against a call that
produced the value.
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
    """One empty, enabled instrumentation session with no process residue."""
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


def _screen(name="probe-0"):
    return SimpleNamespace(
        name=lambda: name,
        geometry=lambda: SimpleNamespace(width=lambda: 1280,
                                         height=lambda: 800),
        devicePixelRatio=lambda: 1.0,
        refreshRate=lambda: 60.0,
    )


# ---------------------------------------------------------------------------
# The canonical watchdog duration, and the rows that predate it
# ---------------------------------------------------------------------------

def test_a_stall_row_without_timestamps_falls_back_to_its_recorded_lateness():
    """Timestamps are the canonical duration; ``late_ms`` is compatibility.

    An in-memory diagnostic row made by an older caller carries only the
    lateness it observed. Reading it as a zero-length stall would hide the
    worst stall of the run from the snapshot's budget verdict.
    """
    # The canonical form: two timestamps, and the duration proved by them.
    assert timing._stall_duration_ms(
        {"started_at": 10.0, "at": 10.25}) == pytest.approx(250.0)

    # No timestamps at all -- the older row shape.
    assert timing._stall_duration_ms({"late_ms": 42.0}) == 42.0
    # Timestamps present but unreadable: still the recorded lateness.
    assert timing._stall_duration_ms(
        {"started_at": 10.0, "at": "not-a-time", "late_ms": 7.5}) == 7.5


def test_a_stall_row_whose_lateness_is_unreadable_reads_as_no_stall():
    """The fallback field is compatibility too, so it can be anything.

    Zero is the only honest answer for a row that carries no duration in
    either form; anything else would be a stall the run never had.
    """
    assert timing._stall_duration_ms({"late_ms": None}) == 0.0
    assert timing._stall_duration_ms({"late_ms": "quite late"}) == 0.0
    assert timing._stall_duration_ms({}) == 0.0
    # Negative lateness is clamped, not reported as a stall running backwards.
    assert timing._stall_duration_ms({"late_ms": -30.0}) == 0.0


def test_the_snapshot_verdict_reads_a_legacy_stall_row_at_its_full_length(
        timing_on):
    """The fallback is what the public artifact's budget verdict is built on."""
    timing._STALLS.append({"at": 1.0, "late_ms": timing.STALL_BUDGET_MS + 50})

    report = timing.snapshot()

    assert report["worst_event_loop_stall_ms"] == timing.STALL_BUDGET_MS + 50
    assert report["stall_budget_met"] is False


# ---------------------------------------------------------------------------
# watch_interactive's front door
# ---------------------------------------------------------------------------

def test_no_readiness_probe_is_installed_without_timing_or_a_widget(
        qtbot, timing_on, monkeypatch):
    """The probe parents itself to the widget and filters its events.

    With timing off it would be pure overhead on every screen built; with no
    widget there is nothing to parent it to, and an unparented event filter
    would outlive the screen it was measuring.
    """
    root, _child = _page()
    qtbot.addWidget(root)
    probe = timing.watch_interactive(root, "interactive module", "on")
    assert probe is not None
    assert len(timing._ACTIVE_PROBES) == 1

    assert timing.watch_interactive(None, "interactive module", "no widget") \
        is None
    assert len(timing._ACTIVE_PROBES) == 1

    monkeypatch.setattr(timing, "ENABLED", False)
    other, _ = _page()
    qtbot.addWidget(other)
    assert timing.watch_interactive(other, "interactive module", "off") is None
    assert len(timing._ACTIVE_PROBES) == 1


# ---------------------------------------------------------------------------
# The snapshot's environment facts
# ---------------------------------------------------------------------------

def test_a_cuda_allocator_that_was_never_initialised_reports_zero(monkeypatch):
    """Torch imported is not torch used, and 0 MB is the true reading.

    ``None`` here would say "could not ask" of a process that asked and got
    a definite answer -- a CPU run would then look like a failed measurement.
    """
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_initialized=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", torch)

    assert timing._gpu_memory_mb() == {
        "allocated_mb": 0.0, "peak_allocated_mb": 0.0}

    # The same reader with an initialised allocator returns real figures, so
    # the zeros above are the uninitialised context and not a dead function.
    torch.cuda = SimpleNamespace(
        is_initialized=lambda: True,
        memory_allocated=lambda: 2 * 1024 * 1024,
        max_memory_allocated=lambda: 3 * 1024 * 1024,
    )
    assert timing._gpu_memory_mb() == {
        "allocated_mb": 2.0, "peak_allocated_mb": 3.0}


def test_a_profile_taken_before_the_application_exists_names_no_display(
        monkeypatch):
    """Qt imported is not Qt running.

    A benchmark that failed during start-up profiles a process whose
    ``QApplication.instance()`` is still ``None``; the display list and the
    platform name are then honestly absent rather than guessed.
    """
    widgets = ModuleType("PySide6.QtWidgets")
    widgets.QApplication = SimpleNamespace(instance=lambda: None)
    monkeypatch.setitem(sys.modules, "PySide6.QtWidgets", widgets)

    profile = timing._hardware_profile()
    assert profile["displays"] == []
    assert profile["qt_platform"] is None

    # The same call once the application is up fills both fields in.
    app = SimpleNamespace(screens=lambda: [_screen()],
                          platformName=lambda: "probe")
    widgets.QApplication = SimpleNamespace(instance=lambda: app)

    profile = timing._hardware_profile()
    assert profile["qt_platform"] == "probe"
    assert [row["name"] for row in profile["displays"]] == ["probe-0"]


def test_a_preferences_module_with_no_level_to_give_leaves_it_unknown(
        monkeypatch):
    """The level is read off whatever is loaded, and never imported for.

    Early in start-up ``spacr.qt.preferences`` can be a module object whose
    body has not finished executing, so the getter is not there to call yet.
    """
    monkeypatch.setitem(sys.modules, "spacr.qt.preferences",
                        ModuleType("spacr.qt.preferences"))
    assert timing._hardware_profile()["performance_level"] is None

    ready = ModuleType("spacr.qt.preferences")
    ready.get_performance_level = lambda: "balanced"
    monkeypatch.setitem(sys.modules, "spacr.qt.preferences", ready)
    assert timing._hardware_profile()["performance_level"] == "balanced"


# ---------------------------------------------------------------------------
# begin
# ---------------------------------------------------------------------------

def test_begin_records_nothing_and_latches_nothing_when_timing_is_off(
        timing_on, monkeypatch):
    """An ordinary ``spacr`` run calls ``begin`` too, and must pay nothing.

    The latch matters as much as the mark: if a disabled ``begin`` set it,
    a session that enabled timing afterwards could never date its clock.
    """
    monkeypatch.setattr(timing, "IMPORT_TIMING_ENABLED", False)
    monkeypatch.setattr(timing.begin, "_done", False, raising=False)
    monkeypatch.delenv("SPACR_TIMING_PROCESS_START", raising=False)
    monkeypatch.setattr(timing, "ENABLED", False)

    timing.begin()

    assert timing._MARKS == []
    assert timing.begin._done is False

    # Enabled, the same call marks the start and latches, so the untouched
    # state above is the disabled guard and not a begin that does nothing.
    monkeypatch.setattr(timing, "ENABLED", True)
    timing.begin()

    assert [mark["name"] for mark in timing._MARKS] == ["timing started"]
    assert timing.begin._done is True
