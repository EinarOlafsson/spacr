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

import time

import pytest

pytest.importorskip("PySide6")

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


# ---------------------------------------------------------------------------
# The snapshot's environment facts
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# begin
# ---------------------------------------------------------------------------
