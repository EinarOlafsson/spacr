"""GIL priority — an interpreter that refuses the switch-interval knob.

``claim()`` lowers ``sys.setswitchinterval`` so the Qt thread keeps waking
while a pure-Python worker runs, and ``release()`` puts it back. Neither may
turn a refusal from the interpreter into a failed pipeline run: a window
that is less smooth is the correct outcome, not a traceback out of a worker.

The refusal is injected by making ``sys.setswitchinterval`` raise, which is
the only way to reach either handler on an interpreter that has the knob.
"""
from __future__ import annotations

import sys

import pytest

from spacr.qt import gil_priority


@pytest.fixture(autouse=True)
def a_clean_counter():
    """Every test starts and ends with no claim outstanding."""
    saved_depth = gil_priority._DEPTH
    saved_restore = gil_priority._RESTORE
    saved_interval = sys.getswitchinterval()
    gil_priority._DEPTH = 0
    gil_priority._RESTORE = None
    yield
    gil_priority._DEPTH = saved_depth
    gil_priority._RESTORE = saved_restore
    sys.setswitchinterval(saved_interval)


def test_a_claim_lowers_the_interval_and_a_release_puts_it_back():
    before = sys.getswitchinterval()
    gil_priority.claim()
    assert sys.getswitchinterval() == pytest.approx(gil_priority.BUSY_INTERVAL)
    assert gil_priority.active() is True
    gil_priority.release()
    assert sys.getswitchinterval() == pytest.approx(before)
    assert gil_priority.active() is False


def test_an_interpreter_without_the_knob_still_lets_the_worker_run(
        monkeypatch, caplog):
    """``claim()`` swallows the refusal and leaves nothing to restore."""
    def refuse(_value):
        raise ValueError("switch interval is not settable here")

    monkeypatch.setattr(sys, "setswitchinterval", refuse)
    with caplog.at_level("DEBUG", logger="spacr.qt.gil_priority"):
        gil_priority.claim()

    assert gil_priority.active() is True
    assert gil_priority._RESTORE is None
    assert any("lower the switch interval" in record.getMessage()
               for record in caplog.records)
    # And the release that follows must be a no-op rather than a second
    # failure, because there is no saved value to write back.
    gil_priority.release()
    assert gil_priority.active() is False


def test_a_refused_restore_does_not_escape_the_release(monkeypatch, caplog):
    """The knob works on the way down and refuses on the way back up."""
    gil_priority.claim()
    assert gil_priority._RESTORE is not None

    def refuse(_value):
        raise RuntimeError("no")

    monkeypatch.setattr(sys, "setswitchinterval", refuse)
    with caplog.at_level("DEBUG", logger="spacr.qt.gil_priority"):
        gil_priority.release()

    assert gil_priority.active() is False
    # The saved value is dropped either way, so the next claim starts fresh
    # instead of restoring a stale interval much later.
    assert gil_priority._RESTORE is None
    assert any("restore the switch interval" in record.getMessage()
               for record in caplog.records)


def test_nested_claims_restore_only_once():
    before = sys.getswitchinterval()
    gil_priority.claim()
    gil_priority.claim()
    gil_priority.release()
    assert sys.getswitchinterval() == pytest.approx(gil_priority.BUSY_INTERVAL)
    gil_priority.release()
    assert sys.getswitchinterval() == pytest.approx(before)


def test_an_extra_release_cannot_drive_the_count_negative():
    gil_priority.release()
    gil_priority.release()
    assert gil_priority._DEPTH == 0
    assert gil_priority.active() is False


def test_the_context_manager_releases_through_an_exception():
    before = sys.getswitchinterval()
    with pytest.raises(ZeroDivisionError):
        with gil_priority.responsive_gui():
            assert gil_priority.active() is True
            1 / 0
    assert gil_priority.active() is False
    assert sys.getswitchinterval() == pytest.approx(before)
