"""Taking back the process hooks removes every copy, and survives what is gone.

``resource_cleanup._uninstall_process_hooks`` is what ``tests/qt/conftest.py``
calls before and after every test, so a budget sweep or a run-registry hook
one test installs cannot fire inside the next. Its docstring promises that
EVERY ``_on_registry_changed`` connection is removed "whatever ``_INSTALLED``
says", because a test that resets the flag and installs again connects a
second copy.

A SECOND COPY WAS NOT REMOVED. On PySide6 6.11 ``disconnect(slot)`` removes
one connection of a slot connected several times and then answers False
while the others are still connected, so "disconnect until PySide6 reports
nothing left" stopped after the first. Measured: three installs, one
uninstall, two receivers still on ``registry().changed``.

The rest holds the teardown to finishing whatever it meets: a timer that was
destroyed with its application, a signal that refuses the disconnect, and a
signal that never says it is empty.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import SIGNAL

from spacr.qt import bridge
from spacr.qt import resource_cleanup as rc

pytestmark = pytest.mark.qt

CHANGED = SIGNAL("changed()")


def _flags_are_clear():
    return (rc._BUDGET_TIMER is None and rc._INSTALLED is False
            and rc._BUDGET_SWEEP_PENDING is False and not rc._SEEN_RUNS)


def test_every_copy_of_the_run_hook_is_taken_back(qapp, monkeypatch):
    """What a test that resets ``_INSTALLED`` to exercise the install does."""
    rc._uninstall_process_hooks()
    before = bridge.registry().receivers(CHANGED)

    for _ in range(3):
        monkeypatch.setattr(rc, "_INSTALLED", False)
        assert rc.install_run_hook() is True
    assert bridge.registry().receivers(CHANGED) == before + 3

    rc._uninstall_process_hooks()

    left = bridge.registry().receivers(CHANGED) - before
    assert left == 0, (
        f"{left} copies of the pre-run cleanup hook are still connected to "
        f"the run registry and fire in every later test")
    assert rc._INSTALLED is False


def test_a_timer_destroyed_with_its_application_is_let_go(qapp, monkeypatch):
    import shiboken6
    from PySide6.QtCore import QTimer

    timer = QTimer()
    shiboken6.delete(timer)
    monkeypatch.setattr(rc, "_BUDGET_TIMER", timer)
    monkeypatch.setattr(rc, "_BUDGET_SWEEP_PENDING", True)
    rc._SEEN_RUNS.add("a run the sweep had seen")

    rc._uninstall_process_hooks()

    assert _flags_are_clear()


class _Changed:
    """A registry signal whose disconnect answers as it is told to."""

    def __init__(self, answer):
        self.answer = answer
        self.attempts = 0

    def disconnect(self, _slot):
        self.attempts += 1
        return self.answer()


def _registry_with(monkeypatch, changed):
    monkeypatch.setattr(bridge, "registry",
                        lambda: SimpleNamespace(changed=changed))


def test_a_signal_that_refuses_the_disconnect_ends_the_attempts(
        qapp, monkeypatch):
    def refuses():
        raise TypeError("disconnect() failed between 'changed' and all slots")

    changed = _Changed(refuses)
    _registry_with(monkeypatch, changed)
    monkeypatch.setattr(rc, "_INSTALLED", True)

    rc._uninstall_process_hooks()

    assert changed.attempts == 1
    assert _flags_are_clear()


def test_a_signal_that_never_says_it_is_empty_is_not_asked_forever(
        qapp, monkeypatch):
    changed = _Changed(lambda: True)
    _registry_with(monkeypatch, changed)
    monkeypatch.setattr(rc, "_INSTALLED", True)

    rc._uninstall_process_hooks()

    assert changed.attempts == 64
    assert _flags_are_clear()
