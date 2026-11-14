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
    # THE REAL SETTER, BOUND BEFORE ANY TEST CAN REPLACE IT. Two tests below
    # inject the interpreter's refusal by monkeypatching
    # ``sys.setswitchinterval`` to raise, and a fixture's teardown runs
    # BEFORE monkeypatch undoes its patches -- so restoring through the name
    # calls the refusal and turns a passing test into a teardown ERROR that
    # names the injected message rather than anything real.
    set_interval = sys.setswitchinterval
    gil_priority._DEPTH = 0
    gil_priority._RESTORE = None
    yield
    gil_priority._DEPTH = saved_depth
    gil_priority._RESTORE = saved_restore
    set_interval(saved_interval)


def test_unweakrefable_and_deleted_busy_owners_do_not_block_imports(monkeypatch):
    from PySide6.QtCore import QObject
    from shiboken6 import delete

    monkeypatch.setattr(gil_priority, '_HOLDS', {})
    gil_priority.hold_background_imports(object())
    assert not gil_priority.background_imports_held()
    owner = QObject()
    gil_priority.hold_background_imports(owner, lambda value: value.objectName() == 'loading')
    delete(owner)
    assert not gil_priority.background_imports_held()
