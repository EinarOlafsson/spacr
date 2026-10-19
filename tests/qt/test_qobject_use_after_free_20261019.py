"""Qt objects are never reached after they are gone (macOS crash reports).

Native crash reports from macOS (PySide6 6.11/6.12) showed SIGSEGV in
``CallableObject_call`` (a Python callable connected as a slot, invoked after
what it captured was destroyed), AppKit ``runModalCleanup`` (a command run
inside a modal session that was already ending), and heap corruption that
surfaced in a widget constructor while the field-animation thread shaded.
These tests pin the fixes: automatic collection cannot be switched back on
behind the GUI-thread policy, a job runner's completion reaches nothing once
the runner is gone, a palette command runs after the palette's ``exec()``
returns, and the shading thread never rings a doorbell the widget dropped.
"""
from __future__ import annotations

import gc
import threading
import time

import pytest
import shiboken6
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication, QWidget

from spacr.qt import gc_policy


@pytest.fixture
def no_policy():
    was_installed = gc_policy.is_installed()
    if was_installed:
        gc_policy.uninstall()
    was_enabled = gc.isenabled()
    gc.enable()
    try:
        yield
    finally:
        if gc_policy.is_installed():
            gc_policy.uninstall()
        if was_enabled:
            gc.enable()
        else:
            gc.disable()
        if was_installed:
            gc_policy.install(QApplication.instance())


def test_a_pause_that_ends_after_the_policy_is_installed_keeps_collection_off(
        qapp, no_policy):
    with gc_policy._automatic_collection_paused():
        assert not gc.isenabled()
        assert gc_policy.install(qapp)
    assert not gc.isenabled(), (
        "a heavy import's pause re-enabled automatic collection behind the "
        "GUI-thread policy")
    gc_policy.uninstall()
    assert gc.isenabled()


def test_pauses_on_two_threads_restore_collection_once(qapp, no_policy):
    inside = threading.Event()
    release = threading.Event()

    def other_thread():
        with gc_policy._automatic_collection_paused():
            inside.set()
            release.wait(5)

    worker = threading.Thread(target=other_thread)
    worker.start()
    assert inside.wait(5)
    with gc_policy._automatic_collection_paused():
        assert not gc.isenabled()
    assert not gc.isenabled(), "the first pause out re-enabled collection"
    release.set()
    worker.join(5)
    assert gc.isenabled()


def test_a_finished_job_reaches_nothing_once_its_runner_is_gone(qapp, qtbot):
    from spacr.qt.job_runner import JobRunner

    owner = QWidget()
    runner = JobRunner(owner, threaded=True)
    calls = []
    runner.job_finished.connect(lambda ok: calls.append(ok))

    def slow():
        time.sleep(0.3)
        return 1

    assert runner.submit(slow, on_done=lambda result: calls.append(result))
    thread = next(iter(runner._jobs.values()))[0]
    shiboken6.delete(owner)
    assert not shiboken6.isValid(runner)
    end = time.monotonic() + 10
    while time.monotonic() < end and not _stopped(thread):
        qapp.processEvents()
        time.sleep(0.01)
    for _ in range(20):
        qapp.processEvents()
    assert calls == []


def _stopped(thread) -> bool:
    try:
        return not thread.isRunning()
    except RuntimeError:
        return True


def test_a_palette_command_runs_after_exec_returns(qapp, monkeypatch):
    from spacr.qt import command_palette as CP
    from spacr.qt import shortcuts

    seen = []
    command = CP.Command(label="Probe", section="Test",
                         action=lambda: seen.append(
                             QApplication.activeModalWidget()))
    modal = []

    def fake_exec(self):
        self._commands = [command]
        self._render(self._commands)
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item is not None and item.data(CP.Qt.UserRole) is command:
                self._list.setCurrentRow(row)
        modal.append(self)
        try:
            self._on_activate()
        finally:
            modal.clear()
        assert seen == [], "the command ran inside the palette's exec()"
        return CP.CommandPalette.Accepted

    monkeypatch.setattr(CP.CommandPalette, "exec", fake_exec)
    monkeypatch.setattr(QApplication, "activeModalWidget",
                        staticmethod(lambda: modal[0] if modal else None))
    window = QWidget()
    try:
        shortcuts._open_palette(window)
    finally:
        window.deleteLater()
    assert seen == [None]


def test_the_field_drops_its_doorbell_before_the_widget_can(qapp):
    from spacr.qt.widgets import ambient

    producer = ambient._FrameProducer.__new__(ambient._FrameProducer)
    producer.doorbell = QObject()
    producer._doorbell_lock = threading.Lock()
    rung = []
    producer.doorbell.objectNameChanged.connect(lambda _n: rung.append(1))
    producer._ring_doorbell()
    assert rung == [1]
    with producer._doorbell_lock:
        producer.doorbell = None
    producer._ring_doorbell()
    assert rung == [1]
    widget = ambient.AmbientWidget()
    try:
        assert widget._doorbell.parent() is None
    finally:
        widget.stop()
        widget.deleteLater()
