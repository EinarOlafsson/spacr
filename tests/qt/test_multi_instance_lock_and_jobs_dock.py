"""Multi-instance locks (a second window on the same project) and the Jobs dock."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from spacr.qt import crash_recovery as cr

_HOLDER = textwrap.dedent("""
    import os, sys
    from PySide6.QtCore import QLockFile
    lock = QLockFile(sys.argv[1])
    lock.setStaleLockTime(0)
    assert lock.tryLock(5000)
    print("ready", flush=True)
    if sys.argv[2] == "die":
        os._exit(0)
    sys.stdin.readline()
    lock.unlock()
""")


def _hold(lock_path, mode="hold"):
    """Run another process that takes ``lock_path`` and keeps or abandons it."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(lock_path), mode],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "ready"
    if mode == "die":
        proc.wait(10)
    return proc


def _release(proc):
    """Let the holder process unlock and exit."""
    proc.stdin.write("\n")
    proc.stdin.flush()
    proc.wait(10)


@pytest.fixture
def locks(tmp_path, monkeypatch):
    """A clean lock folder and no lock state left from another test."""
    monkeypatch.setenv("SPACR_LOCK_DIR", str(tmp_path / "locks"))
    cr._release_all_locks()
    yield tmp_path
    cr._release_all_locks()


def test_a_free_project_is_locked_without_asking(locks, monkeypatch):
    asked = []
    monkeypatch.setattr(cr, "_ask_about_a_locked_project",
                        lambda *a: asked.append(a) or "cancel")
    project = locks / "plate1"
    project.mkdir()
    assert cr._claim_project(None, project) == "locked"
    assert cr._claim_project(None, str(project) + os.sep) == "locked"
    assert asked == []
    assert os.path.isfile(cr._lock_file_for(project))


def test_a_project_held_by_another_window_asks_once(locks, monkeypatch):
    asked = []
    monkeypatch.setattr(cr, "_ask_about_a_locked_project",
                        lambda parent, path, holder:
                        asked.append(holder) or "read_only")
    project = locks / "plate1"
    project.mkdir()
    proc = _hold(cr._lock_file_for(project))
    try:
        assert cr._claim_project(None, project) == "read_only"
        assert cr._claim_project(None, project) == "read_only"
        assert len(asked) == 1
        assert asked[0]["pid"] == proc.pid
    finally:
        _release(proc)
    assert cr._claim_project(None, project) == "locked"


def test_a_stale_lock_from_a_dead_window_is_taken_over(locks, monkeypatch):
    monkeypatch.setattr(cr, "_ask_about_a_locked_project",
                        lambda *a: pytest.fail("a stale lock must not ask"))
    project = locks / "plate1"
    project.mkdir()
    _hold(cr._lock_file_for(project), mode="die")
    assert os.path.isfile(cr._lock_file_for(project))
    assert cr._claim_project(None, project) == "locked"


def test_a_second_instance_does_not_count_the_first_ones_marker(
        tmp_path, locks, monkeypatch):
    monkeypatch.setattr(cr, "_folder", lambda: str(tmp_path))
    proc = _hold(os.path.join(cr._locks_folder(), cr._INSTANCE_LOCK_NAME))
    try:
        (tmp_path / cr._MARKER).write_text("1")
        assert cr.note_that_a_launch_began() == 0
        assert cr._OTHER_INSTANCE["pid"] == proc.pid
        cr.note_a_clean_shutdown()
        assert (tmp_path / cr._MARKER).exists()
    finally:
        _release(proc)
        cr._release_all_locks()
    assert cr._claim_the_instance() == {}
    cr._release_all_locks()


def test_the_warning_dialog_offers_read_only_and_continue(
        qtbot, locks, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    seen = []

    def answer_with(label):
        def fake_exec(box):
            seen.append((box.objectName(), box.text()))
            for button in box.buttons():
                if button.text() == label:
                    button.click()
            return 0
        monkeypatch.setattr(QMessageBox, "exec", fake_exec)

    holder = {"pid": 4242, "host": "here", "app": "spacr"}
    answer_with("Open read-only")
    assert cr._ask_about_a_locked_project(None, "/x", holder) == "read_only"
    answer_with("Continue anyway")
    assert cr._ask_about_a_locked_project(None, "/x", holder) == "continue"
    answer_with("Cancel")
    assert cr._ask_about_a_locked_project(None, "/x", holder) == "cancel"
    assert seen[0][0] == "ProjectLockedDialog"
    assert "4242" in seen[0][1] and "/x" in seen[0][1]


def test_a_run_on_a_read_only_project_is_refused(qtbot, locks, monkeypatch):
    from PySide6.QtWidgets import QWidget

    from spacr.qt.screens import app_screen

    told = []
    monkeypatch.setattr(app_screen.QMessageBox, "information",
                        lambda *a, **k: told.append(a))
    host = QWidget()
    qtbot.addWidget(host)
    monkeypatch.setattr(cr, "_claim_project", lambda parent, path: "read_only")
    claim = app_screen.AppScreen._claim_the_project
    assert claim(host, {"src": [str(locks), "/other"]}) is False
    assert len(told) == 1
    monkeypatch.setattr(cr, "_claim_project", lambda parent, path: "continue")
    assert claim(host, {"src": str(locks)}) is True
    assert claim(host, {"src": ""}) is True


def test_the_queue_opens_read_only_when_another_window_has_it(
        qtbot, locks, monkeypatch):
    from spacr.qt import plate_queue
    from spacr.qt.screens.queue import QueueScreen

    queue_file = locks / "queue.json"
    monkeypatch.setattr(plate_queue, "_queue_path", lambda: queue_file)
    monkeypatch.setattr(cr, "_ask_about_a_locked_project",
                        lambda *a: "read_only")
    proc = _hold(cr._lock_file_for(queue_file))
    try:
        screen = QueueScreen()
        qtbot.addWidget(screen)
        assert screen._queue.read_only is True
        assert not screen._btn_run.isEnabled()
        assert not screen._btn_add.isEnabled()
        screen._queue.save()
        assert not queue_file.exists()
    finally:
        _release(proc)


class _FakeHandle:
    """Just enough of a RunHandle for the Jobs panel."""

    def __init__(self, key, progress=None, visible=True):
        self.app_key = key
        self.progress = progress
        self.user_visible = visible
        self.last_line = f"{key} working"
        self.cancelled = []

    def fraction(self):
        if not self.progress:
            return None
        done, total = self.progress
        return done / total

    def elapsed(self):
        return 3725.0

    def request_cancel(self, reason=""):
        self.cancelled.append(reason)


@pytest.fixture
def fake_registry(monkeypatch):
    """Swap the process-wide run registry for one holding fake jobs."""
    from PySide6.QtCore import QObject, Signal

    from spacr.qt import bridge

    class _Registry(QObject):
        changed = Signal()

        def __init__(self):
            super().__init__()
            self.handles = []

        def active(self):
            return list(self.handles)

    reg = _Registry()
    monkeypatch.setattr(bridge, "_REGISTRY", reg)
    return reg


def test_the_jobs_panel_lists_running_jobs_and_cancels_one(
        qtbot, fake_registry):
    from spacr.qt.widgets.activity_spinner import _JobsPanel

    panel = _JobsPanel()
    qtbot.addWidget(panel)
    assert panel.job_count() == 0
    mask = _FakeHandle("mask", progress=(3, 12))
    measure = _FakeHandle("measure")
    fake_registry.handles = [mask, measure, _FakeHandle("usage", visible=False)]
    fake_registry.changed.emit()
    qtbot.waitUntil(lambda: panel.job_count() == 2, timeout=3000)
    table = panel._table
    assert table.item(0, 0).text() == "mask"
    assert table.cellWidget(0, 1).format() == "3/12"
    assert table.cellWidget(1, 1).maximum() == 0
    assert table.item(0, 2).text() == "1:02:05"
    cancel = table.cellWidget(0, 4)
    cancel.click()
    assert mask.cancelled and not measure.cancelled
    panel.refresh()
    assert not table.cellWidget(0, 4).isEnabled()
    fake_registry.handles = []
    panel.refresh()
    assert panel.job_count() == 0


def test_the_jobs_dock_opens_from_the_window_menu(qtbot, fake_registry):
    from PySide6.QtWidgets import QMainWindow

    from spacr.qt.app import MainWindow

    window = QMainWindow()
    qtbot.addWidget(window)
    dock = MainWindow._show_jobs_dock(window)
    assert dock.objectName() == "JobsDock"
    assert dock.widget().objectName() == "JobsPanel"
    assert MainWindow._show_jobs_dock(window) is dock
    source = open(sys.modules["spacr.qt.app"].__file__).read()
    assert 'setObjectName("ShowJobsAction")' in source
