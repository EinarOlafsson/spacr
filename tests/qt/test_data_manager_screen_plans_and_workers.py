"""Data Manager: planning, confirming a deletion, the note strip and the worker.

The screen owns the only call in spaCR that deletes a user's data, so every
path to it is pinned with the exact sentence it leaves: planning with no
project, a plan with nothing to delete, a cancelled and an accepted
confirmation, a worker that fails or errors, and a close with a thread still
alive. Dialogs are replaced by stand-ins that answer without blocking.

Offscreen, CPU-only, offline.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Signal                    # noqa: E402
from PySide6.QtWidgets import QDialog, QLabel                 # noqa: E402

from spacr import data_manager as dm                          # noqa: E402
from spacr.qt import bridge                                   # noqa: E402
from spacr.qt.screens import data_manager as screen_module    # noqa: E402

from tests.test_data_manager import (build_project,            # noqa: E402
                                     register_pipeline)

pytestmark = pytest.mark.qt


@pytest.fixture()
def project(tmp_path):
    """A registered project on disk."""
    root = str(tmp_path / "plate1")
    build_project(root)
    register_pipeline(root)
    return root


def _files(root):
    return sorted(os.path.join(folder, name)
                  for folder, _dirs, names in os.walk(root) for name in names)


def _answering(dialog_result):
    class _Dialog:
        """A confirmation dialog that answers without being shown."""

        made = []

        def __init__(self, plan, parent=None, *, threaded=True):
            self.plan = plan
            self.threaded = threaded
            self.made.append(self)

        def exec(self):
            return dialog_result

    return _Dialog


class _StubWorker(QObject):
    error = Signal(str)
    finished = Signal(bool)


class _HandThread(QObject):
    """A ``QThread`` stand-in that runs the job body when started.

    ``settle`` decides what ``start`` announces: True or False on
    ``worker.finished``, or None to leave the settling to the test.
    """

    finished = Signal()

    def __init__(self, fn, settings, worker, settle):
        super().__init__()
        self._fn = fn
        self._settings = settings
        self.worker = worker
        self.settle = settle
        self.quit_calls = 0
        self.waited = []

    def start(self):
        self._fn(self._settings)
        if self.settle is not None:
            self.worker.finished.emit(self.settle)

    def quit(self):
        self.quit_calls += 1

    def wait(self, msecs=0):
        self.waited.append(msecs)
        return True


def _threads(monkeypatch, settle):
    made = []

    def _make_thread(fn, settings, app_key="", **kwargs):
        assert kwargs == {"journal": False}
        worker = _StubWorker()
        thread = _HandThread(fn, settings, worker, settle)
        made.append(thread)
        return thread, worker

    monkeypatch.setattr(bridge, "make_thread", _make_thread)
    return made


def _usage(root, **fields):
    return dm.ProjectUsage(root=str(root), total_bytes=2048, total_files=4,
                           **fields)


class _RecordingStyle:
    def __init__(self):
        self.calls = []

    def unpolish(self, widget):
        self.calls.append("unpolish")

    def polish(self, widget):
        self.calls.append("polish")


class _StyledLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorder = _RecordingStyle()

    def style(self):
        return self.recorder


class _StylelessLabel(QLabel):
    def style(self):
        return None


def test_planning_and_confirming_a_prune(qtbot, project, tmp_path,
                                        monkeypatch, subtests):
    """No project asks for one; an empty plan explains; only Accept deletes."""
    with subtests.test(msg="planning without a project asks for one"):
        blank = screen_module.DataManagerScreen(threaded=False)
        qtbot.addWidget(blank)
        assert blank.plan_prune() is False
        assert blank.note_label.text() == "Choose a project folder first."
        assert blank.note_label.property("warn") == "true"
        assert blank.plan is None
        assert blank.confirm_and_prune() is False

    with subtests.test(msg="a plan with nothing to delete says to read why"):
        root = str(tmp_path / "raw_only")
        os.makedirs(os.path.join(root, "orig"))
        with open(os.path.join(root, "orig", "plate1_A01_1.tif"), "wb") as fh:
            fh.write(b"\x00" * 2048)
        raw = screen_module.DataManagerScreen(project=root, threaded=False)
        qtbot.addWidget(raw)
        assert raw.plan_prune() is True
        assert raw.plan.candidates == ()
        assert raw.freed_label.text() == (
            "Nothing here can be deleted safely — see the reasons below.")
        assert raw.prune_table.rowCount() == 0
        dialogs = _answering(QDialog.Accepted)
        monkeypatch.setattr(screen_module, "ConfirmDeleteDialog", dialogs)
        assert raw.confirm_and_prune() is False
        assert dialogs.made == []

    screen = screen_module.DataManagerScreen(project=project, threaded=False)
    qtbot.addWidget(screen)
    assert screen.plan_prune() is True
    plan = screen.plan
    planned, _ = plan.file_list()
    with subtests.test(msg="the plan is shown before anything goes"):
        assert planned
        assert screen.prune_table.rowCount() == len(plan.candidates)
        assert screen.freed_label.text() == (
            f"{dm.human_bytes(plan.total_bytes)} in {plan.total_files:,} "
            f"files can be deleted and made again.")

    with subtests.test(msg="a cancelled confirmation deletes nothing"):
        before = _files(project)
        dialogs = _answering(QDialog.Rejected)
        monkeypatch.setattr(screen_module, "ConfirmDeleteDialog", dialogs)
        assert screen.confirm_and_prune() is False
        assert screen.note_label.text() == "Nothing was deleted."
        assert screen.note_label.property("warn") == "false"
        assert _files(project) == before
        assert screen.plan is plan
        assert [(d.plan, d.threaded) for d in dialogs.made] == [(plan, False)]

    with subtests.test(msg="an accepted confirmation deletes exactly the plan"):
        before = _files(project)
        monkeypatch.setattr(screen_module, "ConfirmDeleteDialog",
                            _answering(QDialog.Accepted))
        assert screen.confirm_and_prune() is True
        assert [p for p in planned if os.path.exists(p)] == []
        assert os.path.isdir(os.path.join(project, "orig"))
        assert screen.plan is None
        assert screen.prune_table.rowCount() == 0
        assert screen.freed_label.text() == (
            f"Freed {dm.human_bytes(plan.total_bytes)} in "
            f"{plan.total_files:,} files.")
        assert sorted(set(before) - set(_files(project))) == sorted(
            p for p in planned if p in before)


def test_the_threaded_path_settles_reports_and_closes(qtbot, project,
                                                     tmp_path, monkeypatch,
                                                     subtests):
    """The worker runs the same job; failure, error text and close all land."""
    with subtests.test(msg="a successful worker settles the scan"):
        _threads(monkeypatch, settle=True)
        widget = screen_module.DataManagerScreen(threaded=True)
        qtbot.addWidget(widget)
        with qtbot.waitSignal(widget.job_finished, timeout=5000) as blocker:
            widget.set_project(project)
        assert blocker.args == [True]
        usage = widget.usage
        assert usage.total_bytes > 0
        assert widget.total_label.text() == (
            f"{dm.human_bytes(usage.total_bytes)} in "
            f"{usage.total_files:,} files")
        assert widget.usage_table.rowCount() == len(
            [k for k in usage.kinds if k.size_bytes or k.n_artifacts])

    root = str(tmp_path)
    monkeypatch.setattr(dm, "scan_project",
                        lambda path, **kw: _usage(path, symlinks=("link",)))
    threads = _threads(monkeypatch, settle=None)
    widget = screen_module.DataManagerScreen(threaded=True)
    qtbot.addWidget(widget)
    settled = []
    widget.job_finished.connect(settled.append)

    with subtests.test(msg="a failed job runs no handler but still reports"):
        widget.set_project(root)
        assert len(threads) == 1
        threads[-1].worker.finished.emit(False)
        assert settled == [False]
        assert widget.usage is None
        assert widget.rescan_button.isEnabled() is True

    with subtests.test(msg="the same job settled as success draws it"):
        widget.set_project(root)
        threads[-1].worker.finished.emit(True)
        assert settled == [False, True]
        assert widget.usage.total_bytes == 2048
        assert widget.total_label.text() == "2.0 kB in 4 files"

    for case_id, text, shown in [
            ("a traceback shows its last line",
             "Traceback (most recent call last):\n  File x\n"
             "OSError: disk is full", "OSError: disk is full"),
            ("an empty error is still a failure", "", "unknown error")]:
        with subtests.test(msg=case_id):
            threads[-1].worker.error.emit(text)
            assert widget.note_label.text() == shown
            assert widget.note_label.property("warn") == "true"

    with subtests.test(msg="closing stops a thread still on the books"):
        widget.set_project(root)
        alive = threads[-1]
        assert len(widget._jobs) == 3
        widget.close()
        assert alive.quit_calls == 1
        assert alive.waited == [2000]
        assert [t.waited for t in threads] == [[2000]] * 3
        assert widget._jobs == []


def test_the_note_strip_lists_what_is_worth_acting_on(qtbot, tmp_path,
                                                     monkeypatch, subtests):
    """Notes join with "; ", omit zero counts and land with or without a style."""
    root = str(tmp_path)
    cases = [
        ("nothing to say", {}, ""),
        ("only a symlink", {"unregistered_bytes": 0, "unregistered_files": 0,
                            "symlinks": ("data -> /elsewhere",)},
         "1 symlink(s), not followed"),
        ("unregistered bytes and a symlink",
         {"unregistered_bytes": 4096, "unregistered_files": 3,
          "symlinks": ("data -> /elsewhere",)},
         "4.1 kB in 3 files has no registry record and is never deleted; "
         "1 symlink(s), not followed"),
    ]
    screen = screen_module.DataManagerScreen(threaded=False)
    qtbot.addWidget(screen)
    for case_id, fields, note in cases:
        with subtests.test(msg=case_id):
            monkeypatch.setattr(dm, "scan_project",
                                lambda path, f=fields, **kw: _usage(path, **f))
            screen.set_project(root)
            assert screen.note_label.text() == note
            assert screen.note_label.property("warn") == "false"

    with subtests.test(msg="a styled label is repolished after the warning"):
        monkeypatch.setattr(screen_module, "QLabel", _StyledLabel)
        styled = screen_module.DataManagerScreen(threaded=False)
        qtbot.addWidget(styled)
        assert styled.set_project(str(tmp_path / "not_a_folder")) is None
        assert styled.note_label.text() == "Choose a project folder first."
        assert styled.note_label.property("warn") == "true"
        assert styled.note_label.recorder.calls[-2:] == ["unpolish", "polish"]

    with subtests.test(msg="a label with no style still says its piece"):
        monkeypatch.setattr(screen_module, "QLabel", _StylelessLabel)
        styleless = screen_module.DataManagerScreen(threaded=False)
        qtbot.addWidget(styleless)
        styleless.set_project(str(tmp_path / "not_a_folder"))
        assert styleless.note_label.style() is None
        assert styleless.note_label.text() == "Choose a project folder first."
        assert styleless.note_label.property("warn") == "true"
