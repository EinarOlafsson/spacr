"""Distributed Jobs screen: creating, editing and deleting a profile, and closing.

The profile editor and the delete confirmation are answered by the test, the
manager runs on real JSON stores in ``tmp_path`` and no command ever runs. Each
action is walked over a table of answers (accept, cancel, decline, a store
that refuses) and the stored profiles, the combo selection, the exact status
sentence and the submit button are read back. Closing must drain every
worker pair -- live, deaf or half-reaped -- and stop polling.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QDialog, QMessageBox

from spacr.qt.screens import distributed_jobs as dj
from spacr.qt.screens.distributed_jobs import (DistributedJobsScreen,
                                               ExecutionProfileDialog)
from spacr.qt.theme import active_palette
from spacr.remote_execution import (ExecutionProfile, JobStore, ProfileStore,
                                    RemoteJobManager)

pytestmark = pytest.mark.qt


class Runner:
    """A scripted stand-in for the shell: no command ever runs."""

    def __init__(self):
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        raise AssertionError("no remote command should run here")


def _profile(name="cloud"):
    return ExecutionProfile(
        name, "command",
        submit_command="cloud-submit {module} {settings}",
        status_command="cloud-status {external_id}",
        cancel_command="cloud-cancel {external_id}",
        log_command="cloud-logs {external_id}",
    )


def _manager(tmp_path, profiles=("cloud",)):
    store = ProfileStore(tmp_path / "profiles.json")
    for name in profiles:
        store.save(_profile(name))
    return RemoteJobManager(store, JobStore(tmp_path / "jobs.json"), Runner())


def _screen(qtbot, tmp_path, profiles=("cloud",), threaded=False):
    widget = DistributedJobsScreen(manager=_manager(tmp_path, profiles),
                                   threaded=threaded, auto_poll=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def answers(monkeypatch):
    """Answer each opened editor in turn with ``(fill, QDialog result)``."""
    pending = []

    def _exec(dialog):
        fill, result = pending.pop(0)
        if fill is not None:
            fill(dialog)
        return result

    monkeypatch.setattr(ExecutionProfileDialog, "exec", _exec, raising=False)
    return pending


def _fill(name, poll=None, slurm=False):
    """Type a profile into the open editor (a full slurm one when asked)."""
    def fill(dialog):
        dialog._name.setText(name)
        if slurm:
            dialog._backend.setCurrentIndex(dialog._backend.findData("slurm"))
            dialog._workdir.setText("/project")
        if poll is not None:
            dialog._poll.setValue(poll)
    return fill


def _refuse(message):
    def refuse(*_args):
        raise OSError(message)
    return refuse


def _names(widget):
    return [p.name for p in widget.manager.profiles.list()]


def _error_style():
    return f"color: {active_palette()['error']};"


def test_creating_a_profile_saves_only_an_accepted_editor(
        qtbot, tmp_path, answers, monkeypatch, subtests):
    """Accept saves and selects; cancel writes nothing; a refusal is said."""
    widget = _screen(qtbot, tmp_path)
    with subtests.test(msg="an accepted editor is saved and selected"):
        answers.append((_fill("second", slurm=True), QDialog.Accepted))
        widget._create_profile()
        assert answers == []
        assert _names(widget) == ["cloud", "second"]
        assert widget.manager.profiles.get("second").workdir == "/project"
        assert widget.manager.profiles.get("second").backend == "slurm"
        assert widget._profile.currentData() == "second"

    with subtests.test(msg="a cancelled editor writes nothing"):
        answers.append((_fill("discarded", slurm=True), QDialog.Rejected))
        widget._create_profile()
        assert answers == []
        assert _names(widget) == ["cloud", "second"]
        assert widget._profile.currentData() == "second"

    with subtests.test(msg="a store that cannot write says so"):
        answers.append((_fill("third", slurm=True), QDialog.Accepted))
        monkeypatch.setattr(widget.manager.profiles, "save",
                            _refuse("read-only filesystem"))
        widget._create_profile()
        assert widget._status.text() == (
            "Could not save profile: OSError: read-only filesystem")
        assert widget._status.styleSheet() == _error_style()
        assert _names(widget) == ["cloud", "second"]


def test_editing_a_profile_renames_by_save_then_delete_and_cancels_cleanly(
        qtbot, tmp_path, answers, monkeypatch, subtests):
    """A rename drops the old name after the save; same name keeps it."""
    widget = _screen(qtbot, tmp_path)
    deleted = []
    real_delete = widget.manager.profiles.delete

    def record(name):
        deleted.append(name)
        return real_delete(name)

    monkeypatch.setattr(widget.manager.profiles, "delete", record)
    cases = [
        ("an accepted edit under the same name keeps it",
         _fill("cloud", poll=45), QDialog.Accepted, [], ["cloud"], "cloud",
         45),
        ("a cancelled edit leaves the stored profile alone",
         _fill("cloud", poll=99), QDialog.Rejected, [], ["cloud"], "cloud",
         45),
        ("a rename writes the new name then drops the old",
         _fill("renamed", poll=30), QDialog.Accepted, ["cloud"], ["renamed"],
         "renamed", 30),
        ("an edit of the renamed one deletes nothing",
         _fill("renamed", poll=45), QDialog.Accepted, ["cloud"], ["renamed"],
         "renamed", 45),
    ]
    for case_id, fill, result, gone, stored, selected, poll in cases:
        with subtests.test(msg=case_id):
            answers.append((fill, result))
            widget._edit_selected_profile()
            assert answers == []
            assert deleted == gone
            assert _names(widget) == stored
            assert widget._profile.currentData() == selected
            assert widget.manager.profiles.get(selected).poll_seconds == poll

    with subtests.test(msg="an edit that cannot be saved keeps the old one"):
        answers.append((_fill("again"), QDialog.Accepted))
        monkeypatch.setattr(widget.manager.profiles, "save",
                            _refuse("read-only filesystem"))
        widget._edit_selected_profile()
        assert widget._status.text() == (
            "Could not update profile: OSError: read-only filesystem")
        assert widget._status.styleSheet() == _error_style()
        assert _names(widget) == ["renamed"]
        assert deleted == ["cloud"]


def test_deleting_a_profile_needs_a_yes_and_reports_a_refusal(
        qtbot, tmp_path, monkeypatch, subtests):
    """No answers keep it, a failing store names its error, Yes removes it."""
    widget = _screen(qtbot, tmp_path)
    asked = []

    def answer(button):
        def question(parent, title, text, *_a, **_k):
            asked.append(text)
            return button
        monkeypatch.setattr(dj.QMessageBox, "question",
                            staticmethod(question))

    with subtests.test(msg="a declined confirmation keeps the profile"):
        answer(QMessageBox.No)
        widget._delete_selected_profile()
        assert asked == ["Delete execution profile 'cloud'?"]
        assert _names(widget) == ["cloud"]
        assert widget._submit.isEnabled() is True

    with subtests.test(msg="a store that cannot delete says which error"):
        answer(QMessageBox.Yes)
        with monkeypatch.context() as patch:
            patch.setattr(widget.manager.profiles, "delete",
                          _refuse("read-only filesystem"))
            widget._delete_selected_profile()
        assert widget._status.text() == (
            "Could not delete profile: OSError: read-only filesystem")
        assert _names(widget) == ["cloud"]

    with subtests.test(msg="a confirmed delete removes it and stops submit"):
        widget._delete_selected_profile()
        assert _names(widget) == []
        assert widget._profile.count() == 0
        assert widget._submit.isEnabled() is False
        assert widget._edit_profile.isEnabled() is False
        assert widget._delete_profile.isEnabled() is False

    with subtests.test(msg="with nothing selected there is nothing to ask"):
        asked.clear()
        monkeypatch.setattr(ExecutionProfileDialog, "exec",
                            lambda dialog: pytest.fail("opened an editor"),
                            raising=False)
        empty = _screen(qtbot, tmp_path / "empty", profiles=())
        empty._edit_selected_profile()
        empty._delete_selected_profile()
        assert asked == []
        assert _names(empty) == []


class _Thread:
    """A finished QThread stand-in counting interruption requests."""

    def __init__(self, deaf=False):
        self.deaf = deaf
        self.interrupted = 0

    def requestInterruption(self):      # noqa: N802 - Qt naming
        self.interrupted += 1
        if self.deaf:
            raise RuntimeError("already gone")

    def isRunning(self):                # noqa: N802 - Qt naming
        return False

    def wait(self, _ms=0):
        return True


class _Worker:
    def __init__(self, deaf=False):
        self.deaf = deaf
        self.cancelled = []

    def request_cancel(self, why):
        self.cancelled.append(why)
        if self.deaf:
            raise RuntimeError("already gone")


def test_closing_drains_every_worker_pair_and_stops_polling(qtbot, tmp_path,
                                                            subtests):
    """Live, deaf and half-reaped pairs are all cancelled and released."""
    with subtests.test(msg="a live, a deaf and a workerless pair"):
        widget = _screen(qtbot, tmp_path)
        widget._timer.start()
        live, deaf, bare = _Thread(), _Thread(deaf=True), _Thread()
        live_worker, deaf_worker = _Worker(), _Worker(deaf=True)
        widget._workers = [(live, live_worker), (deaf, deaf_worker),
                           (bare, None)]
        widget.closeEvent(QCloseEvent())
        assert live_worker.cancelled == ["distributed-jobs screen closed"]
        assert deaf_worker.cancelled == ["distributed-jobs screen closed"]
        assert [live.interrupted, deaf.interrupted, bare.interrupted] == \
            [1, 1, 1]
        assert widget._workers == []
        assert widget._timer.isActive() is False

    with subtests.test(msg="a real threaded call goes with the screen"):
        threaded = _screen(qtbot, tmp_path / "threaded", threaded=True)
        threaded.show()
        qtbot.waitExposed(threaded)
        seen = []
        threaded._start_task("Working…", lambda: "done", seen.append)
        assert len(threaded._workers) == 1
        threaded.closeEvent(QCloseEvent())
        assert threaded._workers == []
        assert threaded._timer.isActive() is False
