"""The Distributed Jobs screen's remaining "nothing happened" paths.

``test_distributed_jobs_screen.py`` drives the happy path and
``test_cov_w3_7_distributed_jobs.py`` drives the refusals. What is left, and
what is driven here, is the handful of places where the screen has to do
NOTHING and do it correctly: a profile editor the user cancelled, a rename
that renames nothing, a finished worker whose callback has already been
dropped, a job whose profile configures no remote command, and a worker pair
whose worker half was reaped before the screen closed. Each of those is a
place where "do nothing" and "quietly destroy the user's profile, or leave a
REST call running with nobody owning it" are one edit apart.

Nothing here reaches a network or a scheduler: the manager's runner is a
scripted callable and both modal dialogs are answered by the test.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QDialog, QDialogButtonBox

from spacr.qt.screens import distributed_jobs as dj
from spacr.qt.screens.distributed_jobs import DistributedJobsScreen, ExecutionProfileDialog
from spacr.remote_execution import (
    ExecutionProfile,
    JobStore,
    ProfileStore,
    RemoteJob,
    RemoteJobManager,
)


class Runner:
    """A scripted stand-in for the shell: no command ever runs."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        return self.results.pop(0)


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


@pytest.fixture
def screen(qtbot, tmp_path):
    """A screen on its own stores, running its work inline."""
    widget = DistributedJobsScreen(manager=_manager(tmp_path),
                                   threaded=False, auto_poll=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def answer_the_editor(monkeypatch):
    """Answer the profile editor the way a person would, in order."""
    def answer(*answers):
        """Each answer is ``(fill_or_None, QDialog result)``, used once."""
        pending = list(answers)

        def _exec(dialog):
            fill, result = pending.pop(0)
            if fill is not None:
                fill(dialog)
            return result

        monkeypatch.setattr(ExecutionProfileDialog, "exec", _exec,
                            raising=False)
        return pending
    return answer


def _fill(name, poll=None):
    """Return a callable that types one profile into the open editor."""
    def fill(dialog):
        dialog._name.setText(name)
        if poll is not None:
            dialog._poll.setValue(poll)
    return fill


def _fill_new(name):
    """Type a complete, valid slurm profile into an empty editor."""
    def fill(dialog):
        dialog._name.setText(name)
        dialog._backend.setCurrentIndex(dialog._backend.findData("slurm"))
        dialog._workdir.setText("/project")
    return fill


# ---------------------------------------------------------------------------
# The profile editor's own buttons
# ---------------------------------------------------------------------------

def test_the_editor_still_saves_when_its_buttons_cannot_be_styled(qtbot,
                                                                  monkeypatch):
    """Styling is decoration; Save and Cancel have to work regardless.

    The dialog looks its standard buttons up by role so it can mark Save
    primary and Cancel destructive. If a style or platform ever hands back
    nothing for that lookup, the guarded styling must be the only thing lost:
    were the ``accepted``/``rejected`` wiring to sit behind the same guard,
    the profile editor would open with two buttons that do nothing and no way
    to save a profile at all -- an unusable screen, not an unstyled one.
    """
    plain = ExecutionProfileDialog()
    qtbot.addWidget(plain)
    box = plain.findChild(QDialogButtonBox)
    assert box.button(QDialogButtonBox.Save).objectName() == "PrimaryButton"
    assert box.button(QDialogButtonBox.Cancel).objectName() == "DangerButton"

    boxes = []

    class Buttonless(QDialogButtonBox):
        """A button box that owns no button object to hand back."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.asked = []
            boxes.append(self)

        def button(self, which):        # noqa: D102 - Qt naming
            self.asked.append(which)
            return None

    monkeypatch.setattr(dj, "QDialogButtonBox", Buttonless)
    dialog = ExecutionProfileDialog()
    qtbot.addWidget(dialog)
    assert len(boxes) == 1, "the dialog did not build its own button box"
    assert boxes[0].asked == [QDialogButtonBox.Save, QDialogButtonBox.Cancel]

    dialog._name.setText("cluster")
    dialog._backend.setCurrentIndex(dialog._backend.findData("slurm"))
    dialog._workdir.setText("/project")
    accepted = []
    dialog.accepted.connect(lambda: accepted.append(True))
    boxes[0].accepted.emit()
    assert accepted == [True], "Save no longer validates and accepts"
    assert dialog.result() == QDialog.Accepted


# ---------------------------------------------------------------------------
# An editor the user cancelled
# ---------------------------------------------------------------------------

def test_a_cancelled_new_profile_writes_nothing_to_disk(screen,
                                                        answer_the_editor):
    """Cancel must mean cancel, or half-typed profiles pile up in the combo.

    The editor is filled in before the user decides. If the screen read the
    dialog's fields without first checking that it was accepted, pressing
    Cancel would still save the half-typed profile, and the profile combo --
    which is what Submit sends a job through -- would fill with entries the
    user explicitly refused.
    """
    remaining = answer_the_editor(
        (_fill_new("second"), QDialog.Accepted),
        (_fill_new("discarded"), QDialog.Rejected),
    )
    screen._create_profile()
    assert [p.name for p in screen.manager.profiles.list()] == \
        ["cloud", "second"]

    screen._create_profile()
    assert remaining == [], "the second editor was never opened"
    assert [p.name for p in screen.manager.profiles.list()] == \
        ["cloud", "second"], "a cancelled editor still wrote a profile"
    assert screen._profile.currentData() == "second"


def test_a_cancelled_edit_leaves_the_stored_profile_untouched(
        screen, answer_the_editor):
    """A cancelled edit that still writes is silent data loss.

    Editing loads the stored profile into the dialog, so every field is
    already populated with something plausible. If Cancel were treated like
    Save, the poll interval, host, and command lines the user was midway
    through changing would be persisted over a working profile and the next
    submission would go somewhere else.
    """
    answer_the_editor((_fill("cloud", poll=45), QDialog.Accepted),
                      (_fill("cloud", poll=99), QDialog.Rejected))
    screen._edit_selected_profile()
    assert screen.manager.profiles.get("cloud").poll_seconds == 45

    screen._edit_selected_profile()
    assert screen.manager.profiles.get("cloud").poll_seconds == 45, \
        "a cancelled editor still overwrote the stored profile"
    assert [p.name for p in screen.manager.profiles.list()] == ["cloud"]


def test_saving_a_profile_under_its_own_name_does_not_delete_it(
        screen, answer_the_editor):
    """The rename cleanup must fire on a rename and only on a rename.

    An edit is stored as "save the new record, then drop the old name". When
    the name did not change, those are the same record: running the cleanup
    anyway would delete the profile the user just saved and leave the screen
    with no profile to submit through. This drives both halves -- a real
    rename, which must drop the old name, and an ordinary edit, which must
    not.
    """
    deleted = []
    original_delete = screen.manager.profiles.delete

    def record(name):
        deleted.append(name)
        return original_delete(name)

    screen.manager.profiles.delete = record
    answer_the_editor((_fill("renamed", poll=30), QDialog.Accepted),
                      (_fill("renamed", poll=45), QDialog.Accepted))

    screen._edit_selected_profile()
    assert deleted == ["cloud"], "the renamed-from profile was left behind"
    assert [p.name for p in screen.manager.profiles.list()] == ["renamed"]

    screen._edit_selected_profile()
    assert deleted == ["cloud"], \
        "an edit that kept the name deleted the profile it had just saved"
    assert screen.manager.profiles.get("renamed").poll_seconds == 45
    assert screen._profile.currentData() == "renamed"


# ---------------------------------------------------------------------------
# A worker whose result nobody is waiting for
# ---------------------------------------------------------------------------

def test_a_finished_worker_with_no_callback_still_frees_the_screen(screen):
    """The screen must unlock even when the result has no home to go to.

    ``_pending_callback`` is cleared or never set for a worker whose caller
    has gone away. Calling ``None`` there would raise straight out of a Qt
    slot, and the exception would leave the screen believing an operation is
    still running: Submit, Refresh, Cancel and Logs all stay disabled and the
    user has to reopen the module to get them back.
    """
    seen = []
    screen._busy = True
    screen._pending_error = ""
    screen._pending_result = "first result"
    screen._pending_callback = seen.append
    screen._finish_task(True)
    assert seen == ["first result"]

    screen._busy = True
    screen._set_busy(True)
    screen._pending_result = "second result"
    screen._pending_callback = None
    screen._finish_task(True)
    assert seen == ["first result"], "a dropped callback was called anyway"
    assert screen._busy is False
    assert screen._refresh.isEnabled(), "the screen stayed locked"
    assert screen._submit.isEnabled()


# ---------------------------------------------------------------------------
# The copyable job record
# ---------------------------------------------------------------------------

def test_only_the_commands_a_profile_actually_configures_are_redacted():
    """The record is meant to be pasted into a bug report unedited.

    Custom command lines can carry hosts, paths and account names, so every
    configured one is replaced. A profile that configures none -- an ssh or
    slurm profile, where those fields are empty -- must not come back with
    four ``<configured command>`` placeholders instead: that would tell a
    maintainer reading the report that commands were set when they were not,
    and send them looking for the wrong bug.
    """
    job = RemoteJob(
        job_id="job-1", module="mask", profile_name="cloud",
        backend="command", status="running",
        profile={"name": "cloud", "backend": "command",
                 "host": "head.example",
                 "submit_command": "cloud-submit --key /home/me/id_rsa",
                 "status_command": "", "cancel_command": "",
                 "log_command": ""},
    )
    record = json.loads(DistributedJobsScreen._job_detail(job))
    assert record["profile"]["submit_command"] == "<configured command>"
    assert "id_rsa" not in json.dumps(record)
    assert record["profile"]["status_command"] == ""
    assert record["profile"]["cancel_command"] == ""
    assert record["profile"]["log_command"] == ""
    assert record["profile"]["host"] == "head.example"
    assert "log_tail" not in record, "the tail belongs after the JSON"

    job.log_tail = "remote line one\n"
    detail = DistributedJobsScreen._job_detail(job)
    assert detail.endswith("--- remote log tail ---\nremote line one\n")
    assert json.loads(detail.split("\n\n--- remote log tail")[0])["status"] \
        == "running"


# ---------------------------------------------------------------------------
# Closing the screen
# ---------------------------------------------------------------------------

def test_a_pair_whose_worker_is_already_gone_is_still_drained(qtbot,
                                                              tmp_path):
    """A half-reaped pair must not stop the screen draining its threads.

    ``_workers`` holds ``(thread, worker)`` pairs and the worker half can
    already be gone by the time the module is closed. Cancelling is attempted
    per pair, but the thread must be interrupted and drained either way: a
    thread that is never drained stays in the process-wide run registry, and
    that registry is what ``MainWindow.closeEvent`` consults when it decides
    whether spaCR is allowed to quit. One missed pair and the application
    refuses to close.
    """
    class StoppedThread:
        """A QThread stand-in that has already finished."""

        def __init__(self):
            self.interrupted = 0

        def requestInterruption(self):      # noqa: N802 - Qt naming
            self.interrupted += 1

        def isRunning(self):                # noqa: N802 - Qt naming
            return False

        def wait(self, _ms=0):
            return True

    class LiveWorker:
        def __init__(self):
            self.cancelled = []

        def request_cancel(self, why):
            self.cancelled.append(why)

    widget = DistributedJobsScreen(manager=_manager(tmp_path),
                                   threaded=False, auto_poll=False)
    qtbot.addWidget(widget)
    with_worker, without_worker = StoppedThread(), StoppedThread()
    worker = LiveWorker()
    widget._workers = [(with_worker, worker), (without_worker, None)]

    widget.closeEvent(QCloseEvent())
    assert worker.cancelled == ["distributed-jobs screen closed"]
    assert with_worker.interrupted == 1
    assert without_worker.interrupted == 1, \
        "the pair with no worker was skipped instead of drained"
    assert widget._workers == []
    assert not widget._timer.isActive()
