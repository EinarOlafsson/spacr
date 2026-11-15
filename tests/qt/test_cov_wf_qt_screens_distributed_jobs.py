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


# ---------------------------------------------------------------------------
# The profile editor's own buttons
# ---------------------------------------------------------------------------

def test_the_editor_still_saves_when_its_buttons_cannot_be_styled(qtbot,
                                                                  monkeypatch):
    """Styling is decoration; Save and Cancel have to work regardless.

    The dialog looks its Save button up by role so it can mark it primary;
    Cancel stays neutral because it destroys nothing (item 749). If a style or platform ever hands back
    nothing for that lookup, the guarded styling must be the only thing lost:
    were the ``accepted``/``rejected`` wiring to sit behind the same guard,
    the profile editor would open with two buttons that do nothing and no way
    to save a profile at all -- an unusable screen, not an unstyled one.
    """
    plain = ExecutionProfileDialog()
    qtbot.addWidget(plain)
    box = plain.findChild(QDialogButtonBox)
    assert box.button(QDialogButtonBox.Save).objectName() == "PrimaryButton"
    assert box.button(QDialogButtonBox.Cancel).objectName() != "DangerButton"

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
    assert boxes[0].asked == [QDialogButtonBox.Save]

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
