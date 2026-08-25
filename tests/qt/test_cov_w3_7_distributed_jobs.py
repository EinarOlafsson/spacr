"""The Distributed Jobs screen where the remote end, or the user, says no.

``tests/qt/test_distributed_jobs_screen.py`` drives the happy path against a
scripted command runner: submit, poll, cancel, and the settings hand-off from
a module screen. What is driven here is the rest of the surface -- the profile
editor's save/rename/delete, every refusal the screen has to state in words,
the worker path taken when the operation is threaded, and the teardown that
must not leave a REST call running with nobody owning it.

Nothing here reaches a network: the manager's runner is a scripted callable,
and the two modal dialogs are answered by the test rather than by a person.
"""
from __future__ import annotations

from collections import deque

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

from spacr.qt.screens import distributed_jobs as dj
from spacr.qt.screens.distributed_jobs import (DistributedJobsScreen,
                                               ExecutionProfileDialog)
from spacr.remote_execution import (CommandResult, ExecutionProfile, JobStore,
                                    ProfileStore, RemoteExecutionError,
                                    RemoteJobManager)


class Runner:
    """A scripted stand-in for the shell: no command ever runs."""

    def __init__(self, *results):
        self.results = deque(results)
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        return self.results.popleft()


def _profile(name="cloud"):
    return ExecutionProfile(
        name, "command",
        submit_command="cloud-submit {module} {settings}",
        status_command="cloud-status {external_id}",
        cancel_command="cloud-cancel {external_id}",
        log_command="cloud-logs {external_id}",
    )


def _manager(tmp_path, runner=None, *, profiles=("cloud",)):
    store = ProfileStore(tmp_path / "profiles.json")
    for name in profiles:
        store.save(_profile(name))
    return RemoteJobManager(store, JobStore(tmp_path / "jobs.json"),
                            runner or Runner())


@pytest.fixture
def screen(qtbot, tmp_path):
    """A screen on its own stores, running its work inline."""
    widget = DistributedJobsScreen(manager=_manager(tmp_path),
                                   threaded=False, auto_poll=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def answer_yes(monkeypatch):
    """Answer the confirmation the way the user would."""
    def choose(button=QMessageBox.Yes):
        monkeypatch.setattr(dj.QMessageBox, "question",
                            staticmethod(lambda *a, **k: button))
    return choose


def _submitted(screen, tmp_path, runner_output="cloud-12\n"):
    """Put one real job record in front of the screen."""
    screen.manager.runner = Runner(CommandResult(0, runner_output))
    screen.configure_submission("mask", {"src": str(tmp_path)})
    screen.submit()
    return screen._jobs[0]


# ---------------------------------------------------------------------------
# The profile editor
# ---------------------------------------------------------------------------

def test_an_existing_profile_is_loaded_back_into_every_field(qtbot):
    profile = ExecutionProfile(
        "cluster", "slurm", host="head", workdir="/project",
        local_root="/local", remote_root="/remote", runner="srun",
        scheduler_options="--time=1", poll_seconds=25)
    dialog = ExecutionProfileDialog(profile=profile)
    qtbot.addWidget(dialog)

    assert dialog._name.text() == "cluster"
    assert dialog._backend.currentData() == "slurm"
    assert dialog._host.text() == "head"
    assert dialog._workdir.text() == "/project"
    assert dialog._local_root.text() == "/local"
    assert dialog._remote_root.text() == "/remote"
    assert dialog._runner.text() == "srun"
    assert dialog._slurm.text() == "--time=1"
    assert dialog._poll.value() == 25
    assert dialog.profile().name == "cluster"


def test_a_backend_the_dialog_does_not_offer_falls_back_to_the_first(qtbot):
    """A profile written by a newer build still opens rather than crashing."""
    profile = ExecutionProfile("odd", "slurm", workdir="/p")
    object.__setattr__(profile, "backend", "quantum")
    dialog = ExecutionProfileDialog(profile=profile)
    qtbot.addWidget(dialog)
    assert dialog._backend.currentIndex() == 0


def test_a_valid_profile_clears_the_error_and_accepts(qtbot):
    dialog = ExecutionProfileDialog()
    qtbot.addWidget(dialog)
    dialog._name.setText("cluster")
    dialog._backend.setCurrentIndex(dialog._backend.findData("slurm"))
    dialog._workdir.setText("/project")
    dialog._error.setText("something earlier went wrong")

    accepted = []
    dialog.accepted.connect(lambda: accepted.append(True))
    dialog._validate_and_accept()
    assert dialog._error.text() == ""
    assert accepted == [True]


# ---------------------------------------------------------------------------
# Loading what is on disk
# ---------------------------------------------------------------------------

def test_unreadable_job_records_are_reported_not_swallowed(qtbot, tmp_path):
    manager = _manager(tmp_path)

    def refuse():
        raise OSError("jobs.json is not readable")

    manager.jobs.list = refuse
    widget = DistributedJobsScreen(manager=manager, threaded=False,
                                   auto_poll=False)
    qtbot.addWidget(widget)
    assert "Could not load distributed job records" in widget._status.text()
    assert "OSError" in widget._status.text()
    assert widget._table.rowCount() == 0


def test_unreadable_profiles_leave_the_submit_button_off(screen):
    def refuse():
        raise OSError("profiles.json is not readable")

    screen.manager.profiles.list = refuse
    screen._reload_profiles()
    assert "Could not load execution profiles" in screen._status.text()
    assert not screen._submit.isEnabled()
    assert not screen._edit_profile.isEnabled()
    assert not screen._delete_profile.isEnabled()


# ---------------------------------------------------------------------------
# Creating, editing and deleting a profile
# ---------------------------------------------------------------------------

@pytest.fixture
def accept_the_editor(monkeypatch):
    """Answer the profile editor as though the user had filled it in."""
    def answer(fill=None, result=QDialog.Accepted):
        def _exec(dialog):
            if fill is not None:
                fill(dialog)
            return result
        monkeypatch.setattr(ExecutionProfileDialog, "exec", _exec,
                            raising=False)
    return answer


def test_a_created_profile_is_saved_and_selected(screen, accept_the_editor):
    def fill(dialog):
        dialog._name.setText("second")
        dialog._backend.setCurrentIndex(dialog._backend.findData("slurm"))
        dialog._workdir.setText("/project")

    accept_the_editor(fill)
    screen._create_profile()
    assert screen.manager.profiles.get("second").workdir == "/project"
    assert screen._profile.currentData() == "second"


def test_a_profile_that_cannot_be_written_says_so(screen, accept_the_editor,
                                                  monkeypatch):
    def fill(dialog):
        dialog._name.setText("second")
        dialog._backend.setCurrentIndex(dialog._backend.findData("slurm"))
        dialog._workdir.setText("/project")

    accept_the_editor(fill)
    monkeypatch.setattr(screen.manager.profiles, "save",
                        lambda profile: (_ for _ in ()).throw(
                            OSError("read-only filesystem")))
    screen._create_profile()
    assert "Could not save profile: OSError" in screen._status.text()


def test_nothing_selected_is_nothing_to_edit_or_delete(qtbot, tmp_path,
                                                       accept_the_editor):
    widget = DistributedJobsScreen(
        manager=_manager(tmp_path, profiles=()), threaded=False,
        auto_poll=False)
    qtbot.addWidget(widget)
    accept_the_editor(lambda dialog: pytest.fail("opened an empty editor"))
    widget._edit_selected_profile()
    widget._delete_selected_profile()


def test_a_profile_that_has_gone_since_the_combo_was_filled(screen,
                                                            monkeypatch):
    monkeypatch.setattr(screen.manager.profiles, "get",
                        lambda name: (_ for _ in ()).throw(
                            RemoteExecutionError("no profile named 'cloud'")))
    screen._edit_selected_profile()
    assert "no profile named 'cloud'" in screen._status.text()


def test_renaming_a_profile_writes_the_new_one_before_dropping_the_old(
        screen, accept_the_editor):
    """A disk error must not erase the only usable profile."""
    accept_the_editor(lambda dialog: dialog._name.setText("renamed"))
    screen._edit_selected_profile()
    assert screen.manager.profiles.get("renamed").name == "renamed"
    assert [p.name for p in screen.manager.profiles.list()] == ["renamed"]
    assert screen._profile.currentData() == "renamed"


def test_an_edit_that_cannot_be_saved_keeps_the_old_profile(screen,
                                                            accept_the_editor,
                                                            monkeypatch):
    accept_the_editor(lambda dialog: dialog._name.setText("renamed"))
    monkeypatch.setattr(screen.manager.profiles, "save",
                        lambda profile: (_ for _ in ()).throw(
                            OSError("read-only filesystem")))
    screen._edit_selected_profile()
    assert "Could not update profile: OSError" in screen._status.text()
    assert screen.manager.profiles.get("cloud").name == "cloud"


def test_a_declined_confirmation_keeps_the_profile(screen, answer_yes):
    answer_yes(QMessageBox.No)
    screen._delete_selected_profile()
    assert [p.name for p in screen.manager.profiles.list()] == ["cloud"]


def test_a_confirmed_delete_removes_it_and_disables_submitting(screen,
                                                               answer_yes):
    answer_yes()
    screen._delete_selected_profile()
    assert screen.manager.profiles.list() == []
    assert not screen._submit.isEnabled()


def test_a_delete_that_fails_says_which_error_it_was(screen, answer_yes,
                                                     monkeypatch):
    answer_yes()
    monkeypatch.setattr(screen.manager.profiles, "delete",
                        lambda name: (_ for _ in ()).throw(
                            OSError("read-only filesystem")))
    screen._delete_selected_profile()
    assert "Could not delete profile: OSError" in screen._status.text()
    assert [p.name for p in screen.manager.profiles.list()] == ["cloud"]


def test_the_chosen_settings_file_lands_in_the_field(screen, monkeypatch,
                                                     tmp_path):
    chosen = tmp_path / "settings.csv"
    monkeypatch.setattr(dj.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(chosen), "")))
    screen._browse_settings()
    assert screen._settings_path.text() == str(chosen)

    monkeypatch.setattr(dj.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    screen._browse_settings()
    assert screen._settings_path.text() == str(chosen), \
        "a cancelled file dialog cleared the path the user already had"


# ---------------------------------------------------------------------------
# Running one operation
# ---------------------------------------------------------------------------

def test_one_operation_at_a_time(screen):
    screen._busy = True
    screen._start_task("Submitting…", lambda: pytest.fail("ran twice"),
                       lambda result: None)
    assert "Another distributed-job operation is running" in \
        screen._status.text()


def test_an_operation_that_raises_is_named_in_the_status_line(screen):
    def explode():
        raise RemoteExecutionError("ssh: connection refused")

    screen._start_task("Polling…", explode,
                       lambda result: pytest.fail("called back on failure"))
    assert not screen._busy
    assert "RemoteExecutionError: ssh: connection refused" in \
        screen._status.text()


def test_a_worker_that_finishes_unsuccessfully_says_so(screen):
    screen._busy = True
    screen._pending_error = ""
    screen._pending_callback = lambda result: pytest.fail("called back")
    screen._finish_task(False)
    assert "Distributed operation failed" in screen._status.text()
    assert not screen._busy


def test_a_threaded_operation_runs_off_the_gui_thread(qtbot, tmp_path):
    """Every network call goes through ``make_thread``; this is that path."""
    widget = DistributedJobsScreen(manager=_manager(tmp_path), threaded=True,
                                   auto_poll=False)
    qtbot.addWidget(widget)
    seen = []
    widget._start_task("Working…", lambda: "done", seen.append)
    assert widget._busy is True
    assert widget._workers, "no worker was registered for the operation"

    qtbot.waitUntil(lambda: not widget._busy, timeout=5000)
    assert seen == ["done"]
    qtbot.waitUntil(lambda: widget._workers == [], timeout=5000)


# ---------------------------------------------------------------------------
# Submitting, cancelling, logs
# ---------------------------------------------------------------------------

def test_a_settings_file_that_is_not_there_is_refused_before_the_network(
        screen, tmp_path):
    screen._settings_path.setText(str(tmp_path / "not-written-yet.json"))
    screen._clear_settings_snapshot()
    screen.submit()
    assert "Choose an existing settings CSV/JSON" in screen._status.text()
    assert screen._table.rowCount() == 0


def test_an_unknown_module_never_reaches_the_remote_end(screen, tmp_path):
    screen.configure_submission("not_a_module", {"src": str(tmp_path)})
    screen._module.addItem("not_a_module", "not_a_module")
    screen._module.setCurrentIndex(screen._module.findData("not_a_module"))
    screen.submit()
    assert "Unknown spaCR module: not_a_module" in screen._status.text()


def test_cancelling_needs_a_job_and_an_unfinished_one(screen, tmp_path,
                                                      answer_yes):
    screen.cancel_selected()
    assert "Select a job first" in screen._status.text()

    job = _submitted(screen, tmp_path)
    job.status = "success"
    screen.manager.jobs.save(job)
    screen._render_jobs(screen.manager.jobs.list())
    answer_yes(QMessageBox.No)
    screen.cancel_selected()
    assert "already success" in screen._status.text()


def test_a_declined_cancellation_leaves_the_job_running(screen, tmp_path,
                                                        answer_yes):
    _submitted(screen, tmp_path)
    answer_yes(QMessageBox.No)
    screen.manager.runner = Runner()      # nothing may be run
    screen.cancel_selected()
    assert screen._table.item(0, 1).text() == "queued"


def test_the_log_needs_a_job_and_lands_in_the_detail_pane(screen, tmp_path):
    screen.refresh_log()
    assert "Select a job first" in screen._status.text()

    _submitted(screen, tmp_path)
    screen.manager.runner = Runner(CommandResult(0, "line one\nline two\n"))
    screen.refresh_log()
    assert "Log refreshed" in screen._status.text()
    assert "remote log tail" in screen._detail.toPlainText()
    assert "line two" in screen._detail.toPlainText()


def test_a_configured_command_is_not_pasted_into_a_bug_report(screen,
                                                              tmp_path):
    job = _submitted(screen, tmp_path)
    detail = screen._job_detail(job)
    assert "<configured command>" in detail
    assert "cloud-submit {module} {settings}" not in detail


# ---------------------------------------------------------------------------
# The table and the record
# ---------------------------------------------------------------------------

def test_with_nothing_selected_the_first_row_is_shown(screen, tmp_path):
    _submitted(screen, tmp_path)
    screen._table.clearSelection()
    screen._table.setCurrentCell(-1, -1)
    screen._render_jobs(screen.manager.jobs.list())
    assert screen._table.currentRow() == 0
    assert screen._cancel.isEnabled()


def test_an_empty_table_disables_every_job_action(screen):
    screen._render_jobs([])
    assert "No distributed jobs have been submitted" in \
        screen._detail.toPlainText()
    assert not screen._cancel.isEnabled()
    assert not screen._logs.isEnabled()
    assert not screen._open_local.isEnabled()


def test_the_local_record_folder_is_opened_only_when_there_is_one(screen,
                                                                  tmp_path,
                                                                  monkeypatch):
    opened = []
    monkeypatch.setattr(dj.QDesktopServices, "openUrl",
                        staticmethod(opened.append))
    screen._open_local_record()
    assert opened == []

    job = _submitted(screen, tmp_path)
    screen._open_local_record()
    assert len(opened) == 1
    assert opened[0].toLocalFile().rstrip("/") == \
        str(dj.Path(job.settings_path).parent)


# ---------------------------------------------------------------------------
# Polling only while it is on screen
# ---------------------------------------------------------------------------

def test_polling_starts_when_the_screen_opens_and_stops_when_it_leaves(
        qtbot, tmp_path):
    widget = DistributedJobsScreen(manager=_manager(tmp_path), threaded=False,
                                   auto_poll=True)
    qtbot.addWidget(widget)
    assert not widget._timer.isActive()

    widget.show()
    qtbot.waitExposed(widget)
    assert widget._timer.isActive()

    widget.hide()
    assert not widget._timer.isActive()


def test_the_poll_interval_follows_the_fastest_profile(screen):
    fast = _profile("fast")
    object.__setattr__(fast, "poll_seconds", 3)
    screen.manager.profiles.save(fast)
    screen._reload_profiles()
    assert screen._timer.interval() == 3000

    screen._update_poll_interval([])
    assert screen._timer.interval() == 10000


def test_closing_the_screen_takes_its_remote_calls_with_it(qtbot, tmp_path):
    """An ownerless REST call keeps the application from being able to quit."""
    from PySide6.QtGui import QCloseEvent

    widget = DistributedJobsScreen(manager=_manager(tmp_path), threaded=True,
                                   auto_poll=False)
    qtbot.addWidget(widget)
    widget.show()
    qtbot.waitExposed(widget)
    widget._start_task("Working…", lambda: "done", lambda result: None)
    assert widget._workers

    widget.closeEvent(QCloseEvent())
    assert widget._workers == []
    assert not widget._timer.isActive()


def test_a_worker_that_will_not_answer_is_still_released(qtbot, tmp_path):
    """Cancellation and interruption are attempts, not preconditions."""
    from PySide6.QtGui import QCloseEvent

    class Deaf:
        def request_cancel(self, _why):
            raise RuntimeError("already gone")

        def requestInterruption(self):      # noqa: N802 - Qt naming
            raise RuntimeError("already gone")

        def isRunning(self):                # noqa: N802 - Qt naming
            return False

        def wait(self, _ms=0):
            return True

    widget = DistributedJobsScreen(manager=_manager(tmp_path), threaded=False,
                                   auto_poll=False)
    qtbot.addWidget(widget)
    widget._workers = [(Deaf(), Deaf())]
    widget.closeEvent(QCloseEvent())
    assert widget._workers == []
