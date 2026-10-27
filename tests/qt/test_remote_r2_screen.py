"""Item 732 in the Distributed Jobs screen: jobs in a glass card, the console
in place of the details pane, followed logs, the System panel, resources and
the job-script editor, allocations, and the profile editor's sign-in methods.

Every remote is fake: a queued runner answers for ssh and the scheduler, and
the user, account and host names are invented.
"""
from __future__ import annotations

from collections import deque
import base64
import logging
import threading

import pytest
from PySide6.QtWidgets import QDialog, QPlainTextEdit

from spacr.qt.screens import distributed_jobs as ui
from spacr.qt.screens.distributed_jobs import DistributedJobsScreen, ExecutionProfileDialog
from spacr.qt.widgets import Card, ConsolePanel
from spacr.remote_execution import (
    CommandResult,
    ExecutionProfile,
    JobStore,
    ProfileStore,
    RemoteJobManager,
)

FAKE_ACCOUNT = "labfake0"
PROBE = f"""##host
cl-login1.example.edu
##user
ada_fake
##cpus
36
##mem_mb
191000
##gpu
##sinfo
standard*|up|14-00:00:00|300|36|180000|(null)
gpu|up|14-00:00:00|20|40|370000|gpu:v100:2(S:0-1)
##assoc
{FAKE_ACCOUNT}|gpu|normal,priority
##pbs
##lsf
##end
"""


class Runner:
    def __init__(self, *results):
        self.results = deque(results)
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        return self.results.popleft()


def _log_reply(start, text):
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return CommandResult(0, f"{start}\n{encoded}")


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACR_REMOTE_STATE_DIR", str(tmp_path / "state"))
    return tmp_path


def _screen(qtbot, tmp_path, runner, *profiles, preview=False, reuse=False, threaded=False):
    store = ProfileStore(tmp_path / "profiles.json")
    for profile in () if reuse else profiles or (
            ExecutionProfile("fakelakes", "slurm", host="fakelakes", workdir="/home/ada_fake/w"),):
        store.save(profile)
    manager = RemoteJobManager(store, JobStore(tmp_path / "jobs.json"), runner)
    screen = DistributedJobsScreen(manager=manager, threaded=threaded, auto_poll=False, preview=preview)
    qtbot.addWidget(screen)
    return screen


def test_jobs_sit_in_a_glass_card_and_the_console_replaces_the_details_pane(qtbot, qt_theme_applied, state):
    screen = _screen(qtbot, state, Runner())
    card = screen._table.parentWidget()
    while card is not None and not isinstance(card, Card):
        card = card.parentWidget()
    assert card is screen._jobs_card and card.objectName() == "Card"
    assert isinstance(screen._console, ConsolePanel)
    names = [pane.name for pane in screen._body_splitter.panes()]
    assert names == ["Jobs", "Console", "System"]
    assert not [w for w in screen.findChildren(QPlainTextEdit)
                if w.accessibleName() == "Distributed job details and log"]


def test_a_submission_prints_the_event_and_script_then_follows_the_log(qtbot, qt_theme_applied, state, tmp_path):
    settings = tmp_path / "s.json"
    settings.write_text('{"src": "/home/ada_fake/plate"}')
    runner = Runner(CommandResult(0), CommandResult(0, "4242\n"))
    screen = _screen(qtbot, state, runner)
    screen._module.setCurrentIndex(screen._module.findData("mask"))
    screen._settings_path.setText(str(settings))
    screen._env_edit.setPlainText("module load mamba\nmamba activate spacr")
    screen.submit()
    job = screen.manager.jobs.list()[0]
    text = screen._console.as_text()
    assert "4242" in text and "mamba activate spacr" in text and "#SBATCH --job-name" in text
    assert screen.manager.profiles.get("fakelakes").env_setup == "module load mamba\nmamba activate spacr"
    assert job.job_id in screen._follow
    first_log = "segmenting well 1\n"
    first_offset = len(first_log.encode("utf-8"))
    runner.results.extend([CommandResult(0, "RUNNING\n"), _log_reply(0, first_log)])
    screen.refresh()
    assert "segmenting well 1" in screen._console.as_text()
    assert screen._follow[job.job_id] == first_offset
    errors = []
    screen._console.append_error = errors.append
    failed_log = "spacr-run: command not found\n"
    final_offset = first_offset + len(failed_log.encode("utf-8"))
    runner.results.extend([CommandResult(0, ""), CommandResult(0, "FAILED|127:0\n"),
                           _log_reply(first_offset, failed_log)])
    screen.refresh()
    assert "spacr-run: command not found" in screen._console.as_text()
    assert len(errors) == 1 and "exit code 127" in errors[0] and "mamba activate spacr" in errors[0]
    assert job.job_id in screen._follow, "the final read must confirm that the log is drained"
    runner.results.append(_log_reply(final_offset, ""))
    screen.refresh()
    assert job.job_id not in screen._follow
    calls = len(runner.calls)
    screen.refresh()
    assert len(runner.calls) == calls, "a finished job is neither polled nor followed again"


def test_a_terminal_log_retries_a_failed_fetch_and_drains_every_chunk_before_retiring(
        qtbot, qt_theme_applied, state, monkeypatch):
    runner = Runner(CommandResult(0), CommandResult(0, "4243\n"))
    screen = _screen(qtbot, state, runner)
    screen.configure_submission("mask", {"src": "/invented/plate"})
    screen.submit()
    job = screen.manager.jobs.list()[0]
    job.status = "success"
    screen.manager.jobs.save(job)
    displayed = []
    display_stdout = screen._console._display_stdout

    def observe_display(text):
        displayed.append(text)
        display_stdout(text)

    monkeypatch.setattr(screen._console, "_display_stdout", observe_display)
    first = "first full chunk\n"
    first += "x" * (262144 - len(first))
    second = "last chunk with €\n"
    first_offset = len(first.encode("utf-8"))
    final_offset = first_offset + len(second.encode("utf-8"))
    runner.results.extend([
        CommandResult(1, "", "invented transient log retrieval error"),
        _log_reply(0, first), _log_reply(first_offset, second),
        _log_reply(final_offset, ""),
    ])
    screen.refresh()
    assert screen._follow[job.job_id] == 0
    assert "invented transient log retrieval error" in screen._console.as_text()
    screen.refresh()
    assert screen._follow[job.job_id] == first_offset
    screen.refresh()
    assert screen._follow[job.job_id] == final_offset
    header = f"[{job.job_id[:12]} {job.module}]\n"
    notice = ui.tr("Job {job} ({module}) is now {status}.",
                   job=job.job_id[:12], module=job.module, status="success") + "\n"
    assert displayed == [notice, header + first + "\n", header + second]
    text = screen._console.as_text()
    assert second.rstrip("\n") in text
    screen.refresh()
    assert job.job_id not in screen._follow
    calls = len(runner.calls)
    screen.refresh()
    assert len(runner.calls) == calls
    assert not runner.results


def test_the_preview_is_confirmed_before_anything_is_sent(qtbot, qt_theme_applied, state, tmp_path, monkeypatch):
    settings = tmp_path / "s.json"
    settings.write_text('{"src": "/p"}')
    runner = Runner(CommandResult(0), CommandResult(0, "51\n"))
    screen = _screen(qtbot, state, runner, preview=True)
    screen._settings_path.setText(str(settings))
    shown = []
    monkeypatch.setattr(ui._ScriptPreviewDialog, "exec",
                        lambda dialog: shown.append(dialog.text.toPlainText()) or QDialog.Rejected)
    screen.submit()
    assert runner.calls == [] and "nothing was sent" in screen._status.text()
    assert "exec spacr-run" in shown[0]
    monkeypatch.setattr(ui._ScriptPreviewDialog, "exec", lambda dialog: QDialog.Accepted)
    screen.submit()
    assert len(runner.calls) == 2 and runner.calls[1][1]["input_text"].startswith("#!/bin/bash -l")


def test_a_custom_script_without_settings_is_saved_and_previewed(qtbot, qt_theme_applied, state):
    screen = _screen(qtbot, state, Runner())
    screen._module.setCurrentIndex(screen._module.findData("mask"))
    screen._script_edit.setPlainText("#!/bin/bash -l\n{env_setup}\nmy-tool {module} {settings_path}\n")
    screen._save_script()
    assert "my-tool {module}" in screen.manager.profiles.get("fakelakes").job_script
    preview = screen._preview_script()
    assert "my-tool mask /home/ada_fake/w/spacr-jobs/preview/settings.json" in preview
    assert "--settings" not in preview
    screen._use_script_preset()
    screen._save_script()
    assert screen.manager.profiles.get("fakelakes").job_script == "", "the unchanged preset is stored as the default"


def test_an_unknown_placeholder_is_a_warning_not_a_save(qtbot, qt_theme_applied, state):
    screen = _screen(qtbot, state, Runner())
    screen._script_edit.setPlainText("spacr-run {module} {setings_path}\n")
    screen._save_script()
    assert "setings_path" in screen._status.text()
    assert screen.manager.profiles.get("fakelakes").job_script == ""


def test_resources_are_saved_per_profile_and_checked_against_the_cluster(qtbot, qt_theme_applied, state):
    runner = Runner(CommandResult(0, PROBE))
    screen = _screen(qtbot, state, runner)
    screen._read_cluster()
    assert screen._system_table.rowCount() == 2
    assert screen._system_table.item(1, 0).text() == "gpu"
    assert screen._system_table.item(1, 4).text() == "2"
    assert FAKE_ACCOUNT in screen._system_accounts.text() and "priority" in screen._system_accounts.text()
    assert "cl-login1.example.edu" in screen._system_summary.text()
    assert screen._res_partition.findText("gpu") >= 0 and screen._res_account.findText(FAKE_ACCOUNT) >= 0
    screen._res_partition.setEditText("gpu")
    screen._res_account.setEditText(FAKE_ACCOUNT)
    screen._res_gpus.setValue(4)
    screen._res_cpus.setValue(20)
    screen._res_mem.setText("90G")
    screen._res_time.setText("48:00:00")
    warnings = []
    screen._console.append_warning = warnings.append
    screen._save_resources()
    saved = screen.manager.profiles.get("fakelakes").resources
    assert saved == {"partition": "gpu", "account": FAKE_ACCOUNT, "gpus": "4", "cpus": "20",
                     "mem": "90G", "time": "48:00:00"}
    assert any("4 GPUs" in item for item in warnings)
    screen._res_gpus.setValue(2)
    assert screen._check_resources() == []
    rebuilt = _screen(qtbot, state, Runner(), reuse=True)
    assert rebuilt._system_table.rowCount() == 2, "the last probe is remembered per profile"
    assert rebuilt._res_gpus.value() == 4


def test_an_allocation_is_started_listed_and_used(qtbot, qt_theme_applied, state, tmp_path, monkeypatch):
    settings = tmp_path / "s.json"
    settings.write_text('{"src": "/p"}')
    runner = Runner(CommandResult(0), CommandResult(0, "7001\n"))
    screen = _screen(qtbot, state, runner)
    screen._res_gpus.setValue(1)
    screen._request_allocation()
    assert "sleep infinity" in runner.calls[1][1]["input_text"]
    assert screen._run_inside.count() == 1
    runner.results.extend([CommandResult(0, "RUNNING\n"), CommandResult(0, "0\n")])
    screen.refresh()
    assert screen._run_inside.count() == 2 and screen._copy_shell.isEnabled()
    screen._run_inside.setCurrentIndex(1)
    copied = []
    monkeypatch.setattr(ui.QApplication.clipboard(), "setText", copied.append, raising=False)
    screen._copy_allocation_shell()
    assert copied and "srun --jobid=7001 --overlap --pty bash -l" in copied[0]
    screen._settings_path.setText(str(settings))
    runner.results.extend([CommandResult(0), CommandResult(0, "9090\n")])
    screen.submit()
    assert "srun --jobid=7001 --overlap" in runner.calls[-1][0][-1]


def test_the_profile_dialog_offers_every_sign_in_method(qtbot, qt_theme_applied):
    dialog = ExecutionProfileDialog()
    qtbot.addWidget(dialog)
    modes = [dialog._auth_mode.itemData(index) for index in range(dialog._auth_mode.count())]
    assert modes == ["openssh", "interactive", "gssapi", "custom"]
    dialog._name.setText("corp")
    dialog._backend.setCurrentIndex(dialog._backend.findData("slurm"))
    dialog._host.setText("corp-hpc")
    dialog._workdir.setText("/w")
    dialog._auth_mode.setCurrentIndex(dialog._auth_mode.findData("custom"))
    assert dialog._ssh_command.isEnabled() and not dialog._ssh_keychain.isEnabled()
    dialog._ssh_command.setText("corp-ssh {host} {command}")
    profile = dialog.profile()
    assert profile.auth_mode == "custom" and profile.ssh_command == "corp-ssh {host} {command}"
    assert not profile.ssh_interactive
    dialog._ssh_command.setText("sshpass -p secret ssh {host}")
    dialog._validate_and_accept()
    assert "password" in dialog._error.text()
    dialog._auth_mode.setCurrentIndex(dialog._auth_mode.findData("gssapi"))
    assert dialog.profile().auth_mode == "gssapi" and dialog.profile().ssh_command == ""


def test_a_preset_fills_environment_resources_and_login_hosts(qtbot, qt_theme_applied):
    dialog = ExecutionProfileDialog()
    qtbot.addWidget(dialog)
    dialog._preset.setCurrentIndex(dialog._preset.findData("slurm_module_mamba"))
    dialog._apply_preset(dialog._preset.currentIndex())
    dialog._name.setText("lakes")
    dialog._host.setText("lakes")
    dialog._workdir.setText("/home/ada_fake")
    profile = dialog.profile()
    assert profile.backend == "slurm" and "mamba activate spacr" in profile.env_setup
    assert profile.resources["gpus"] == "1" and profile.login_hosts == "*login*"
    restored = ExecutionProfileDialog(profile=profile)
    qtbot.addWidget(restored)
    assert restored.profile() == profile


def test_direct_run_is_an_explicit_warned_ssh_only_opt_in(qtbot, qt_theme_applied):
    dialog = ExecutionProfileDialog()
    qtbot.addWidget(dialog)
    dialog._backend.setCurrentIndex(dialog._backend.findData("slurm"))
    assert not dialog._direct_run.isEnabled()
    dialog._backend.setCurrentIndex(dialog._backend.findData("ssh"))
    assert dialog._direct_run.isEnabled() and not dialog._direct_run.isChecked()
    dialog._direct_run.setChecked(True)
    assert "login-node refusal" in dialog._error.text() and "banned" in dialog._error.text()
    dialog._direct_run.setChecked(False)
    assert dialog._error.text() == ""


class DelayedRunner(Runner):
    def __init__(self, *results):
        super().__init__(*results)
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, argv, **kwargs):
        self.entered.set()
        assert self.release.wait(5), "the test must release its own probe"
        return super().__call__(argv, **kwargs)


def test_a_late_cluster_probe_is_cached_for_its_profile_without_replacing_another(
        qtbot, qt_theme_applied, state):
    first = ExecutionProfile("a-cluster", "slurm", host="a.example", workdir="/w")
    second = ExecutionProfile("b-cluster", "pbs", host="b.example", workdir="/w")
    cached_second = {"host": "b.example", "scheduler": "pbs", "pbs_queues": ["bqueue"]}
    ui._save_cluster_cache(second.name, cached_second)
    cached_second = ui._load_cluster_cache(second.name)
    runner = DelayedRunner(CommandResult(0, PROBE))
    screen = _screen(qtbot, state, runner, first, second, threaded=True)
    screen._read_cluster()
    try:
        qtbot.waitUntil(runner.entered.is_set, timeout=5000)
        screen._profile.setCurrentIndex(screen._profile.findData(second.name))
        screen._res_cpus.setValue(7)
        before = screen._system_summary.text()
        assert screen._cluster == cached_second
    finally:
        runner.release.set()
        qtbot.waitUntil(lambda: not screen._busy and not screen._workers, timeout=5000)
    assert screen._profile.currentData() == second.name
    assert screen._cluster == cached_second
    assert screen._system_summary.text() == before
    assert screen._system_table.item(0, 0).text() == "bqueue"
    assert screen._res_cpus.value() == 7
    assert ui._load_cluster_cache(second.name) == cached_second
    assert ui._load_cluster_cache(first.name)["host"] == "cl-login1.example.edu"
    screen._profile.setCurrentIndex(screen._profile.findData(first.name))
    assert screen._cluster["host"] == "cl-login1.example.edu"
    assert screen._system_table.rowCount() == 2


def test_a_logged_operation_failure_has_one_central_occurrence_and_real_console_output(
        qtbot, qt_theme_applied, state, monkeypatch):
    from spacr import crashreport
    from spacr.logging_util import _AutomaticErrorHandler
    from spacr.remote_execution import RemoteExecutionError

    screen = _screen(qtbot, state, Runner(), threaded=True)
    logger = logging.Logger("spacr.qt.screens.distributed_jobs")
    logger.addHandler(_AutomaticErrorHandler())
    monkeypatch.setattr(ui, "LOG", logger)
    calls = []

    def dispatched(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(crashreport, "_dispatch_error", dispatched)
    failure = RemoteExecutionError("invented remote endpoint refused the operation")

    def fail():
        raise failure

    screen._start_task("Testing an isolated remote failure", fail, pytest.fail)
    qtbot.waitUntil(lambda: not screen._busy and not screen._workers, timeout=5000)
    assert len(calls) == 1
    assert calls[0][1]["exception"] is failure
    assert calls[0][1]["event_id"] == failure._spacr_error_occurrence
    assert "spaCR ERROR" in screen._console.as_text()
    assert str(failure) in screen._console.as_text()
    assert str(failure) in screen._status.text()
