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
            ExecutionProfile("fakelakes", "slurm", host="fakelakes", workdir="/projects/ada_fake/w"),):
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
    settings.write_text('{"src": "/projects/ada_fake/plate"}')
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
    assert "my-tool mask /projects/ada_fake/w/spacr-jobs/preview/settings.json" in preview
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
    dialog._workdir.setText("/projects/ada_fake")
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


def _raise_owned_remote_error(error, *args, **kwargs):
    raise error


def test_editor_presets_and_readonly_preview_preserve_real_script_text(qtbot, qt_theme_applied, state):
    from PySide6.QtWidgets import QDialogButtonBox
    from PySide6.QtGui import QTextCursor

    screen = _screen(qtbot, state, Runner())
    key, lines = next((key, value) for key, value in ui._ENV_PRESETS.items() if value)
    screen._env_preset.setCurrentIndex(screen._env_preset.findData(key))
    screen._env_edit.setPlainText('echo before\n')
    screen._add_env_lines()
    assert screen._env_edit.toPlainText() == 'echo before\n' + lines
    screen._env_edit.clear()
    screen._add_env_lines()
    assert screen._env_edit.toPlainText() == lines
    screen._script_edit.setPlainText('run ')
    screen._script_edit.moveCursor(QTextCursor.End)
    placeholder = str(screen._placeholder.currentData())
    screen._insert_placeholder()
    assert screen._script_edit.toPlainText() == 'run {' + placeholder + '}'
    screen._script_edit.setPlainText('#!/bin/bash\n{run_command}\n')
    text = screen._preview_script()
    assert text.startswith('#!/bin/bash') and '{run_command}' not in text and 'spacr-run' in text
    assert text.rstrip() in screen._console.as_text()
    preview = ui._ScriptPreviewDialog(screen, text, confirm=False)
    qtbot.addWidget(preview)
    assert preview.text.isReadOnly() and preview.text.toPlainText() == text
    buttons = preview.findChild(QDialogButtonBox)
    assert buttons.button(QDialogButtonBox.Close) is not None
    assert buttons.button(QDialogButtonBox.Cancel) is None
    assert screen.manager.jobs.list() == []


def test_missing_profile_and_preview_errors_never_submit(qtbot, qt_theme_applied, state, monkeypatch):
    import functools
    from spacr.remote_execution import RemoteExecutionError

    runner = Runner()
    screen = _screen(qtbot, state, runner)
    error = RemoteExecutionError('preview unavailable')
    monkeypatch.setattr(ui, '_preview_job_script', functools.partial(_raise_owned_remote_error, error))
    assert screen._preview_script() == ''
    assert 'preview unavailable' in screen._console.as_text()
    monkeypatch.setattr(screen.manager.profiles, 'get', functools.partial(_raise_owned_remote_error, error))
    assert screen._current_profile() is None
    with pytest.raises(RemoteExecutionError, match='Create an execution profile first'):
        screen._profile_transfer_document('connection')
    screen._read_cluster()
    assert 'Choose an SSH' in screen._status.text()
    screen._copy_allocation_shell()
    assert 'Choose a running allocation' in screen._status.text()
    assert screen.manager.jobs.list() == [] and runner.calls == []


def test_signin_copy_preserves_clipboard_on_validation_error(qtbot, qt_theme_applied, monkeypatch):
    import functools
    from PySide6.QtWidgets import QApplication
    from spacr.remote_execution import RemoteExecutionError

    dialog = ExecutionProfileDialog(profile=ExecutionProfile('cluster', 'ssh', host='fake-host', workdir='/work'))
    qtbot.addWidget(dialog)
    expected = ui._signin_command(dialog._connection_profile())
    dialog._copy_signin_command()
    assert QApplication.clipboard().text() == expected
    assert expected in dialog._error.text()
    monkeypatch.setattr(ui, '_signin_command', functools.partial(
        _raise_owned_remote_error, RemoteExecutionError('sign-in command unavailable')))
    dialog._copy_signin_command()
    assert dialog._error.text() == 'sign-in command unavailable'
    assert QApplication.clipboard().text() == expected


def test_failed_late_cluster_cache_does_not_replace_selected_profile_state(qtbot, qt_theme_applied, state, monkeypatch):
    import functools
    from spacr.remote_execution import RemoteExecutionError, _parse_cluster_probe

    screen = _screen(qtbot, state, Runner())
    original = screen._cluster
    profile_before = screen.manager.profiles.path.read_bytes()
    monkeypatch.setattr(ui, '_save_cluster_cache', functools.partial(
        _raise_owned_remote_error, RemoteExecutionError('cluster cache unavailable')))
    screen._cluster_read('another-profile', _parse_cluster_probe(PROBE))
    assert screen._cluster is original
    assert 'another-profile' in screen._status.text()
    assert 'cluster cache unavailable' in screen._console.as_text()
    assert screen.manager.profiles.path.read_bytes() == profile_before
    assert screen.manager.jobs.list() == []


def _dismiss_readonly_script_preview(dialog):
    from PySide6.QtWidgets import QDialogButtonBox

    assert dialog.text.isReadOnly() and 'spacr-run' in dialog.text.toPlainText()
    assert dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Close) is not None
    dialog.reject()
    return dialog.result()


def test_unknown_preset_and_unchecked_direct_run_preserve_other_validation(qtbot, qt_theme_applied):
    dialog = ExecutionProfileDialog(profile=ExecutionProfile('cluster', 'ssh', host='fake-host', workdir='/work'))
    qtbot.addWidget(dialog)
    original = dialog.profile()
    dialog._error.setText('An unrelated validation error')
    dialog._warn_direct_run(False)
    assert dialog._error.text() == 'An unrelated validation error'
    dialog._apply_preset(0)
    assert dialog.profile() == original


def test_missing_or_invalid_editor_state_refuses_operations_before_transport(qtbot, qt_theme_applied, state, monkeypatch):
    import functools
    from spacr.remote_execution import RemoteExecutionError

    runner = Runner()
    screen = _screen(qtbot, state, runner)
    before = screen.manager.profiles.path.read_bytes()
    screen._profile.clear()
    assert screen._store_editors('resources') is None
    screen._save_resources()
    screen._request_allocation()
    assert screen._preview_script() == ''
    assert screen.manager.profiles.path.read_bytes() == before
    screen._reload_profiles('fakelakes')
    screen.configure_submission('mask', {'src': '/data/input'})
    previous_env = screen._env_edit.toPlainText()
    screen._env_edit.setPlainText('\0invalid environment')
    screen.submit()
    assert 'must be text' in screen._status.text()
    assert screen.manager.jobs.list() == []
    screen._env_edit.setPlainText(previous_env)
    screen._preview = True
    monkeypatch.setattr(ui, '_preview_job_script', functools.partial(
        _raise_owned_remote_error, RemoteExecutionError('owned preview refusal')))
    screen.submit()
    assert 'owned preview refusal' in screen._status.text()
    assert screen.manager.jobs.list() == [] and runner.calls == []


def test_preview_button_displays_readonly_script_without_submitting(qtbot, qt_theme_applied, state, monkeypatch):
    screen = _screen(qtbot, state, Runner(), preview=True)
    monkeypatch.setattr(ui._ScriptPreviewDialog, 'exec', _dismiss_readonly_script_preview)
    text = screen._preview_script()
    assert 'spacr-run' in text and '{run_command}' not in text
    previews = screen.findChildren(ui._ScriptPreviewDialog)
    assert len(previews) == 1 and previews[0].text.toPlainText() == text
    assert previews[0].result() == QDialog.Rejected
    assert screen.manager.jobs.list() == [] and screen.manager.runner.calls == []


def test_cluster_read_warns_for_actual_current_resource_request(qtbot, qt_theme_applied, state):
    from spacr.remote_execution import _parse_cluster_probe

    screen = _screen(qtbot, state, Runner())
    assert screen._check_resources() == [] and 'Read cluster first' in screen._status.text()
    cluster = _parse_cluster_probe(PROBE)
    screen._res_cpus.setValue(37)
    screen._cluster_read('fakelakes', cluster)
    assert screen._cluster == cluster
    problems = screen._check_resources()
    assert any('37 CPUs' in problem and '36' in problem for problem in problems)
    assert '37 CPUs' in screen._console.as_text()
    assert screen.manager.jobs.list() == [] and screen.manager.runner.calls == []


def test_selected_active_job_arms_follow_and_keeps_failure_details_private(qtbot, qt_theme_applied, state, monkeypatch):
    from dataclasses import replace
    from spacr.remote_execution import RemoteJob

    screen = _screen(qtbot, state, Runner())
    profile = screen._current_profile().to_dict()
    profile['ssh_command'] = 'private-command-not-for-display'
    job = RemoteJob('abcdef123456', 'mask', 'fakelakes', 'ssh', status='running',
                    profile=profile, log_reference='/work/job.log', error='known failure',
                    created_utc='not-a-time', updated_utc='also-not-a-time')
    screen.manager.jobs.save(job)
    screen._render_jobs([job], select=job.job_id)
    assert screen._follow[job.job_id] == 0
    texts = [screen._table.item(0, column).text() for column in range(screen._table.columnCount())]
    assert 'not-a-time' in texts
    summary = screen._job_summary(job)
    assert 'log /work/job.log' in summary and 'known failure' in summary and 'remote folder' not in summary
    errors = []
    monkeypatch.setattr(screen._console, 'append_error', errors.append)
    screen._last_status[job.job_id] = 'running'
    screen._report_transition(replace(job, status='failed', error=''))
    assert len(errors) == 1 and 'failed with exit code ?' in errors[0]
    assert '<configured command>' in errors[0] and 'private-command-not-for-display' not in errors[0]
    without_log = replace(job, job_id='fedcba654321', status='queued', error='', log_reference='')
    screen.manager.jobs.save(without_log)
    screen._submitted(without_log)
    assert without_log.job_id not in screen._follow
    assert 'log ' not in screen._job_summary(without_log)
