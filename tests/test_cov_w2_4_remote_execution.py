"""Remote execution — the refusals, the outages, and the real subprocess.

``tests/test_remote_execution.py`` covers the happy submissions. This file
goes after everything that goes wrong, because every path here ends in a
message a user reads on a screen with no traceback under it:

* :func:`_run_command` against the ACTUAL subprocess module -- a missing
  program, a timeout and a file that is not executable, so the three error
  sentences are the ones a real ``execve`` produces;
* the profile validator's remaining refusals, each asserted to name the
  field to fix;
* the state store's failure modes: unreadable JSON, an unwritable
  directory, and an interpreter with no ``fcntl``;
* the four backends' error branches -- a scheduler that returns an unsafe
  job ID, a log that does not exist yet, an exit code that will not parse;
* :class:`RemoteJobManager`'s rule that a transient outage must NOT turn a
  still-running remote job into a permanent failure.

Nothing here opens a network connection: the command runner is the seam the
module was built with, and it is injected.
"""
from __future__ import annotations

import builtins
import os
import sys
from collections import deque
from pathlib import Path

import pytest

from spacr import remote_execution as rx
from spacr.remote_execution import (
    CommandResult,
    ExecutionProfile,
    JobStore,
    ProfileStore,
    RemoteExecutionError,
    RemoteJob,
    RemoteJobManager,
)


class QueueRunner:
    """Deterministic command runner that records every argument vector."""

    def __init__(self, *results):
        self.results = deque(results)
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), dict(kwargs)))
        if not self.results:
            raise AssertionError(f"unexpected command: {argv}")
        result = self.results.popleft()
        if isinstance(result, Exception):
            raise result
        return result


# ---------------------------------------------------------------------------
# _run_command, against the real subprocess module
# ---------------------------------------------------------------------------

def test_a_real_command_returns_its_code_and_both_streams():
    result = rx._run_command(
        [sys.executable, "-c",
         "import sys; sys.stdout.write('out'); sys.stderr.write('err'); "
         "sys.exit(3)"],
        timeout=30.0)

    assert isinstance(result, CommandResult)
    assert result.returncode == 3
    assert result.stdout == "out"
    assert result.stderr == "err"


def test_stdin_reaches_the_command():
    result = rx._run_command(
        [sys.executable, "-c", "import sys; print(sys.stdin.read().strip())"],
        input_text="the settings payload", timeout=30.0)
    assert result.stdout.strip() == "the settings payload"


def test_a_missing_program_says_which_one_and_what_to_do():
    with pytest.raises(RemoteExecutionError) as caught:
        rx._run_command(["spacr-no-such-program-anywhere"])
    assert "spacr-no-such-program-anywhere" in str(caught.value)
    assert "execution profile" in str(caught.value)


def test_a_command_that_hangs_is_timed_out_and_named():
    with pytest.raises(RemoteExecutionError) as caught:
        rx._run_command(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=1.0)
    assert "Command timed out after 1s" in str(caught.value)
    assert sys.executable in str(caught.value)


@pytest.mark.xfail(strict=True, reason="the message names the requested "
                                       "timeout, not the one that was waited")
def test_a_timeout_message_names_the_time_that_was_actually_waited():
    """``timeout`` is floored at one second before the call, but the sentence
    is formatted from the raw argument, so a 0.1 s request that waited a
    whole second reports 0.1 s -- a number the user cannot reconcile with
    the wall clock."""
    with pytest.raises(RemoteExecutionError) as caught:
        rx._run_command(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=0.1)
    assert "timed out after 1s" in str(caught.value)


def test_a_file_that_cannot_be_executed_is_reported_not_raised(tmp_path):
    """A profile pointing at a data file is an ordinary configuration slip."""
    not_a_program = tmp_path / "settings.json"
    not_a_program.write_text("{}")

    with pytest.raises(RemoteExecutionError) as caught:
        rx._run_command([str(not_a_program)])

    assert str(not_a_program) in str(caught.value)
    assert "PermissionError" in str(caught.value)


# ---------------------------------------------------------------------------
# state_directory
# ---------------------------------------------------------------------------

def test_the_state_directory_follows_the_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SPACR_REMOTE_STATE_DIR", str(tmp_path / "portable"))
    assert rx.state_directory() == tmp_path / "portable"


def test_the_state_directory_follows_xdg_when_there_is_no_override(
        monkeypatch, tmp_path):
    monkeypatch.delenv("SPACR_REMOTE_STATE_DIR", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assert rx.state_directory() == tmp_path / "state" / "spacr" / "remote"


def test_the_state_directory_falls_back_to_the_home_convention(monkeypatch,
                                                               tmp_path):
    monkeypatch.delenv("SPACR_REMOTE_STATE_DIR", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    assert rx.state_directory() == (
        tmp_path / ".local" / "state" / "spacr" / "remote")


# ---------------------------------------------------------------------------
# Text and path validation
# ---------------------------------------------------------------------------

def test_a_required_field_left_blank_names_itself():
    with pytest.raises(RemoteExecutionError, match="SSH host is required."):
        rx._safe_text("", "SSH host")


@pytest.mark.parametrize("hostile", ["a\nb", "a\rb", "a\x00b"])
def test_a_newline_or_nul_is_refused_because_it_would_split_a_command(
        hostile):
    with pytest.raises(RemoteExecutionError, match="newlines or NUL"):
        rx._safe_text(hostile, "SSH host")


def test_a_blank_field_is_allowed_when_the_caller_says_so():
    assert rx._safe_text("", "Log command", allow_empty=True) == ""


def test_a_trailing_slash_is_trimmed_but_the_root_survives():
    assert rx._path_text("/work/dir/", "Remote work directory") == "/work/dir"
    assert rx._path_text("/", "Remote work directory") == "/"


# ---------------------------------------------------------------------------
# ExecutionProfile.validate
# ---------------------------------------------------------------------------

def test_an_unknown_backend_lists_the_ones_there_are():
    with pytest.raises(RemoteExecutionError) as caught:
        ExecutionProfile("x", "carrier-pigeon", host="h",
                         workdir="/w").validate()
    assert "carrier-pigeon" in str(caught.value)
    assert "ssh, slurm, command" in str(caught.value)


@pytest.mark.parametrize("seconds", [1, 0, 3601, -5])
def test_a_poll_interval_outside_the_range_is_refused(seconds):
    with pytest.raises(RemoteExecutionError, match="2–3600"):
        ExecutionProfile("x", "ssh", host="h", workdir="/w",
                         poll_seconds=seconds).validate()


def test_a_cancel_command_that_names_no_job_cannot_cancel_one():
    with pytest.raises(RemoteExecutionError, match="{external_id}"):
        ExecutionProfile(
            "x", "command",
            submit_command="cloud submit {settings}",
            status_command="cloud status {external_id}",
            cancel_command="cloud cancel-everything",
        ).validate()


def test_a_job_id_pattern_that_is_not_a_regex_is_refused():
    with pytest.raises(RemoteExecutionError,
                       match="Job-ID regular expression is invalid"):
        ExecutionProfile(
            "x", "command",
            submit_command="cloud submit {settings}",
            status_command="cloud status {external_id}",
            cancel_command="cloud cancel {external_id}",
            job_id_pattern="job-(",
        ).validate()


def test_a_complete_command_profile_validates():
    profile = ExecutionProfile(
        "cloud", "command",
        submit_command="cloud submit {settings} --module {module}",
        status_command="cloud status {external_id}",
        cancel_command="cloud cancel {external_id}",
        log_command="cloud logs {external_id}",
        job_id_pattern=r"job-(\d+)",
    ).validate()
    assert profile.name == "cloud"


# ---------------------------------------------------------------------------
# RemoteJob.from_dict
# ---------------------------------------------------------------------------

def test_a_stored_status_this_version_does_not_know_becomes_unknown():
    """A newer spaCR's state name must not be shown as if it were current."""
    job = RemoteJob.from_dict({"job_id": "a", "module": "mask",
                               "profile_name": "p", "backend": "ssh",
                               "status": "hibernating",
                               "a_field_from_the_future": 1})
    assert job.status == "unknown"


# ---------------------------------------------------------------------------
# The state files
# ---------------------------------------------------------------------------

def test_a_lock_that_cannot_be_opened_names_the_path(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")

    with pytest.raises(RemoteExecutionError, match="Could not open state lock"):
        with rx._file_lock(blocker / "sub" / "jobs.json.lock"):
            pass


def test_an_interpreter_with_no_fcntl_still_takes_the_lock(tmp_path,
                                                           monkeypatch):
    """Windows has no ``fcntl``; the store must still work there."""
    real_import = builtins.__import__

    def without_fcntl(name, *args, **kwargs):
        if name == "fcntl":
            raise ImportError("no fcntl on this platform")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_fcntl)

    store = ProfileStore(tmp_path / "profiles.json")
    store.save(ExecutionProfile("w", "ssh", host="h", workdir="/w"))
    assert [p.name for p in store.list()] == ["w"]


def test_a_state_file_that_is_not_json_names_it(tmp_path):
    path = tmp_path / "jobs.json"
    path.write_text("{ this was edited by hand")

    with pytest.raises(RemoteExecutionError) as caught:
        JobStore(path).list()
    assert str(path) in str(caught.value)
    assert "JSONDecodeError" in str(caught.value)


def test_a_state_file_that_cannot_be_written_names_it(tmp_path):
    unwritable = tmp_path / "read_only"
    unwritable.mkdir()
    os.chmod(unwritable, 0o500)
    try:
        with pytest.raises(RemoteExecutionError, match="Could not write"):
            rx._write_json_atomic(unwritable / "jobs.json", {"jobs": []})
    finally:
        os.chmod(unwritable, 0o700)


def test_a_failed_replace_leaves_no_temporary_file_behind(tmp_path,
                                                          monkeypatch):
    """A half-written state file is worse than no write at all."""
    def refuse(_source, _target):
        raise OSError("the filesystem went away")

    monkeypatch.setattr(rx.os, "replace", refuse)
    target = tmp_path / "jobs.json"

    with pytest.raises(RemoteExecutionError, match="Could not write"):
        rx._write_json_atomic(target, {"jobs": []})

    assert list(tmp_path.iterdir()) == []


def test_a_profile_that_does_not_exist_says_so_by_name(tmp_path):
    with pytest.raises(RemoteExecutionError,
                       match="'workstation' does not exist"):
        ProfileStore(tmp_path / "profiles.json").get("workstation")


def test_a_job_that_does_not_exist_says_so_by_id(tmp_path):
    with pytest.raises(RemoteExecutionError, match="'abc' was not found"):
        JobStore(tmp_path / "jobs.json").get("abc")


def test_deleting_a_profile_reports_whether_there_was_one(tmp_path):
    store = ProfileStore(tmp_path / "profiles.json")
    store.save(ExecutionProfile("w", "ssh", host="h", workdir="/w"))
    assert store.delete("W") is True
    assert store.delete("W") is False


# ---------------------------------------------------------------------------
# Path mapping
# ---------------------------------------------------------------------------

def test_a_path_on_another_volume_is_left_alone(monkeypatch):
    """Windows raises on two paths with different drives; POSIX does not."""
    def different_drives(_paths):
        raise ValueError("Paths don't have the same drive")

    monkeypatch.setattr(rx.os.path, "commonpath", different_drives)
    assert rx._map_path_string("/mnt/lab/plate", "/mnt/lab",
                               "/cluster") == "/mnt/lab/plate"


def test_the_root_itself_maps_to_the_remote_root():
    assert rx._map_path_string("/mnt/lab", "/mnt/lab",
                               "/cluster/lab") == "/cluster/lab"


# ---------------------------------------------------------------------------
# Command templates
# ---------------------------------------------------------------------------

def test_a_template_with_an_unbalanced_quote_is_named():
    with pytest.raises(RemoteExecutionError,
                       match="Submit command cannot be parsed"):
        rx._split_template('cloud submit "unterminated', "Submit command")


def test_an_empty_template_is_named():
    with pytest.raises(RemoteExecutionError, match="Submit command is empty."):
        rx._split_template("   ", "Submit command")


def test_a_template_that_starts_with_a_flag_has_no_program():
    with pytest.raises(RemoteExecutionError, match="must begin with a program"):
        rx._split_template("--flag value", "Submit command")


def test_options_may_start_with_a_flag_because_they_are_not_a_program():
    assert rx._split_template("--partition gpu", "Slurm options",
                              require_program=False) == ["--partition", "gpu"]


def test_an_unknown_placeholder_is_named_in_braces():
    with pytest.raises(RemoteExecutionError,
                       match=r"unknown placeholder \{queue\}"):
        rx._render_template("cloud submit {queue}", {"job_id": "a"},
                            "Submit command")


def test_broken_placeholder_syntax_is_reported_not_raised_raw():
    with pytest.raises(RemoteExecutionError,
                       match="invalid placeholder syntax"):
        rx._render_template("cloud submit {", {"job_id": "a"},
                            "Submit command")


def test_a_placeholder_is_substituted_inside_one_argument():
    """Shell operators in a value have no special meaning as one argv item."""
    argv = rx._render_template(
        "cloud submit --id {external_id}",
        {"external_id": "a b; rm -rf /"}, "Submit command")
    assert argv == ["cloud", "submit", "--id", "a b; rm -rf /"]


# ---------------------------------------------------------------------------
# _require_ok and _safe_external_id
# ---------------------------------------------------------------------------

def test_a_failed_command_quotes_only_the_last_1200_characters():
    """A megabyte of stderr in a dialog is a dialog nobody can close."""
    with pytest.raises(RemoteExecutionError) as caught:
        rx._require_ok(CommandResult(1, "", "x" * 5000), "SSH submission")
    message = str(caught.value)
    assert "exit code 1" in message
    assert message.split(": ", 1)[1] == "x" * 1200


def test_a_failure_with_no_output_still_says_the_code():
    with pytest.raises(RemoteExecutionError,
                       match=r"SSH submission failed with exit code 2\."):
        rx._require_ok(CommandResult(2, "", ""), "SSH submission")


def test_a_scheduler_returning_a_hostile_cluster_name_is_refused():
    with pytest.raises(RemoteExecutionError, match="unsafe cluster name"):
        rx._safe_external_id("123;a cluster; rm -rf /",
                             allow_slurm_cluster_suffix=True)


def test_a_slurm_cluster_suffix_is_stripped_from_the_job_id():
    assert rx._safe_external_id("4711;cluster-a",
                                allow_slurm_cluster_suffix=True) == "4711"


@pytest.mark.parametrize("hostile", ["", "   ", "-oProxyCommand=x", "a b"])
def test_a_scheduler_returning_an_unsafe_job_id_is_refused(hostile):
    with pytest.raises(RemoteExecutionError, match="unsafe or empty job ID"):
        rx._safe_external_id(hostile)


# ---------------------------------------------------------------------------
# _normalise_state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("EXIT:0", ("success", 0)),
    ("EXIT:137", ("failed", 137)),
    ("EXIT:not-a-number", ("failed", None)),
    ("COMPLETED", ("success", 0)),
    ("OUT_OF_MEMORY", ("failed", None)),
    ("CANCELLED+", ("cancelled", None)),
    ("PENDING", ("pending", None)),
    ("RUNNING\nRUNNING", ("running", None)),
    ("something the cloud invented", ("unknown", None)),
    ("", ("unknown", None)),
])
def test_every_scheduler_vocabulary_maps_onto_the_compact_model(text,
                                                                expected):
    assert rx._normalise_state(text) == expected


# ---------------------------------------------------------------------------
# The abstract backend
# ---------------------------------------------------------------------------

def test_the_base_backend_declares_its_three_operations_unimplemented():
    backend = rx._Backend(QueueRunner())
    profile = ExecutionProfile("w", "ssh", host="h", workdir="/w")
    job = RemoteJob("id", "mask", "w", "ssh")

    for call in (lambda: backend.submit(profile, job, "{}"),
                 lambda: backend.refresh(profile, job),
                 lambda: backend.cancel(profile, job)):
        with pytest.raises(NotImplementedError):
            call()
    # Logs are not abstract: the stored tail is the honest default.
    job.log_tail = "last known output"
    assert backend.logs(profile, job, 10) == "last known output"


def test_an_unsupported_backend_is_named():
    profile = ExecutionProfile("w", "ssh", host="h", workdir="/w")
    broken = ExecutionProfile.__new__(ExecutionProfile)
    object.__setattr__(broken, "backend", "telepathy")
    with pytest.raises(RemoteExecutionError, match="telepathy"):
        rx._backend(broken, QueueRunner())
    assert isinstance(rx._backend(profile, QueueRunner()), rx._SSHBackend)


# ---------------------------------------------------------------------------
# Backend error branches
# ---------------------------------------------------------------------------

@pytest.fixture
def ssh_profile():
    return ExecutionProfile("w", "ssh", host="lab-ws", workdir="/work")


@pytest.fixture
def ssh_job():
    return RemoteJob("job1234abcd", "mask", "w", "ssh", external_id="4242",
                     remote_job_dir="/work/spacr-jobs/job1234abcd",
                     log_reference="/work/spacr-jobs/job1234abcd/job.log")


def test_a_workstation_returning_a_non_numeric_pid_is_refused(ssh_profile):
    runner = QueueRunner(CommandResult(0, "", ""),
                         CommandResult(0, "not_a_pid\n", ""))
    job = RemoteJob("job1234abcd", "mask", "w", "ssh")

    with pytest.raises(RemoteExecutionError, match="non-numeric process ID"):
        rx._SSHBackend(runner).submit(ssh_profile, job, "{}")


def test_cancelling_a_process_that_already_exited_is_not_an_error(
        ssh_profile, ssh_job):
    runner = QueueRunner(CommandResult(1, "", "kill: No such process"))
    rx._SSHBackend(runner).cancel(ssh_profile, ssh_job)
    assert ssh_job.status == "cancelled"


def test_a_cancellation_that_really_failed_is_raised(ssh_profile, ssh_job):
    runner = QueueRunner(CommandResult(255, "", "Permission denied"))
    with pytest.raises(RemoteExecutionError, match="SSH cancellation failed"):
        rx._SSHBackend(runner).cancel(ssh_profile, ssh_job)


def test_the_ssh_log_tail_is_bounded_at_both_ends(ssh_profile, ssh_job):
    """``tail -n 0`` prints nothing and ``-n 1000000000`` is the whole file."""
    runner = QueueRunner(CommandResult(0, "line\n", ""))
    assert rx._SSHBackend(runner).logs(ssh_profile, ssh_job, 0) == "line\n"
    # The host makes this an ssh invocation, so the tail command is the
    # single quoted argument at the end.
    assert runner.calls[0][0][:2] == ["ssh", "lab-ws"]
    assert "tail -n 1 --" in runner.calls[0][0][2]

    runner = QueueRunner(CommandResult(0, "line\n", ""))
    rx._SSHBackend(runner).logs(ssh_profile, ssh_job, 10 ** 9)
    assert "tail -n 10000 --" in runner.calls[0][0][2]


def test_a_profile_with_no_host_runs_the_command_locally(ssh_job):
    """A shared filesystem needs no ssh hop, and the argv says so."""
    local = ExecutionProfile("local", "slurm", workdir="/work")
    runner = QueueRunner(CommandResult(0, "line\n", ""))

    rx._SSHBackend(runner).logs(local, ssh_job, 5)

    argv = runner.calls[0][0]
    assert argv[0] == "tail"
    assert argv[argv.index("-n") + 1] == "5"


@pytest.fixture
def slurm_profile():
    return ExecutionProfile("hpc", "slurm", host="login", workdir="/work")


def test_an_exit_code_slurm_cannot_spell_leaves_the_state_alone(
        slurm_profile):
    """``sacct`` printing ``COMPLETED|weird`` must not lose the state."""
    job = RemoteJob("j", "mask", "hpc", "slurm", external_id="4711")
    runner = QueueRunner(CommandResult(1, "", "not in queue"),
                         CommandResult(0, "COMPLETED|weird:0\n", ""))

    rx._SlurmBackend(runner).refresh(slurm_profile, job)

    assert job.status == "success"
    assert job.exit_code == 0


def test_a_slurm_log_that_does_not_exist_yet_says_so(slurm_profile):
    job = RemoteJob("j", "mask", "hpc", "slurm", external_id="4711",
                    log_reference="/work/job.log")
    runner = QueueRunner(CommandResult(1, "", "tail: No such file"))

    assert rx._SlurmBackend(runner).logs(slurm_profile, job, 50) == (
        "The Slurm log has not been created yet.")


def test_a_slurm_log_that_failed_for_another_reason_is_raised(slurm_profile):
    job = RemoteJob("j", "mask", "hpc", "slurm", external_id="4711",
                    log_reference="/work/job.log")
    runner = QueueRunner(CommandResult(1, "", "Permission denied"))

    with pytest.raises(RemoteExecutionError, match="Slurm log retrieval"):
        rx._SlurmBackend(runner).logs(slurm_profile, job, 50)


def test_a_slurm_log_is_returned_when_it_is_there(slurm_profile):
    job = RemoteJob("j", "mask", "hpc", "slurm", external_id="4711",
                    log_reference="/work/job.log")
    runner = QueueRunner(CommandResult(0, "the tail\n", ""))
    assert rx._SlurmBackend(runner).logs(slurm_profile, job, 50) == "the tail\n"


@pytest.fixture
def cloud_profile():
    return ExecutionProfile(
        "cloud", "command",
        submit_command="cloud submit {settings}",
        status_command="cloud status {external_id}",
        cancel_command="cloud cancel {external_id}",
    )


def test_a_command_profile_with_no_log_command_says_how_to_add_one(
        cloud_profile):
    job = RemoteJob("j", "mask", "cloud", "command", external_id="job-7")
    text = rx._CommandBackend(QueueRunner()).logs(cloud_profile, job, 50)
    assert "no log command" in text
    assert "{external_id}" in text


def test_a_command_profile_log_is_fetched_when_there_is_one():
    profile = ExecutionProfile(
        "cloud", "command",
        submit_command="cloud submit {settings}",
        status_command="cloud status {external_id}",
        cancel_command="cloud cancel {external_id}",
        log_command="cloud logs {external_id}",
    ).validate()
    job = RemoteJob("j", "mask", "cloud", "command", external_id="job-7")
    runner = QueueRunner(CommandResult(0, "cloud output\n", ""))

    assert rx._CommandBackend(runner).logs(profile, job, 50) == "cloud output\n"
    assert runner.calls[0][0] == ["cloud", "logs", "job-7"]


# ---------------------------------------------------------------------------
# RemoteJobManager
# ---------------------------------------------------------------------------

def _manager(tmp_path, profile, runner):
    profiles = ProfileStore(tmp_path / "profiles.json")
    profiles.save(profile)
    jobs = JobStore(tmp_path / "jobs.json")
    return RemoteJobManager(profiles, jobs, runner)


def test_a_module_that_is_not_headless_is_refused_before_anything_is_written(
        tmp_path, ssh_profile):
    manager = _manager(tmp_path, ssh_profile, QueueRunner())

    with pytest.raises(RemoteExecutionError, match="spacr-run --list"):
        manager.submit("not_a_module", {}, "w")

    assert manager.jobs.list() == []


def test_a_local_job_record_that_cannot_be_written_names_the_directory(
        tmp_path, ssh_profile, monkeypatch):
    manager = _manager(tmp_path, ssh_profile, QueueRunner())
    real_mkdir = Path.mkdir

    def refuse(self, *args, **kwargs):
        # Only the per-job record directory: the profile store's own lock
        # has to keep working or the failure lands in the wrong place.
        if self.parent.name == "jobs":
            raise OSError("no space left on device")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", refuse)

    with pytest.raises(RemoteExecutionError) as caught:
        manager.submit("mask", {"src": "/data"}, "w")
    assert "Could not prepare local job record" in str(caught.value)
    assert "OSError" in str(caught.value)


def test_a_backend_crash_that_is_not_ours_is_wrapped_and_recorded(
        tmp_path, ssh_profile):
    """The job must survive as ``failed`` with the reason, not vanish."""
    manager = _manager(tmp_path, ssh_profile,
                       QueueRunner(ZeroDivisionError("a bug in the runner")))

    with pytest.raises(RemoteExecutionError, match="ZeroDivisionError"):
        manager.submit("mask", {"src": "/data"}, "w")

    job = manager.jobs.list()[0]
    assert job.status == "failed"
    assert "a bug in the runner" in job.error


def _finished_job(manager, status="running"):
    job = RemoteJob("job1234abcd", "mask", "w", "ssh", status=status,
                    external_id="4242",
                    remote_job_dir="/work/spacr-jobs/job1234abcd",
                    log_reference="/work/spacr-jobs/job1234abcd/job.log",
                    profile=ExecutionProfile("w", "ssh", host="lab-ws",
                                             workdir="/work").to_dict())
    manager.jobs.save(job)
    return job


def test_a_finished_job_is_not_polled_again(tmp_path, ssh_profile):
    runner = QueueRunner()
    manager = _manager(tmp_path, ssh_profile, runner)
    _finished_job(manager, status="success")

    job = manager.refresh("job1234abcd")

    assert job.status == "success"
    assert runner.calls == []


def test_a_log_that_is_not_there_yet_does_not_hide_the_status(tmp_path,
                                                              ssh_profile):
    runner = QueueRunner(CommandResult(0, "RUNNING\n", ""),
                         CommandResult(1, "", "tail: cannot open"))
    manager = _manager(tmp_path, ssh_profile, runner)
    _finished_job(manager)

    job = manager.refresh("job1234abcd")

    assert job.status == "running"
    assert job.log_tail.startswith("Log not available yet:")
    assert job.error == ""


def test_a_transient_outage_does_not_fail_a_still_running_job(tmp_path,
                                                              ssh_profile):
    """The rule this module is arranged around."""
    runner = QueueRunner(CommandResult(255, "", "ssh: connect: timed out"))
    manager = _manager(tmp_path, ssh_profile, runner)
    _finished_job(manager)

    job = manager.refresh("job1234abcd", include_logs=False)

    assert job.status == "running"
    assert "SSH status check failed" in job.error


def test_a_finished_job_is_not_cancelled_again(tmp_path, ssh_profile):
    runner = QueueRunner()
    manager = _manager(tmp_path, ssh_profile, runner)
    _finished_job(manager, status="cancelled")

    assert manager.cancel("job1234abcd").status == "cancelled"
    assert runner.calls == []


def test_a_cancellation_the_backend_refuses_is_raised_and_recorded(
        tmp_path, ssh_profile):
    runner = QueueRunner(CommandResult(255, "", "Permission denied"))
    manager = _manager(tmp_path, ssh_profile, runner)
    _finished_job(manager)

    with pytest.raises(RemoteExecutionError, match="SSH cancellation"):
        manager.cancel("job1234abcd")

    job = manager.jobs.get("job1234abcd")
    assert job.status == "running"
    assert "Permission denied" in job.error


def test_a_cancellation_crash_that_is_not_ours_is_wrapped(tmp_path,
                                                          ssh_profile):
    runner = QueueRunner(TypeError("the runner was handed the wrong thing"))
    manager = _manager(tmp_path, ssh_profile, runner)
    _finished_job(manager)

    with pytest.raises(RemoteExecutionError, match="TypeError"):
        manager.cancel("job1234abcd")

    assert "TypeError" in manager.jobs.get("job1234abcd").error


def test_a_temporary_that_cannot_be_removed_does_not_mask_the_write_error(
        tmp_path, monkeypatch):
    """The user needs to hear why the write failed, not why the cleanup did."""
    monkeypatch.setattr(rx.os, "replace",
                        lambda *_a: (_ for _ in ()).throw(
                            OSError("the filesystem went away")))
    monkeypatch.setattr(rx.os, "unlink",
                        lambda *_a: (_ for _ in ()).throw(
                            OSError("and the temporary is stuck too")))

    with pytest.raises(RemoteExecutionError) as caught:
        rx._write_json_atomic(tmp_path / "jobs.json", {"jobs": []})

    assert "the filesystem went away" in str(caught.value)
    assert "stuck too" not in str(caught.value)


def test_refresh_all_polls_only_the_jobs_that_can_still_change(tmp_path,
                                                              ssh_profile):
    runner = QueueRunner(CommandResult(0, "EXIT:0\n", ""))
    manager = _manager(tmp_path, ssh_profile, runner)
    profile = ssh_profile.to_dict()
    for job_id, status in (("aaaa1111", "running"), ("bbbb2222", "success"),
                           ("cccc3333", "cancelled")):
        manager.jobs.save(RemoteJob(job_id, "mask", "w", "ssh", status=status,
                                    external_id="42",
                                    remote_job_dir="/work/j",
                                    profile=profile))

    jobs = manager.refresh_all()

    assert len(runner.calls) == 1
    assert {job.job_id: job.status for job in jobs}["aaaa1111"] == "success"


def test_the_log_tail_is_kept_beside_the_job(tmp_path, ssh_profile):
    runner = QueueRunner(CommandResult(0, "the last hundred lines\n", ""))
    manager = _manager(tmp_path, ssh_profile, runner)
    _finished_job(manager)

    text = manager.logs("job1234abcd", lines=100)

    assert text == "the last hundred lines\n"
    assert manager.jobs.get("job1234abcd").log_tail == text


def test_a_successful_cancellation_clears_the_previous_error(tmp_path,
                                                             ssh_profile):
    manager = _manager(tmp_path, ssh_profile,
                       QueueRunner(CommandResult(0, "", "")))
    job = _finished_job(manager)
    job.error = "an earlier poll timed out"
    manager.jobs.save(job)

    cancelled = manager.cancel("job1234abcd")

    assert cancelled.status == "cancelled"
    assert cancelled.error == ""
