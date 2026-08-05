"""Backend, persistence and CLI tests for distributed spaCR execution."""
from __future__ import annotations

import json
from collections import deque

import pytest

from spacr.remote_execution import (
    CommandResult,
    ExecutionProfile,
    JobStore,
    ProfileStore,
    RemoteExecutionError,
    RemoteJob,
    RemoteJobManager,
    map_settings_paths,
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


def _manager(tmp_path, profile, runner):
    profiles = ProfileStore(tmp_path / "profiles.json")
    profiles.save(profile)
    jobs = JobStore(tmp_path / "jobs.json")
    return RemoteJobManager(profiles, jobs, runner)


def test_map_settings_paths_recurses_without_rewriting_outside_root():
    settings = {
        "src": "/mnt/lab/exp/plate",
        "models": ["/mnt/lab/models/a", "/opt/model"],
        "nested": {"paths": ("/mnt/lab", "ordinary text")},
        "/mnt/lab/key-is-not-a-path": "/mnt/lab/value",
    }
    mapped = map_settings_paths(settings, "/mnt/lab", "/cluster/lab")
    assert mapped["src"] == "/cluster/lab/exp/plate"
    assert mapped["models"] == ["/cluster/lab/models/a", "/opt/model"]
    assert mapped["nested"]["paths"] == ["/cluster/lab", "ordinary text"]
    assert "/mnt/lab/key-is-not-a-path" in mapped
    assert mapped["/mnt/lab/key-is-not-a-path"] == "/cluster/lab/value"


@pytest.mark.parametrize(
    "profile,message",
    [
        (ExecutionProfile("bad/name", "ssh", host="h", workdir="/w"),
         "Profile name"),
        (ExecutionProfile("x", "ssh", workdir="/w"), "needs a host"),
        (ExecutionProfile("x", "ssh", host="-oProxy=x", workdir="/w"),
         "unsupported"),
        (ExecutionProfile("x", "slurm", workdir="relative"),
         "absolute POSIX"),
        (ExecutionProfile("x", "slurm", workdir="/w",
                          local_root="/a"), "both local and remote"),
        (ExecutionProfile("x", "slurm", workdir="/w",
                          local_root="relative", remote_root="/remote"),
         "Local dataset root must be absolute"),
        (ExecutionProfile(
            "x", "command",
            submit_command="cloud submit",
            status_command="cloud status {external_id}",
            cancel_command="cloud cancel {external_id}",
        ), "{settings}"),
        (ExecutionProfile(
            "x", "command",
            submit_command="cloud submit {settings}",
            status_command="cloud status",
            cancel_command="cloud cancel {external_id}",
        ), "{external_id}"),
    ],
)
def test_profile_validation_rejects_unsafe_or_incomplete_profiles(
    profile, message
):
    with pytest.raises(RemoteExecutionError, match=message):
        profile.validate()


def test_profile_and_job_stores_replace_atomically(tmp_path):
    profiles = ProfileStore(tmp_path / "profiles.json")
    profiles.save(ExecutionProfile("gpu", "ssh", host="a", workdir="/work"))
    profiles.save(ExecutionProfile("GPU", "ssh", host="b", workdir="/work"))
    assert len(profiles.list()) == 1
    assert profiles.get("gpu").host == "b"
    assert profiles.delete("GpU")
    assert profiles.list() == []
    assert not profiles.delete("missing")

    jobs = JobStore(tmp_path / "jobs.json")
    job = RemoteJob("abcdef", "mask", "gpu", "ssh")
    jobs.save(job)
    job.status = "running"
    jobs.save(job)
    assert len(jobs.list()) == 1
    assert jobs.get("abc").status == "running"
    jobs.save(RemoteJob("abc999", "measure", "gpu", "ssh"))
    with pytest.raises(RemoteExecutionError, match="ambiguous"):
        jobs.get("abc")


def test_ssh_submit_maps_settings_polls_exit_and_retrieves_log(tmp_path):
    profile = ExecutionProfile(
        "workstation", "ssh", host="scientist@gpu",
        workdir="/shared/work", local_root="/mnt/lab",
        remote_root="/shared/lab",
    )
    runner = QueueRunner(
        CommandResult(0),                  # upload
        CommandResult(0, "4242\n"),        # launch
        CommandResult(0, "EXIT:0\n"),      # poll
        CommandResult(0, "all done\n"),    # tail
    )
    manager = _manager(tmp_path, profile, runner)
    job = manager.submit(
        "mask", {"src": "/mnt/lab/plate-a", "random_state": 7},
        "workstation",
    )
    assert job.status == "running"
    assert job.external_id == "4242"
    assert job.remote_settings_path.endswith("/settings.json")
    payload = json.loads(
        runner.calls[0][1]["input_text"]
    )
    assert payload["src"] == "/shared/lab/plate-a"
    assert runner.calls[0][0][0:2] == ["ssh", "scientist@gpu"]
    # Fixed remote script values are quoted, and the command is not passed via
    # subprocess shell=True.
    assert "shell" not in runner.calls[1][1]

    refreshed = manager.refresh(job.job_id)
    assert refreshed.status == "success"
    assert refreshed.exit_code == 0
    assert refreshed.log_tail == "all done\n"


def test_ssh_cancel_uses_external_pid_and_marks_cancelled(tmp_path):
    profile = ExecutionProfile(
        "gpu", "ssh", host="gpu", workdir="/work"
    )
    runner = QueueRunner(
        CommandResult(0), CommandResult(0, "991\n"), CommandResult(0)
    )
    manager = _manager(tmp_path, profile, runner)
    job = manager.submit("measure", {"src": "/same/path"}, "gpu")
    cancelled = manager.cancel(job.job_id)
    assert cancelled.status == "cancelled"
    assert "kill -TERM 991" in runner.calls[-1][0][-1]


def test_slurm_submit_falls_back_to_accounting_after_queue(tmp_path):
    profile = ExecutionProfile(
        "cluster", "slurm", host="login", workdir="/project/lab",
        scheduler_options="--partition=gpu --gres=gpu:1",
    )
    runner = QueueRunner(
        CommandResult(0),                    # upload
        CommandResult(0, "12345;alpha\n"),   # sbatch
        CommandResult(0, ""),                # squeue: left queue
        CommandResult(0, "COMPLETED|0:0\n"), # sacct
    )
    manager = _manager(tmp_path, profile, runner)
    job = manager.submit("umap", {"src": "/project/lab/p"}, "cluster")
    assert job.status == "queued"
    assert job.external_id == "12345"
    sbatch = runner.calls[1]
    assert "sbatch --parsable" in sbatch[0][-1]
    assert "--partition=gpu" in sbatch[0][-1]
    assert "exec spacr-run umap --settings" in sbatch[1]["input_text"]

    refreshed = manager.refresh(job.job_id, include_logs=False)
    assert refreshed.status == "success"
    assert refreshed.exit_code == 0


def test_slurm_cancel_and_pending_queue_state(tmp_path):
    profile = ExecutionProfile(
        "local-slurm", "slurm", workdir="/project"
    )
    runner = QueueRunner(
        CommandResult(0), CommandResult(0, "88\n"),
        CommandResult(0, "PENDING\n"), CommandResult(0),
    )
    manager = _manager(tmp_path, profile, runner)
    job = manager.submit("mask", {"src": "/project/p"}, "local-slurm")
    assert manager.refresh(job.job_id, include_logs=False).status == "pending"
    assert manager.cancel(job.job_id).status == "cancelled"
    assert runner.calls[-1][0] == ["scancel", "88"]


def _command_profile():
    return ExecutionProfile(
        "cloud", "command",
        submit_command=(
            "cloud submit --module {module} --settings {settings}"
        ),
        status_command="cloud status {external_id}",
        cancel_command="cloud cancel {external_id}",
        log_command="cloud logs {external_id}",
        job_id_pattern=r'"jobId":\s*"(?P<id>[A-Za-z0-9-]+)"',
    )


def test_custom_cloud_backend_templates_are_argument_vectors(tmp_path):
    runner = QueueRunner(
        CommandResult(0, '{"jobId": "cloud-77"}\n'),
        CommandResult(0, "RUNNING\n"),
        CommandResult(0, "epoch 2\n"),
        CommandResult(0),
    )
    manager = _manager(tmp_path, _command_profile(), runner)
    job = manager.submit("classify", {"src": "/data/plate"}, "cloud")
    assert job.external_id == "cloud-77"
    submit_argv = runner.calls[0][0]
    assert submit_argv[:4] == ["cloud", "submit", "--module", "classify"]
    assert submit_argv[-1].endswith("/settings.json")
    assert "shell" not in runner.calls[0][1]
    assert manager.refresh(job.job_id, include_logs=False).status == "running"
    assert manager.logs(job.job_id) == "epoch 2\n"
    assert manager.cancel(job.job_id).status == "cancelled"


def test_transient_poll_failure_is_visible_but_not_a_false_job_failure(tmp_path):
    runner = QueueRunner(
        CommandResult(0, "cloud-9\n"),
        RemoteExecutionError("network unavailable"),
    )
    profile = ExecutionProfile(
        "cloud", "command",
        submit_command="cloud-submit {settings}",
        status_command="cloud-status {external_id}",
        cancel_command="cloud-cancel {external_id}",
    )
    manager = _manager(tmp_path, profile, runner)
    job = manager.submit("mask", {"src": "/data"}, "cloud")
    refreshed = manager.refresh(job.job_id, include_logs=False)
    assert refreshed.status == "queued"
    assert "network unavailable" in refreshed.error


def test_unsafe_scheduler_identifier_is_rejected_and_job_is_inspectable(tmp_path):
    runner = QueueRunner(CommandResult(0), CommandResult(0, "12; rm -rf /\n"))
    profile = ExecutionProfile("gpu", "ssh", host="gpu", workdir="/work")
    manager = _manager(tmp_path, profile, runner)
    with pytest.raises(RemoteExecutionError, match="unsafe"):
        manager.submit("mask", {"src": "/data"}, "gpu")
    [job] = manager.jobs.list()
    assert job.status == "failed"
    assert "unsafe" in job.error


@pytest.mark.parametrize("identifier", ["--help", "pid_12"])
def test_ssh_requires_a_numeric_non_option_process_id(
    tmp_path, identifier
):
    runner = QueueRunner(CommandResult(0), CommandResult(0, identifier + "\n"))
    profile = ExecutionProfile("gpu", "ssh", host="gpu", workdir="/work")
    manager = _manager(tmp_path, profile, runner)
    with pytest.raises(RemoteExecutionError):
        manager.submit("mask", {"src": "/data"}, "gpu")
    assert manager.jobs.list()[0].status == "failed"


def test_cli_profile_add_and_list_share_state(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SPACR_REMOTE_STATE_DIR", str(tmp_path))
    from spacr import cli_remote

    assert cli_remote.main([
        "profile", "add", "gpu", "--backend", "ssh",
        "--host", "gpu", "--workdir", "/work",
    ]) == 0
    assert cli_remote.main(["profile", "list", "--json"]) == 0
    output = capsys.readouterr().out
    assert '"name": "gpu"' in output
    assert '"backend": "ssh"' in output


def test_cli_unknown_profile_error_is_nonzero(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SPACR_REMOTE_STATE_DIR", str(tmp_path))
    from spacr import cli_remote

    assert cli_remote.main(["profile", "delete", "missing"]) == 2
    assert "does not exist" in capsys.readouterr().err
