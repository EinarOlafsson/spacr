"""How a distributed job's state is read from Slurm and shown by ``spacr-remote``.

``spacr.remote_execution`` keeps one :class:`RemoteJob` per submission and
refreshes it from the scheduler; ``spacr.cli_remote`` renders it as one
terminal row and, for ``watch``, reports a finished job's error and log. The
scheduler is replaced by a deterministic command runner and the job manager
by a namespace, so every argument vector and every printed line is exact.
"""
from __future__ import annotations

from types import SimpleNamespace

from spacr import cli_remote
from spacr import remote_execution as rx


class _QueueRunner:
    """Deterministic command runner; records every argument vector."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        if not self.results:
            raise AssertionError(f"unexpected command: {argv}")
        return self.results.pop(0)


SQUEUE = ["squeue", "-h", "-j", "4711", "-o", "%T"]
SACCT = ["sacct", "-n", "-X", "-j", "4711", "--format=State,ExitCode",
         "--parsable2"]
GONE = rx.CommandResult(1, "", "slurm_load_jobs error")


def test_slurm_refresh_reads_the_queue_then_the_accounting_record(subtests):
    """A queued job is read from squeue; a finished one from sacct, exit code only when printed."""
    cases = [
        ("still in the queue", [rx.CommandResult(0, "RUNNING\n", "")],
         "running", None, [SQUEUE]),
        ("cancelled before it ran: State with no ExitCode column",
         [GONE, rx.CommandResult(0, "CANCELLED\n", "")],
         "cancelled", None, [SQUEUE, SACCT]),
        ("failed with an exit code", [GONE, rx.CommandResult(0, "FAILED|2:0\n",
                                                             "")],
         "failed", 2, [SQUEUE, SACCT]),
        ("completed, blank lines skipped",
         [GONE, rx.CommandResult(0, "\nCOMPLETED|0:0\n", "")],
         "success", 0, [SQUEUE, SACCT]),
        ("an unparseable exit code keeps the state's own",
         [GONE, rx.CommandResult(0, "FAILED|x:0\n", "")],
         "failed", None, [SQUEUE, SACCT]),
        ("an exit column without a colon is not read",
         [GONE, rx.CommandResult(0, "COMPLETED|7\n", "")],
         "success", 0, [SQUEUE, SACCT]),
        ("through a login host, each command is one quoted ssh argument",
         [GONE, rx.CommandResult(0, "FAILED|2:0\n", "")], "failed", 2,
         [["ssh", "login", " ".join(SQUEUE)], ["ssh", "login", " ".join(SACCT)]]),
    ]
    for case_id, results, status, exit_code, calls in cases:
        with subtests.test(msg=case_id):
            host = "login" if calls[0][0] == "ssh" else ""
            profile = rx.ExecutionProfile("hpc", "slurm", host=host,
                                          workdir="/work")
            job = rx.RemoteJob("j", "mask", "hpc", "slurm", external_id="4711")
            runner = _QueueRunner(*results)
            rx._SlurmBackend(runner).refresh(profile, job)
            assert (job.status, job.exit_code) == (status, exit_code)
            assert runner.calls == calls


def test_a_job_row_shows_its_remote_id_or_a_dash_and_any_error(subtests):
    """One fixed-width row per job: id, status, module, profile, remote id, error."""
    cases = [
        ("no remote id yet", dict(),
         "abcdef012345  submitting  measure             cloud               —"),
        ("a remote id and an error", dict(external_id="4242",
                                          error="SSHError: connection refused",
                                          status="failed"),
         "abcdef012345  failed      measure             cloud               "
         "4242  SSHError: connection refused"),
    ]
    for case_id, extra, row in cases:
        with subtests.test(msg=case_id):
            job = rx.RemoteJob(job_id="abcdef0123456789", module="measure",
                               profile_name="cloud", backend="command",
                               **extra)
            assert cli_remote._job_row(job) == row


def test_watch_reports_a_finished_jobs_error_and_log_once(monkeypatch, capsys,
                                                          subtests):
    """A terminal job is refreshed once; its error goes to stderr, its log to stdout."""
    cases = [
        ("failed, with logs", "failed", "scheduler refused",
         ["--logs", "--lines", "9"], cli_remote.EXIT_RUNTIME,
         "scheduler refused\n", "last log"),
        ("succeeded, no logs asked", "success", "", [], cli_remote.EXIT_OK,
         "", None),
    ]
    for case_id, status, error, flags, code, err, log in cases:
        with subtests.test(msg=case_id):
            job = rx.RemoteJob("abcdef123456", "mask", "cluster", "slurm",
                               status=status, error=error)
            calls = []
            manager = SimpleNamespace(
                refresh=lambda key, **kw: calls.append(("refresh", key, kw))
                or job,
                logs=lambda key, lines: calls.append(("logs", key, lines))
                or "last log")
            monkeypatch.setattr(cli_remote, "RemoteJobManager",
                                lambda: manager)
            assert cli_remote.main(["watch", job.job_id, *flags]) == code
            out = capsys.readouterr()
            expected_out = f"{job.updated_utc}  {status}\n"
            if log is not None:
                expected_out += log + "\n"
            assert out.out == expected_out
            assert out.err == err
            expected_calls = [("refresh", job.job_id,
                               {"include_logs": False})]
            if log is not None:
                expected_calls.append(("logs", job.job_id, 9))
            assert calls == expected_calls
