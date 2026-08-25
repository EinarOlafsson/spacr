"""``spacr-remote``: the Qt-free client for distributed spaCR jobs.

Every test here drives :func:`spacr.cli_remote.main` with a real argument
vector against a real on-disk profile/job store redirected by
``SPACR_REMOTE_STATE_DIR``, and the ``command`` backend is pointed at real
local programs (``/bin/echo`` and a small shell script). Nothing about the
persistence layer, the argument parser, the backend dispatch or the exit
codes is stubbed, so a change to any of them shows up here.
"""
from __future__ import annotations

import json
import os
import stat
import textwrap

import pytest

from spacr import cli_remote
from spacr.remote_execution import RemoteExecutionError


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """Point the profile/job stores at a throwaway directory."""
    target = tmp_path / "remote-state"
    target.mkdir()
    monkeypatch.setenv("SPACR_REMOTE_STATE_DIR", str(target))
    return target


@pytest.fixture
def status_script(tmp_path):
    """An executable status command that reports RUNNING once, then COMPLETED.

    A real program rather than a patched manager: the ``watch`` loop is only
    interesting when a job genuinely changes state between two polls.
    """
    script = tmp_path / "status.sh"
    script.write_text(textwrap.dedent("""\
        #!/bin/sh
        counter="$0.count"
        n=$(cat "$counter" 2>/dev/null || echo 0)
        n=$((n+1))
        echo "$n" > "$counter"
        if [ "$n" -ge 2 ]; then echo COMPLETED; else echo RUNNING; fi
    """), encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script


def _add_command_profile(name="cloud", *, status_command=None,
                         log_command="/bin/echo tail-of-{external_id}"):
    """Register a ``command`` profile whose commands are real local programs."""
    argv = [
        "profile", "add", name, "--backend", "command",
        "--submit-command", "/bin/echo JOBID=4242 for {settings}",
        "--status-command", status_command or "/bin/echo COMPLETED {external_id}",
        "--cancel-command", "/bin/echo cancelling {external_id}",
        "--job-id-pattern", r"JOBID=(\d+)",
    ]
    if log_command:
        argv += ["--log-command", log_command]
    assert cli_remote.main(argv) == cli_remote.EXIT_OK


@pytest.fixture
def settings_file(tmp_path):
    """A real spaCR settings file the ``submit`` path can resolve."""
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"src": str(tmp_path)}), encoding="utf-8")
    return path


def _submitted_job_id(capsys):
    """Pull the local job ID out of the submit confirmation line."""
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.startswith("Submitted ")
    return line.split(" as ", 1)[1].split(" ", 1)[0]


# --------------------------------------------------------------------------
# profile
# --------------------------------------------------------------------------


def test_an_empty_profile_list_says_so(state_dir, capsys):
    """A brand-new install lists no profiles rather than an empty table."""
    assert cli_remote.main(["profile", "list"]) == cli_remote.EXIT_OK
    out = capsys.readouterr().out
    assert "NAME" in out and "(no execution profiles)" in out


def test_profile_add_then_list_shows_the_backend_and_host(state_dir, capsys):
    """A saved profile appears in the human table with its backend and host."""
    assert cli_remote.main([
        "profile", "add", "workstation", "--backend", "ssh",
        "--host", "gpubox", "--workdir", "/scratch/spacr",
    ]) == cli_remote.EXIT_OK
    assert "Saved ssh execution profile 'workstation'." in capsys.readouterr().out

    assert cli_remote.main(["profile", "list"]) == cli_remote.EXIT_OK
    out = capsys.readouterr().out
    assert "workstation" in out and "ssh" in out and "gpubox" in out
    assert "(no execution profiles)" not in out


def test_a_profile_without_a_host_reads_as_local_or_custom(state_dir, capsys):
    """A ``command`` profile has no host, so the table says where it runs."""
    _add_command_profile("cloud")
    capsys.readouterr()
    assert cli_remote.main(["profile", "list"]) == cli_remote.EXIT_OK
    assert "local/custom" in capsys.readouterr().out


def test_profile_list_json_is_machine_readable(state_dir, capsys):
    """``--json`` emits the stored profile records rather than the table."""
    _add_command_profile("cloud")
    capsys.readouterr()
    assert cli_remote.main(["profile", "list", "--json"]) == cli_remote.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert [item["name"] for item in payload] == ["cloud"]
    assert payload[0]["backend"] == "command"
    assert payload[0]["job_id_pattern"] == r"JOBID=(\d+)"


def test_every_add_flag_reaches_the_saved_profile(state_dir, capsys):
    """The ``profile add`` arguments map one-for-one onto the stored record."""
    assert cli_remote.main([
        "profile", "add", "hpc", "--backend", "slurm", "--host", "login01",
        "--workdir", "/work/spacr", "--local-root", "/data",
        "--remote-root", "/mnt/data", "--runner", "spacr-run",
        "--scheduler-options", "--partition gpu --time 4:00:00",
        "--poll-seconds", "30",
    ]) == cli_remote.EXIT_OK
    capsys.readouterr()
    cli_remote.main(["profile", "list", "--json"])
    saved = json.loads(capsys.readouterr().out)[0]
    assert saved["backend"] == "slurm"
    assert saved["workdir"] == "/work/spacr"
    assert saved["local_root"] == "/data"
    assert saved["remote_root"] == "/mnt/data"
    assert saved["scheduler_options"] == "--partition gpu --time 4:00:00"
    assert saved["poll_seconds"] == 30


def test_profile_delete_removes_it(state_dir, capsys):
    """Deleting a profile reports it and empties the list."""
    _add_command_profile("cloud")
    capsys.readouterr()
    assert cli_remote.main(["profile", "delete", "cloud"]) == cli_remote.EXIT_OK
    assert "Deleted execution profile 'cloud'." in capsys.readouterr().out
    cli_remote.main(["profile", "list"])
    assert "(no execution profiles)" in capsys.readouterr().out


def test_deleting_a_profile_that_never_existed_is_a_usage_error(state_dir,
                                                                capsys):
    """A missing profile is reported on stderr with the usage exit code."""
    assert cli_remote.main(["profile", "delete", "ghost"]) == cli_remote.EXIT_USAGE
    assert "does not exist" in capsys.readouterr().err


def test_an_invalid_profile_is_refused_rather_than_saved(state_dir, capsys):
    """A profile that fails validation never reaches the store."""
    assert cli_remote.main([
        "profile", "add", "broken", "--backend", "ssh",
    ]) == cli_remote.EXIT_USAGE
    assert "needs a host" in capsys.readouterr().err
    cli_remote.main(["profile", "list"])
    assert "(no execution profiles)" in capsys.readouterr().out


def test_an_unknown_profile_subcommand_is_rejected_by_the_handler(state_dir):
    """``_cmd_profile`` refuses a namespace it does not recognise.

    The parser makes the subcommand mandatory, so this reaches the handler
    only by calling it directly -- which is what a future subcommand added to
    the parser but not the handler would do.
    """
    import argparse

    args = argparse.Namespace(profile_command="rename", json=False)
    with pytest.raises(RemoteExecutionError, match="list, add or delete"):
        cli_remote._cmd_profile(args)


# --------------------------------------------------------------------------
# submit
# --------------------------------------------------------------------------


def test_submitting_an_unknown_module_names_the_way_to_the_list(state_dir,
                                                                settings_file,
                                                                capsys):
    """A misspelled module is a usage error that points at ``spacr-run --list``."""
    _add_command_profile("cloud")
    capsys.readouterr()
    assert cli_remote.main([
        "submit", "not_a_module", "--settings", str(settings_file),
        "--profile", "cloud",
    ]) == cli_remote.EXIT_USAGE
    err = capsys.readouterr().err
    assert "not_a_module" in err and "spacr-run --list" in err


def test_an_unreadable_settings_file_is_reported_as_one_sentence(
        state_dir, capsys, tmp_path):
    """A ``SettingsError`` becomes a remote-execution error, not a traceback."""
    _add_command_profile("cloud")
    capsys.readouterr()
    assert cli_remote.main([
        "submit", "measure", "--settings", str(tmp_path / "nope.json"),
        "--profile", "cloud",
    ]) == cli_remote.EXIT_USAGE
    err = capsys.readouterr().err
    assert "settings file not found" in err


def test_a_bad_override_is_reported_before_anything_is_submitted(
        state_dir, settings_file, capsys):
    """``--set`` values are validated locally so a cluster job never starts wrong."""
    _add_command_profile("cloud")
    capsys.readouterr()
    assert cli_remote.main([
        "submit", "measure", "--settings", str(settings_file),
        "--profile", "cloud", "--set", "this_key_does_not_exist=1",
    ]) == cli_remote.EXIT_USAGE
    assert capsys.readouterr().err.startswith("error: ")


def test_submit_runs_the_profile_command_and_records_the_job(
        state_dir, settings_file, capsys):
    """A full submission through the ``command`` backend persists a queued job."""
    _add_command_profile("cloud")
    capsys.readouterr()
    assert cli_remote.main([
        "submit", "measure", "--settings", str(settings_file),
        "--profile", "cloud",
    ]) == cli_remote.EXIT_OK
    out = capsys.readouterr().out
    assert out.startswith("Submitted measure as ")
    assert "(cloud/4242)" in out

    cli_remote.main(["list", "--json"])
    jobs = json.loads(capsys.readouterr().out)
    assert len(jobs) == 1
    assert jobs[0]["module"] == "measure"
    assert jobs[0]["external_id"] == "4242"
    assert jobs[0]["status"] == "queued"
    # The resolved settings were written beside the job store, not into the
    # user's normal state directory.
    assert os.path.exists(jobs[0]["settings_path"])
    assert str(state_dir) in jobs[0]["settings_path"]


def test_submitting_to_a_profile_that_does_not_exist_is_a_usage_error(
        state_dir, settings_file, capsys):
    """The profile is resolved before the job record is created."""
    assert cli_remote.main([
        "submit", "measure", "--settings", str(settings_file),
        "--profile", "ghost",
    ]) == cli_remote.EXIT_USAGE
    assert "does not exist" in capsys.readouterr().err


# --------------------------------------------------------------------------
# list / status / cancel / logs
# --------------------------------------------------------------------------


def test_an_empty_job_list_says_so(state_dir, capsys):
    """No jobs prints the header and an explicit empty note."""
    assert cli_remote.main(["list"]) == cli_remote.EXIT_OK
    out = capsys.readouterr().out
    assert "JOB" in out and "(no distributed jobs)" in out


def test_the_job_table_carries_id_status_module_and_remote_id(
        state_dir, settings_file, capsys):
    """One job renders as one compact row naming its remote ID."""
    _add_command_profile("cloud")
    capsys.readouterr()
    cli_remote.main(["submit", "measure", "--settings", str(settings_file),
                     "--profile", "cloud"])
    job_id = _submitted_job_id(capsys)

    assert cli_remote.main(["list"]) == cli_remote.EXIT_OK
    out = capsys.readouterr().out
    assert job_id[:12] in out
    assert "queued" in out and "measure" in out and "cloud" in out
    assert "4242" in out


def test_a_job_with_no_remote_id_renders_a_dash(state_dir):
    """``_job_row`` shows an em dash where a remote ID has not arrived yet."""
    from spacr.remote_execution import RemoteJob

    job = RemoteJob(job_id="abcdef0123456789", module="measure",
                    profile_name="cloud", backend="command")
    row = cli_remote._job_row(job)
    assert row.startswith("abcdef012345")
    assert "—" in row
    assert row.rstrip().endswith("—")


def test_a_job_error_is_appended_to_its_row(state_dir):
    """A row that carries an error shows it after the remote ID."""
    from spacr.remote_execution import RemoteJob

    job = RemoteJob(job_id="abcdef0123456789", module="measure",
                    profile_name="cloud", backend="command",
                    external_id="4242", error="SSHError: connection refused")
    row = cli_remote._job_row(job)
    assert row.rstrip().endswith("SSHError: connection refused")
    assert "4242" in row


def test_list_refresh_polls_the_active_jobs(state_dir, settings_file, capsys):
    """``--refresh`` runs each active job's status command before printing."""
    _add_command_profile("cloud")
    capsys.readouterr()
    cli_remote.main(["submit", "measure", "--settings", str(settings_file),
                     "--profile", "cloud"])
    capsys.readouterr()

    assert cli_remote.main(["list", "--refresh", "--json"]) == cli_remote.EXIT_OK
    jobs = json.loads(capsys.readouterr().out)
    # The profile's status command echoes COMPLETED, so the queued job is
    # now terminal -- refresh actually ran, it did not just reprint.
    assert jobs[0]["status"] == "success"


def test_status_prints_the_row_and_the_log_tail(state_dir, settings_file,
                                                capsys):
    """``status --logs`` shows the compact row followed by the retrieved tail."""
    _add_command_profile("cloud")
    capsys.readouterr()
    cli_remote.main(["submit", "measure", "--settings", str(settings_file),
                     "--profile", "cloud"])
    job_id = _submitted_job_id(capsys)

    assert cli_remote.main(["status", job_id, "--logs"]) == cli_remote.EXIT_OK
    out = capsys.readouterr().out
    assert job_id[:12] in out
    assert "Log tail:" in out
    assert "tail-of-4242" in out


def test_status_json_prints_one_record(state_dir, settings_file, capsys):
    """``status --json`` emits the whole job record and no table."""
    _add_command_profile("cloud")
    capsys.readouterr()
    cli_remote.main(["submit", "measure", "--settings", str(settings_file),
                     "--profile", "cloud"])
    job_id = _submitted_job_id(capsys)

    assert cli_remote.main(["status", job_id, "--json"]) == cli_remote.EXIT_OK
    out = capsys.readouterr().out
    assert "JOB           STATUS" not in out
    record = json.loads(out)
    assert record["job_id"] == job_id
    assert record["status"] == "success"


def test_a_failed_job_makes_status_exit_nonzero(state_dir, settings_file,
                                                capsys):
    """``status`` is the exit code of the remote job, so scripts can branch on it."""
    _add_command_profile("cloud",
                         status_command="/bin/echo FAILED {external_id}")
    capsys.readouterr()
    cli_remote.main(["submit", "measure", "--settings", str(settings_file),
                     "--profile", "cloud"])
    job_id = _submitted_job_id(capsys)

    assert cli_remote.main(["status", job_id]) == cli_remote.EXIT_RUNTIME
    assert "failed" in capsys.readouterr().out


def test_status_of_an_unknown_job_is_a_usage_error(state_dir, capsys):
    """An unrecognised job ID is a sentence on stderr, not a traceback."""
    assert cli_remote.main(["status", "deadbeef"]) == cli_remote.EXIT_USAGE
    assert "was not found" in capsys.readouterr().err


def test_cancel_prints_the_new_state(state_dir, settings_file, capsys):
    """Cancelling runs the profile's cancel command and persists ``cancelled``."""
    _add_command_profile("cloud")
    capsys.readouterr()
    cli_remote.main(["submit", "measure", "--settings", str(settings_file),
                     "--profile", "cloud"])
    job_id = _submitted_job_id(capsys)

    assert cli_remote.main(["cancel", job_id]) == cli_remote.EXIT_OK
    assert capsys.readouterr().out.strip() == f"{job_id}: cancelled"

    cli_remote.main(["list", "--json"])
    assert json.loads(capsys.readouterr().out)[0]["status"] == "cancelled"


def test_logs_prints_the_retrieved_tail(state_dir, settings_file, capsys):
    """``logs`` prints exactly what the profile's log command produced."""
    _add_command_profile("cloud")
    capsys.readouterr()
    cli_remote.main(["submit", "measure", "--settings", str(settings_file),
                     "--profile", "cloud"])
    job_id = _submitted_job_id(capsys)

    assert cli_remote.main(["logs", job_id, "--lines", "5"]) == cli_remote.EXIT_OK
    assert "tail-of-4242" in capsys.readouterr().out


def test_logs_without_a_log_command_explains_how_to_add_one(
        state_dir, settings_file, capsys):
    """A command profile with no log command says what is missing."""
    _add_command_profile("cloud", log_command="")
    capsys.readouterr()
    cli_remote.main(["submit", "measure", "--settings", str(settings_file),
                     "--profile", "cloud"])
    job_id = _submitted_job_id(capsys)

    assert cli_remote.main(["logs", job_id]) == cli_remote.EXIT_OK
    assert "no log command" in capsys.readouterr().out


# --------------------------------------------------------------------------
# watch
# --------------------------------------------------------------------------


def test_watch_polls_until_the_job_is_terminal(state_dir, settings_file,
                                               status_script, capsys,
                                               monkeypatch):
    """``watch`` keeps polling while the job runs and stops when it succeeds."""
    _add_command_profile(
        "cloud", status_command=f"{status_script} " + "{external_id}")
    capsys.readouterr()
    cli_remote.main(["submit", "measure", "--settings", str(settings_file),
                     "--profile", "cloud"])
    job_id = _submitted_job_id(capsys)

    slept = []
    monkeypatch.setattr("time.sleep", slept.append)

    assert cli_remote.main(["watch", job_id, "--interval", "1"]) \
        == cli_remote.EXIT_OK
    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert len(lines) == 2, lines
    assert lines[0].endswith("running")
    assert lines[1].endswith("success")
    # One poll interval elapsed, floored at two seconds however small
    # ``--interval`` was.
    assert slept == [2]


def test_watch_returns_nonzero_for_a_job_that_failed(state_dir, settings_file,
                                                     capsys):
    """A watched job that ends in failure exits nonzero on the first poll."""
    _add_command_profile("cloud",
                         status_command="/bin/echo FAILED {external_id}")
    capsys.readouterr()
    cli_remote.main(["submit", "measure", "--settings", str(settings_file),
                     "--profile", "cloud"])
    job_id = _submitted_job_id(capsys)

    assert cli_remote.main(["watch", job_id]) == cli_remote.EXIT_RUNTIME
    assert capsys.readouterr().out.strip().endswith("failed")


def test_watch_can_print_the_log_tail_when_it_finishes(state_dir,
                                                       settings_file, capsys):
    """``watch --logs`` retrieves the tail once the job reaches a terminal state."""
    _add_command_profile("cloud")
    capsys.readouterr()
    cli_remote.main(["submit", "measure", "--settings", str(settings_file),
                     "--profile", "cloud"])
    job_id = _submitted_job_id(capsys)

    assert cli_remote.main(["watch", job_id, "--logs", "--lines", "10"]) \
        == cli_remote.EXIT_OK
    assert "tail-of-4242" in capsys.readouterr().out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def test_an_interrupt_is_reported_rather_than_a_traceback(state_dir, capsys,
                                                          monkeypatch):
    """Ctrl-C during a long poll exits with the runtime code and one word."""
    def interrupt(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_remote, "_cmd_list", interrupt)
    assert cli_remote.main(["list"]) == cli_remote.EXIT_RUNTIME
    assert capsys.readouterr().err.strip() == "interrupted"


def test_a_missing_command_is_an_argparse_usage_error(state_dir):
    """The subcommand is mandatory: no command exits 2 through argparse."""
    with pytest.raises(SystemExit) as excinfo:
        cli_remote.main([])
    assert excinfo.value.code == 2


def test_the_parser_names_every_shipped_subcommand():
    """``build_parser`` exposes the documented command set."""
    parser = cli_remote.build_parser()
    actions = [action for action in parser._actions
               if getattr(action, "choices", None)
               and "submit" in getattr(action, "choices", {})]
    assert actions, "no subcommand action on the parser"
    assert set(actions[0].choices) == {
        "profile", "submit", "list", "status", "cancel", "logs", "watch"}
