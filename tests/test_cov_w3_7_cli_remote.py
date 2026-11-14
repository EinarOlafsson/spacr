"""``spacr-remote``: the Qt-free client for distributed spaCR jobs.

The handler, row-formatting and argument-helper edges.  Every command run
end to end through ``main`` is table-driven in ``tests/test_cli_remote_client.py``.
"""
from __future__ import annotations

import pytest

from spacr import cli_remote
from spacr.remote_execution import RemoteExecutionError

pytestmark = pytest.mark.posix_only


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """Point the profile/job stores at a throwaway directory."""
    target = tmp_path / "remote-state"
    target.mkdir()
    monkeypatch.setenv("SPACR_REMOTE_STATE_DIR", str(target))
    return target


# --------------------------------------------------------------------------
# profile
# --------------------------------------------------------------------------


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
# job table rows
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# parser and argument helpers
# --------------------------------------------------------------------------


def test_the_parser_names_every_shipped_subcommand():
    """``build_parser`` exposes the documented command set."""
    parser = cli_remote.build_parser()
    actions = [action for action in parser._actions
               if getattr(action, "choices", None)
               and "submit" in getattr(action, "choices", {})]
    assert actions, "no subcommand action on the parser"
    assert set(actions[0].choices) == {
        "profile", "submit", "list", "status", "cancel", "logs", "watch",
        "allocate", "resources", "script"}


def test_template_file_and_resource_argument_errors_are_explicit(tmp_path):
    template = tmp_path / 'job.sh'
    template.write_text('#!/bin/sh\necho café\n', encoding='utf-8')
    assert cli_remote._read_text(str(template)) == '#!/bin/sh\necho café\n'
    with pytest.raises(RemoteExecutionError, match='Could not read') as caught:
        cli_remote._read_text(str(tmp_path / 'missing.sh'))
    assert isinstance(caught.value.__cause__, OSError)
    for value in ('missing-separator', '=empty-key'):
        with pytest.raises(RemoteExecutionError, match='KEY=VALUE'):
            cli_remote._resources_from_args([value])
    assert cli_remote._resources_from_args([' cpus = 4 ']) == {'cpus': '4'}


def test_watch_reports_terminal_error_and_requested_logs_without_polling(monkeypatch, capsys):
    from types import SimpleNamespace
    from spacr.remote_execution import RemoteJob

    job = RemoteJob('abcdef123456', 'mask', 'cluster', 'slurm', status='failed', error='scheduler refused')
    calls = []
    manager = SimpleNamespace(refresh=lambda key, **kwargs: calls.append((key, kwargs)) or job,
                              logs=lambda key, lines: calls.append((key, lines)) or 'last log')
    monkeypatch.setattr(cli_remote, 'RemoteJobManager', lambda: manager)
    assert cli_remote.main(['watch', job.job_id, '--logs', '--lines', '9']) == cli_remote.EXIT_RUNTIME
    out = capsys.readouterr()
    assert 'failed' in out.out and 'last log' in out.out and 'scheduler refused' in out.err
    assert calls == [(job.job_id, {'include_logs': False}), (job.job_id, 9)]
