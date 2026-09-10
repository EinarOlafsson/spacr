"""``python -m spacr.cli_remote`` is a supported way to reach the client.

The console script is the documented entry, but the module form is what a
scheduler or a container ``CMD`` reaches for, and it goes through a different
line of code: the module body's own ``__main__`` guard rather than the
setuptools shim. It has to convert the handler's return value into a process
exit code, because a scheduler reads the exit status and nothing else.
"""
from __future__ import annotations

import runpy
import sys

import pytest


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """Point the profile/job stores at a throwaway directory."""
    target = tmp_path / "remote-state"
    target.mkdir()
    monkeypatch.setenv("SPACR_REMOTE_STATE_DIR", str(target))
    return target


def _run_module(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["spacr-remote", *argv])
    with pytest.raises(SystemExit) as leaving:
        runpy.run_module("spacr.cli_remote", run_name="__main__")
    return leaving.value.code


def test_the_module_entry_point_exits_zero_on_a_command_that_worked(
        state_dir, monkeypatch, capsys):
    """A successful listing leaves the process with status 0.

    Returning the handler's value without raising would leave the exit code
    at 0 for failures too, which is the one thing a scheduler watches.
    """
    code = _run_module(monkeypatch, ["profile", "list"])

    assert code == 0
    assert "(no execution profiles)" in capsys.readouterr().out


def test_the_module_entry_point_exits_nonzero_when_the_job_is_unknown(
        state_dir, monkeypatch, capsys):
    """A usage failure reaches the shell as a non-zero status, with a reason."""
    code = _run_module(monkeypatch, ["status", "no-such-job"])

    assert code != 0
    assert "no-such-job" in capsys.readouterr().err
