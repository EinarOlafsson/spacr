"""``python -m spacr.cli_workspace``: the module-level entry point.

The console script installed as ``spacr-workspace`` calls :func:`main`, but a
user who has the source and no installed entry point runs the module directly
instead. That path goes through the ``if __name__ == "__main__"`` guard, which
turns the integer :func:`main` returns into the process exit status. These
tests execute the module under the name ``__main__`` and read that status
back, so the guard is held to the same exit-code contract as :func:`main`.
"""
from __future__ import annotations

import json
import runpy
import sys
import warnings

import pytest

from spacr.workspace import DOC_NAME


def _run_as_main(argv):
    """Execute ``spacr.cli_workspace`` under the name ``__main__``.

    :param argv: the command-line words after the program name.
    :returns: the :class:`SystemExit` the guard raised.
    """
    saved_argv = sys.argv
    saved_module = sys.modules.pop("spacr.cli_workspace", None)
    sys.argv = ["spacr-workspace", *argv]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_module("spacr.cli_workspace", run_name="__main__")
        return excinfo.value
    finally:
        sys.argv = saved_argv
        if saved_module is not None:
            sys.modules["spacr.cli_workspace"] = saved_module


def test_running_the_module_on_a_missing_run_folder_exits_two(tmp_path, capsys):
    """A folder that is not there is exit 2, and the message names the path."""
    missing = tmp_path / "no_such_run"

    exit_exc = _run_as_main([str(missing)])

    assert exit_exc.code == 2
    assert str(missing) in capsys.readouterr().err


def test_running_the_module_on_a_saved_workspace_exits_zero(tmp_path, capsys):
    """A run that carries a workspace prints it and exits 0."""
    run = tmp_path / "run"
    run.mkdir()
    document = {
        "version": 1,
        "panels": [{"kind": "regression", "title": "ols_4"}],
        "files": [{"role": "database", "path": "measurements.db"}],
    }
    (run / DOC_NAME).write_text(json.dumps(document), encoding="utf-8")

    exit_exc = _run_as_main([str(run), "--json"])

    assert exit_exc.code == 0
    assert json.loads(capsys.readouterr().out) == document
