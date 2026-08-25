"""``spacr repro`` resolves a run by basename and exits with main()'s code.

Two paths that a user meets on the command line and that nothing else covers:
typing just the run folder's name instead of its full path, and running the
module as a script, where the return value has to become the process's exit
status rather than being printed and discarded.
"""
from __future__ import annotations

import io
import json
import runpy
import sys
import warnings
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import pytest

from spacr import cli_repro


@pytest.fixture
def run_folder(tmp_path, monkeypatch):
    """A real run journal under a private runs root, returned as its Path."""
    from spacr import run_journal as rj
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(rj, "runs_root", lambda: root)
    monkeypatch.setattr(cli_repro, "runs_root", lambda: root)
    with rj.open_run("mask", {"src": str(tmp_path / "data"), "n": 5}) as run:
        pass
    return run.dir


def test_a_run_can_be_named_by_its_basename(run_folder, monkeypatch):
    """A bare folder name resolves under the runs root, not the cwd.

    The name printed in every spaCR log line is the basename; requiring the
    full path would make the CLI unusable from anywhere but the runs root.
    """
    monkeypatch.chdir(run_folder.parent.parent)
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = cli_repro.main([run_folder.name, "--show"])

    assert code == 0
    out = buf.getvalue()
    assert f"run:       {run_folder.name}" in out
    assert "app:       mask" in out
    manifest = json.loads((run_folder / "manifest.json").read_text())
    assert f"n_settings:{manifest['n_settings']}" in out


def test_an_unknown_basename_is_still_an_error(tmp_path, monkeypatch):
    """A name that exists under neither the cwd nor the runs root exits 2."""
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(cli_repro, "runs_root", lambda: root)
    err = io.StringIO()
    with redirect_stderr(err):
        code = cli_repro.main(["not-a-run"])
    assert code == 2
    assert "no such run folder: not-a-run" in err.getvalue()


def test_running_the_module_as_a_script_exits_with_main_s_code(tmp_path,
                                                               monkeypatch):
    """``python -m spacr.cli_repro`` must fail the shell, not just print.

    The module's ``__main__`` guard wraps ``main()`` in ``SystemExit``; without
    it a bad run folder would print an error and still exit 0, and every
    script that checks ``$?`` would treat a failed replay as a success.
    """
    from spacr import run_journal as rj
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setattr(rj, "runs_root", lambda: root)
    monkeypatch.setattr(sys, "argv", ["spacr-repro", "definitely-missing"])

    err = io.StringIO()
    with warnings.catch_warnings():
        # runpy re-executes an already-imported module; that notice is not
        # what this test is about.
        warnings.simplefilter("ignore", RuntimeWarning)
        with redirect_stderr(err), pytest.raises(SystemExit) as excinfo:
            runpy.run_module("spacr.cli_repro", run_name="__main__")

    assert excinfo.value.code == 2
    assert "no such run folder: definitely-missing" in err.getvalue()
