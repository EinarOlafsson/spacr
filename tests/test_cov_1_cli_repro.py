"""``spacr repro`` resolves a run by basename and exits with main()'s code.

Two paths that a user meets on the command line and that nothing else covers:
typing just the run folder's name instead of its full path, and running the
module as a script, where the return value has to become the process's exit
status rather than being printed and discarded.
"""
from __future__ import annotations

import io
import runpy
import sys
import warnings
from contextlib import redirect_stderr

import pytest


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
