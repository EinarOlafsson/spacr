"""``python -m spacr`` dispatch: which screen each subcommand opens on.

Every window subcommand opens the one Qt application and lands on a named
tab, so a script still saying ``python -m spacr mask`` reaches the Mask
screen rather than an import error. The dispatch table and the parser's
choices are two lists that have to agree, and this drives the path taken
when they do not.
"""
from __future__ import annotations

import runpy
import sys

import pytest


def test_running_the_module_exits_with_the_code_main_returned(
        monkeypatch, capsys):
    """``python -m spacr version`` exits 0 after printing the version."""
    from spacr.version import version_str

    monkeypatch.setattr(sys, "argv", ["spacr", "version"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("spacr.__main__", run_name="__main__")
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == version_str
