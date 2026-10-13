"""``python -m spacr.sweep_child`` turns the trial's status into an exit code.

The module-scope entry point is what the sweep parent actually launches, so
the usage error it reports has to arrive as a nonzero exit status rather
than as a return value nobody reads.
"""
from __future__ import annotations

import runpy
import sys

import pytest


def test_running_the_module_as_a_script_raises_the_status_it_returns(
        monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["spacr.sweep_child", "only-one-path"])

    with pytest.raises(SystemExit) as caught:
        runpy.run_module("spacr.sweep_child", run_name="__main__")

    assert caught.value.code == 2, (
        "two paths are required; anything else is a usage error")
    assert "usage: python -m spacr.sweep_child" in capsys.readouterr().err
