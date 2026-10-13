"""The module run as a script, which is the only way the installers call it.

The desktop installers run outside the environment they build, so they reach
this module as ``python -m spacr.install_profile`` and read the profile back
off stdout. That entry point is a different code path from calling
:func:`spacr.install_profile.main` -- it goes through the ``__main__`` guard
and turns the return code into a ``SystemExit`` -- and an installer that
stopped exiting zero would look like a failed install on somebody's machine.
Driven with ``runpy`` in-process, the same way ``spacr.doctor``'s script entry
point is driven.
"""
from __future__ import annotations

import json
import runpy
import sys

import pytest


def test_running_the_module_as_a_script_writes_the_profile_and_exits_zero(
        tmp_path, monkeypatch, capsys):
    """The ``__main__`` guard writes the file and exits with main's code."""
    target = tmp_path / "install-profile.json"
    monkeypatch.setattr(sys, "argv", [
        "spacr.install_profile",
        "--path", str(target),
        "--requested", "cu124",
        "--detected", "nvidia",
        "--consent-collected", "1",
        "--share-diagnostics", "0",
        "--report-issues", "1",
        "--sign-in-now", "0",
    ])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("spacr.install_profile", run_name="__main__")

    assert excinfo.value.code == 0
    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["schema"] == 1
    assert written["requested_backend"] == "cu124"
    assert written["detected_accelerator"] == "nvidia"
    assert written["consent"] == {
        "collected": True,
        "share_diagnostics": False,
        "report_issues": True,
        "sign_in_now": False,
    }
    # The installer logs what was recorded by reading stdout, so the echoed
    # JSON has to be the profile that reached the disk, not a summary of it.
    echoed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert echoed == written


def test_running_the_module_as_a_script_refuses_an_unknown_accelerator(
        tmp_path, monkeypatch):
    """argparse rejects the value before any file is created."""
    target = tmp_path / "install-profile.json"
    monkeypatch.setattr(sys, "argv", [
        "spacr.install_profile",
        "--path", str(target),
        "--requested", "cpu",
        "--detected", "quantum",
    ])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("spacr.install_profile", run_name="__main__")

    assert excinfo.value.code == 2
    assert not target.exists()
