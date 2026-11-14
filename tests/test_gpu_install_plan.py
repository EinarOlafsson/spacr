"""The GPU button decides before it installs. Instruction 95.

    "if pressed it should activate the GPU cuML version, if dependencies are
     not installed then it should check the python version and install if
     compatible other wise tell the user what they need to do."

Four outcomes, and each has to be a different sentence, because "it did not
work" is the same message for all four and actionable for none.
"""
from __future__ import annotations

import sys

import pytest

from spacr.gpu_reduce import (
    SUPPORTED_PYTHON,
    install_command,
    install_plan,
    python_supported,
)


def test_the_supported_interpreters_are_the_ones_the_wheel_declares():
    """Read off cuml-cu12's metadata, not guessed: it carries classifiers
    for 3.11 and 3.12 only."""
    assert SUPPORTED_PYTHON == ((3, 11), (3, 12))


def test_python_supported_matches_the_running_interpreter():
    assert python_supported() is (sys.version_info[:2] in SUPPORTED_PYTHON)


def test_nothing_is_installed_by_deciding(monkeypatch):
    """A function that decided AND installed could not be asked what would
    happen without it happening."""
    import subprocess

    def refuse(*args, **kwargs):
        raise AssertionError("deciding must not start a process")

    for name in ("run", "call", "check_call", "check_output", "Popen"):
        monkeypatch.setattr(subprocess, name, refuse)
    for supported in (True, False):
        monkeypatch.setattr("spacr.gpu_reduce.python_supported", lambda ok=supported: ok)
        assert install_plan()["action"] in {"wrong_python", "install", "ready", "no_device"}


def test_the_command_is_shown_rather_than_hidden():
    command = install_command()
    assert command[0] == sys.executable
    assert command[-1] == "spacr[rapids]"


@pytest.mark.parametrize("action", ["ready", "install", "wrong_python",
                                    "no_device"])
def test_every_outcome_carries_its_own_message(action, monkeypatch):
    import types

    monkeypatch.setattr("spacr.gpu_reduce.rapids_available",
                        lambda: action == "ready")
    monkeypatch.setattr("spacr.gpu_reduce.python_supported",
                        lambda: action != "wrong_python")
    if action == "no_device":
        monkeypatch.setitem(sys.modules, "cuml", types.SimpleNamespace())
    else:
        monkeypatch.delitem(sys.modules, "cuml", raising=False)
    plan = install_plan()
    assert plan["action"] == action
    assert plan["message"].strip()
