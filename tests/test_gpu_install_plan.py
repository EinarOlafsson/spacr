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


def test_an_unsupported_interpreter_says_what_to_do(monkeypatch):
    """"Make a 3.11 environment" is actionable; a pip resolver error is
    not, and pip is what the user would otherwise meet."""
    monkeypatch.setattr("spacr.gpu_reduce.python_supported", lambda: False)
    monkeypatch.setattr("spacr.gpu_reduce.rapids_available", lambda: False)
    plan = install_plan()
    assert plan["action"] == "wrong_python"
    assert "3.11" in plan["message"]
    assert "conda create" in plan["message"]


def test_a_supported_interpreter_offers_to_install(monkeypatch):
    monkeypatch.setattr("spacr.gpu_reduce.python_supported", lambda: True)
    monkeypatch.setattr("spacr.gpu_reduce.rapids_available", lambda: False)
    plan = install_plan()
    assert plan["action"] == "install"


def test_the_size_is_stated_before_the_download_starts(monkeypatch):
    """A multi-gigabyte download with no progress reads as a hang."""
    monkeypatch.setattr("spacr.gpu_reduce.python_supported", lambda: True)
    monkeypatch.setattr("spacr.gpu_reduce.rapids_available", lambda: False)
    message = install_plan()["message"]
    assert "GIGABYTES" in message


def test_the_restart_is_stated_too(monkeypatch):
    """pip can upgrade numpy and scipy underneath a process that has already
    imported them, and this one has."""
    monkeypatch.setattr("spacr.gpu_reduce.python_supported", lambda: True)
    monkeypatch.setattr("spacr.gpu_reduce.rapids_available", lambda: False)
    assert "RESTARTED" in install_plan()["message"]


def test_a_working_install_is_ready_rather_than_offered_again(monkeypatch):
    monkeypatch.setattr("spacr.gpu_reduce.rapids_available", lambda: True)
    assert install_plan()["action"] == "ready"


def test_cuml_without_a_device_is_not_an_install_problem(monkeypatch):
    """Installing again cannot conjure a GPU, so it must not be offered."""
    import types

    monkeypatch.setattr("spacr.gpu_reduce.rapids_available", lambda: False)
    monkeypatch.setitem(sys.modules, "cuml", types.SimpleNamespace())
    plan = install_plan()
    assert plan["action"] == "no_device"
    assert "nvidia-smi" in plan["message"]


def test_nothing_is_installed_by_deciding(monkeypatch):
    """A function that decided AND installed could not be asked what would
    happen without it happening."""
    import inspect

    from spacr import gpu_reduce

    source = inspect.getsource(gpu_reduce.install_plan)
    for forbidden in ("subprocess", "check_call", "pip.main"):
        assert forbidden not in source


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
