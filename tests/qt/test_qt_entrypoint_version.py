"""Tests for the lightweight ``spacr`` Qt console entry point."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest


@pytest.mark.parametrize("flag", ["-v", "-version", "--version"])
def test_version_flags_print_only_version_without_launching_gui(
    flag,
    monkeypatch,
    capsys,
):
    import spacr.qt as qt
    import spacr.version as version

    launched = False
    fake_app = ModuleType("spacr.qt.app")

    def fail_if_launched(_argv):
        nonlocal launched
        launched = True
        pytest.fail("version flags must exit before launching Qt")

    fake_app.launch = fail_if_launched
    monkeypatch.setitem(sys.modules, "spacr.qt.app", fake_app)
    monkeypatch.setattr(version, "get_version", lambda: "9.8.7")

    assert qt.run([flag]) == 0
    assert capsys.readouterr().out == "9.8.7\n"
    assert launched is False


def test_version_flag_uses_process_arguments(monkeypatch, capsys):
    import spacr.qt as qt
    import spacr.version as version

    monkeypatch.setattr(sys, "argv", ["spacr", "-v"])
    monkeypatch.setattr(version, "get_version", lambda: "1.2.3")

    assert qt.run() == 0
    assert capsys.readouterr().out == "1.2.3\n"
