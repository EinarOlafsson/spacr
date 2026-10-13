"""No test can start a real vendor installer by accident.

Review of item 420, 2026-09-19: Install became the default button of the
prompt a provider that is not set up opens, and the only guard against a
test pressing it for real lived in the two test files written with it. The
root conftest now refuses ``spacr.qt.ai.cli_install._spawn`` for every test;
these check that it does, from a file with no guard of its own.
"""
from __future__ import annotations

import subprocess

import pytest

from spacr.qt.ai import cli_install, providers


def test_the_installer_seam_is_refused_inside_a_test():
    assert cli_install._spawn is not subprocess.Popen
    with pytest.raises(AssertionError, match="real installer"):
        cli_install._spawn(["true"])


def test_an_install_a_test_reaches_by_accident_is_refused():
    plan = cli_install.InstallPlan(
        providers.github_cli(), providers.InstallMethod(("true",), "true"),
        "linux")
    with pytest.raises(AssertionError, match="real installer"):
        cli_install.run_install(plan)
