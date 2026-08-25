"""Upgrade plumbing: where the checkout guard looks, and how a plan is read.

Every function here takes its input by injection -- a runner, a fake
``importlib.metadata`` distribution, a ``sys.path`` -- so the real code runs
against real strings without a packaging tool ever being launched.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from spacr import updater


# ---------------------------------------------------------------------------
# Where spaCR is installed from
# ---------------------------------------------------------------------------

class _Distribution:
    """Stands in for the installed distribution's metadata files."""

    def __init__(self, payload):
        self._payload = payload

    def read_text(self, name):
        return self._payload if name == "direct_url.json" else None


def _fake_metadata(monkeypatch, payload):
    import importlib.metadata as md

    monkeypatch.setattr(md, "distribution", lambda _name: _Distribution(payload))


def test_a_pip_editable_install_names_its_working_tree(monkeypatch, tmp_path):
    checkout = tmp_path / "my checkout"
    checkout.mkdir()
    _fake_metadata(monkeypatch, (
        '{"dir_info": {"editable": true}, "url": "file://'
        + str(checkout).replace(" ", "%20") + '"}'))
    assert updater.editable_install_location() == str(checkout)


def test_an_editable_install_recorded_without_a_file_url_is_still_named(
        monkeypatch):
    _fake_metadata(monkeypatch, (
        '{"dir_info": {"editable": true}, "url": "git+https://x/y"}'))
    assert updater.editable_install_location() == "git+https://x/y"


def test_an_editable_record_with_no_url_falls_through_to_the_path_check(
        monkeypatch):
    _fake_metadata(monkeypatch, '{"dir_info": {"editable": true}}')
    # The repository this suite runs from IS a checkout, so the second check
    # answers where the first said nothing usable.
    assert updater.editable_install_location() == os.path.abspath(
        os.path.dirname(os.path.dirname(os.path.abspath(
            sys.modules["spacr"].__file__))))


def test_a_non_editable_record_is_not_a_checkout(monkeypatch):
    _fake_metadata(monkeypatch, '{"dir_info": {"editable": false}}')
    assert updater.editable_install_location() is not None  # source tree


def test_unreadable_metadata_does_not_stop_the_path_check(monkeypatch):
    import importlib.metadata as md

    def refuse(_name):
        raise md.PackageNotFoundError("spacr")

    monkeypatch.setattr(md, "distribution", refuse)
    assert updater.editable_install_location() is not None


def test_a_spacr_that_cannot_be_imported_answers_nothing(monkeypatch):
    _fake_metadata(monkeypatch, None)
    monkeypatch.setitem(sys.modules, "spacr", None)
    assert updater.editable_install_location() is None


def test_a_package_under_site_packages_is_not_a_checkout(monkeypatch,
                                                         tmp_path):
    """An ordinary install must never be reported as a working tree."""
    import types

    site = tmp_path / "site-packages"
    (site / "spacr").mkdir(parents=True)
    (site / "spacr" / "__init__.py").write_text("")
    # A .git alongside it would fool the marker check on its own.
    (site / ".git").mkdir()
    fake = types.ModuleType("spacr")
    fake.__file__ = str(site / "spacr" / "__init__.py")

    _fake_metadata(monkeypatch, None)
    monkeypatch.setitem(sys.modules, "spacr", fake)
    monkeypatch.setattr(sys, "path", ["", str(site)])
    assert updater.editable_install_location() is None


def test_a_tree_with_no_git_and_no_pyproject_is_not_a_checkout(monkeypatch,
                                                              tmp_path):
    import types

    root = tmp_path / "somewhere"
    (root / "spacr").mkdir(parents=True)
    fake = types.ModuleType("spacr")
    fake.__file__ = str(root / "spacr" / "__init__.py")
    _fake_metadata(monkeypatch, None)
    monkeypatch.setitem(sys.modules, "spacr", fake)
    monkeypatch.setattr(sys, "path", [str(tmp_path / "lib")])
    assert updater.editable_install_location() is None
    (root / "pyproject.toml").write_text("[project]\n")
    assert updater.editable_install_location() == str(root)


# ---------------------------------------------------------------------------
# Running the packaging tool
# ---------------------------------------------------------------------------

def test_a_missing_packaging_tool_is_reported_not_raised(monkeypatch):
    def missing(*_args, **_kwargs):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(subprocess, "run", missing)
    code, output = updater.run_install_command(["uv", "pip", "install", "spacr"])
    assert code == 1
    assert output.startswith("Could not run uv:")


def test_an_install_that_never_answers_says_how_long_it_waited(monkeypatch):
    def hang(*_args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="pip", timeout=kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", hang)
    code, output = updater.run_install_command(["pip"], timeout=180.0)
    assert code == 1
    assert output == "The command timed out after 3 minutes."


def test_a_failed_install_keeps_both_streams(monkeypatch):
    class Completed:
        returncode = 2
        stdout = "collecting spacr\n"
        stderr = "ERROR: could not resolve\n"

    monkeypatch.setattr(subprocess, "run",
                        lambda *_a, **_k: Completed())
    code, output = updater.run_install_command(["pip"])
    assert code == 2
    assert output == "collecting spacr\nERROR: could not resolve\n"


# ---------------------------------------------------------------------------
# Environment questions
# ---------------------------------------------------------------------------

def test_a_version_lookup_that_throws_answers_none(monkeypatch):
    import importlib.metadata as md

    def explode(_name):
        raise RuntimeError("metadata store is corrupt")

    monkeypatch.setattr(md, "version", explode)
    assert updater.installed_version("numpy") is None


def test_an_absent_package_has_no_version():
    assert updater.installed_version("this-package-is-not-installed") is None


def test_pip_is_reported_present_in_this_interpreter():
    assert updater.pip_available() is True


def test_a_broken_spec_lookup_answers_no_pip(monkeypatch):
    import importlib.util

    def explode(_name):
        raise ValueError("__spec__ is not set")

    monkeypatch.setattr(importlib.util, "find_spec", explode)
    assert updater.pip_available() is False


def test_without_pip_or_uv_the_dry_run_still_asks_pip(monkeypatch):
    """There is no third tool: the last resort is the command that exists."""
    monkeypatch.setattr(updater, "pip_available", lambda: False)
    monkeypatch.setattr(updater, "find_uv", lambda: None)
    args = updater.dry_run_command("cuml-cu12")
    assert args[:2] == [sys.executable, "-m"]
    assert "--dry-run" in args and args[-1] == "cuml-cu12"


def test_with_uv_and_no_pip_the_dry_run_goes_through_uv(monkeypatch):
    monkeypatch.setattr(updater, "pip_available", lambda: False)
    monkeypatch.setattr(updater, "find_uv", lambda: "/opt/uv")
    args = updater.dry_run_command("cuml-cu12")
    assert args[0] == "/opt/uv"
    assert "--dry-run" in args and args[-1] == "cuml-cu12"


# ---------------------------------------------------------------------------
# Reading a plan
# ---------------------------------------------------------------------------

def test_each_kind_of_change_describes_itself():
    added = updater.PackageChange("cuml-cu12", None, "26.8.0")
    moved = updater.PackageChange("numpy", "1.26.4", "2.2.6")
    same = updater.PackageChange("pandas", "2.2.3", "2.2.3")
    assert added.describe() == "cuml-cu12 26.8.0 (new)"
    assert moved.describe() == "numpy 1.26.4 -> 2.2.6"
    assert same.describe() == "pandas 2.2.3 (unchanged)"
    assert added.is_addition and not added.is_move
    assert moved.is_move and moved.protected
    assert not same.is_move


def test_an_addition_with_no_version_still_reads():
    assert updater.PackageChange("mystery", None, None).describe() == (
        "mystery ? (new)")


def test_a_plan_that_changes_nothing_says_so():
    plan = updater.DryRun("cuml-cu12", True, ())
    assert "change nothing" in plan.summary()


def test_a_dry_run_whose_tool_is_missing_is_not_allowed():
    def missing(*_args, **_kwargs):
        raise FileNotFoundError("no uv here")

    plan = updater.dry_run_install("cuml-cu12", runner=missing)
    assert plan.ok is False
    assert "Could not run" in plan.error


def test_a_resolver_that_never_answers_names_the_budget():
    def hang(*_args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="pip", timeout=kwargs["timeout"])

    plan = updater.dry_run_install("cuml-cu12", timeout=42.0, runner=hang)
    assert plan.ok is False
    assert plan.error == "The resolver did not answer within 42 seconds."


class _Completed:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_output_that_holds_no_plan_is_refused_rather_than_guessed():
    plan = updater.dry_run_install(
        "cuml-cu12",
        runner=lambda *_a, **_k: _Completed(stdout="all done, nothing to say"))
    assert plan.ok is False
    assert plan.error == "The packaging tool produced no readable plan."
    assert plan.raw == "all done, nothing to say"


def test_a_failed_resolve_quotes_the_last_thing_pip_said():
    tail = "\n".join(f"line {n}" for n in range(12))
    plan = updater.dry_run_install(
        "cuml-cu12",
        runner=lambda *_a, **_k: _Completed(stderr=tail, returncode=1))
    assert plan.ok is False
    assert plan.error.splitlines() == [f"line {n}" for n in range(6, 12)]


def test_a_resolver_that_failed_silently_still_says_something():
    assert updater._resolver_error("   \n\n") == (
        "The packaging tool failed and said nothing.")


def test_the_pip_report_is_found_among_pips_own_chatter():
    """pip writes progress before the document and a summary after it."""
    report = (
        'Collecting cuml-cu12\n'
        '{"not_the_report": 1}\n'
        'not json at all {oops\n'
        '{"install": [{"metadata": {"name": "cuml-cu12", '
        '"version": "26.8.0"}}, {"metadata": {"version": "1.0"}}]}\n'
        'Would install cuml-cu12-26.8.0\n')
    changes = updater._parse_pip_report(report)
    assert [change.name for change in changes] == ["cuml-cu12"]
    assert changes[0].proposed == "26.8.0"
    assert changes[0].is_addition


def test_output_with_no_json_document_is_not_a_pip_report():
    assert updater._parse_pip_report("Collecting cuml-cu12\n") is None
    assert updater._parse_pip_report("") is None


def test_uv_output_merges_the_two_halves_of_a_version_move():
    changes = updater._parse_uv_dry_run(
        "Resolved 3 packages\n"
        " + cuml-cu12==26.8.0\n"
        " - numpy==1.26.4\n"
        " + numpy==2.2.6\n"
        " - obsolete-thing==0.1\n"
        "Would install 2 packages\n")
    by_name = {change.name: change for change in changes}
    assert by_name["numpy"].current == "1.26.4"
    assert by_name["numpy"].proposed == "2.2.6"
    assert by_name["cuml-cu12"].is_addition
    assert by_name["obsolete-thing"].proposed is None
    assert by_name["obsolete-thing"].current == "0.1"


def test_output_uv_never_wrote_is_not_a_uv_plan():
    assert updater._parse_uv_dry_run("Resolved 3 packages\n") is None
    assert updater._parse_uv_dry_run("") is None
