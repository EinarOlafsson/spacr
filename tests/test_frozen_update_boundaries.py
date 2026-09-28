"""Frozen update discovery must not launch a GUI executable as Python."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from spacr import install_cleanup as cleanup
from spacr import updater


def test_current_debian_package_is_recognized_as_the_running_frozen_app(tmp_path):
    """dpkg's new /opt layout must enter the installer-helper path."""
    root = tmp_path / "opt" / "spacr"
    root.mkdir(parents=True)
    (root / "spacr").write_text("frozen app fixture")
    packages = SimpleNamespace(version=lambda name: "1.5.1.0" if name == "spacr" else None,
                               owns=lambda name, path: name == "spacr" and path == str(root / "spacr"))
    machine = cleanup._Machine(
        platform="linux", environ={}, fs_root=str(tmp_path), packages=packages,
        running_prefix=str(root / "_internal"), executable=str(root / "spacr"),
    )
    records = cleanup._find_deb(machine)
    assert len(records) == 1
    assert records[0].root == str(root)
    assert records[0].running
    assert records[0].registrations == ("deb:spacr",)


def test_legacy_stdeb_layout_is_retained_despite_an_unowned_opt_directory(tmp_path):
    """A python3-spacr record cannot acquire another package's /opt directory."""
    root = tmp_path / "opt" / "spacr"
    root.mkdir(parents=True)
    (root / "spacr").write_text("unrelated frozen fixture")
    legacy = tmp_path / "usr/lib/python3/dist-packages/spacr"
    packages = SimpleNamespace(version=lambda name: "1.5.0.1" if name == "python3-spacr" else None)
    machine = cleanup._Machine(
        platform="linux", environ={}, fs_root=str(tmp_path), packages=packages,
        running_prefix=str(legacy),
    )
    record, = cleanup._find_deb(machine)
    assert record.root == str(legacy)
    assert record.running


def test_spacr_package_name_alone_does_not_own_a_foreign_opt_directory(tmp_path):
    """A legacy package named spacr also needs an actual dpkg file witness."""
    root = tmp_path / "opt/spacr"
    root.mkdir(parents=True)
    (root / "spacr").write_text("unowned application")
    packages = SimpleNamespace(version=lambda name: "1.5.0.1" if name == "spacr" else None,
                               owns=lambda name, path: False)
    machine = cleanup._Machine(platform="linux", environ={}, fs_root=str(tmp_path),
                               packages=packages)
    record, = cleanup._find_deb(machine)
    assert record.root != str(root)


def test_dpkg_ownership_requires_an_exact_file_and_successful_query():
    """A prefix match or failed package-manager query does not prove ownership."""
    calls = []

    def query(args):
        """Supply one dpkg file list without launching the package manager."""
        calls.append(args)
        return 0, "/opt/spacr/spacr-other\n/opt/spacr/spacr\n"

    packages = cleanup._SystemPackages(query, which=lambda name: "/usr/bin/dpkg-query")
    assert packages.owns("spacr", "/opt/spacr/spacr")
    assert not packages.owns("spacr", "/opt/spacr/spac")
    assert calls[0] == ["dpkg-query", "-L", "spacr"]
    packages._runner = lambda args: (1, "/opt/spacr/spacr\n")
    assert not packages.owns("spacr", "/opt/spacr/spacr")


def test_frozen_executable_location_identifies_a_temporary_extraction_runtime(tmp_path, monkeypatch):
    """One-file extraction prefixes must not hide the actual installed executable."""
    executable = tmp_path / "installed" / "spacr"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))
    machine = cleanup._Machine(environ={}, fs_root=str(tmp_path),
                               running_prefix=str(tmp_path / "_MEI123"),
                               executable=str(executable))
    assert cleanup._running(machine, str(executable.parent))


def test_frozen_upgrader_never_constructs_or_runs_a_pip_command(monkeypatch):
    """Neither uv discovery nor subprocess execution should precede refusal."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    def forbidden(*args, **kwargs):
        """Fail before any real discovery, installer or child process can run."""
        pytest.fail("frozen update reached an interpreter-based update path")

    monkeypatch.setattr(updater, "find_uv", forbidden)
    monkeypatch.setattr(updater, "run_install_command", forbidden)
    monkeypatch.setattr(updater, "editable_install_location", forbidden)
    with pytest.raises(RuntimeError, match="not a Python environment"):
        updater.upgrade_command(target_version="1.5.1.1")
    code, message = updater.run_pip_upgrade(target_version="1.5.1.1")
    assert code != 0 and "installer" in message


def test_helper_cannot_use_frozen_executable_even_outside_removed_roots(tmp_path, monkeypatch):
    """The helper needs an interpreter, not merely an executable outside the app."""
    executable = tmp_path / "elsewhere" / "spacr"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))
    machine = cleanup._Machine(environ={}, fs_root=str(tmp_path), executable=str(executable))
    command, _, error = cleanup._helper_command([], str(tmp_path), "helper.py", "plan.json", machine)
    assert command is None
    assert "frozen application bundle is required" in error


def test_absent_archived_helper_source_returns_a_plan_error_without_spawning(tmp_path, monkeypatch):
    """Missing frozen source is an actionable refusal before any deletion."""
    root = tmp_path / "installed"
    root.mkdir()
    marker = root / "analysis.txt"
    marker.write_text("preserve")
    record = cleanup.InstallRecord(kind="installer", layout="linux-online", platform="linux",
                                   root=str(root), running=True)
    machine = cleanup._Machine(environ={}, fs_root=str(tmp_path), executable=str(root / "spacr"))
    monkeypatch.setattr(cleanup, "__file__", str(root / "archive" / "install_cleanup.pyc"))

    def no_spawn(*args, **kwargs):
        """Reject any attempt to continue an unusable frozen update plan."""
        pytest.fail("incomplete frozen updater was spawned")

    plan = cleanup.start_update_helper([record], "1.5.1.1", workdir=str(tmp_path / "helper"),
                                       system=machine, spawn=no_spawn)
    assert plan["command"] is None
    assert "standalone updater source is unavailable" in plan["error"]
    assert marker.read_text() == "preserve"
