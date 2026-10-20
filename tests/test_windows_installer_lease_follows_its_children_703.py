"""A Windows installer lease stays live while a process its owner started still runs.

1.5.1.6 design: killing the PowerShell installer (Task Manager, a closed console) does not end its uv, pip or
Python children, and Windows keeps the dead parent's id in their records. The installer now runs inside a
kill-on-close Job Object, so Windows ends those children with it, and every Windows lease records tree=1: it
is taken over only when the owner is gone and no process whose parent id is the owner's, created at or after
the owner, is alive. An unreadable process table, a child stamped another way, or an unknown tree value
refuses. A lease from 1.5.1.5, which recorded no tree, is judged by its pid as before.

These run on any system: the Windows owner record, process table and stamps are fakes.
"""
from pathlib import Path

import pytest

from spacr import install_cleanup as ic

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "packaging/online/install_spacr_windows.ps1"
OWNER = 4100
STARTED = "filetime:133000000000000000"


@pytest.fixture
def windows(monkeypatch):
    """Judge leases the way a Windows helper does, with a fake process table."""
    record = f"pid={OWNER + 1}\nstarted=filetime:1\nhost=WINHOST\ntree=1\n"
    monkeypatch.setattr(ic, "_lease_owner_text", lambda: record)
    monkeypatch.setattr(ic, "_lease_file_in_use", lambda path: False)
    monkeypatch.setattr(ic, "_process_gone", lambda pid: True)
    table = {"children": []}
    monkeypatch.setattr(ic, "_windows_children", lambda pid: table["children"] if pid == OWNER else [])
    return table


def _lease(tmp_path, text):
    path = tmp_path / ".app.spacr-installer-lock"
    path.write_text(text)
    return str(path)


def _owner(tree="tree=1\n"):
    return f"pid={OWNER}\nstarted={STARTED}\nhost=WINHOST\n" + tree


def test_a_dead_owner_with_no_children_is_taken_over(tmp_path, windows):
    assert ic._lease_owner_dead(_lease(tmp_path, _owner())) is True


def test_a_surviving_child_keeps_the_lease(tmp_path, windows):
    windows["children"] = [(5200, "filetime:133000000000000500")]
    assert ic._lease_owner_dead(_lease(tmp_path, _owner())) is False


def test_a_child_older_than_the_owner_belonged_to_an_earlier_holder_of_its_pid(tmp_path, windows):
    windows["children"] = [(5200, "filetime:132999999999999999")]
    assert ic._lease_owner_dead(_lease(tmp_path, _owner())) is True


@pytest.mark.parametrize("children", [None, [(5200, "ticks:5")], [(5200, "filetime:x")]])
def test_an_unreadable_table_or_another_stamp_refuses(tmp_path, windows, children):
    windows["children"] = children
    assert ic._lease_owner_dead(_lease(tmp_path, _owner())) is False


@pytest.mark.parametrize("tree", ["tree=0\n", "tree=job\n", "tree=1\npgid=7\n"])
def test_an_unknown_tree_value_refuses(tmp_path, windows, tree):
    assert ic._lease_owner_dead(_lease(tmp_path, _owner(tree))) is False


def test_a_lease_from_before_the_tree_record_is_judged_by_its_pid(tmp_path, windows):
    windows["children"] = [(5200, "filetime:133000000000000500")]
    assert ic._lease_owner_dead(_lease(tmp_path, _owner(tree=""))) is True


def test_a_live_owner_is_never_taken_over_whatever_its_children(tmp_path, windows, monkeypatch):
    monkeypatch.setattr(ic, "_process_gone", lambda pid: False)
    monkeypatch.setattr(ic, "_process_started", lambda pid: STARTED)
    assert ic._lease_owner_dead(_lease(tmp_path, _owner())) is False


def test_the_owner_stamp_kind_must_be_filetime(tmp_path, windows):
    lease = _lease(tmp_path, f"pid={OWNER}\nstarted=ticks:5\nhost=WINHOST\ntree=1\n")
    assert ic._lease_owner_dead(lease) is False


def test_children_are_unknown_off_windows():
    if ic.os.name == "nt":
        pytest.skip("reads the real Windows process table")
    assert ic._windows_children(1) is None
    assert ic._owner_children_live(1, STARTED) is None


@pytest.mark.skipif(ic.os.name != "nt", reason="reads the real Windows process table")
def test_an_exited_process_still_referenced_by_a_handle_is_gone():
    """Found natively 2026-10-19: a killed update helper whose process object stayed alive (another process
    still held a handle to it) kept its runtime lease forever, because OpenProcess and GetProcessTimes succeed
    on an exited process. Its exit code decides."""
    import subprocess
    import sys

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(timeout=60)
    assert ic._process_started(child.pid) == ""
    assert ic._process_gone(child.pid) is True
    assert ic._process_started(ic.os.getpid()).startswith("filetime:")
    assert ic._process_gone(ic.os.getpid()) is False


def test_the_powershell_installer_follows_the_same_rule():
    source = WINDOWS.read_text(encoding="utf-8")
    assert "function Enable-SpacrInstallerJob" in source and "$null = Enable-SpacrInstallerJob" in source
    assert source.index("$null = Enable-SpacrInstallerJob") < source.index("Open-SpacrInstallerLease $OuterLease")
    assert "[int]0x2000" in source and "SetInformationJobObject($Job, 9," in source
    assert 'host=$env:COMPUTERNAME`ntree=1`n' in source
    assert "ParentProcessId = $OwnerId" in source and "-ge $Since" in source
    assert 'if ($Tree -ne "1") { return $false }' in source
    assert "($false -eq (Test-SpacrOwnerChildrenLive" in source


def test_the_powershell_installer_sweeps_only_what_a_taken_over_owner_left():
    """Mirrors spacr_remove_abandoned_staging: a killed installer's staging runtime, staging profile and TEMP
    work folder go only for an owner whose lease was just taken over, after every lease is held, and only while
    that owner is still gone. Found natively on Windows 11 2026-10-19: four killed installs left four of each."""
    source = WINDOWS.read_text(encoding="utf-8")
    take_over = source.index("if (Test-SpacrLeaseOwnerDead $LeasePath) {")
    assert source.index("Add-SpacrAbandonedOwner $LeasePath", take_over) < source.index(
        "Remove-Item -LiteralPath $LeasePath -Force", take_over)
    assert source.count("Add-SpacrAbandonedOwner $LeasePath") == 1
    runtime_lease = source.index("Open-SpacrInstallerLease $RuntimeLease")
    sweep = source.index("    Remove-SpacrAbandonedStaging\n")
    assert runtime_lease < sweep < source.index("$InstallerStarted = $true")
    body = source[source.index("function Remove-SpacrAbandonedStaging"):]
    body = body[:body.index("\n}\n")]
    assert "if ($OwnerId -eq $PID) { continue }" in body
    assert '($Now -ne "" -and $Now -eq $Owner.Started)' in body
    assert 'Remove-SpacrAbandonedPath $InstallRoot (".venv-staging-" + $OwnerId)' in body
    assert 'Remove-SpacrAbandonedPath $InstallRoot (".install-profile-staging-" + $OwnerId + ".json")' in body
    assert 'Remove-SpacrAbandonedPath $env:TEMP ("spacr-install-" + $OwnerId)' in body
    remover = source[source.index("function Remove-SpacrAbandonedPath"):]
    remover = remover[:remover.index("\n}\n")]
    assert "GetDirectoryName($Path).TrimEnd(\"\\\") -ne $Base) { return }" in remover
    assert "[System.IO.FileAttributes]::ReparsePoint) { return }" in remover
    assert remover.index("ReparsePoint) { return }") < remover.index("Remove-SpacrTree $Path")
    tree = source[source.index("function Remove-SpacrTree"):]
    tree = tree[:tree.index("\n}\n")]
    assert "Delete($Path, $true)" not in source and "Delete($Folder, $true)" not in tree
    link = tree.index("if ($Entry.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {")
    assert link < tree.index("[System.IO.Directory]::Delete($Entry.FullName, $false)") < tree.index(
        "} else {", link) < tree.index("Remove-SpacrTree $Entry.FullName")
    assert tree.count("Remove-SpacrTree $Entry.FullName") == 1
    assert tree.rstrip().endswith("[System.IO.Directory]::Delete($Folder, $false)")
    assert 'Get-SpacrInstallerMessage "staging_removed"' in remover
    assert "'^[1-9][0-9]*$'" in source[source.index("function Add-SpacrAbandonedOwner"):]
