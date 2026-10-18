"""Item 703: finders, removal leases and the macOS phase relay at their edges.

The finders see a running frozen bundle outside ``/Applications`` and the
all-users Start Menu of a real Windows computer. The removal lease releases
what it took even when a lock vanished or a handle cannot be closed, and an
unreadable installation folder is reported instead of removed. The macOS
phase relay stops on its deadline, on oversized or garbled helper output and
on a helper that will not exit, and reports the helper's own error text.

Everything below ``tmp_path``; the shell scripts run here are inert and
change nothing outside their own pipes.
"""
from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from spacr import install_cleanup as ic
from tests.test_install_cleanup import Box, linux_online
from tests.test_install_cleanup import (
    _nothing_here_reaches_this_computer as _nothing_here_reaches_this_computer,
)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX shell and descriptor fixtures")


def test_a_running_frozen_bundle_outside_applications_is_found(tmp_path, monkeypatch):
    box = Box(tmp_path)
    app = box.home / "Applications" / "spaCR.app"
    executable = box.write(app / "Contents" / "MacOS" / "spaCR")
    machine = box.machine("macos", executable=str(executable))
    monkeypatch.setattr(ic.sys, "frozen", True, raising=False)
    monkeypatch.setattr(ic.sys, "executable", str(executable))
    records = ic._find_macos_apps(machine)
    found = [record for record in records if str(app) in record.registrations]
    assert len(found) == 1 and found[0].root == str(app) and found[0].layout == "macos-app"
    other = box.write(box.home / "tools" / "MacOS" / "spaCR")
    monkeypatch.setattr(ic.sys, "executable", str(other))
    machine = box.machine("macos", executable=str(other))
    assert not [record for record in ic._find_macos_apps(machine)
                if str(box.home / "tools") in record.root]


def test_a_real_windows_computer_also_reads_the_all_users_start_menu(tmp_path, monkeypatch):
    program_files = tmp_path / "Program Files"
    root = program_files / "spaCR"
    (root / "Uninstall.exe").parent.mkdir(parents=True)
    (root / "spacr.exe").write_text("desktop")
    (root / "Uninstall.exe").write_text("uninstaller")
    common = tmp_path / "Common Programs"
    common.mkdir()
    (common / "spaCR.lnk").write_text(str(root))
    monkeypatch.setenv("PROGRAMFILES", str(program_files))
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData"))
    machine = ic._Machine(platform="windows", registry=SimpleNamespace(), packages=None)
    assert machine.real
    asked = []
    monkeypatch.setattr(ic, "_windows_known_folder",
                        lambda identifier: asked.append(identifier) or str(common))
    monkeypatch.setattr(ic, "_running", lambda machine, path: False)
    with monkeypatch.context() as patched:
        patched.setattr(ic.os, "name", "nt")
        records = ic._find_windows_offline(machine)
    assert asked == [0x17]
    assert [record.shortcuts for record in records] == [(str(common / "spaCR.lnk"),)]
    assert records[0].notes == () and records[0].needs_admin


def test_a_windows_launch_folder_is_checked_by_its_own_access_rules(tmp_path, monkeypatch):
    checked = []
    monkeypatch.setattr(ic, "_windows_pip_private_directory", lambda path: checked.append(path))
    with monkeypatch.context() as patched:
        patched.setattr(ic.os, "name", "nt")
        handshake = ic._UpdateLaunchHandshake(str(tmp_path), "c" * 64)
    assert checked == [str(tmp_path)] and handshake.root == str(tmp_path)


def _installer(box):
    root = linux_online(box, "1.5.0.7")
    return root, ic.InstallRecord(kind="installer", layout="linux-online", platform="linux",
                                  root=str(root))


def test_a_lock_that_cannot_be_inspected_is_released_and_reported(tmp_path, monkeypatch):
    box = Box(tmp_path)
    root, record = _installer(box)
    machine = box.machine("windows")
    real_fstat = os.fstat
    closed = []
    real_close = os.close
    monkeypatch.setattr(ic.os, "fstat", lambda fd: (_ for _ in ()).throw(OSError(5, "I/O error")))
    monkeypatch.setattr(ic.os, "close", lambda fd: closed.append(fd) or real_close(fd))
    with pytest.raises(ic._RemovalLeaseUnavailable):
        with ic._removal_transaction_lease(record, machine):
            pytest.fail("the lease was granted")
    monkeypatch.setattr(ic.os, "fstat", real_fstat)
    assert len(closed) == 1


def test_a_lock_removed_by_someone_else_and_an_unclosable_handle_are_tolerated(
        tmp_path, monkeypatch):
    box = Box(tmp_path)
    root, record = _installer(box)
    machine = box.machine("windows")
    lock = root.parent / ("." + root.name + ".spacr-installer-lock")
    real_close = os.close
    leaked = []

    def stubborn(fd):
        leaked.append(fd)
        raise OSError(9, "Bad file descriptor")

    with ic._removal_transaction_lease(record, machine) as lease:
        assert lock.is_file() and lease == {"remove_empty_root": False}
        os.unlink(lock)
        os.unlink(root / ".venv.spacr-update-lock")
        monkeypatch.setattr(ic.os, "close", stubborn)
    monkeypatch.setattr(ic.os, "close", real_close)
    for fd in leaked:
        real_close(fd)
    assert len(leaked) == 2 and not lock.exists() and root.is_dir()


def test_posix_locks_that_vanished_are_tolerated(tmp_path):
    box = Box(tmp_path)
    root, record = _installer(box)
    machine = box.machine("linux")
    lock = root.parent / ("." + root.name + ".spacr-installer-lock")
    with ic._removal_transaction_lease(record, machine):
        os.rmdir(lock)
    assert not lock.exists() and root.is_dir()


def test_an_installation_folder_that_cannot_be_listed_is_kept_and_reported(tmp_path, monkeypatch):
    box = Box(tmp_path)
    root, record = _installer(box)
    real_listdir = os.listdir

    def denied(path):
        if os.fspath(path) == str(root):
            raise PermissionError(13, "Permission denied", str(root))
        return real_listdir(path)

    monkeypatch.setattr(ic.os, "listdir", denied)
    report = ic._remove_install_unleased(record, system=box.machine("linux"))
    assert report.failed and report.failed[0][0] == str(root)
    assert "close anything using it" in report.failed[0][1]
    assert (root / "venv" / "bin" / "python").exists()


class _Phase:
    def __init__(self):
        self.sent = []

    def _send(self, phase, **details):
        self.sent.append(phase)

    def _receive(self, expected, **kwargs):
        return {"phase": "activate" if "activate" in expected else "confirmed"}


def test_the_relay_waits_for_slow_output_and_reports_the_helper_error(monkeypatch):
    monkeypatch.setattr(ic, "_wait_for_exit", lambda pid, **kwargs: True)
    script = 'sleep 0.5; printf "almost\\n"; echo "the journal is locked" >&2; exit 3'
    with pytest.raises(RuntimeError, match="The macOS transaction stopped: the journal is locked"):
        ic._macos_phased_transaction(script, _Phase(), "1.5.1.5", 4242)


@pytest.mark.parametrize("script", [
    "head -c 66000 /dev/zero | tr '\\0' a",
    "i=0; while [ $i -lt 3900 ]; do echo receipt-line-$i; i=$((i+1)); done",
])
def test_oversized_helper_output_is_refused(script):
    with pytest.raises(ValueError, match="invalid transaction receipt"):
        ic._macos_phased_transaction(script, _Phase(), "1.5.1.5", 4242)


def test_a_relay_past_its_deadline_stops(monkeypatch):
    clock = iter([0.0] + [20000.0] * 10)
    monkeypatch.setattr(ic.time, "monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError, match="did not finish before its deadline"):
        ic._macos_phased_transaction("exec cat > /dev/null", _Phase(), "1.5.1.5", 4242)


class _Child:
    """A real shell child whose stdin cannot be closed and that will not exit."""

    def __init__(self, argv, **options):
        self.process = subprocess.Popen(argv, **options)
        self.stdout = self.process.stdout
        self.stdin = SimpleNamespace(write=self.process.stdin.write,
                                     flush=self.process.stdin.flush, close=self._close)

    def _close(self):
        self.process.stdin.close()
        raise BrokenPipeError(32, "the helper already closed its input")

    def wait(self, timeout):
        self.process.wait(timeout=5)
        raise subprocess.TimeoutExpired("sh", timeout)


def test_a_helper_that_will_not_exit_retains_its_recovery_files():
    with pytest.raises(ic._UpdateActivationUnconfirmed, match="still running"):
        ic._macos_phased_transaction("printf done", _Phase(), "1.5.1.5", 4242, popen=_Child)
