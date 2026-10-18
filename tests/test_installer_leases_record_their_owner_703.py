"""Installer leases record their owner, and only a provably abandoned lease is taken over.

Maintainer decision 2026-10-18 (macOS native matrix): a killed installer left
.installing / .spacr-installer-lock behind and every later install said
"already running" forever. A lease now records pid, start time and host. A
later installer clears it only when that owner is provably gone on this host
and the lease holds nothing else; a live owner, a rollback, another host or a
lease that recorded no owner is never taken over.
"""
import os
from pathlib import Path
import platform
import select
import subprocess
import sys

import pytest

from spacr import install_cleanup as ic

ROOT = Path(__file__).resolve().parents[1]
UNIX = ROOT / "packaging/online/install_spacr_unix.sh"
WRAPPER = ROOT / "packaging/online/install_for_macos_user.sh"
WINDOWS = ROOT / "packaging/online/install_spacr_windows.ps1"
pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX lease contract; Windows is covered by source checks")
PLATFORM = "macos" if sys.platform == "darwin" else "linux"


def _lease_source():
    return UNIX.read_text().split("# @SPACR_INSTALLER_LEASE_BEGIN@\n", 1)[1].split("# @SPACR_INSTALLER_LEASE_END@", 1)[0]


def _take(tmp_path, lease, extra=""):
    script = tmp_path / "take.sh"
    script.write_text("set -Eeuo pipefail\nPLATFORM=" + PLATFORM + f"\nINSTALL_ROOT='{tmp_path / 'scratch-root'}'\n"
                      + "spacr_say() { printf '%s\\n' \"$*\" >&2; }\n"
                      + _lease_source() + extra + f'\nspacr_take_installer_lease "{lease}"\nprintf "acquired\\n"\n')
    return subprocess.run(["bash", str(script)], text=True, capture_output=True, timeout=20)


def _started(pid):
    return " ".join(subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True).stdout.split())


def _ended_pid():
    child = subprocess.Popen(["sleep", "0"])
    started = _started(child.pid)
    child.wait()
    return child.pid, started or "Thu Jan  1 00:00:00 1970"


def _stamp(lease, pid, started, host=None):
    lease.mkdir(mode=0o700)
    (lease / "owner").write_text(f"pid={pid}\nstarted={started}\nhost={host or platform.node()}\n")


def test_a_new_lease_records_its_owner(tmp_path):
    lease = tmp_path / ".app.spacr-installer-lock"
    result = _take(tmp_path, lease)
    assert result.returncode == 0, result.stderr
    fields = dict(line.split("=", 1) for line in (lease / "owner").read_text().splitlines())
    assert set(fields) == {"pid", "started", "host"} and fields["host"] == platform.node()
    assert fields["pid"].isdigit() and fields["started"]


def test_a_lease_whose_owner_ended_is_taken_over(tmp_path):
    lease = tmp_path / ".app.spacr-installer-lock"
    pid, started = _ended_pid()
    _stamp(lease, pid, started)
    result = _take(tmp_path, lease)
    assert result.returncode == 0, result.stderr
    assert "lease_cleared" in result.stderr
    assert (lease / "owner").read_text() != f"pid={pid}\nstarted={started}\nhost={platform.node()}\n"


def test_a_reused_pid_with_another_start_time_is_taken_over(tmp_path):
    lease = tmp_path / ".app.spacr-installer-lock"
    _stamp(lease, os.getpid(), "Mon Jan  1 00:00:00 2001")
    assert _take(tmp_path, lease).returncode == 0


def test_a_live_owner_is_never_taken_over_and_is_named(tmp_path):
    lease = tmp_path / ".app.spacr-installer-lock"
    holder = subprocess.Popen(["bash", "-c", "read -r line"], stdin=subprocess.PIPE, text=True)
    try:
        _stamp(lease, holder.pid, _started(holder.pid))
        before = (lease / "owner").read_text()
        result = _take(tmp_path, lease)
        assert result.returncode != 0
        assert "already_running" in result.stderr and f"lease_held {lease} {holder.pid}" in result.stderr
        assert (lease / "owner").read_text() == before
    finally:
        holder.communicate("done\n", timeout=10)


@pytest.mark.parametrize("evidence", ["rollback", "journal.json", "unknown-evidence"])
def test_a_retained_rollback_or_any_evidence_is_never_taken_over(tmp_path, evidence):
    lease = tmp_path / ".app.spacr-installer-lock"
    pid, started = _ended_pid()
    _stamp(lease, pid, started)
    (lease / evidence).write_text("preserve")
    result = _take(tmp_path, lease)
    assert result.returncode != 0
    assert (lease / evidence).read_text() == "preserve"


def test_another_hosts_lease_and_an_ownerless_lease_are_refused(tmp_path):
    other = tmp_path / ".other.spacr-installer-lock"
    pid, started = _ended_pid()
    _stamp(other, pid, started, host="another-computer")
    assert _take(tmp_path, other).returncode != 0
    legacy = tmp_path / ".legacy.spacr-installer-lock"
    legacy.mkdir()
    result = _take(tmp_path, legacy)
    assert result.returncode != 0 and f"lease_held_unknown {legacy}" in result.stderr
    assert legacy.is_dir()


def test_a_retained_rollback_marks_every_lease(tmp_path):
    lease = tmp_path / ".app.spacr-installer-lock"
    extra = ('\nSPACR_INSTALLER_LEASE="' + str(lease) + '"\nspacr_take_installer_lease "$SPACR_INSTALLER_LEASE"'
             '\nSPACR_INSTALLER_LEASE_ID="$(spacr_installer_lease_identity "$SPACR_INSTALLER_LEASE")"'
             '\nwork_dir=/retained/work\nSPACR_INSTALLER_RETAIN_LEASE=1\nspacr_release_installer_lease\nexit 0\n')
    result = _take(tmp_path, tmp_path / "unused", extra)
    assert result.returncode == 0, result.stderr
    assert (lease / "rollback").read_text() == "/retained/work\n"
    assert ic._lease_owner_dead(str(lease)) is False


def test_python_and_shell_record_the_same_owner_fields(tmp_path):
    lease = tmp_path / ".py.spacr-update-lock"
    ic._take_lease_dir(str(lease))
    fields = ic._lease_fields((lease / "owner").read_text())
    assert fields["pid"] == str(os.getpid()) and fields["host"] == platform.node()
    assert fields["started"] == _started(os.getpid())
    ic._release_lease_dir(str(lease))
    assert not lease.exists()


def test_python_takes_over_only_a_dead_owner(tmp_path):
    dead = tmp_path / ".dead.spacr-update-lock"
    pid, started = _ended_pid()
    _stamp(dead, pid, started)
    ic._take_lease_dir(str(dead))
    assert ic._lease_fields((dead / "owner").read_text())["pid"] == str(os.getpid())
    holder = subprocess.Popen(["bash", "-c", "read -r line"], stdin=subprocess.PIPE, text=True)
    try:
        live = tmp_path / ".live.spacr-update-lock"
        _stamp(live, holder.pid, _started(holder.pid))
        with pytest.raises(FileExistsError, match=str(holder.pid)):
            ic._take_lease_dir(str(live))
    finally:
        holder.communicate("done\n", timeout=10)
    retained = tmp_path / ".retained.spacr-update-lock"
    _stamp(retained, pid, started)
    ic._retain_lease_dir(str(retained), "rollback pending")
    with pytest.raises(FileExistsError):
        ic._take_lease_dir(str(retained))


def test_the_macos_first_run_check_lets_only_a_dead_owner_through(tmp_path):
    support = tmp_path / "support"
    support.mkdir()
    wrapper = support / "install-for-user.sh"
    wrapper.write_text(WRAPPER.read_text())
    (support / "installer_messages.sh").write_text('spacr_say() { printf "%s\\n" "$*"; }\n')
    backend = support / "install-online.sh"
    backend.write_text("#!/bin/sh\nprintf 'backend\\n'\n")
    backend.chmod(0o755)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    env = dict(os.environ, SPACR_USER_INSTALL_ROOT=str(runtime), SPACR_NO_RELAUNCH="1")
    pid, started = _ended_pid()
    _stamp(runtime / ".installing", pid, started)
    passed = subprocess.run(["/bin/sh", str(wrapper)], env=env, text=True, capture_output=True, timeout=20)
    assert passed.returncode == 0 and "backend" in passed.stdout
    (runtime / ".installing" / "rollback").write_text("pending")
    refused = subprocess.run(["/bin/sh", str(wrapper)], env=env, text=True, capture_output=True, timeout=20)
    assert refused.returncode == 75 and "lease_held" in refused.stdout


def test_the_windows_installer_follows_the_same_contract():
    source = WINDOWS.read_text()
    assert "function Test-SpacrLeaseOwnerDead" in source
    assert 'pid=$PID`nstarted=$(Get-SpacrProcessStarted $PID)`nhost=$env:COMPUTERNAME' in source
    assert '"rollback=$WorkDir`n"' in source
    assert '"filetime:" + $Process.StartTime.ToFileTimeUtc()' in source and "Get-Process" not in source
