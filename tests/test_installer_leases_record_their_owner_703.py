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


def _take(tmp_path, lease, extra="", env=None):
    script = tmp_path / "take.sh"
    script.write_text("set -Eeuo pipefail\nPLATFORM=" + PLATFORM + f"\nINSTALL_ROOT='{tmp_path / 'scratch-root'}'\n"
                      + "spacr_say() { printf '%s\\n' \"$*\" >&2; }\n"
                      + _lease_source() + extra + f'\nspacr_take_installer_lease "{lease}"\nprintf "acquired\\n"\n')
    return subprocess.run(["bash", str(script)], text=True, capture_output=True, timeout=20, env=env)


def _started(pid):
    return ic._process_started(pid)


#: The stamp kind this system records ("ticks" on Linux, "lstart" elsewhere), and one of the other kind.
KIND = (_started(os.getpid()) or "ticks:0").partition(":")[0]
OTHER_KIND = "lstart:Mon Jan  1 00:00:00 2001" if KIND == "ticks" else "ticks:1"


def _ended_pid():
    child = subprocess.Popen(["sleep", "0"])
    started = _started(child.pid)
    child.wait()
    return child.pid, started or KIND + ":1"


def _stamp(lease, pid, started, host=None, pidns=None):
    lease.mkdir(mode=0o700)
    namespace = ic._pid_namespace() if pidns is None else pidns
    (lease / "owner").write_text(f"pid={pid}\nstarted={started}\nhost={host or platform.node()}\npidns={namespace}\n")


def test_a_new_lease_records_its_owner(tmp_path):
    lease = tmp_path / ".app.spacr-installer-lock"
    result = _take(tmp_path, lease)
    assert result.returncode == 0, result.stderr
    fields = dict(line.split("=", 1) for line in (lease / "owner").read_text().splitlines())
    assert set(fields) - {"pgid"} == {"pid", "started", "host", "pidns"} and fields["host"] == platform.node()
    assert fields["pid"].isdigit() and fields["started"].startswith(KIND + ":")
    assert fields["pidns"] == ic._pid_namespace()


def test_a_lease_whose_owner_ended_is_taken_over(tmp_path):
    lease = tmp_path / ".app.spacr-installer-lock"
    pid, started = _ended_pid()
    _stamp(lease, pid, started)
    result = _take(tmp_path, lease)
    assert result.returncode == 0, result.stderr
    assert "lease_cleared" in result.stderr
    assert ic._lease_fields((lease / "owner").read_text())["pid"] != str(pid)
    assert not Path(str(lease) + ".takeover").exists()


def test_a_reused_pid_with_another_start_time_is_taken_over(tmp_path):
    lease = tmp_path / ".app.spacr-installer-lock"
    _stamp(lease, os.getpid(), KIND + ":1")
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
    assert fields["started"] == _started(os.getpid()) and fields["pidns"] == ic._pid_namespace()
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


def _holder():
    return subprocess.Popen(["bash", "-c", "read -r line"], stdin=subprocess.PIPE, text=True)


def test_a_live_owner_checked_from_another_locale_and_time_zone_is_never_taken_over(tmp_path):
    """The start stamp must not move with LANG or TZ, or a live installer would look like a reused pid."""
    lease = tmp_path / ".app.spacr-installer-lock"
    holder = _holder()
    try:
        _stamp(lease, holder.pid, _started(holder.pid))
        env = dict(os.environ, LANG="de_DE.UTF-8", LC_ALL="de_DE.UTF-8", TZ="Pacific/Kiritimati")
        result = _take(tmp_path, lease, env=env)
        assert result.returncode != 0 and f"lease_held {lease} {holder.pid}" in result.stderr
        assert ic._lease_fields((lease / "owner").read_text())["pid"] == str(holder.pid)
    finally:
        holder.communicate("done\n", timeout=10)


def test_a_live_owner_stamped_another_way_is_never_taken_over(tmp_path):
    lease = tmp_path / ".app.spacr-installer-lock"
    holder = _holder()
    try:
        _stamp(lease, holder.pid, OTHER_KIND)
        assert _take(tmp_path, lease).returncode != 0
        assert ic._lease_owner_dead(str(lease)) is False
    finally:
        holder.communicate("done\n", timeout=10)


def test_a_lease_from_another_pid_namespace_is_refused(tmp_path):
    """A container that shares this host name sees other pids; its lease is never judged from here."""
    lease = tmp_path / ".app.spacr-installer-lock"
    pid, started = _ended_pid()
    _stamp(lease, pid, started, pidns="pid:[4026532999]")
    assert _take(tmp_path, lease).returncode != 0
    assert ic._lease_owner_dead(str(lease)) is False
    assert ic._lease_fields((lease / "owner").read_text())["pid"] == str(pid)


def test_a_takeover_in_progress_makes_a_second_installer_refuse(tmp_path):
    lease = tmp_path / ".app.spacr-installer-lock"
    pid, started = _ended_pid()
    _stamp(lease, pid, started)
    guard = Path(str(lease) + ".takeover")
    guard.mkdir(mode=0o700)
    result = _take(tmp_path, lease)
    assert result.returncode != 0 and guard.is_dir()
    assert ic._lease_fields((lease / "owner").read_text())["pid"] == str(pid)
    with pytest.raises(FileExistsError, match="takeover"):
        ic._take_lease_dir(str(lease))
    assert ic._lease_fields((lease / "owner").read_text())["pid"] == str(pid)


def test_python_refuses_another_users_lease_and_an_unreadable_process(tmp_path, monkeypatch):
    pid, started = _ended_pid()
    lease = tmp_path / ".dead.spacr-update-lock"
    _stamp(lease, pid, started)
    assert ic._lease_owner_dead(str(lease)) is True
    real = os.geteuid()
    monkeypatch.setattr(ic.os, "geteuid", lambda: real + 1)
    assert ic._lease_owner_dead(str(lease)) is False
    monkeypatch.setattr(ic.os, "geteuid", lambda: real)
    ic._lease_owner_text()
    monkeypatch.setattr(ic, "_process_gone", lambda pid: False)
    monkeypatch.setattr(ic, "_process_started", lambda pid: None)
    assert ic._lease_owner_dead(str(lease)) is False


def test_a_signal_the_system_refuses_is_not_proof_that_a_process_is_gone(monkeypatch):
    def refuse(pid, signal):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(ic.os, "kill", refuse)
    assert ic._process_gone(12345) is False


def test_an_open_windows_lease_file_is_never_taken_over(tmp_path, monkeypatch):
    pid, started = _ended_pid()
    lease = tmp_path / ".app.spacr-installer-lock"
    record = "".join(line + "\n" for line in ic._lease_owner_text().splitlines() if not line.startswith("pgid="))
    lease.write_text(record.replace(f"pid={os.getpid()}", f"pid={pid}"))
    assert ic._lease_owner_dead(str(lease)) is True
    monkeypatch.setattr(ic, "_lease_file_in_use", lambda path: True)
    assert ic._lease_owner_dead(str(lease)) is False


def _future(lease, pid, started):
    """A lease from a later installer: everything this version records, plus a field it does not know."""
    _stamp(lease, pid, started)
    with open(lease / "owner", "a") as handle:
        handle.write("futurefield=1\n")


def test_a_later_installers_lease_with_a_dead_pid_names_no_holder(tmp_path):
    """Found on a Mac 2026-10-19: 1.5.1.5 refused a 1.5.1.6 lease (pgid=) forever, naming its dead pid."""
    lease = tmp_path / ".app.spacr-installer-lock"
    pid, started = _ended_pid()
    _future(lease, pid, started)
    result = _take(tmp_path, lease)
    assert result.returncode != 0 and f"lease_held_unknown {lease}" in result.stderr
    assert f"lease_held {lease}" not in result.stderr and (lease / "owner").is_file()
    assert ic._lease_owner_dead(str(lease)) is False
    message = ic._lease_held_message(str(lease))
    assert str(pid) not in message and "remove that folder" in message


def test_a_later_installers_lease_with_a_live_pid_names_it(tmp_path):
    lease = tmp_path / ".app.spacr-installer-lock"
    holder = _holder()
    try:
        _future(lease, holder.pid, _started(holder.pid))
        result = _take(tmp_path, lease)
        assert result.returncode != 0 and f"lease_held {lease} {holder.pid}" in result.stderr
        assert ic._lease_owner_dead(str(lease)) is False
        assert str(holder.pid) in ic._lease_held_message(str(lease))
    finally:
        holder.communicate("done\n", timeout=10)


@pytest.mark.parametrize("alive", [False, True])
def test_the_macos_first_run_check_reports_a_later_lease_the_same_way(tmp_path, alive):
    support = tmp_path / "support"
    support.mkdir()
    wrapper = support / "install-for-user.sh"
    wrapper.write_text(WRAPPER.read_text())
    (support / "installer_messages.sh").write_text('spacr_say() { printf "%s\\n" "$*"; }\n')
    (support / "install-online.sh").write_text("#!/bin/sh\nprintf 'backend\\n'\n")
    (support / "install-online.sh").chmod(0o755)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    env = dict(os.environ, SPACR_USER_INSTALL_ROOT=str(runtime), SPACR_NO_RELAUNCH="1")
    holder = _holder() if alive else None
    try:
        pid, started = (holder.pid, _started(holder.pid)) if alive else _ended_pid()
        _future(runtime / ".installing", pid, started)
        refused = subprocess.run(["/bin/sh", str(wrapper)], env=env, text=True, capture_output=True, timeout=20)
        assert refused.returncode == 75 and "backend" not in refused.stdout
        if alive:
            assert f"lease_held {runtime / '.installing'} {pid}" in refused.stdout
        else:
            assert f"lease_held_unknown {runtime / '.installing'}" in refused.stdout
    finally:
        if holder is not None:
            holder.communicate("done\n", timeout=10)


def test_the_windows_installer_names_no_dead_holder_for_a_later_lease():
    source = WINDOWS.read_text()
    assert "function Test-SpacrLeaseHasUnknownFields" in source
    assert "'^(pid|started|host|tree|rollback)='" in source
    assert '-and (Test-SpacrLeaseHasUnknownFields $LeasePath)) { $Owner = "" }' in source
