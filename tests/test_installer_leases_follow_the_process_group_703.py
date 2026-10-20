"""A lease stays live while its installer's process group has a member, and abandoned staging is removed.

Found natively on an Intel Mac 2026-10-18: SIGKILL of the installer during `uv pip install` left uv, the log
tee and a subshell running, reparented to launchd and still writing into the install root. The owner pid was
provably gone, so the next installer took the leases over and two installers ran in one root and cache. A
lease now records the installer's process group; it is live while any member owned by this user survives,
other than the checking installer's own ancestors and descendants. A lease that recorded no group is judged
by its pid alone, as before. Once every lease is held, the staging runtime of an owner whose leases were just
taken over is removed.

The process table is faked through a `ps` on PATH, so these run the same way on Linux and macOS.
"""
import os
from pathlib import Path
import platform
import subprocess
import sys

import pytest

from spacr import install_cleanup as ic

ROOT = Path(__file__).resolve().parents[1]
UNIX = ROOT / "packaging/online/install_spacr_unix.sh"
WRAPPER = ROOT / "packaging/online/install_for_macos_user.sh"
pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
PLATFORM = "macos" if sys.platform == "darwin" else "linux"
REAL_PS = "/bin/ps" if os.path.exists("/bin/ps") else "/usr/bin/ps"
GROUP = 424242
UID = os.geteuid()


def _fake_ps(tmp_path, rows, *, broken=False, lineage_group=None):
    """A ps that answers group queries from ``rows`` and passes everything else to the real ps.

    ``rows`` are (pid, ppid, pgid, uid) tuples. With ``lineage_group`` the table also lists every ancestor of
    the process that ran ps in that group, as a real process table would.
    """
    table = tmp_path / "ps-table"
    table.write_text("".join(f"{a} {b} {c} {d}\n" for a, b, c, d in rows))
    tool = tmp_path / "bin"
    tool.mkdir(exist_ok=True)
    ps = tool / "ps"
    ps.write_text(f"""#!/bin/sh
case "$*" in
  "-A -o pid=,ppid=,pgid=,uid=")
    {"exit 1" if broken else ""}
    cat "{table}"
    if [ -n "$LINEAGE_GROUP" ]; then
        p="$PPID"
        while [ -n "$p" ] && [ "$p" != 0 ] && [ "$p" != 1 ]; do
            q="$({REAL_PS} -o ppid= -p "$p" | tr -d ' ')"
            printf '%s %s %s %s\n' "$p" "${{q:-1}}" "$LINEAGE_GROUP" "{UID}"
            p="$q"
        done
    fi
    exit 0 ;;
  "-o pgid= -p "*) printf '{GROUP}\\n'; exit 0 ;;
esac
exec {REAL_PS} "$@"
""")
    ps.chmod(0o755)
    return dict(os.environ, PATH=f"{tool}:{os.environ['PATH']}", LINEAGE_GROUP=str(lineage_group or ""))


def _lease_source():
    return UNIX.read_text(encoding="utf-8").split("# @SPACR_INSTALLER_LEASE_BEGIN@\n", 1)[1].split("# @SPACR_INSTALLER_LEASE_END@", 1)[0]


def _take(tmp_path, lease, env, root=None):
    root = root or tmp_path / "scratch-root"
    script = tmp_path / "take.sh"
    script.write_text("set -Eeuo pipefail\nPLATFORM=" + PLATFORM + f"\nINSTALL_ROOT='{root}'\n"
                      + "spacr_say() { printf '%s\\n' \"$*\" >&2; }\n"
                      + _lease_source() + f'\nspacr_take_installer_lease "{lease}"\nprintf "acquired\\n"\n')
    return subprocess.run(["bash", str(script)], text=True, capture_output=True, timeout=20, env=env)


def _ended_pid():
    child = subprocess.Popen(["sleep", "0"])
    started = ic._process_started(child.pid)
    child.wait()
    return child.pid, started


def _stamp(lease, pid, started, group=GROUP):
    lease.mkdir(mode=0o700)
    text = f"pid={pid}\nstarted={started}\nhost={platform.node()}\npidns={ic._pid_namespace()}\n"
    if group is not None:
        text += f"pgid={group}\n"
    (lease / "owner").write_text(text)


def test_a_new_lease_records_its_process_group(tmp_path):
    env = _fake_ps(tmp_path, [])
    lease = tmp_path / ".app.spacr-installer-lock"
    assert _take(tmp_path, lease, env).returncode == 0
    assert f"pgid={GROUP}" in (lease / "owner").read_text(encoding="utf-8").splitlines()


def test_a_surviving_child_keeps_the_lease_and_is_named(tmp_path):
    pid, started = _ended_pid()
    lease = tmp_path / ".app.spacr-installer-lock"
    _stamp(lease, pid, started)
    env = _fake_ps(tmp_path, [(4242, 1, GROUP, UID)])
    result = _take(tmp_path, lease, env)
    assert result.returncode != 0
    assert f"lease_held {lease} 4242" in result.stderr
    assert (lease / "owner").read_text(encoding="utf-8").startswith(f"pid={pid}\n")


def test_an_empty_group_is_taken_over_and_its_staging_removed(tmp_path):
    pid, started = _ended_pid()
    root = tmp_path / "root"
    root.mkdir()
    (root / f".venv-staging-{pid}" / "bin").mkdir(parents=True)
    (root / f".install-profile-staging-{pid}.json").write_text("{}")
    live = os.getpid()
    (root / f".venv-staging-{live}").mkdir()
    _stamp(root.parent / f".{root.name}.spacr-installer-lock", pid, started)
    _stamp(root / ".venv.spacr-update-lock", pid, started)
    if PLATFORM == "macos":
        _stamp(root / ".installing", pid, started)
    env = _fake_ps(tmp_path, [(777, 1, GROUP + 1, UID)])
    script = tmp_path / "acquire.sh"
    script.write_text("set -Eeuo pipefail\nPLATFORM=" + PLATFORM + f"\nINSTALL_ROOT='{root}'\n"
                      + "spacr_say() { printf '%s\\n' \"$*\" >&2; }\n" + _lease_source() + "\nexit 0\n")
    result = subprocess.run(["bash", str(script)], text=True, capture_output=True, timeout=20, env=env)
    assert result.returncode == 0, result.stderr
    assert "lease_cleared" in result.stderr and f"staging_removed {root}/.venv-staging-{pid}" in result.stderr
    assert not (root / f".venv-staging-{pid}").exists()
    assert not (root / f".install-profile-staging-{pid}.json").exists()
    assert (root / f".venv-staging-{live}").is_dir()


def test_another_users_process_in_the_group_does_not_count(tmp_path):
    pid, started = _ended_pid()
    lease = tmp_path / ".app.spacr-installer-lock"
    _stamp(lease, pid, started)
    env = _fake_ps(tmp_path, [(4242, 1, GROUP, UID + 1)])
    assert _take(tmp_path, lease, env).returncode == 0


def test_an_unreadable_process_table_refuses(tmp_path):
    pid, started = _ended_pid()
    lease = tmp_path / ".app.spacr-installer-lock"
    _stamp(lease, pid, started)
    env = _fake_ps(tmp_path, [], broken=True)
    assert _take(tmp_path, lease, env).returncode != 0
    assert lease.is_dir()


def test_a_lease_that_recorded_no_group_is_judged_by_its_pid(tmp_path):
    pid, started = _ended_pid()
    lease = tmp_path / ".app.spacr-installer-lock"
    _stamp(lease, pid, started, group=None)
    env = _fake_ps(tmp_path, [(4242, 1, GROUP, UID)])
    assert _take(tmp_path, lease, env).returncode == 0


def test_the_checking_installer_and_its_own_lineage_do_not_count(tmp_path):
    """Two installers run one after another from a shell without job control share a group."""
    pid, started = _ended_pid()
    lease = tmp_path / ".app.spacr-installer-lock"
    _stamp(lease, pid, started)
    env = _fake_ps(tmp_path, [(777, 1, GROUP + 1, UID)], lineage_group=GROUP)
    assert _take(tmp_path, lease, env).returncode == 0


def test_python_follows_the_same_group_rule(tmp_path, monkeypatch):
    pid, started = _ended_pid()
    monkeypatch.setattr(ic, "_OWN_LEASE_RECORD", {})
    monkeypatch.setattr(ic, "_process_started", lambda p: started if int(p) == pid else "x:1")
    lease = tmp_path / ".py.spacr-update-lock"
    _stamp(lease, pid, started)
    monkeypatch.setenv("PATH", _fake_ps(tmp_path, [(4242, 1, GROUP, UID)])["PATH"])
    assert ic._lease_owner_dead(str(lease)) is False
    monkeypatch.setenv("PATH", _fake_ps(tmp_path, [(4242, 1, GROUP, UID + 1)])["PATH"])
    assert ic._lease_owner_dead(str(lease)) is True
    monkeypatch.setenv("PATH", _fake_ps(tmp_path, [], broken=True)["PATH"])
    assert ic._lease_owner_dead(str(lease)) is False
    legacy = tmp_path / ".legacy.spacr-update-lock"
    _stamp(legacy, pid, started, group=None)
    assert ic._lease_owner_dead(str(legacy)) is True


def test_python_records_its_group():
    assert f"pgid={os.getpgrp()}" in ic._lease_owner_text().splitlines()


def test_the_macos_first_run_check_follows_the_group_too(tmp_path):
    support = tmp_path / "support"
    support.mkdir()
    wrapper = support / "install-for-user.sh"
    wrapper.write_text(WRAPPER.read_text(encoding="utf-8"))
    (support / "installer_messages.sh").write_text('spacr_say() { printf "%s\\n" "$*"; }\n')
    backend = support / "install-online.sh"
    backend.write_text("#!/bin/sh\nprintf 'backend\\n'\n")
    backend.chmod(0o755)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    pid, started = _ended_pid()
    _stamp(runtime / ".installing", pid, started)
    for rows, expected in (([(4242, 1, GROUP, UID)], 75), ([(777, 1, GROUP + 1, UID)], 0)):
        env = dict(_fake_ps(tmp_path, rows), SPACR_USER_INSTALL_ROOT=str(runtime), SPACR_NO_RELAUNCH="1")
        result = subprocess.run(["/bin/sh", str(wrapper)], env=env, text=True, capture_output=True, timeout=20)
        assert result.returncode == expected, result.stdout + result.stderr


def test_a_parent_cycle_in_the_process_table_ends_and_refuses(tmp_path):
    """ps does not snapshot atomically; a reused pid can make two rows each other's parent. The walk ends and
    the member counts as live."""
    pid, started = _ended_pid()
    lease = tmp_path / ".app.spacr-installer-lock"
    _stamp(lease, pid, started)
    env = _fake_ps(tmp_path, [(5001, 5002, GROUP, UID), (5002, 5001, GROUP + 7, UID)])
    result = _take(tmp_path, lease, env)
    assert result.returncode != 0 and "5001" in result.stderr
    assert lease.is_dir()
