"""Item 703: private runtime staging, recovery and Debian replacement guards refuse before mutation."""
import errno
import json
import os
import stat
import subprocess
import sys
import time
import types
from dataclasses import asdict

import pytest

from spacr import install_cleanup as ic
from tests.test_703_cleanup_a_phase_channel import _Proxy
from tests.test_debian_update_transaction_703 import Packages
from tests.test_macos_runtime_transaction_703 import _snapshot
from tests.test_pip_runtime_transaction_703 import installation  # noqa: F401
from tests.test_retained_runtime_recovery_703 import updated  # noqa: F401


def _machine(tmp_path, platform="linux"):
    return ic._Machine(platform=platform, environ={"HOME": str(tmp_path / "user")}, fs_root=str(tmp_path))


def _fake_runtime(tmp_path):
    prefix = tmp_path / "runtime" / "venv"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "lib").mkdir()
    (prefix / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    (prefix / "bin" / "python").write_text("#!/bin/sh\n")
    (prefix / "lib" / "module.py").write_text("VALUE = 1\n")
    record = ic.InstallRecord(kind="environment", layout="venv", platform="linux", root=str(prefix),
                              python=str(prefix / "bin" / "python"), version="1.0.0", running=True)
    return prefix, record


@pytest.mark.parametrize("reply", [(1, "probe failed"), (0, '{"verified": false, "version": "1.0.0"}'),
                                   (0, "not json"), (0, "[1, 2]")])
def test_retiring_identity_refuses_unverified_probe_output(tmp_path, monkeypatch, reply):
    calls = []
    monkeypatch.setattr(ic, "_run", lambda argv, timeout: calls.append(argv) or reply)
    with pytest.raises(ValueError, match="previous_runtime_identity_unverified"):
        ic._retiring_runtime_identity(str(tmp_path))
    assert calls and calls[0][0] == os.path.join(str(tmp_path), "bin", "python")


def test_retiring_identity_accepts_verified_probe(tmp_path, monkeypatch):
    identity = {"verified": True, "version": "1.2.3", "prefix": str(tmp_path)}
    monkeypatch.setattr(ic, "_run", lambda argv, timeout: (0, json.dumps(identity)))
    assert ic._retiring_runtime_identity(str(tmp_path)) == identity


def _macos_root(tmp_path):
    root = tmp_path / "spaCR online"
    for relative in ("venv/bin/python", "bootstrap/uv"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\n")
        path.chmod(0o755)
    return root


def test_macos_online_commands_require_the_private_interpreter(tmp_path):
    root = _macos_root(tmp_path)
    record = ic.InstallRecord(kind="installer", layout="macos-online", platform="macos", root=str(root))
    with pytest.raises(ValueError, match="does not match its private Python"):
        ic._macos_online_commands(record, "2.0.0", str(tmp_path / "work"))


def test_macos_online_commands_refuse_workdir_inside_the_environment(tmp_path):
    root = _macos_root(tmp_path)
    record = ic.InstallRecord(kind="installer", layout="macos-online", platform="macos", root=str(root),
                              python=str(root / "venv" / "bin" / "python"))
    with pytest.raises(ValueError, match="outside the installed environment"):
        ic._macos_online_commands(record, "2.0.0", str(root / "updater"))


def test_runtime_staging_accepts_relative_path_configuration_files(tmp_path):
    prefix, _record = _fake_runtime(tmp_path)
    (prefix / "lib" / "local.pth").write_text("../shared-packages\n")
    assert ic._runtime_staging_allowed(str(prefix), _machine(tmp_path)) is None
    (prefix / "lib" / "absolute.pth").write_text(os.path.realpath(prefix) + "/lib\n")
    with pytest.raises(ValueError, match="absolute links"):
        ic._runtime_staging_allowed(str(prefix), _machine(tmp_path))


def test_recovery_refuses_a_journal_readable_by_others(updated):
    prefix, transaction, machine, _original, run = updated
    (transaction / "transaction.json").chmod(0o644)
    before = _snapshot(prefix.parent)
    with pytest.raises(ValueError, match="owner-only journal"):
        ic._recover_runtime_update(str(transaction), 43210, machine=machine, runner=run, wait=lambda pid: True)
    assert _snapshot(prefix.parent) == before


@pytest.mark.parametrize("pid", [0, True, "43210"])
def test_recovery_requires_a_real_previous_desktop_process(updated, pid):
    prefix, transaction, machine, _original, run = updated
    before = _snapshot(prefix.parent)
    with pytest.raises(ValueError, match="previous desktop process identity"):
        ic._recover_runtime_update(str(transaction), pid, machine=machine, runner=run, wait=lambda pid: True)
    assert _snapshot(prefix.parent) == before


def test_recovery_preserves_both_runtimes_when_retained_runtime_is_unverified(updated):
    prefix, transaction, machine, _original, _run = updated
    before = _snapshot(prefix), _snapshot(transaction / "runtime")
    probes = []
    with pytest.raises(RuntimeError, match="retained runtime could not be verified"):
        ic._recover_runtime_update(str(transaction), 43210, machine=machine,
                                   runner=lambda argv: probes.append(argv) or (1, "refused"),
                                   wait=lambda pid: True)
    assert (_snapshot(prefix), _snapshot(transaction / "runtime")) == before
    assert len(probes) == 1 and "-B" in probes[0]
    assert json.loads((transaction / "transaction.json").read_text())["state"] == "committed"
    assert not (prefix.parent / ("." + prefix.name + ".spacr-update-lock")).exists()


def test_recovery_journal_removes_its_temporary_file_when_replacement_fails(tmp_path, monkeypatch):
    def refuse(_source, _target):
        raise OSError(errno.EROFS, "read-only")

    monkeypatch.setattr(ic, "os", _Proxy(os, replace=refuse))
    with pytest.raises(OSError):
        ic._runtime_recovery_journal(str(tmp_path), {"state": "staging"})
    assert list(tmp_path.iterdir()) == []


def test_saved_wgpu_omission_accepts_uv_environment_banner():
    output = ("Using Python 3.11.9 environment at: /private/venv\nChecked 42 packages in 3ms\n"
              "Found 1 incompatibility\nThe package `spacr` requires `wgpu>=0.19`, but it's not installed\n")
    assert ic._only_saved_wgpu_omission(1, output) is True
    assert ic._only_saved_wgpu_omission(1, output.replace("Found 1 incompatibility", "Found 2 incompatibilities")) is False


def _lock(prefix):
    return prefix.parent / ("." + prefix.name + ".spacr-update-lock")


@pytest.mark.parametrize("identity,message", [(None, "previous_runtime_identity_unverified"),
                                              ({"version": "3.0.0"}, "newer than the requested release")])
def test_runtime_update_refuses_unverified_or_newer_current_runtime(tmp_path, monkeypatch, identity, message):
    prefix, record = _fake_runtime(tmp_path)
    monkeypatch.setattr(ic, "_retiring_runtime_identity", lambda path: identity)
    calls = []
    with pytest.raises(ValueError, match=message):
        ic._transactional_runtime_update(record, "2.0.0", [record.python, "-I", "-m", "pip", "install", "--upgrade",
                                                           "spacr==2.0.0"], machine=_machine(tmp_path),
                                         runner=lambda argv: calls.append(argv) or (0, ""))
    assert calls == []
    assert not _lock(prefix).exists()
    assert sorted(path.name for path in prefix.parent.iterdir()) == ["venv"]


def test_runtime_update_refuses_staged_source_whose_timestamp_cannot_change(tmp_path, monkeypatch):
    prefix, record = _fake_runtime(tmp_path)
    original = prefix / "lib" / "module.py"
    monkeypatch.setattr(ic, "_retiring_runtime_identity", lambda path: {"version": "1.0.0"})
    monkeypatch.setattr(ic, "os", _Proxy(os, utime=lambda *args, **kwargs: None))

    def run(argv):
        if argv[4:6] == ["install", "--upgrade"]:
            staged = os.path.join(os.path.dirname(os.path.dirname(argv[0])), "lib", "module.py")
            details = os.stat(original)
            with open(staged, "w") as stream:
                stream.write("VALUE = 2\n")
            os.utime(staged, ns=(details.st_atime_ns, details.st_mtime_ns))
        return 0, "ok"

    with pytest.raises(OSError) as raised:
        ic._transactional_runtime_update(record, "2.0.0", [record.python, "-I", "-m", "pip", "install", "--upgrade",
                                                           "spacr==2.0.0"], machine=_machine(tmp_path), runner=run)
    assert raised.value.errno == errno.EIO
    assert original.read_text() == "VALUE = 1\n"
    assert sorted(path.name for path in prefix.parent.iterdir()) == ["venv"]


def test_macos_online_update_refuses_unverified_current_runtime(tmp_path, monkeypatch):
    record = ic.InstallRecord(kind="installer", layout="macos-online", platform="macos",
                              root=str(tmp_path), python=str(tmp_path / "venv" / "bin" / "python"), running=True)
    monkeypatch.setattr(ic, "_macos_online_update_record", lambda records, system, strict: record)
    monkeypatch.setattr(ic, "_macos_online_commands", lambda *args: (["install"], ["probe"], ["relaunch"]))
    monkeypatch.setattr(ic, "_retiring_runtime_identity", lambda path: None)
    plan = {"records": [], "version": "2.0.0", "workdir": str(tmp_path), "install": ["install"],
            "relaunch": ["relaunch"], "fetch": None}
    with pytest.raises(ValueError, match="previous_runtime_identity_unverified"):
        ic._run_macos_online_update(plan, types.SimpleNamespace(platform="macos"))


def test_debian_frontend_lock_refuses_a_group_writable_lock(tmp_path):
    lock = tmp_path / "lock-frontend"
    lock.write_text("")
    lock.chmod(0o666)
    entered = []
    with pytest.raises(OSError, match="unsafe Debian frontend lock"):
        with ic._debian_frontend_lock(str(lock)):
            entered.append(True)
    assert entered == []


def test_debian_archive_requires_private_staging_before_download(tmp_path):
    tmp_path.chmod(0o755)
    fetched = []
    with pytest.raises(ValueError, match="private trusted staging"):
        ic._debian_prepare_archive("2.0.0", str(tmp_path), fetch=lambda url, target: fetched.append(url),
                                   published_digest=lambda url: "0" * 64)
    assert fetched == []


def test_debian_archive_refuses_shared_writable_download(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)

    def fetch(url, target):
        with open(target, "wb") as stream:
            stream.write(b"package")
        os.chmod(target, 0o664)

    with pytest.raises(ValueError, match="unsafe Debian archive staging identity"):
        ic._debian_prepare_archive("2.0.0", str(stage), fetch=fetch, published_digest=lambda url: "0" * 64)
    assert (stage / "spacr_2.0.0_amd64.deb").read_bytes() == b"package"


def test_debian_metadata_refuses_other_architectures_without_running_dpkg():
    calls = []
    with pytest.raises(ValueError, match="unsupported Debian package architecture"):
        ic._debian_package_metadata("new.deb", "2.0.0", "arm64", run=lambda argv: calls.append(argv) or (0, ""))
    assert calls == []


@pytest.mark.parametrize("reply", [(0, "install ok installed\n2.0.0\n"), (1, "install ok installed\n2.0.0\namd64\n"),
                                   (0, "install ok installed\n2.0.0\ni386\n")])
def test_debian_state_refuses_ambiguous_package_query(reply):
    with pytest.raises(ValueError, match="ambiguous"):
        ic._debian_installed_state(run=lambda argv: reply)


def test_debian_installed_version_requires_configured_package():
    with pytest.raises(ValueError, match="not a configured native Debian package"):
        ic._debian_installed_version(run=lambda argv: (0, "deinstall ok config-files\n2.0.0\namd64\n"))
    assert ic._debian_installed_version(run=lambda argv: (0, "install ok installed\n2.0.0\namd64\n")) == "2.0.0"


class _NotNewer(Packages):
    def __call__(self, command):
        if command[1] == "--compare-versions":
            self.calls.append(command)
            return 1, ""
        return super().__call__(command)


def test_debian_transaction_refuses_release_that_is_not_newer():
    runner = _NotNewer()
    with pytest.raises(ValueError, match="not newer than the installed version"):
        ic._debian_package_transaction("new.deb", "old.deb", "1.5.1.4", "1.5.1.5",
                                       verify=lambda version: True, run=runner)
    assert not any(command[1] == "--install" for command in runner.calls)


def test_debian_unconfirmed_desktop_retains_new_package_without_journal():
    runner = Packages()

    def unconfirmed():
        raise ic._UpdateActivationUnconfirmed("desktop not confirmed")

    with pytest.raises(ic._UpdateActivationUnconfirmed):
        ic._debian_package_transaction("new.deb", "old.deb", "1.5.1.4", "1.5.1.5",
                                       verify=lambda version: True, run=runner, after_activate=unconfirmed)
    assert runner.version == "1.5.1.5"
    assert sum(command[1] == "--install" for command in runner.calls) == 1


def test_debian_unverified_restoration_survives_journal_failure():
    runner = Packages()
    states = []

    def journal(state):
        states.append(state)
        if state == "recovery-required":
            raise OSError("journal disk failure")

    with pytest.raises(RuntimeError, match="both archives are retained") as raised:
        ic._debian_package_transaction("new.deb", "old.deb", "1.5.1.4", "1.5.1.5",
                                       verify=lambda version: False, run=runner, journal=journal)
    assert "restored Debian identity" in str(raised.value.__cause__)
    assert states == ["installing", "recovery-required"]
    assert runner.version == "1.5.1.4"


def test_debian_restored_package_is_reported_even_when_journal_fails():
    runner = Packages(replacement=1)
    states = []

    def journal(state):
        states.append(state)
        if state == "restored":
            raise OSError("journal disk failure")

    with pytest.raises(RuntimeError, match="previous package and runtime were restored"):
        ic._debian_package_transaction("new.deb", "old.deb", "1.5.1.4", "1.5.1.5",
                                       verify=lambda version: True, run=runner, journal=journal)
    assert states == ["installing", "restored"]
    assert runner.version == "1.5.1.4"


@pytest.mark.skipif(os.name == "nt" or os.getuid() == 0, reason="needs an unprivileged POSIX account")
def test_debian_desktop_verification_refuses_redirected_root(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="must not be redirected"):
        ic._debian_verify_desktop("2.0.0", os.getuid(), root=str(link),
                                  invoke=lambda *args, **kwargs: pytest.fail("verification ran"))


@pytest.mark.skipif(os.name == "nt" or os.getuid() == 0, reason="needs an unprivileged POSIX account")
def test_debian_desktop_verification_refuses_user_owned_application(tmp_path):
    root = tmp_path / "spacr"
    root.mkdir()
    (root / "spacr").write_text("#!/bin/sh\n")
    with pytest.raises(ValueError, match="not a trusted system package"):
        ic._debian_verify_desktop("2.0.0", os.getuid(), root=str(root),
                                  invoke=lambda *args, **kwargs: pytest.fail("verification ran"))


def _debian_planner(monkeypatch, *, marker_owner=0, pkexec=True):
    import io

    record = ic.InstallRecord(kind="installer", layout="linux-deb", platform="linux", root="/opt/spacr",
                              version="1.5.1.4", registrations=("deb:spacr",), running=True)
    monkeypatch.setattr(ic, "sys", _Proxy(sys, frozen=True))
    monkeypatch.setattr(ic, "_user_data", lambda machine: [])
    monkeypatch.setattr(ic, "_debian_check_package_owner", lambda version, run: "/opt/spacr/_internal/spacr-update-helper")
    monkeypatch.setattr(ic, "_run", lambda argv: (0, ""))
    monkeypatch.setattr(ic, "open", lambda *args, **kwargs: io.StringIO(
        json.dumps({"schema": 1, "adapter": "linux-deb-v1"})), raising=False)

    def lstat(path):
        if path == "/opt/spacr/update-protocol.json":
            return os.stat_result((stat.S_IFREG | 0o644, 1, 1, 1, marker_owner, 0, 48, 0, 0, 0))
        return os.lstat(path)

    paths = _Proxy(os.path, isfile=lambda path: pkexec if path == "/usr/bin/pkexec" else os.path.isfile(path))
    monkeypatch.setattr(ic, "os", _Proxy(os, lstat=lstat, path=paths,
                                         access=lambda path, mode: pkexec if path == "/usr/bin/pkexec" else os.access(path, mode)))
    return [record], types.SimpleNamespace(platform="linux", executable="/opt/spacr/spacr")


@pytest.mark.skipif(os.name == "nt" or os.getuid() == 0, reason="needs an unprivileged POSIX account")
def test_debian_planner_refuses_marker_not_owned_by_root(monkeypatch):
    records, machine = _debian_planner(monkeypatch, marker_owner=1000)
    with pytest.raises(ValueError, match="protocol marker is not trusted"):
        ic._debian_update_commands(records, "1.5.1.5", machine)


@pytest.mark.skipif(os.name == "nt" or os.getuid() == 0, reason="needs an unprivileged POSIX account")
def test_debian_planner_requires_graphical_authorization(monkeypatch):
    records, machine = _debian_planner(monkeypatch, pkexec=False)
    with pytest.raises(ValueError, match="graphical administrator authorization is unavailable"):
        ic._debian_update_commands(records, "1.5.1.5", machine)
    records, machine = _debian_planner(monkeypatch)
    command, relaunch = ic._debian_update_commands(records, "1.5.1.5", machine)
    assert command[:3] == ["/usr/bin/pkexec", "/opt/spacr/_internal/spacr-update-helper", "prepare-debian-root"]
    assert relaunch == ["/opt/spacr/spacr"]


def _debian_plan(tmp_path):
    tmp_path.chmod(0o700)
    tmp_path = type(tmp_path)(os.path.realpath(tmp_path))
    helper = str(tmp_path / "spacr-update-helper")
    record = ic.InstallRecord(kind="installer", layout="linux-deb", platform="linux", root="/opt/spacr",
                              version="1.5.1.4", registrations=("deb:spacr",), running=True)
    command, relaunch = ["/usr/bin/pkexec", "/opt/spacr/_internal/spacr-update-helper", "fixture"], ["/opt/spacr/spacr"]
    plan = {"adapter": "linux-deb-v1", "version": "1.5.1.5", "pid": 123456, "records": [asdict(record)],
            "ticked": [], "workdir": str(tmp_path),
            "handshake": {"schema": 1, "token": "a" * 64, "expires": time.time() + 600},
            "command": [helper, "run-plan", str(tmp_path / "plan.json")], "frozen_application": True,
            "fetch": None, "install": command, "relaunch": relaunch}
    return plan, helper, command, relaunch


def test_debian_update_must_run_from_its_extracted_helper(tmp_path):
    plan, _helper, _command, _relaunch = _debian_plan(tmp_path)
    with pytest.raises(ValueError, match="extracted private helper"):
        ic._run_debian_update(plan, types.SimpleNamespace(platform="linux"))


@pytest.mark.parametrize("fault", ["changed", "no-receipt", "invalid-receipt"])
def test_debian_update_rechecks_identity_and_requires_a_receipt(tmp_path, monkeypatch, fault):
    plan, helper, command, relaunch = _debian_plan(tmp_path)
    monkeypatch.setattr(ic, "sys", _Proxy(sys, executable=helper))
    planned = []

    def commands(*_args, **_kwargs):
        planned.append(True)
        if fault == "changed" and len(planned) > 1:
            return command + ["other"], relaunch
        return command, relaunch

    monkeypatch.setattr(ic, "_debian_update_commands", commands)

    def coordinate(plan_value, handshake, operation, *, check, launch, wait=None):
        assert isinstance(handshake, ic._FrozenUpdateHandshake)
        check()
        return ("" if fault == "no-receipt" else "installer log\nnot json"), None

    monkeypatch.setattr(ic, "_coordinate_native_update", coordinate)
    expected = {"changed": (ValueError, "identity changed after shutdown"),
                "no-receipt": (RuntimeError, "no verified receipt"),
                "invalid-receipt": (RuntimeError, "no verified receipt")}[fault]
    with pytest.raises(expected[0], match=expected[1]):
        ic._run_debian_update(plan, types.SimpleNamespace(platform="linux"), published_digest=lambda url: "0" * 64)
    assert len(planned) == 2


def _root_stat(directory, mode, uid=0):
    return os.stat_result(((stat.S_IFDIR if directory else stat.S_IFREG) | mode, 1, 1, 1, uid, 0, 64, 0, 0, 0))


def test_debian_root_apply_refuses_a_different_desktop_owner(monkeypatch):
    stage = "/var/lib/spacr-updates/transaction-fixture"
    helper = stage + "/spacr-update-helper"
    stats = {stage: _root_stat(True, 0o700), "/var/lib/spacr-updates": _root_stat(True, 0o700),
             helper: _root_stat(False, 0o755)}
    monkeypatch.setattr(ic, "_debian_authenticated_user", lambda uid: uid)
    monkeypatch.setattr(ic, "sys", _Proxy(sys, executable=helper))
    monkeypatch.setattr(ic, "os", _Proxy(os, lstat=lambda path: stats[path],
                                         path=_Proxy(os.path, realpath=lambda path: path)))
    monkeypatch.setattr(ic, "_desktop_process_identity", lambda pid: (2000, "/opt/spacr/spacr"))
    monkeypatch.setattr(ic, "_UpdatePhaseClient", lambda *args: pytest.fail("connected to a foreign desktop"))
    monkeypatch.setattr(ic, "_debian_write_journal", lambda *args: pytest.fail("journal written"))
    with pytest.raises(ValueError, match="could not be identified"):
        ic._debian_apply_root_transaction(stage, "1.5.1.5", "1.5.1.4", 1000, phase_channel="endpoint", desktop_pid=4242)


def _metadata_runner(calls, extract_code=0):
    def run(argv):
        calls.append(argv)
        if argv[1] == "--field":
            return 0, "Package: spacr\nVersion: 1.5.1.5\nArchitecture: amd64\n"
        return extract_code, "extract log"
    return run


def test_debian_payload_staging_requires_root_owned_opt(monkeypatch):
    calls = []
    monkeypatch.setattr(ic, "os", _Proxy(os, lstat=lambda path: _root_stat(True, 0o755, uid=1000)))
    monkeypatch.setattr(ic.tempfile, "mkdtemp", lambda **kwargs: pytest.fail("staged under an untrusted /opt"))
    with pytest.raises(ValueError, match="not a trusted system package"):
        ic._debian_stage_payload("new.deb", "1.5.1.5", 1000, run=_metadata_runner(calls))
    assert [argv[1] for argv in calls] == ["--field"]


@pytest.mark.parametrize("extract_code,verified", [(2, True), (0, False)])
def test_debian_payload_staging_removes_unverified_bundle(tmp_path, monkeypatch, extract_code, verified):
    calls = []
    staged = tmp_path / ".spacr-staged-fixture"
    monkeypatch.setattr(ic, "os", _Proxy(os, lstat=lambda path: _root_stat(True, 0o755) if path == "/opt" else os.lstat(path)))

    def mkdtemp(prefix, dir):
        assert (prefix, dir) == (".spacr-staged-", "/opt")
        staged.mkdir()
        return str(staged)

    monkeypatch.setattr(ic.tempfile, "mkdtemp", mkdtemp)
    checked = []
    with pytest.raises(RuntimeError, match="could not be verified"):
        ic._debian_stage_payload("new.deb", "1.5.1.5", 1000, run=_metadata_runner(calls, extract_code),
                                 verify=lambda version, uid, root: checked.append(root) or verified)
    assert not staged.exists()
    assert checked == ([] if extract_code else [os.path.join(str(staged), "opt", "spacr")])


def test_debian_journal_removes_temporary_record_when_replacement_fails(tmp_path, monkeypatch):
    def refuse(_source, _target):
        raise OSError(errno.EROFS, "read-only")

    monkeypatch.setattr(ic, "os", _Proxy(os, replace=refuse))
    with pytest.raises(OSError):
        ic._debian_write_journal(str(tmp_path), {"state": "prepared"})
    assert list(tmp_path.iterdir()) == []


def test_debian_root_preparation_forwards_the_phase_channel(tmp_path, monkeypatch):
    from contextlib import nullcontext

    parent = "/var/lib/spacr-updates"
    tmp_path = type(tmp_path)(os.path.realpath(tmp_path))
    source = tmp_path / "installed-helper"
    source.write_bytes(b"helper bytes")
    stage = tmp_path / "transaction-fixture"
    stage.mkdir()
    monkeypatch.setattr(ic, "_debian_authenticated_user", lambda uid: uid)
    monkeypatch.setattr(ic, "_debian_frontend_lock", nullcontext)
    monkeypatch.setattr(ic, "_debian_check_package_owner", lambda version: str(source))
    monkeypatch.setattr(ic, "sys", _Proxy(sys, executable=str(source)))
    paths = _Proxy(os.path, lexists=lambda path: path == parent or os.path.lexists(path),
                   realpath=lambda path: path if path == parent else os.path.realpath(path))
    monkeypatch.setattr(ic, "os", _Proxy(os, path=paths, mkdir=lambda *args: pytest.fail("created system folder"),
                                         lstat=lambda path: _root_stat(True, 0o700) if path == parent else os.lstat(path)))
    monkeypatch.setattr(ic.tempfile, "mkdtemp", lambda prefix, dir: str(stage))
    launched = []

    def run(command, **kwargs):
        launched.append((command, kwargs["env"]))
        return types.SimpleNamespace(returncode=0, stdout="receipt")

    monkeypatch.setattr(ic, "subprocess", _Proxy(subprocess, run=run))
    assert ic._debian_prepare_root_transaction("1.5.1.5", "1.5.1.4", 1000,
                                               phase_channel="/run/user/1000/phase", desktop_pid=4242) == (0, "receipt")
    (command, environment), = launched
    assert command[0] == str(stage / "spacr-update-helper")
    assert command[-4:] == ["--phase-channel", "/run/user/1000/phase", "--desktop-pid", "4242"]
    assert environment["PKEXEC_UID"] == "1000"
    assert (stage / "spacr-update-helper").read_bytes() == b"helper bytes"
