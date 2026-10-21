"""Item 703: Windows private-pip, native ACL, frozen payload and POSIX pip update guards."""
import ctypes
import errno
import hashlib
import json
import os
import shutil
import types
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from spacr import install_cleanup as cleanup
from tests.test_703_cleanup_a_phase_channel import _Proxy
from tests.test_frozen_windows_install_transaction_703 import identity, payload
from tests.test_windows_private_pip_703 import USER, NativeAclModel, PackageRunner, plan_for, runtime, snapshot  # noqa: F401


class _AclModel(NativeAclModel):
    def __init__(self, ctypes_module, defect):
        super().__init__(ctypes_module, None)
        self.my_defect = defect
        self.conversions = 0
        if defect == "bad_header":
            self.acl = ctypes_module.create_string_buffer(b"\x02\x00" + (4).to_bytes(2, "little") + b"\x01\x00\x00\x00")

    def GetTokenInformation(self, token, kind, buffer, size, needed):
        if self.my_defect == "token_size":
            self.ctypes.cast(needed, self.ctypes.POINTER(self.ctypes.c_uint32)).contents.value = 0
            return 0
        return super().GetTokenInformation(token, kind, buffer, size, needed)

    def ConvertSidToStringSidW(self, sid, text):
        self.conversions += 1
        if self.my_defect == "ace_sid_error" and self.conversions == 3:
            return 0
        if (self.my_defect, self.conversions) in {("user_null", 2), ("ace_null", 3)}:
            return 1
        return super().ConvertSidToStringSidW(sid, text)


def _read_acl(tmp_path, monkeypatch, defect):
    native_os = types.SimpleNamespace(**vars(os))
    native_os.name = "nt"
    monkeypatch.setattr(cleanup, "os", native_os)
    model = _AclModel(ctypes, defect)
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, **kwargs: model.kernel if name == "kernel32" else model.security,
                        raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)
    return model, lambda: cleanup._windows_pip_acl(str(tmp_path))


@pytest.mark.parametrize("defect", ["token_size", "bad_header", "ace_sid_error"])
def test_native_acl_reader_refuses_malformed_token_or_dacl(tmp_path, monkeypatch, defect):
    model, read = _read_acl(tmp_path, monkeypatch, defect)
    with pytest.raises((ValueError, OSError)):
        read()
    assert model.closed == [999]
    assert ctypes.addressof(model.descriptor) in model.freed


@pytest.mark.parametrize("defect", ["user_null", "ace_null"])
def test_native_acl_reader_reports_empty_sid_text_without_freeing_it(tmp_path, monkeypatch, defect):
    model, read = _read_acl(tmp_path, monkeypatch, defect)
    result = read()
    assert result["owner"] == USER and result["has_dacl"] is True
    assert result["user"] == (None if defect == "user_null" else USER)
    assert [rule["sid"] for rule in result["rules"]] == [USER if defect == "user_null" else None]
    assert len(model.freed) == 3


def test_windows_private_directory_must_be_a_directory(runtime):
    with pytest.raises(ValueError):
        cleanup._windows_pip_private_directory(str(runtime.python))
    assert cleanup._windows_pip_private_directory(str(runtime.prefix)) == str(runtime.prefix)


def test_windows_pip_commands_refuse_non_environment_records(runtime):
    with pytest.raises(ValueError):
        cleanup._windows_pip_commands(replace(runtime.record, kind="installer"), "1.5.1.5",
                                      str(runtime.helper), runtime.machine)


def test_windows_pip_commands_require_the_console_interpreter(runtime):
    windowed = runtime.python.with_name("pythonw.exe")
    home = runtime.prefix.parent.parent
    machine = cleanup._Machine(platform="windows", environ={"USERPROFILE": str(home)}, fs_root=str(home.parent),
                               running_prefix=str(runtime.prefix), executable=str(windowed))
    record = replace(runtime.record, python=str(windowed))
    assert cleanup._windows_pip_commands(record, "1.5.1.5", str(runtime.helper), machine)[0][0] == str(runtime.python)
    runtime.python.unlink()
    with pytest.raises(ValueError):
        cleanup._windows_pip_commands(record, "1.5.1.5", str(runtime.helper), machine)


def test_windows_pip_helper_runs_only_on_windows(runtime):
    with pytest.raises(ValueError):
        cleanup._windows_pip_helper(runtime.record, str(runtime.helper / "spacr_helper.py"), "plan.json",
                                    types.SimpleNamespace(platform="linux"))


def test_windows_pip_helper_requires_an_outside_base_interpreter(runtime, monkeypatch, tmp_path):
    empty = tmp_path / "empty base"
    empty.mkdir()
    helper = tmp_path / "outside helper"
    helper.mkdir()
    (helper / "spacr_helper.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(cleanup.sys, "base_prefix", str(empty))
    ran = []
    monkeypatch.setattr(cleanup, "_run", lambda *args, **kwargs: ran.append(args) or (0, "{}"))
    with pytest.raises(ValueError):
        cleanup._windows_pip_helper(runtime.record, str(helper / "spacr_helper.py"), "plan.json", runtime.machine)
    assert ran == []


@pytest.mark.parametrize("phase_channel,desktop_pid", [("C:\\phase.txt", 1), ("C:\\phase.txt", True),
                                                       ("relative\\phase.txt", 4242), ('C:\\pha"se', 4242)])
def test_windows_frozen_install_refuses_invalid_phase_arguments(monkeypatch, phase_channel, desktop_pid):
    monkeypatch.setattr(cleanup, "_windows_authorized_process", lambda *args, **kwargs: pytest.fail("elevated"))
    with pytest.raises(ValueError, match="invalid Windows installer path"):
        cleanup._windows_frozen_install("C:\\Downloads\\spaCR-setup.exe", "C:\\Program Files\\spaCR",
                                        phase_channel=phase_channel, desktop_pid=desktop_pid)


def test_windows_frozen_install_passes_the_phase_channel_to_the_elevated_installer(monkeypatch):
    calls = []
    monkeypatch.setattr(cleanup, "_windows_authorized_process", lambda installer, parameters, timeout: calls.append(
        (installer, parameters)) or (0, ""))
    assert cleanup._windows_frozen_install("C:\\Downloads\\setup.exe", "C:\\spaCR",
                                           phase_channel="C:\\phase.txt", desktop_pid=4242) == (0, "")
    assert calls == [("C:\\Downloads\\setup.exe", '/S /SPACRUPDATE="C:\\phase.txt" /SPACRPID=4242 /D=C:\\spaCR')]


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), True, "60", 5_000_000])
def test_windows_authorized_process_refuses_invalid_timeouts(monkeypatch, timeout):
    monkeypatch.setattr(cleanup, "_windows_process_elevated", lambda: pytest.fail("token inspected"))
    with pytest.raises(ValueError, match="invalid Windows installer timeout"):
        cleanup._windows_authorized_process("C:\\setup.exe", "/S", timeout=timeout)


@pytest.mark.parametrize("installer,parameters", [("setup.exe", "/S"), ("C:\\set\nup.exe", "/S"),
                                                  ("C:\\setup.exe", "/S\r\n/D=C:\\x")])
def test_windows_authorized_process_refuses_unpinned_installer_text(monkeypatch, installer, parameters):
    monkeypatch.setattr(cleanup, "_windows_process_elevated", lambda: False)
    monkeypatch.setattr(cleanup, "_windows_update_library", lambda name: pytest.fail("native library loaded"))
    with pytest.raises(ValueError, match="invalid Windows installer path"):
        cleanup._windows_authorized_process(installer, parameters)


@pytest.mark.parametrize("lease", [{"schema": 1, "token": "short", "expires": 5},
                                   {"schema": 2, "token": "a" * 64, "expires": 5},
                                   {"schema": 1, "token": "a" * 64, "expires": float("inf")}])
def test_windows_pip_handshake_refuses_invalid_leases(lease, monkeypatch):
    monkeypatch.setattr(cleanup, "_windows_pip_private_directory", lambda path: pytest.fail("directory inspected"))
    with pytest.raises(ValueError, match="invalid frozen updater readiness lease"):
        cleanup._WindowsPipHandshake({"handshake": lease, "workdir": "C:\\work", "pid": 1})


@pytest.mark.parametrize("reply", [(0, '{"verified": false, "scripts": []}'), (1, '{"verified": true, "scripts": []}'),
                                   (0, '{"verified": true, "scripts": "all"}')])
def test_windows_entrypoint_relocation_requires_verified_receipt(tmp_path, reply):
    with pytest.raises(ValueError, match="windows_entrypoint_verification_failed"):
        cleanup._windows_pip_scripts(str(tmp_path / "staged"), str(tmp_path / "venv"), lambda argv: reply)


def _transaction(runtime, monkeypatch, runner, *, install=None, activate=None):
    monkeypatch.setattr(cleanup, "_windows_pip_scripts", lambda staged, prefix, run: {"verified": True, "scripts": []})
    plan = plan_for(runtime)
    return cleanup._windows_pip_transaction(runtime.record, plan["version"], install or plan["install"],
                                            machine=runtime.machine, runner=runner, activate=activate)


def _lock(runtime):
    return runtime.prefix.parent / ("." + runtime.prefix.name + ".spacr-update-lock")


def test_windows_transaction_refuses_unverified_previous_runtime(runtime, monkeypatch):
    monkeypatch.setattr(cleanup, "_retiring_runtime_identity", lambda path: None)
    before = snapshot(runtime.prefix)
    runner = PackageRunner(runtime)
    with pytest.raises(ValueError, match="previous_runtime_identity_unverified"):
        _transaction(runtime, monkeypatch, runner)
    assert runner.commands == [] and snapshot(runtime.prefix) == before
    assert not _lock(runtime).exists()


def test_windows_transaction_refuses_command_for_another_release(runtime, monkeypatch):
    before = snapshot(runtime.prefix)
    runner = PackageRunner(runtime)
    install = plan_for(runtime)["install"][:-1] + ["spacr==1.5.1.6"]
    with pytest.raises(ValueError, match="does not target this private runtime"):
        _transaction(runtime, monkeypatch, runner, install=install)
    assert runner.commands == [] and snapshot(runtime.prefix) == before
    root, = runtime.prefix.parent.glob(".spacr-windows-update-*")
    assert json.loads((root / "transaction.json").read_text(encoding="utf-8"))["state"] == "staging_failed"


def test_windows_transaction_refuses_stage_whose_timestamp_cannot_move(runtime, monkeypatch):
    source = runtime.package / "__init__.py"
    os.utime(source, (1_700_000_000, 1_700_000_000))
    before = snapshot(runtime.prefix)
    runner = PackageRunner(runtime)

    def collision(argv):
        code, output = runner(argv)
        if argv[1:5] == ["-I", "-m", "pip", "install"]:
            staged = Path(argv[0]).parent.parent / "Lib/site-packages/spacr/__init__.py"
            os.utime(staged, (1_700_000_000, 1_700_000_000))
        return code, output

    monkeypatch.setattr(cleanup, "os", _Proxy(os, utime=lambda *args, **kwargs: None))
    with pytest.raises(OSError) as raised:
        _transaction(runtime, monkeypatch, collision)
    assert raised.value.errno == errno.EIO
    assert snapshot(runtime.prefix) == before


def test_windows_transaction_journals_an_unstoppable_new_desktop(runtime, monkeypatch):
    def still_running(version):
        raise cleanup._UpdateLaunchStillRunning(4321)

    with pytest.raises(cleanup._UpdateLaunchStillRunning):
        _transaction(runtime, monkeypatch, PackageRunner(runtime), activate=still_running)
    root, = runtime.prefix.parent.glob(".spacr-windows-update-*")
    journal = json.loads((root / "transaction.json").read_text(encoding="utf-8"))
    assert (journal["state"], journal["launch_pid"]) == ("launch_unconfirmed", 4321)
    assert (runtime.package / "__init__.py").read_text(encoding="utf-8") == '__version__ = "1.5.1.5"\n'
    assert not _lock(runtime).exists()


def test_windows_rollback_stops_when_the_stage_name_was_reused(runtime, monkeypatch):
    before = snapshot(runtime.prefix)
    runner = PackageRunner(runtime, "final_import")

    def reuse(argv):
        if argv[-1] == "import spacr.qt" and Path(argv[0]).parent.parent == runtime.prefix:
            root, = runtime.prefix.parent.glob(".spacr-windows-update-*")
            (root / "staged").mkdir()
        return runner(argv)

    with pytest.raises(RuntimeError, match="windows_runtime_rollback_requires_review"):
        _transaction(runtime, monkeypatch, reuse)
    root, = runtime.prefix.parent.glob(".spacr-windows-update-*")
    assert snapshot(root / "previous") == before
    assert (runtime.package / "__init__.py").read_text(encoding="utf-8") == '__version__ = "1.5.1.5"\n'
    assert _lock(runtime).exists()


def _windows_update(runtime, monkeypatch, *, plan=None, identities=None, runner=None):
    plan = plan or plan_for(runtime)
    if identities is not None:
        values = list(identities)
        monkeypatch.setattr(cleanup, "_retiring_runtime_identity", lambda path: values.pop(0) if len(values) > 1 else values[0])
    return cleanup._run_windows_pip_update(plan, runtime.machine, run=runner or PackageRunner(runtime))


@pytest.mark.parametrize("change", ["frozen", "linux", "two"])
def test_windows_update_requires_one_running_unfrozen_windows_environment(runtime, monkeypatch, change):
    plan = plan_for(runtime)
    machine = runtime.machine
    if change == "frozen":
        plan["frozen_application"] = True
    elif change == "linux":
        machine = types.SimpleNamespace(platform="linux")
    else:
        plan["records"] = plan["records"] * 2
    with pytest.raises(ValueError):
        cleanup._run_windows_pip_update(plan, machine, run=PackageRunner(runtime))


def test_windows_update_refuses_an_unapproved_command(runtime, monkeypatch):
    plan = plan_for(runtime)
    plan["install"] = plan["install"] + ["--pre"]
    with pytest.raises(ValueError, match="does not target this private runtime"):
        _windows_update(runtime, monkeypatch, plan=plan)


@pytest.mark.parametrize("identity_value,message", [(None, "previous_runtime_identity_unverified"),
                                                    ({"verified": True, "version": "1.5.1.6"}, "newer")])
def test_windows_update_refuses_unverified_or_newer_runtime(runtime, monkeypatch, identity_value, message):
    runner = PackageRunner(runtime)
    with pytest.raises(ValueError, match=message):
        _windows_update(runtime, monkeypatch, identities=[identity_value], runner=runner)
    assert runner.commands == []


def test_windows_update_requires_pip_in_the_private_runtime(runtime, monkeypatch):
    def runner(argv):
        return (1, "No module named pip") if argv[-1] == "import pip" else (0, "")

    with pytest.raises(RuntimeError, match="No module named pip"):
        _windows_update(runtime, monkeypatch, runner=runner)


def test_windows_update_rechecks_runtime_identity_after_shutdown(runtime, monkeypatch):
    events = []

    class Handshake:
        def __init__(self, plan):
            events.append("lease")

        def check(self):
            events.append("check")

        def ready(self):
            events.append("ready")

        def wait_for_shutdown(self, wait):
            events.append("shutdown")

    def transaction(record, version, install, *, machine, runner, before_activate, activate):
        before_activate()
        return pytest.fail("activation continued")

    monkeypatch.setattr(cleanup, "_WindowsPipHandshake", Handshake)
    monkeypatch.setattr(cleanup, "_windows_pip_transaction", transaction)
    with pytest.raises(ValueError, match="previous_runtime_identity_unverified"):
        _windows_update(runtime, monkeypatch, identities=[
            {"verified": True, "version": "1.5.1.4"}, {"verified": True, "version": "1.5.1.3"}])
    assert events == ["lease", "check", "ready", "shutdown"]


def _posix_runtime(tmp_path, version="1.0.0"):
    base = Path(os.path.realpath(tmp_path))
    prefix = base / "private" / "venv"
    (prefix / "bin").mkdir(parents=True)
    python = prefix / "bin" / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    (prefix / "pyvenv.cfg").write_text("include-system-site-packages = false\n", encoding="utf-8")
    for directory in (prefix, prefix.parent):
        directory.chmod(0o755)
    record = cleanup.InstallRecord(kind="environment", layout="venv", platform="linux", root=str(prefix),
                                   python=str(python), version=version, running=True)
    machine = cleanup._Machine(platform="linux", environ={"HOME": str(base / "user")}, fs_root=str(base),
                               executable=str(python), running_prefix=str(prefix))
    return record, machine, prefix, base


def test_pip_commands_refuse_records_that_are_not_running_environments(tmp_path):
    record, machine, _prefix, base = _posix_runtime(tmp_path)
    with pytest.raises(ValueError, match="package manager"):
        cleanup._pip_environment_commands(replace(record, kind="installer"), "2.0.0", str(base / "work"), machine)
    with pytest.raises(ValueError, match="package manager"):
        cleanup._pip_environment_commands(record, "2.0.0", str(base / "work"),
                                          types.SimpleNamespace(platform="freebsd"))


@pytest.mark.posix_only
def test_pip_commands_refuse_updater_folder_inside_environment(tmp_path):
    record, machine, prefix, _base = _posix_runtime(tmp_path)
    with pytest.raises(ValueError, match="outside the installed environment"):
        cleanup._pip_environment_commands(record, "2.0.0", str(prefix / "updater"), machine)


@pytest.mark.posix_only
def test_pip_commands_without_recorded_version_and_with_newer_runtime(tmp_path):
    record, machine, prefix, base = _posix_runtime(tmp_path)
    install, relaunch = cleanup._pip_environment_commands(replace(record, version=None), "2.0.0",
                                                          str(base / "work"), machine)
    assert install == [record.python, "-I", "-m", "pip", "install", "--upgrade", "spacr==2.0.0"]
    assert relaunch == [record.python, "-I", "-m", "spacr.qt"]
    with pytest.raises(ValueError, match="newer than the requested release"):
        cleanup._pip_environment_commands(replace(record, version="3.0.0"), "2.0.0", str(base / "work"), machine)


@pytest.mark.parametrize("reply", [(0, "not json"), (0, '{"verified": true}'), (0, "[]")])
def test_pip_environment_identity_refuses_unreadable_probe(tmp_path, reply):
    record, _machine, _prefix, _base = _posix_runtime(tmp_path)
    probes = []
    with pytest.raises(RuntimeError, match="could not be identified"):
        cleanup._pip_environment_identity(record, lambda argv: probes.append(argv) or reply)
    assert probes and probes[0][2] == "-B"


def _pip_plan(record, machine, base):
    install, relaunch = cleanup._pip_environment_commands(record, "2.0.0", str(base / "work"), machine)
    return {"records": [asdict(record)], "version": "2.0.0", "workdir": str(base / "work"),
            "install": install, "relaunch": relaunch, "fetch": None}


@pytest.mark.posix_only
@pytest.mark.parametrize("change", ["frozen", "install"])
def test_pip_update_refuses_frozen_or_altered_plans(tmp_path, change):
    record, machine, _prefix, base = _posix_runtime(tmp_path)
    plan = _pip_plan(record, machine, base)
    if change == "frozen":
        plan["frozen_application"] = True
    else:
        plan["install"] = plan["install"] + ["--index-url", "https://example.invalid"]
    ran = []
    with pytest.raises(ValueError):
        cleanup._run_pip_environment_update(plan, machine, run=lambda argv: ran.append(argv) or (0, ""))
    assert ran == []


@pytest.mark.posix_only
def test_pip_update_rechecks_interpreter_identity_after_shutdown(tmp_path, monkeypatch):
    record, machine, _prefix, base = _posix_runtime(tmp_path)
    plan = _pip_plan(record, machine, base)
    identities = [{"version": "1.0.0", "prefix": record.root}, {"version": "1.0.0", "prefix": "/elsewhere"}]
    monkeypatch.setattr(cleanup, "_pip_environment_identity", lambda record, runner: identities.pop(0))
    events = []

    class Handshake:
        def __init__(self, plan_value):
            assert plan_value is plan

        def check(self):
            events.append("check")

        def ready(self):
            events.append("ready")

        def wait_for_shutdown(self, wait):
            events.append("shutdown")

    def transaction(record, version, install, *, machine, runner, before_activate, activate):
        before_activate()
        return pytest.fail("activation continued")

    monkeypatch.setattr(cleanup, "_FrozenUpdateHandshake", Handshake)
    monkeypatch.setattr(cleanup, "_transactional_runtime_update", transaction)
    with pytest.raises(RuntimeError, match="could not be identified"):
        cleanup._run_pip_environment_update(plan, machine, run=lambda argv: (0, ""))
    assert events == ["check", "ready", "shutdown"] and identities == []


def _manifest(root, version):
    files = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in root.rglob("*") if path.is_file() and path.name != "spacr-payload.json"}
    (root / "spacr-payload.json").write_text(json.dumps({"schema": 1, "version": version, "files": files}), encoding="utf-8")


def test_frozen_payload_refuses_oversized_manifest(tmp_path):
    root = payload(tmp_path / "bundle", "2.0.0")
    with open(root / "spacr-payload.json", "r+b") as stream:
        stream.truncate(17 * 1024 * 1024)
    with pytest.raises(ValueError, match="frozen_payload_manifest_too_large"):
        cleanup._frozen_windows_payload(root, "2.0.0")


@pytest.mark.posix_only
@pytest.mark.parametrize("name", ["c:stream.dll", "back\\slash.dll"])
def test_frozen_payload_refuses_names_windows_would_resolve_elsewhere(tmp_path, name):
    root = payload(tmp_path / "bundle", "2.0.0", manifest=False)
    (root / "_internal" / name).write_bytes(b"payload")
    _manifest(root, "2.0.0")
    with pytest.raises(ValueError, match="frozen_payload_path_outside_application"):
        cleanup._frozen_windows_payload(root, "2.0.0")


@pytest.mark.parametrize("missing", ["spacr.exe", "_internal"])
def test_frozen_payload_requires_executable_and_internal_files(tmp_path, missing):
    root = tmp_path / "bundle"
    (root / "_internal").mkdir(parents=True)
    if missing != "spacr.exe":
        (root / "spacr.exe").write_bytes(b"exe")
    if missing != "_internal":
        (root / "_internal" / "runtime.dll").write_bytes(b"dll")
    _manifest(root, "2.0.0")
    with pytest.raises(ValueError, match="frozen_payload_incomplete"):
        cleanup._frozen_windows_payload(root, "2.0.0")


def test_frozen_identity_refuses_an_oversized_receipt(tmp_path):
    root = payload(tmp_path / "bundle", "2.0.0")

    def oversized(argv):
        Path(argv[-1]).write_text(json.dumps({"padding": "x" * 20000}), encoding="utf-8")
        return 0, ""

    with pytest.raises(ValueError, match="frozen_application_identity_unverified"):
        cleanup._frozen_windows_identity(root, "2.0.0", oversized)
    assert cleanup._frozen_windows_identity(root, "2.0.0", identity)["version"] == "2.0.0"


def test_previous_frozen_version_of_an_empty_folder_is_none(tmp_path):
    assert cleanup._frozen_windows_previous_version(tmp_path) is None


def test_previous_frozen_version_refuses_internal_without_executable(tmp_path):
    (tmp_path / "_internal").mkdir()
    with pytest.raises(ValueError, match="previous_frozen_application_identity_unverified"):
        cleanup._frozen_windows_previous_version(tmp_path)


@pytest.mark.needs_symlink
def test_previous_frozen_version_refuses_linked_executable(tmp_path):
    target = tmp_path / "elsewhere.exe"
    target.write_bytes(b"exe")
    root = tmp_path / "install"
    root.mkdir()
    (root / "spacr.exe").symlink_to(target)
    with pytest.raises(ValueError, match="previous_frozen_application_identity_unverified"):
        cleanup._frozen_windows_previous_version(root)


def test_previous_frozen_version_refuses_metadata_that_is_not_a_file(tmp_path):
    root = tmp_path / "install"
    (root / "_internal" / "spacr-1.0.0.dist-info" / "METADATA").mkdir(parents=True)
    (root / "spacr.exe").write_bytes(b"exe")
    with pytest.raises(ValueError, match="previous_frozen_application_identity_unverified"):
        cleanup._frozen_windows_previous_version(root)


def test_previous_frozen_version_ignores_foreign_distributions_and_requires_one(tmp_path):
    root = tmp_path / "install"
    foreign = root / "_internal" / "spacr_plugin-3.0.dist-info" / "METADATA"
    foreign.parent.mkdir(parents=True)
    foreign.write_text("Name: spacr-plugin\nVersion: 3.0\n", encoding="utf-8")
    (root / "spacr.exe").write_bytes(b"exe")
    with pytest.raises(ValueError, match="previous_frozen_application_identity_unverified"):
        cleanup._frozen_windows_previous_version(root)
    own = root / "_internal" / "spacr-1.5.1.4.dist-info" / "METADATA"
    own.parent.mkdir()
    own.write_text("Name: spacr\nVersion: 1.5.1.4\n", encoding="utf-8")
    assert cleanup._frozen_windows_previous_version(root) == "1.5.1.4"


def test_frozen_install_into_a_new_folder_has_no_previous_payload(tmp_path):
    source = payload(tmp_path / "download", "2.0.0")
    destination = tmp_path / "Programs" / "spaCR"
    result = cleanup._install_frozen_windows(source, destination, "2.0.0", run=identity)
    assert result == {"ok": True, "version": "2.0.0", "previous_runtime": None}
    assert (destination / "spacr.exe").read_text(encoding="utf-8") == "2.0.0"
    assert not list(destination.parent.glob(".spacr-*"))


@pytest.mark.needs_symlink
def test_frozen_install_refuses_linked_destination_parent(tmp_path):
    source = payload(tmp_path / "download", "2.0.0")
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "linked").symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="frozen_install_reparse_path"):
        cleanup._install_frozen_windows(source, tmp_path / "linked" / "spaCR", "2.0.0", run=identity)
    assert list(real.iterdir()) == []


def test_frozen_install_refuses_source_inside_destination(tmp_path):
    destination = tmp_path / "spaCR"
    source = payload(destination / "download", "2.0.0")
    with pytest.raises(ValueError, match="frozen_install_requires_outside_helper"):
        cleanup._install_frozen_windows(source, destination, "2.0.0", run=identity)


def _installed(tmp_path, *, manifest=False):
    source = payload(tmp_path / "download", "1.5.1.5")
    old = payload(tmp_path / "installed", "1.5.1.4", manifest=manifest)
    (old / "Uninstall.exe").write_bytes(b"existing uninstaller")
    return source, old


def test_frozen_install_refuses_installed_uninstaller_that_is_not_a_file(tmp_path):
    source, old = _installed(tmp_path)
    (source / "Uninstall.exe").write_bytes(b"new uninstaller")
    (old / "spacr-uninstaller.json").mkdir()
    with pytest.raises(ValueError, match="frozen_uninstaller_candidate_invalid"):
        cleanup._install_frozen_windows(source, old, "1.5.1.5", run=identity)
    assert (old / "spacr.exe").read_text(encoding="utf-8") == "1.5.1.4"


def test_frozen_install_refuses_uninstaller_changed_while_copying(tmp_path, monkeypatch):
    source, old = _installed(tmp_path)
    (source / "Uninstall.exe").write_bytes(b"new uninstaller")

    def tampered_copy(origin, target):
        shutil.copytree(origin, target)
        (Path(target) / "Uninstall.exe").write_bytes(b"tampered uninstaller")

    monkeypatch.setattr(cleanup, "shutil", _Proxy(shutil, copytree=tampered_copy))
    with pytest.raises(ValueError, match="frozen_uninstaller_candidate_changed"):
        cleanup._install_frozen_windows(source, old, "1.5.1.5", run=identity)
    assert (old / "Uninstall.exe").read_bytes() == b"existing uninstaller"
    assert not list(old.parent.glob(".spacr-frozen-install-*"))


def test_frozen_install_stops_when_installed_application_changes_during_approval(tmp_path):
    source, old = _installed(tmp_path)

    def changed():
        (old / "spacr.exe").write_text("1.5.1.4 modified by another process", encoding="utf-8")

    with pytest.raises(ValueError, match="changed during staging"):
        cleanup._install_frozen_windows(source, old, "1.5.1.5", run=identity, before_activate=changed)
    assert (old / "spacr.exe").read_text(encoding="utf-8") == "1.5.1.4 modified by another process"
    assert not list(old.parent.glob(".spacr-frozen-install-*"))


def test_frozen_install_retains_backup_that_changed_after_activation(tmp_path):
    source, old = _installed(tmp_path, manifest=True)

    def added_to_backup():
        backup, = old.parent.glob(".spacr-frozen-install-*/previous")
        (backup / "_internal" / "late.dll").write_bytes(b"unexpected")

    result = cleanup._install_frozen_windows(source, old, "1.5.1.5", run=identity, after_activate=added_to_backup)
    backup = Path(result["previous_runtime"])
    assert (backup / "spacr.exe").read_text(encoding="utf-8") == "1.5.1.4"
    assert (backup / "_internal" / "late.dll").read_bytes() == b"unexpected"
    assert (old / "spacr.exe").read_text(encoding="utf-8") == "1.5.1.5"


def test_frozen_install_retains_known_backup_when_it_cannot_be_removed(tmp_path, monkeypatch):
    source, old = _installed(tmp_path, manifest=True)

    def refuse(path, **kwargs):
        raise OSError(errno.EBUSY, "in use", str(path))

    monkeypatch.setattr(cleanup, "shutil", _Proxy(shutil, rmtree=refuse))
    result = cleanup._install_frozen_windows(source, old, "1.5.1.5", run=identity)
    backup = Path(result["previous_runtime"])
    assert (backup / "spacr.exe").read_text(encoding="utf-8") == "1.5.1.4"
    assert (old / "spacr.exe").read_text(encoding="utf-8") == "1.5.1.5"


def test_frozen_install_keeps_lock_and_recovery_when_restoration_fails(tmp_path, monkeypatch):
    source, old = _installed(tmp_path)

    def wrong_identity(argv):
        if Path(argv[0]).parent != old.resolve():
            return identity(argv)
        Path(argv[-1]).write_text(json.dumps({"schema": 1, "frozen": True, "version": "0.0.1"}), encoding="utf-8")
        return 0, ""

    def replace_file(origin, target):
        if Path(origin).parent.name == "previous":
            raise OSError(errno.EACCES, "denied", str(origin))
        return os.replace(origin, target)

    monkeypatch.setattr(cleanup, "os", _Proxy(os, replace=replace_file))
    with pytest.raises(ValueError, match="frozen_application_identity_mismatch"):
        cleanup._install_frozen_windows(source, old, "1.5.1.5", run=wrong_identity)
    transaction, = old.parent.glob(".spacr-frozen-install-*")
    assert (transaction / "previous" / "spacr.exe").read_text(encoding="utf-8") == "1.5.1.4"
    assert (old.parent / ".installed.spacr-install-lock").is_dir()
    assert not (old / "spacr.exe").exists()
