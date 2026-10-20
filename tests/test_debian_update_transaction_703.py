"""Debian package identity and rollback without touching the host database."""
import importlib.util
import hashlib
from pathlib import Path
import sys
import shutil
import subprocess
import json
import time
from dataclasses import asdict
import io
import stat
from types import SimpleNamespace

import pytest


def _module():
    source = Path(__file__).parents[1] / "spacr" / "install_cleanup.py"
    spec = importlib.util.spec_from_file_location("debian_update_fixture", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ic():
    module = _module()
    module.tr = lambda source, **values: str(source).format(**values)
    return module


class Packages:
    def __init__(self, *, replacement=0, rollback=0, initial="1.5.1.4"):
        self.version = initial
        self.replacement = replacement
        self.rollback = rollback
        self.calls = []

    def __call__(self, command):
        self.calls.append(command)
        if command[0].endswith("dpkg-deb"):
            version = "1.5.1.5" if command[2].endswith("new.deb") else "1.5.1.4"
            return 0, f"Package: spacr\nVersion: {version}\nArchitecture: amd64\n"
        if command[0].endswith("dpkg-query"):
            return 0, f"install ok installed\n{self.version}\namd64\n"
        if command[1] == "--compare-versions":
            return 0, ""
        assert command[1] == "--install"
        new = command[2].endswith("new.deb")
        self.version = "1.5.1.5" if new else "1.5.1.4"
        return self.replacement if new else self.rollback, "fixture install"


def _transaction(ic, runner, verify=lambda version: True):
    return ic._debian_package_transaction("new.deb", "old.deb", "1.5.1.4", "1.5.1.5",
                                          verify=verify, run=runner)


def test_commits_only_verified_replacement(ic):
    runner = Packages()
    observed = []
    result = _transaction(ic, runner, lambda version: observed.append(version) or True)
    assert result["state"] == "committed"
    assert observed == ["1.5.1.5"]
    assert sum(command[1] == "--install" for command in runner.calls) == 1


@pytest.mark.parametrize("failure", ["install", "runtime"])
def test_failed_replacement_restores_and_verifies_previous_package(ic, failure):
    runner = Packages(replacement=1 if failure == "install" else 0)
    observed = []
    def verify(version):
        observed.append(version)
        return version != "1.5.1.5" if failure == "runtime" else True
    with pytest.raises(RuntimeError, match="previous package and runtime were restored"):
        _transaction(ic, runner, verify)
    assert runner.version == "1.5.1.4"
    assert observed[-1] == "1.5.1.4"


def test_failed_rollback_is_not_reported_as_restored(ic):
    runner = Packages(replacement=1, rollback=1)
    with pytest.raises(RuntimeError, match="both archives are retained"):
        _transaction(ic, runner)


def test_changed_installation_refuses_before_dpkg_mutation(ic):
    runner = Packages(initial="1.5.1.6")
    with pytest.raises(ValueError, match="changed during preparation"):
        _transaction(ic, runner)
    assert not any(command[1] == "--install" for command in runner.calls)


@pytest.mark.parametrize("metadata", [
    "Package: other\nVersion: 1.5.1.5\nArchitecture: amd64\n",
    "Package: spacr\nVersion: 1.5.1.4\nArchitecture: amd64\n",
    "Package: spacr\nVersion: 1.5.1.5\nArchitecture: arm64\n",
    "Package: spacr\nPackage: spacr\nVersion: 1.5.1.5\nArchitecture: amd64\n",
])
def test_wrong_or_ambiguous_archive_refuses(ic, metadata):
    with pytest.raises(ValueError):
        ic._debian_package_metadata("new.deb", "1.5.1.5", "amd64", run=lambda argv: (0, metadata))


def test_rollback_does_not_overwrite_foreign_package(ic):
    runner = Packages(replacement=1)
    def changed(command):
        result = runner(command)
        if command[1] == "--install":
            runner.version = "1.5.1.6"
        return result
    with pytest.raises(RuntimeError, match="both archives are retained"):
        _transaction(ic, changed)
    assert runner.version == "1.5.1.6"
    assert sum(command[1] == "--install" for command in runner.calls) == 1


@pytest.mark.posix_only
def test_prepared_archive_requires_exact_hash_and_fixed_release_url(ic, tmp_path):
    tmp_path.chmod(0o700)
    content = b"isolated fixture, not an installable package"
    calls = []
    def fetch(url, target):
        calls.append(url)
        Path(target).write_bytes(content)
        Path(target).chmod(0o600)
    result = ic._debian_prepare_archive("1.5.1.5", tmp_path, fetch=fetch,
        published_digest=lambda url: hashlib.sha256(content).hexdigest())
    assert calls == [ic._RELEASE_DOWNLOAD + "/v1.5.1.5/spacr_1.5.1.5_amd64.deb"]
    assert Path(result["path"]).read_bytes() == content


@pytest.mark.posix_only
@pytest.mark.parametrize("digest", [None, "", "g" * 64, "a" * 63])
def test_missing_or_invalid_hash_refuses_before_download(ic, tmp_path, digest):
    tmp_path.chmod(0o700)
    def unexpected(*args):
        pytest.fail("unverified package was downloaded")
    with pytest.raises(ValueError, match="published SHA256"):
        ic._debian_prepare_archive("1.5.1.5", tmp_path, fetch=unexpected,
                                   published_digest=lambda url: digest)
    assert not list(tmp_path.iterdir())


@pytest.mark.posix_only
def test_bad_download_retained_without_installation(ic, tmp_path):
    tmp_path.chmod(0o700)
    def fetch(url, path):
        Path(path).write_bytes(b"wrong")
        Path(path).chmod(0o600)
    with pytest.raises(ValueError, match="published SHA256"):
        ic._debian_prepare_archive("1.5.1.5", tmp_path, fetch=fetch,
                                   published_digest=lambda url: "0" * 64)
    assert (tmp_path / "spacr_1.5.1.5_amd64.deb").read_bytes() == b"wrong"


@pytest.mark.posix_only
def test_symlink_archive_is_not_followed(ic, tmp_path):
    tmp_path.chmod(0o700)
    foreign = tmp_path / "foreign"
    foreign.write_bytes(b"foreign data")
    def fetch(url, path):
        Path(path).symlink_to(foreign)
    with pytest.raises(OSError):
        ic._debian_prepare_archive("1.5.1.5", tmp_path, fetch=fetch,
                                   published_digest=lambda url: "0" * 64)
    assert foreign.read_bytes() == b"foreign data"


@pytest.mark.posix_only
def test_existing_archive_refuses_before_fetch(ic, tmp_path):
    tmp_path.chmod(0o700)
    archive = tmp_path / "spacr_1.5.1.5_amd64.deb"
    archive.write_bytes(b"previous evidence")
    with pytest.raises(ValueError, match="existing archive"):
        ic._debian_prepare_archive("1.5.1.5", tmp_path,
            published_digest=lambda url: "0" * 64, fetch=lambda *args: pytest.fail("overwritten"))
    assert archive.read_bytes() == b"previous evidence"


@pytest.mark.needs_symlink
def test_frontend_lock_refuses_symlink_and_releases_after_failure(ic, tmp_path):
    lock = tmp_path / "lock"
    lock.write_bytes(b"")
    lock.chmod(0o600)
    alias = tmp_path / "alias"
    alias.symlink_to(lock)
    with pytest.raises(OSError):
        with ic._debian_frontend_lock(alias):
            pytest.fail("symlink lock accepted")
    with pytest.raises(RuntimeError):
        with ic._debian_frontend_lock(lock):
            raise RuntimeError("fixture interrupted")
    with ic._debian_frontend_lock(lock):
        pass


@pytest.mark.skipif(not all(shutil.which(name) for name in ("fakeroot", "dpkg", "dpkg-deb")),
                    reason="requires unprivileged Debian packaging tools")
@pytest.mark.parametrize("reject_new_runtime", [False, True])
def test_actual_private_dpkg_replacement_and_rollback(ic, tmp_path, reject_new_runtime):
    tool = Path(__file__).parents[1] / "tools" / "accept_frozen_debian_update.py"
    spec = importlib.util.spec_from_file_location("private_debian_acceptance", tool)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    packages = {}
    for version in ("1.5.1.4", "1.5.1.5"):
        stage = tmp_path / version
        (stage / "DEBIAN").mkdir(parents=True)
        (stage / "DEBIAN" / "control").write_text(
            f"Package: spacr\nVersion: {version}\nArchitecture: amd64\n"
            "Maintainer: fixture <test@example.org>\nDescription: isolated fixture\n", encoding="utf-8")
        bundle = stage / "opt" / "spacr"
        bundle.mkdir(parents=True)
        (bundle / "spacr").write_text(version, encoding="utf-8")
        (bundle / ("old-only" if version.endswith("4") else "new-only")).write_text(version, encoding="utf-8")
        archive = tmp_path / f"spacr_{version}_amd64.deb"
        subprocess.run(["dpkg-deb", "--root-owner-group", "-Znone", "--build", str(stage), str(archive)],
                       check=True, capture_output=True, timeout=20)
        packages[version] = archive
    database = driver.PrivateDpkg(tmp_path / "private-root")
    assert database.run("--install", str(packages["1.5.1.4"]))["exit"] == 0
    def run(command):
        if command[0].endswith("dpkg-query"):
            command = [*command[:1], f"--admindir={database.admin}", *command[1:]]
        elif command[:2] == ["/usr/bin/dpkg", "--install"]:
            command = database.command(*command[1:])
        result = subprocess.run(command, text=True, capture_output=True, timeout=20)
        return result.returncode, result.stdout + result.stderr
    def verify(version):
        assert (database.root / "opt/spacr/spacr").read_text(encoding="utf-8") == version
        return not (reject_new_runtime and version == "1.5.1.5")
    arguments = (str(packages["1.5.1.5"]), str(packages["1.5.1.4"]), "1.5.1.4", "1.5.1.5")
    if reject_new_runtime:
        with pytest.raises(RuntimeError, match="previous package and runtime were restored"):
            ic._debian_package_transaction(*arguments, verify=verify, run=run)
        expected = "1.5.1.4"
    else:
        assert ic._debian_package_transaction(*arguments, verify=verify, run=run)["state"] == "committed"
        expected = "1.5.1.5"
    assert driver.compare(driver.package_inventory(packages[expected]), database.root)["exact"]
    assert database.status()["version"] == expected


@pytest.mark.posix_only
def test_privileged_runner_drops_inherited_environment(ic, monkeypatch):
    monkeypatch.setattr(ic.os, "geteuid", lambda: 0)
    observed = []
    def invoke(command, **kwargs):
        observed.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="fixture")
    monkeypatch.setattr(ic.subprocess, "run", invoke)
    assert ic._debian_root_runner(["/usr/bin/dpkg", "--compare-versions", "2", "gt", "1"]) == (0, "fixture")
    environment = observed[0][1]["env"]
    assert environment["DPKG_FRONTEND_LOCKED"] == "1"
    assert environment["DEBIAN_FRONTEND"] == "noninteractive"
    assert "PYTHONPATH" not in environment and "LD_PRELOAD" not in environment
    with pytest.raises(ValueError):
        ic._debian_root_runner(["/tmp/arbitrary-command"])
    assert len(observed) == 1


def test_desktop_probe_executes_only_as_nonroot_with_private_profile(ic, tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    root.mkdir(mode=0o755)
    executable = root / "spacr"
    executable.write_bytes(b"not executed")
    executable.chmod(0o755)
    original = ic.os.lstat
    def trusted(path, *args, **kwargs):
        details = original(path, *args, **kwargs)
        if Path(path) in {root, executable}:
            fields = list(details)
            fields[4] = 0
            return ic.os.stat_result(fields)
        return details
    monkeypatch.setattr(ic.os, "lstat", trusted)
    monkeypatch.setattr(ic.os, "chown", lambda *args: None)
    observed = []
    uid = ic.os.getuid()
    assert uid > 0, "fixture must run without root"
    def invoke(command, **kwargs):
        observed.append((command, kwargs))
        assert kwargs["user"] == uid
        assert kwargs["extra_groups"] == []
        assert Path(kwargs["env"]["HOME"]).is_dir()
        assert kwargs["env"]["HOME"] != str(Path.home())
        assert "LD_LIBRARY_PATH" not in kwargs["env"]
        return SimpleNamespace(returncode=0, stdout="1.5.1.5\n")
    assert ic._debian_verify_desktop("1.5.1.5", uid, root=str(root), invoke=invoke)
    assert not Path(observed[0][1]["env"]["HOME"]).exists()
    with pytest.raises(ValueError, match="desktop user"):
        ic._debian_verify_desktop("1.5.1.5", 0, root=str(root), invoke=invoke)
    assert len(observed) == 1


@pytest.mark.posix_only
@pytest.mark.parametrize("command", ["prepare-debian-root", "apply-debian-root"])
def test_root_commands_refuse_unprivileged_execution_without_application_import(ic, command, capsys):
    ic.tr = ic._cleanup_translate
    assert ic.os.geteuid() != 0
    result = ic._main([command, "--version", "1.5.1.5", "--previous-version", "1.5.1.4",
                       "--desktop-uid", str(ic.os.getuid()), "--stage", "/never-created"])
    assert result == 6
    assert "graphical authorization" in capsys.readouterr().out
    assert ic._PRIVILEGED_DEBIAN_HELPER is False


def test_root_dispatch_never_imports_application_even_for_invalid_version(ic, monkeypatch, capsys):
    import builtins

    ic.tr = ic._cleanup_translate
    original_import = builtins.__import__
    application_imports = []
    def guarded_import(name, *args, **kwargs):
        if name.startswith("spacr"):
            application_imports.append(name)
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    def prepare(version, previous, uid):
        assert ic._PRIVILEGED_DEBIAN_HELPER is True
        ic._runtime_requested_version(version)
        pytest.fail("invalid version accepted")
    monkeypatch.setattr(ic, "_debian_prepare_root_transaction", prepare)
    assert ic._main(["prepare-debian-root", "--version", "../../invalid", "--previous-version", "1.5.1.4",
                     "--desktop-uid", "1000"]) == 6
    assert ic._PRIVILEGED_DEBIAN_HELPER is False
    assert "Nothing was changed" in capsys.readouterr().out
    assert application_imports == []


@pytest.mark.posix_only
@pytest.mark.parametrize("authenticated", [None, "0", "1001", "01000"])
def test_root_account_binding_refuses_missing_or_different_pkexec_identity(ic, monkeypatch, authenticated):
    monkeypatch.setattr(ic.os, "geteuid", lambda: 0)
    if authenticated is None:
        monkeypatch.delenv("PKEXEC_UID", raising=False)
    else:
        monkeypatch.setenv("PKEXEC_UID", authenticated)
    with pytest.raises(ValueError, match="graphical authorization"):
        ic._debian_authenticated_user(1000)


@pytest.mark.parametrize("failure", [None, "cancel", "checksum", "authorization", "receipt", "tampered",
    "old-desktop", "exited-desktop", "no-window"])
@pytest.mark.posix_only
def test_debian_handoff_requires_checksums_approval_and_verified_receipt(ic, tmp_path, monkeypatch, failure):
    tmp_path.chmod(0o700)
    helper = str(tmp_path / "spacr-update-helper")
    monkeypatch.setattr(ic.sys, "executable", helper)
    record = ic.InstallRecord(kind="installer", layout="linux-deb", platform="linux",
        root="/opt/spacr", version="1.5.1.4", registrations=("deb:spacr",), running=True)
    command = ["/usr/bin/pkexec", "/opt/spacr/_internal/spacr-update-helper", "fixture"]
    relaunch = ["/opt/spacr/spacr"]
    monkeypatch.setattr(ic, "_debian_update_commands", lambda *args, **kwargs: (command, relaunch))
    plan = {"adapter": "linux-deb-v1", "version": "1.5.1.5", "pid": 123456,
        "records": [asdict(record)], "ticked": [], "workdir": str(tmp_path),
        "handshake": {"schema": 1, "token": "a" * 64, "expires": time.time() + 600},
        "command": [helper, "run-plan", str(tmp_path / "plan.json")],
        "frozen_application": True, "fetch": None, "install": list(command), "relaunch": relaunch}
    if failure == "tampered":
        plan["install"] = ["/tmp/arbitrary"]
    events = []
    original_ready = ic._FrozenUpdateHandshake.ready
    def ready(handshake):
        original_ready(handshake)
        events.append("ready")
        if failure == "cancel":
            handshake.cancel()
        else:
            handshake.approve()
    monkeypatch.setattr(ic._FrozenUpdateHandshake, "ready", ready)
    monkeypatch.setattr(ic, "_desktop_process_identity", lambda _pid: (ic.os.getuid(), relaunch[0]))
    monkeypatch.setattr(ic, "_wait_for_exit", lambda _pid, **_kwargs: "exited" in events)
    def run(argv):
        assert argv[:len(command)] == command
        assert events == []
        events.append("authorized")
        if failure == "authorization":
            return 126, "authorization cancelled"
        endpoint = argv[argv.index("--phase-channel") + 1]
        with ic._UpdatePhaseClient(endpoint, plan["pid"], relaunch[0]) as channel:
            events.append("staged")
            ic._native_update_ready(channel, "1.5.1.5", plan["pid"])
            events.append("installed")
            ic._native_update_activated(channel, "1.5.1.5")
        return 0, json.dumps({"state": "stopped" if failure == "receipt" else "committed", "version": "1.5.1.5"})
    def wait(pid):
        assert pid == 123456
        events.append("exited")
        return True
    def spawn(argv, environment, cwd):
        assert argv == relaunch and cwd == str(tmp_path)
        assert environment["PYINSTALLER_RESET_ENVIRONMENT"] == "1"
        folder, binding = json.loads(environment["SPACR_UPDATE_LAUNCH"])
        channel = ic._UpdateLaunchHandshake(folder, binding)
        assert channel._read("launch-request.json")["details"] == {
            "version": "1.5.1.5", "executable": relaunch[0], "root": "/opt/spacr", "frozen": True}
        actual = {"version": "1.5.1.4" if failure == "old-desktop" else "1.5.1.5",
            "executable": relaunch[0], "module": "/opt/spacr/_internal/spacr/__init__.py", "pid": 45678}
        channel._write("launch-result.json", "starting" if failure == "no-window" else "ready", details=actual)
        process = SimpleNamespace(pid=45678, poll=lambda: 1 if failure == "exited-desktop" else None)
        if failure == "no-window":
            monkeypatch.setattr(ic.time, "time", lambda: channel.expires + 1)
        def stop():
            events.append("stopped")
            process.poll = lambda: -15
        process.terminate = stop
        process.wait = lambda timeout: -15
        events.append("relaunched")
        return process
    arguments = dict(wait=wait, run=run, spawn=spawn,
        published_digest=lambda url: None if failure == "checksum" else "0" * 64)
    machine = SimpleNamespace(home=str(tmp_path))
    if failure is None:
        assert ic._run_debian_update(plan, machine, **arguments)[0] == "1.5.1.5"
        assert events == ["authorized", "staged", "ready", "exited", "installed", "relaunched"]
    else:
        with pytest.raises((ValueError, RuntimeError)):
            ic._run_debian_update(plan, machine, **arguments)
        if failure in {"old-desktop", "exited-desktop", "no-window"}:
            assert "relaunched" in events
            assert ("stopped" in events) is (failure != "exited-desktop")
            retained = ic._FrozenUpdateHandshake(plan)._read("debian-transaction.json")
            assert retained is None
        elif failure != "receipt":
            assert "relaunched" not in events
        if failure in {"checksum", "tampered"}:
            assert "authorized" not in events


def test_initial_journal_failure_prevents_package_mutation(ic):
    runner = Packages()
    def journal(state):
        raise OSError("fixture disk failure")
    with pytest.raises(OSError, match="fixture disk"):
        ic._debian_package_transaction("new.deb", "old.deb", "1.5.1.4", "1.5.1.5",
            verify=lambda version: True, run=runner, journal=journal)
    assert not any(command[1] == "--install" for command in runner.calls)


def test_commit_journal_failure_restores_previous_package(ic):
    runner = Packages()
    states = []
    def journal(state):
        states.append(state)
        if state == "committed":
            raise OSError("fixture commit recording failure")
    with pytest.raises(RuntimeError, match="previous package and runtime were restored"):
        ic._debian_package_transaction("new.deb", "old.deb", "1.5.1.4", "1.5.1.5",
            verify=lambda version: True, run=runner, journal=journal)
    assert runner.version == "1.5.1.4"
    assert states == ["installing", "committed", "restored"]


def test_rollback_failure_retains_recovery_state(ic):
    runner = Packages(replacement=1, rollback=1)
    states = []
    with pytest.raises(RuntimeError, match="both archives are retained"):
        ic._debian_package_transaction("new.deb", "old.deb", "1.5.1.4", "1.5.1.5",
            verify=lambda version: True, run=runner, journal=states.append)
    assert states == ["installing", "recovery-required"]


@pytest.mark.posix_only
def test_journal_replacement_is_complete_private_and_retains_archives(ic, tmp_path):
    archive = tmp_path / "previous.deb"
    archive.write_bytes(b"retained rollback archive")
    for state in ("prepared", "installing", "restored"):
        ic._debian_write_journal(tmp_path, {"state": state, "rollback_archive": str(archive)})
        journal = tmp_path / "journal.json"
        assert json.loads(journal.read_text(encoding="utf-8")) == {"state": state, "rollback_archive": str(archive)}
        assert journal.stat().st_mode & 0o077 == 0
        assert archive.read_bytes() == b"retained rollback archive"
    assert not list(tmp_path.glob(".journal-*"))


@pytest.mark.parametrize("fault", [None, "marker", "owner", "changed-version", "not-newer", "user-data", "other-copy"])
def test_marked_debian_planner_binds_package_helper_and_desktop(ic, monkeypatch, fault):
    monkeypatch.setattr(ic.sys, "frozen", True, raising=False)
    record = ic.InstallRecord(kind="installer", layout="linux-deb", platform="linux",
        root="/opt/spacr", version="1.5.1.4", registrations=("deb:spacr",), running=True)
    machine = SimpleNamespace(platform="linux", executable="/opt/spacr/spacr")
    calls = []
    def run(command):
        calls.append(command)
        if command[0].endswith("dpkg-query"):
            if command[1] == "--show":
                version = "1.5.1.6" if fault == "changed-version" else "1.5.1.4"
                return 0, f"install ok installed\n{version}\namd64\n"
            assert command[1] == "--search"
            return 0, ("other" if fault == "owner" else "spacr") + ": " + command[2] + "\n"
        assert command[:2] == ["/usr/bin/dpkg", "--compare-versions"]
        return (1 if fault == "not-newer" else 0), ""
    monkeypatch.setattr(ic, "_run", run)
    monkeypatch.setattr(ic, "_user_data", lambda machine: ["/opt/spacr/models"] if fault == "user-data" else [])
    original_stat = ic.os.lstat
    def trusted_stat(path, *args, **kwargs):
        path = str(path)
        if path == "/opt" or path.startswith("/opt/spacr"):
            directory = path in {"/opt", "/opt/spacr", "/opt/spacr/_internal"}
            mode = (stat.S_IFDIR if directory else stat.S_IFREG) | 0o755
            return ic.os.stat_result((mode, 1, 1, 1, 0, 0, 64, 0, 0, 0))
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(ic.os, "lstat", trusted_stat)
    original_isfile, original_access = ic.os.path.isfile, ic.os.access
    monkeypatch.setattr(ic.os.path, "isfile", lambda path: True if path == "/usr/bin/pkexec" else original_isfile(path))
    monkeypatch.setattr(ic.os, "access", lambda path, mode: True if path == "/usr/bin/pkexec" else original_access(path, mode))
    marker = {"schema": 1, "adapter": "unsupported" if fault == "marker" else "linux-deb-v1"}
    monkeypatch.setattr(ic, "open", lambda *args, **kwargs: io.StringIO(json.dumps(marker)), raising=False)
    records = [record, record] if fault == "other-copy" else [record]
    if fault is not None:
        with pytest.raises(ValueError):
            ic._debian_update_commands(records, "1.5.1.5", machine)
    else:
        command, relaunch = ic._debian_update_commands(records, "1.5.1.5", machine)
        assert command[:3] == ["/usr/bin/pkexec", "/opt/spacr/_internal/spacr-update-helper", "prepare-debian-root"]
        assert command[-2:] == ["--desktop-uid", str(ic.os.getuid())]
        assert relaunch == ["/opt/spacr/spacr"]
    assert not any(command[1] == "--install" for command in calls)


@pytest.fixture
def privileged_debian_files(ic, tmp_path, monkeypatch):
    """Model root metadata only; keep every filesystem operation in owned storage."""
    import contextlib

    native_os = ic.os
    paths = SimpleNamespace(**vars(native_os.path))
    operations = SimpleNamespace(**vars(native_os))
    owners, events, calls = {}, [], []
    state = SimpleNamespace(version="1.5.1.4", package_owner="spacr", child_code=0, locked=0)

    def physical(path):
        value = native_os.fspath(path)
        if value == str(tmp_path) or value.startswith(str(tmp_path) + "/"):
            return value
        assert value == "/opt" or value.startswith("/opt/") or value.startswith("/var/lib/"), value
        return str(tmp_path / value.lstrip("/"))

    def logical(path):
        return "/" + str(Path(path).relative_to(tmp_path))

    def details(path):
        result = native_os.lstat(physical(path))
        values = list(result)
        values[4] = owners.get(native_os.fspath(path), 0)
        return native_os.stat_result(values)

    def descriptor_details(descriptor):
        values = list(native_os.fstat(descriptor))
        values[4] = 0
        return native_os.stat_result(values)

    paths.realpath = lambda path: logical(native_os.path.realpath(physical(path)))
    paths.lexists = lambda path: native_os.path.lexists(physical(path))
    operations.path = paths
    operations.lstat = details
    operations.fstat = descriptor_details
    operations.geteuid = lambda: 0
    operations.environ = dict(native_os.environ, PKEXEC_UID=str(native_os.getuid()),
                              PYTHONPATH="untrusted inherited value", LD_PRELOAD="untrusted inherited value")
    operations.mkdir = lambda path, mode=0o777: native_os.mkdir(physical(path), mode)
    operations.open = lambda path, flags, mode=0o777: native_os.open(physical(path), flags, mode)
    operations.replace = lambda source, target: native_os.replace(physical(source), physical(target))
    operations.unlink = lambda path: native_os.unlink(physical(path))
    operations.chmod = lambda path, mode: native_os.chmod(physical(path), mode)
    native_shutil = ic.shutil
    monkeypatch.setattr(ic, "shutil", SimpleNamespace(**dict(vars(native_shutil),
        rmtree=lambda path: native_shutil.rmtree(physical(path)))))
    monkeypatch.setattr(ic, "os", operations)
    monkeypatch.setattr(ic, "sys", SimpleNamespace(executable="/opt/spacr/_internal/spacr-update-helper"))

    original_tempfile = ic.tempfile
    def temporary_directory(*, prefix, dir):
        return logical(original_tempfile.mkdtemp(prefix=prefix, dir=physical(dir)))
    def temporary_file(*, prefix, dir):
        descriptor, path = original_tempfile.mkstemp(prefix=prefix, dir=physical(dir))
        return descriptor, logical(path)
    monkeypatch.setattr(ic, "tempfile", SimpleNamespace(mkdtemp=temporary_directory, mkstemp=temporary_file))

    parent = Path(physical("/var/lib/spacr-updates"))
    parent.mkdir(parents=True, mode=0o700)
    lock = Path(physical("/var/lib/dpkg/lock-frontend"))
    lock.parent.mkdir(parents=True)
    lock.write_bytes(b"")
    lock.chmod(0o600)
    installed = Path(physical("/opt/spacr/_internal/spacr-update-helper"))
    installed.parent.mkdir(parents=True)
    for directory in (installed.parent, installed.parent.parent, installed.parent.parent.parent):
        directory.chmod(0o755)
    installed.write_bytes(b"owned trusted helper fixture; never executable code")
    installed.chmod(0o755)
    application = Path(physical("/opt/spacr/spacr"))
    application.write_bytes(b"owned application fixture; never executed")
    application.chmod(0o755)
    real_lock = ic._debian_frontend_lock
    @contextlib.contextmanager
    def locked():
        with real_lock():
            state.locked += 1
            events.append("lock-enter")
            try:
                yield
            finally:
                state.locked -= 1
                events.append("lock-exit")
    monkeypatch.setattr(ic, "_debian_frontend_lock", locked)

    def invoke(command, **kwargs):
        calls.append((list(command), kwargs))
        assert kwargs["cwd"] == "/" and kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["stdout"] == subprocess.PIPE and kwargs["stderr"] == subprocess.STDOUT
        assert kwargs["text"] is True
        assert "PYTHONPATH" not in kwargs["env"] and "LD_PRELOAD" not in kwargs["env"]
        if command[0].startswith("/var/lib/spacr-updates/transaction-"):
            assert state.locked == 0
            assert Path(physical(command[0])).read_bytes() == installed.read_bytes()
            events.append("copied-helper")
            return SimpleNamespace(returncode=state.child_code, stdout="bounded child result")
        if command[:2] == ["/usr/bin/dpkg-deb", "--extract"]:
            assert state.locked == 0
            target = Path(physical(command[3])) / "opt/spacr"
            target.mkdir(parents=True)
            (target / "spacr").write_bytes(b"new staged application fixture")
            events.append("staged-payload")
            return SimpleNamespace(returncode=0, stdout="modeled extraction")
        assert state.locked == 1 or command[:2] == ["/usr/bin/dpkg-deb", "--field"], "package mutation escaped the frontend lock"
        assert kwargs["env"]["DPKG_FRONTEND_LOCKED"] == "1"
        assert kwargs["timeout"] == 1800
        if command[:2] == ["/usr/bin/dpkg-query", "--show"]:
            return SimpleNamespace(returncode=0, stdout=f"install ok installed\n{state.version}\namd64\n")
        if command[:2] == ["/usr/bin/dpkg-query", "--search"]:
            return SimpleNamespace(returncode=0, stdout=state.package_owner + ": " + command[2] + "\n")
        if command[:2] == ["/usr/bin/dpkg-deb", "--field"]:
            version = "1.5.1.5" if command[2].endswith("1.5.1.5_amd64.deb") else "1.5.1.4"
            return SimpleNamespace(returncode=0, stdout=f"Package: spacr\nVersion: {version}\nArchitecture: amd64\n")
        if command[:2] == ["/usr/bin/dpkg", "--compare-versions"]:
            return SimpleNamespace(returncode=0, stdout="")
        assert command[:2] == ["/usr/bin/dpkg", "--install"], command
        state.version = "1.5.1.5" if command[2].endswith("1.5.1.5_amd64.deb") else "1.5.1.4"
        events.append("install:" + state.version)
        return SimpleNamespace(returncode=0, stdout="modeled package replacement")
    monkeypatch.setattr(ic, "subprocess", SimpleNamespace(run=invoke, DEVNULL=subprocess.DEVNULL,
        PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT))

    def download(url, target):
        assert state.locked == 0
        events.append("download:" + Path(target).name)
        Path(physical(target)).write_bytes(url.encode())
        Path(physical(target)).chmod(0o600)
    monkeypatch.setattr(ic, "_download", download)
    monkeypatch.setattr(ic, "_published_digest", lambda url: hashlib.sha256(url.encode()).hexdigest())
    real_journal = ic._debian_write_journal
    journals = []
    def journal(stage, data):
        events.append("journal:" + data["state"])
        journals.append(dict(data))
        real_journal(stage, data)
    monkeypatch.setattr(ic, "_debian_write_journal", journal)
    def verify(version, uid):
        assert state.locked == 1 and uid == native_os.getuid()
        events.append("verify:" + version)
        return state.version == version
    monkeypatch.setattr(ic, "_debian_verify_desktop", verify)
    return SimpleNamespace(physical=physical, parent=parent, installed=installed, owners=owners,
        state=state, calls=calls, events=events, journals=journals, uid=native_os.getuid())


@pytest.mark.posix_only
@pytest.mark.parametrize("exit_code", [0, 17])
def test_root_preparation_copies_trusted_bytes_and_uses_only_fixed_child_argv(ic, privileged_debian_files, exit_code):
    fixture = privileged_debian_files
    fixture.parent.rmdir()
    fixture.state.child_code = exit_code
    before = fixture.installed.read_bytes()
    assert ic._debian_prepare_root_transaction("1.5.1.5", "1.5.1.4", fixture.uid) == (exit_code, "bounded child result")
    stages = list(fixture.parent.iterdir())
    assert len(stages) == 1 and stages[0].name.startswith("transaction-")
    copied = stages[0] / "spacr-update-helper"
    assert copied.read_bytes() == before == fixture.installed.read_bytes()
    assert copied.stat().st_ino != fixture.installed.stat().st_ino
    assert stat.S_IMODE(copied.stat().st_mode) == 0o700
    assert stat.S_IMODE(fixture.parent.stat().st_mode) == 0o700
    command, options = fixture.calls[-1]
    stage = "/var/lib/spacr-updates/" + stages[0].name
    assert command == [stage + "/spacr-update-helper", "apply-debian-root", "--stage", stage,
        "--version", "1.5.1.5", "--previous-version", "1.5.1.4", "--desktop-uid", str(fixture.uid)]
    assert options["env"] == {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C.UTF-8",
        "HOME": "/root", "PKEXEC_UID": str(fixture.uid), "PYINSTALLER_RESET_ENVIRONMENT": "1"}
    assert options["timeout"] == 7200
    assert fixture.events == ["lock-enter", "lock-exit", "copied-helper"]


@pytest.mark.posix_only
@pytest.mark.parametrize("fault", ["owner", "mode", "not-directory", "symlink"])
def test_root_preparation_refuses_untrusted_system_staging(ic, privileged_debian_files, fault):
    fixture = privileged_debian_files
    if fault == "owner":
        fixture.owners["/var/lib/spacr-updates"] = fixture.uid
    elif fault == "mode":
        fixture.parent.chmod(0o755)
    else:
        fixture.parent.rmdir()
        if fault == "not-directory":
            fixture.parent.write_bytes(b"not a directory")
        else:
            target = fixture.parent.with_name("untrusted-target")
            target.mkdir(mode=0o700)
            fixture.parent.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="private system directory"):
        ic._debian_prepare_root_transaction("1.5.1.5", "1.5.1.4", fixture.uid)
    assert fixture.calls == []


@pytest.mark.posix_only
@pytest.mark.parametrize("fault", ["owner", "mode", "symlink", "wrong-helper", "package-owner", "changed-version"])
def test_root_preparation_refuses_changed_installed_helper_identity(ic, privileged_debian_files, fault):
    fixture = privileged_debian_files
    source = "/opt/spacr/_internal/spacr-update-helper"
    if fault == "owner":
        fixture.owners[source] = fixture.uid
    elif fault == "mode":
        fixture.installed.chmod(0o777)
    elif fault == "symlink":
        target = fixture.installed.with_name("different-helper")
        fixture.installed.rename(target)
        fixture.installed.symlink_to(target)
    elif fault == "wrong-helper":
        ic.sys.executable = "/opt/spacr/spacr"
    elif fault == "package-owner":
        fixture.state.package_owner = "unrelated-package"
    else:
        fixture.state.version = "1.5.1.6"
    with pytest.raises(ValueError):
        ic._debian_prepare_root_transaction("1.5.1.5", "1.5.1.4", fixture.uid)
    assert list(fixture.parent.iterdir()) == []
    assert "copied-helper" not in fixture.events
    assert not any(command[:2] == ["/usr/bin/dpkg", "--install"] for command, _ in fixture.calls)


@pytest.fixture
def privileged_debian_stage(ic, privileged_debian_files):
    fixture = privileged_debian_files
    stage = fixture.parent / "transaction-owned"
    stage.mkdir(mode=0o700)
    helper = stage / "spacr-update-helper"
    helper.write_bytes(fixture.installed.read_bytes())
    helper.chmod(0o700)
    fixture.stage = "/var/lib/spacr-updates/" + stage.name
    fixture.stage_path = stage
    fixture.helper = helper
    ic.sys.executable = fixture.stage + "/spacr-update-helper"
    return fixture


@pytest.mark.posix_only
def test_root_apply_records_archives_and_lock_before_verified_commit(ic, privileged_debian_stage):
    fixture = privileged_debian_stage
    result = ic._debian_apply_root_transaction(fixture.stage, "1.5.1.5", "1.5.1.4", fixture.uid)
    assert result["state"] == "committed" and result["version"] == "1.5.1.5"
    assert fixture.events == ["journal:preparing", "download:spacr_1.5.1.5_amd64.deb",
        "download:spacr_1.5.1.4_amd64.deb", "journal:prepared", "lock-enter",
        "journal:installing", "install:1.5.1.5", "verify:1.5.1.5", "journal:committed", "lock-exit"]
    assert [data["state"] for data in fixture.journals] == ["preparing", "prepared", "installing", "committed"]
    journal = json.loads((fixture.stage_path / "journal.json").read_text(encoding="utf-8"))
    assert journal == fixture.journals[-1]
    assert journal["desktop_uid"] == fixture.uid and journal["previous_version"] == "1.5.1.4"
    for key, version in (("replacement", "1.5.1.5"), ("rollback", "1.5.1.4")):
        record = journal[key]
        payload = Path(fixture.physical(record["path"])).read_bytes()
        assert record["version"] == version and hashlib.sha256(payload).hexdigest() == record["sha256"]
    assert not list(fixture.stage_path.glob(".journal-*"))


@pytest.mark.parametrize("fault", ["wrong-root", "stage-owner", "stage-mode", "stage-symlink",
    "parent-owner", "parent-mode", "helper-owner", "helper-mode", "helper-symlink", "wrong-helper"])
@pytest.mark.posix_only
def test_root_apply_refuses_untrusted_stage_before_download_or_package_commands(ic, privileged_debian_stage, fault):
    fixture = privileged_debian_stage
    if fault == "wrong-root":
        moved = fixture.parent.parent / "wrong-root"
        moved.mkdir()
        fixture.stage_path.rename(moved / fixture.stage_path.name)
        fixture.stage = "/var/lib/wrong-root/" + fixture.stage_path.name
        ic.sys.executable = fixture.stage + "/spacr-update-helper"
    elif fault == "stage-owner":
        fixture.owners[fixture.stage] = fixture.uid
    elif fault == "stage-mode":
        fixture.stage_path.chmod(0o755)
    elif fault == "stage-symlink":
        target = fixture.stage_path.with_name("redirected-stage")
        fixture.stage_path.rename(target)
        fixture.stage_path.symlink_to(target, target_is_directory=True)
    elif fault == "parent-owner":
        fixture.owners["/var/lib/spacr-updates"] = fixture.uid
    elif fault == "parent-mode":
        fixture.parent.chmod(0o755)
    elif fault == "helper-owner":
        fixture.owners[fixture.stage + "/spacr-update-helper"] = fixture.uid
    elif fault == "helper-mode":
        fixture.helper.chmod(0o777)
    elif fault == "helper-symlink":
        target = fixture.helper.with_name("redirected-helper")
        fixture.helper.rename(target)
        fixture.helper.symlink_to(target)
    else:
        ic.sys.executable = "/opt/spacr/_internal/spacr-update-helper"
    with pytest.raises(ValueError):
        ic._debian_apply_root_transaction(fixture.stage, "1.5.1.5", "1.5.1.4", fixture.uid)
    assert fixture.events == [] and fixture.calls == [] and fixture.journals == []


@pytest.mark.posix_only
def test_root_apply_initial_journal_failure_stops_all_download_and_package_work(ic, privileged_debian_stage, monkeypatch):
    fixture = privileged_debian_stage
    def refused(stage, data):
        assert data["state"] == "preparing"
        raise OSError("owned fixture journal unavailable")
    monkeypatch.setattr(ic, "_debian_write_journal", refused)
    with pytest.raises(OSError, match="journal unavailable"):
        ic._debian_apply_root_transaction(fixture.stage, "1.5.1.5", "1.5.1.4", fixture.uid)
    assert fixture.events == [] and fixture.calls == []
    assert sorted(path.name for path in fixture.stage_path.iterdir()) == ["spacr-update-helper"]


@pytest.mark.parametrize("startup", ["failed", "uncertain"])
def test_desktop_acknowledgment_controls_debian_rollback_and_retention(ic, startup):
    runner = Packages()
    states = []
    def activated():
        assert runner.version == "1.5.1.5"
        assert "committed" not in states
        if startup == "uncertain":
            raise ic._UpdateActivationUnconfirmed("child may still be running")
        raise RuntimeError("owned child stopped after failed startup")
    with pytest.raises(RuntimeError):
        ic._debian_package_transaction("new.deb", "old.deb", "1.5.1.4", "1.5.1.5",
            verify=lambda version: runner.version == version, run=runner,
            journal=states.append, after_activate=activated)
    assert runner.version == ("1.5.1.4" if startup == "failed" else "1.5.1.5")
    assert states[-1] == ("restored" if startup == "failed" else "desktop-unconfirmed")
    assert "committed" not in states


@pytest.mark.posix_only
@pytest.mark.parametrize("cancel", [False, True])
def test_root_stages_verified_payload_before_restart_and_never_mutates_on_cancel(
        ic, privileged_debian_stage, monkeypatch, cancel):
    fixture = privileged_debian_stage
    stage = fixture.stage
    monkeypatch.setattr(ic, "_desktop_process_identity", lambda _pid: (fixture.uid, "/opt/spacr/spacr"))
    monkeypatch.setattr(ic, "_wait_for_exit", lambda _pid, **_kwargs: "restart" in fixture.events)
    original = Path(fixture.physical("/opt/spacr/spacr")).read_bytes()
    original_verify = ic._debian_verify_desktop
    def verify(version, uid, **kwargs):
        if "root" in kwargs:
            assert fixture.state.locked == 0 and uid == fixture.uid
            assert Path(fixture.physical(kwargs["root"] + "/spacr")).read_bytes() == b"new staged application fixture"
            assert Path(fixture.physical("/opt/spacr/spacr")).read_bytes() == original
            fixture.events.append("verified-stage")
            return True
        return original_verify(version, uid)
    monkeypatch.setattr(ic, "_debian_verify_desktop", verify)
    class Phase:
        def __init__(self, descriptor, pid, executable):
            assert (descriptor, pid, executable) == ("private-descriptor", 4242, "/opt/spacr/spacr")
        def _send(self, phase, **details):
            assert details["version"] == "1.5.1.5"
            if phase == "prepared":
                assert fixture.state.version == "1.5.1.4"
                assert "verified-stage" in fixture.events
            else:
                assert phase == "activated" and fixture.state.version == "1.5.1.5"
            fixture.events.append(phase)
        def _receive(self, expected, **kwargs):
            if "activate" in expected:
                if cancel:
                    return {"phase": "cancel"}
                fixture.events.append("restart")
                return {"phase": "activate"}
            assert expected == {"confirmed", "failed", "retain"}
            fixture.events.append("new-main-window")
            return {"phase": "confirmed"}
        def __exit__(self):
            fixture.events.append("channel-closed")
    monkeypatch.setattr(ic, "_UpdatePhaseClient", Phase)
    if cancel:
        with pytest.raises(RuntimeError, match="cancelled"):
            ic._debian_apply_root_transaction(stage, "1.5.1.5", "1.5.1.4", fixture.uid,
                phase_channel="private-descriptor", desktop_pid=4242)
        assert fixture.state.version == "1.5.1.4"
        assert not any(event.startswith("install:") for event in fixture.events)
    else:
        receipt = ic._debian_apply_root_transaction(stage, "1.5.1.5", "1.5.1.4", fixture.uid,
            phase_channel="private-descriptor", desktop_pid=4242)
        assert receipt["state"] == "committed"
        required = ["staged-payload", "verified-stage", "prepared", "restart", "install:1.5.1.5",
                    "activated", "new-main-window", "journal:committed"]
        assert [fixture.events.index(event) for event in required] == sorted(fixture.events.index(event) for event in required)
    assert not list(Path(fixture.physical("/opt")).glob(".spacr-staged-*"))
    assert list(Path(fixture.physical(stage)).glob("*.deb"))
