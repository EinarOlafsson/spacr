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


@pytest.mark.parametrize("digest", [None, "", "g" * 64, "a" * 63])
def test_missing_or_invalid_hash_refuses_before_download(ic, tmp_path, digest):
    tmp_path.chmod(0o700)
    def unexpected(*args):
        pytest.fail("unverified package was downloaded")
    with pytest.raises(ValueError, match="published SHA256"):
        ic._debian_prepare_archive("1.5.1.5", tmp_path, fetch=unexpected,
                                   published_digest=lambda url: digest)
    assert not list(tmp_path.iterdir())


def test_bad_download_retained_without_installation(ic, tmp_path):
    tmp_path.chmod(0o700)
    def fetch(url, path):
        Path(path).write_bytes(b"wrong")
        Path(path).chmod(0o600)
    with pytest.raises(ValueError, match="published SHA256"):
        ic._debian_prepare_archive("1.5.1.5", tmp_path, fetch=fetch,
                                   published_digest=lambda url: "0" * 64)
    assert (tmp_path / "spacr_1.5.1.5_amd64.deb").read_bytes() == b"wrong"


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


def test_existing_archive_refuses_before_fetch(ic, tmp_path):
    tmp_path.chmod(0o700)
    archive = tmp_path / "spacr_1.5.1.5_amd64.deb"
    archive.write_bytes(b"previous evidence")
    with pytest.raises(ValueError, match="existing archive"):
        ic._debian_prepare_archive("1.5.1.5", tmp_path,
            published_digest=lambda url: "0" * 64, fetch=lambda *args: pytest.fail("overwritten"))
    assert archive.read_bytes() == b"previous evidence"


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
            "Maintainer: fixture <test@example.org>\nDescription: isolated fixture\n")
        bundle = stage / "opt" / "spacr"
        bundle.mkdir(parents=True)
        (bundle / "spacr").write_text(version)
        (bundle / ("old-only" if version.endswith("4") else "new-only")).write_text(version)
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
        assert (database.root / "opt/spacr/spacr").read_text() == version
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


@pytest.mark.parametrize("authenticated", [None, "0", "1001", "01000"])
def test_root_account_binding_refuses_missing_or_different_pkexec_identity(ic, monkeypatch, authenticated):
    monkeypatch.setattr(ic.os, "geteuid", lambda: 0)
    if authenticated is None:
        monkeypatch.delenv("PKEXEC_UID", raising=False)
    else:
        monkeypatch.setenv("PKEXEC_UID", authenticated)
    with pytest.raises(ValueError, match="graphical authorization"):
        ic._debian_authenticated_user(1000)


@pytest.mark.parametrize("failure", [None, "cancel", "checksum", "authorization", "receipt", "tampered"])
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
    def run(argv):
        assert argv == command
        assert events == ["ready", "exited"]
        events.append("authorized")
        if failure == "authorization":
            return 126, "authorization cancelled"
        return 0, json.dumps({"state": "stopped" if failure == "receipt" else "committed", "version": "1.5.1.5"})
    def wait(pid):
        assert pid == 123456
        events.append("exited")
        return True
    arguments = dict(wait=wait, run=run, spawn=lambda *args: events.append("relaunched"),
        published_digest=lambda url: None if failure == "checksum" else "0" * 64)
    machine = SimpleNamespace(home=str(tmp_path))
    if failure is None:
        assert ic._run_debian_update(plan, machine, **arguments)[0] == "1.5.1.5"
        assert events == ["ready", "exited", "authorized", "relaunched"]
    else:
        with pytest.raises((ValueError, RuntimeError)):
            ic._run_debian_update(plan, machine, **arguments)
        assert "relaunched" not in events
        if failure in {"cancel", "checksum", "tampered"}:
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


def test_journal_replacement_is_complete_private_and_retains_archives(ic, tmp_path):
    archive = tmp_path / "previous.deb"
    archive.write_bytes(b"retained rollback archive")
    for state in ("prepared", "installing", "restored"):
        ic._debian_write_journal(tmp_path, {"state": state, "rollback_archive": str(archive)})
        journal = tmp_path / "journal.json"
        assert json.loads(journal.read_text()) == {"state": state, "rollback_archive": str(archive)}
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
