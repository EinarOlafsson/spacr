"""macOS adapter contracts without real hdiutil, elevation or application execution."""

import json
import os
from pathlib import Path
import plistlib
import shutil
import sys

import pytest

from spacr import install_cleanup as cleanup

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX private-directory and symlink fixtures")


def _bundle(path, version):
    """Write inert bundle-shaped bytes solely for parser and transaction tests.

    :param path: fixture bundle directory to create.
    :param version: exact fixture release recorded in plist and distribution metadata.
    """
    contents = path / "Contents"
    executable = contents / "MacOS/spacr"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(f"not executable code: {version}".encode())
    executable.chmod(0o755)
    info = {"CFBundleIdentifier": "com.einarolafsson.spacr", "CFBundleExecutable": "spacr",
            **cleanup._macos_version_fields(version)}
    (contents / "Info.plist").write_bytes(plistlib.dumps(info))
    metadata = contents / "Resources" / f"spacr-{version}.dist-info" / "METADATA"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(f"Name: spacr\nVersion: {version}\n", encoding="utf-8")
    return path


def _native(argv, **options):
    """Permit only verification tool intents; never launch a subprocess.

    :param argv: requested native-tool argument vector.
    :param options: runner keyword arguments accepted for interface compatibility.
    """
    if argv == ["/usr/bin/uname", "-m"]:
        return "arm64\n"
    assert argv[0] in {"/usr/bin/lipo", "/usr/bin/codesign"}
    if argv[0] == "/usr/bin/lipo":
        assert argv[2:] == ["-verify_arch", "arm64"] and argv[1].endswith("/Contents/MacOS/spacr")
    return ""


def _swap(left, right):
    """Model directory exchange for deterministic failure tests, not native atomicity proof.

    :param left: first fixture directory to exchange.
    :param right: second fixture directory to exchange.
    """
    temporary = str(right) + ".test-exchange"
    os.rename(left, temporary)
    os.rename(right, left)
    os.rename(temporary, right)


def _transaction(tmp_path):
    """Prepare two independently versioned inert bundles and private sibling storage.

    :param tmp_path: pytest-owned temporary parent directory.
    """
    target = _bundle(tmp_path / "spaCR.app", "1.5.1.0")
    (target / "analysis.csv").write_text("keep my results", encoding="utf-8")
    transaction = tmp_path / ".spacr-update-fixture"
    transaction.mkdir(mode=0o700)
    staged = _bundle(transaction / "previous-or-new.app", "1.5.1.1")
    return target, staged, transaction


@pytest.mark.parametrize("version,short,build", [
    ("1.5.1.0", "1.5.1", "105.1.0"), ("1.5.1.1", "1.5.1", "105.1.1"),
    ("2.0.0.0", "2.0.0", "200.0.0"), ("99.99.99.99", "99.99.99", "9999.99.99"),
])
def test_exact_package_version_maps_to_bounded_apple_fields(version, short, build):
    """The build mapping preserves the fourth component within Apple's4/2/2 digit limits.

    :param version: exact package release under test.
    :param short: expected marketing version.
    :param build: expected native build version.
    """
    assert cleanup._macos_version_fields(version) == {
        "CFBundleShortVersionString": short, "CFBundleVersion": build, "SPACRPackageVersion": version}


@pytest.mark.parametrize("version", ["0.5.1.0", "100.0.0.0", "1.100.0.0", "1.0.100.0", "1.0.0.100", "1.5.1rc1"])
def test_unrepresentable_native_versions_fail_without_truncation(version):
    """Version information must never be dropped to squeeze an invalid mapping through.

    :param version: unsupported package-version example.
    """
    with pytest.raises(ValueError):
        cleanup._macos_version_fields(version)


def test_version_build_mapping_has_a_unique_inverse():
    """Recover the exact normalized four-part tuple at all relevant component boundaries."""
    for major in (1, 9, 10, 99):
        for minor in (0, 1, 9, 99):
            for patch in (0, 1, 99):
                for revision in (0, 1, 99):
                    version = f"{major}.{minor}.{patch}.{revision}"
                    packed, native_patch, native_revision = map(int, cleanup._macos_version_fields(version)["CFBundleVersion"].split("."))
                    assert (packed // 100, packed % 100, native_patch, native_revision) == (major, minor, patch, revision)


def test_native_identity_requires_exact_distribution_and_full_package_version(tmp_path):
    """Matching marketing versions cannot hide a different internal release.

    :param tmp_path: pytest-owned bundle fixture directory.
    """
    app = _bundle(tmp_path / "spaCR.app", "1.5.1.0")
    with pytest.raises(ValueError, match="bundle version"):
        cleanup._macos_bundle_identity(str(app), "1.5.1.1", run=_native)


def test_outside_payload_symlinks_are_refused_but_old_user_links_are_retained(tmp_path):
    """Incoming links remain confined while old unrelated content is inventoried without traversal.

    :param tmp_path: pytest-owned bundle and external-file fixture directory.
    """
    app = _bundle(tmp_path / "spaCR.app", "1.5.1.1")
    external = tmp_path / "private.csv"
    external.write_text("not package data", encoding="utf-8")
    (app / "user-link").symlink_to(external)
    with pytest.raises(ValueError, match="outside"):
        cleanup._macos_tree(str(app))
    result = cleanup._macos_tree(str(app), allow_external_links=True)
    assert result["user-link"]["kind"] == "symlink"
    assert not any(entry.get("sha256") for path, entry in result.items() if path == "user-link")


def test_user_inventory_distinguishes_bundle_root_permissions(tmp_path):
    """Changing only the application root's mode must change its full inventory.

    :param tmp_path: pytest-owned inert application fixture directory.
    """
    app = _bundle(tmp_path / "spaCR.app", "1.5.1.1")
    app.chmod(0o755)
    before = cleanup._macos_tree(str(app))
    app.chmod(0o700)
    after = cleanup._macos_tree(str(app))
    assert before != after
    assert before["."]["mode"] == 0o755 and after["."]["mode"] == 0o700


def test_user_inventory_refuses_an_unreadable_subtree(tmp_path, monkeypatch):
    """A real os.walk traversal must propagate a child scandir error, not omit files.

    :param tmp_path: pytest-owned bundle with a retained unknown-data subtree.
    :param monkeypatch: injects an OS directory-read failure at that subtree only.
    """
    app = _bundle(tmp_path / "spaCR.app", "1.5.1.1")
    blocked = app / "unreadable"
    blocked.mkdir()
    retained = blocked / "analysis.csv"
    retained.write_text("keep unknown data", encoding="utf-8")
    scandir = cleanup.os.scandir

    def denied(path):
        """Model the kernel's denied directory read without relying on the runner UID.

        :param path: actual path requested by the unmodified os.walk traversal.
        """
        if os.fspath(path) == str(blocked):
            raise PermissionError("unreadable payload subtree")
        return scandir(path)

    monkeypatch.setattr(cleanup.os, "scandir", denied)
    with pytest.raises(PermissionError, match="unreadable payload subtree"):
        cleanup._macos_tree(str(app), allow_external_links=True)
    assert retained.read_text(encoding="utf-8") == "keep unknown data"


def test_verified_exchange_retains_entire_old_bundle_and_unknown_files(tmp_path):
    """The completed target is new; the old tree and results remain in its recorded backup.

    :param tmp_path: pytest-owned transaction fixture directory.
    """
    target, staged, transaction = _transaction(tmp_path)
    expected = cleanup._macos_tree(str(staged))
    receipt = cleanup._macos_replace_user_bundle(str(target), str(staged), str(transaction), "1.5.1.1", expected,
                                                 run=_native, swap=_swap)
    assert cleanup._macos_tree(str(target)) == expected
    assert (Path(receipt["backup"]) / "analysis.csv").read_text(encoding="utf-8") == "keep my results"
    assert json.loads((transaction / "journal.json").read_text())["state"] == "committed"


def test_post_exchange_verification_failure_rolls_back_actual_directories(tmp_path):
    """A zero swap outcome does not bypass installed native identity checks.

    :param tmp_path: pytest-owned transaction fixture directory.
    """
    target, staged, transaction = _transaction(tmp_path)
    before = cleanup._macos_tree(str(target), allow_external_links=True)
    expected = cleanup._macos_tree(str(staged))

    def verify(argv, **options):
        """Fail only the actual installed new bundle's native signature verification.

        :param argv: requested native verification command.
        :param options: runner keyword arguments passed to the inert native stub.
        """
        if argv[0] == "/usr/bin/codesign" and argv[-1] == str(target):
            raise RuntimeError("signature verification failed")
        return _native(argv, **options)

    with pytest.raises(RuntimeError, match="signature"):
        cleanup._macos_replace_user_bundle(str(target), str(staged), str(transaction), "1.5.1.1", expected,
                                            run=verify, swap=_swap)
    assert cleanup._macos_tree(str(target), allow_external_links=True) == before
    assert json.loads((transaction / "journal.json").read_text())["state"] == "rolled-back"


def test_crash_after_atomic_exchange_is_recoverable_from_the_written_journal(tmp_path):
    """The prepared journal plus actual directory inodes distinguish an interrupted exchange.

    :param tmp_path: pytest-owned transaction fixture directory.
    """
    target, staged, transaction = _transaction(tmp_path)
    old = cleanup._macos_tree(str(target), allow_external_links=True)

    def crash(left, right):
        """Simulate process loss immediately after filesystem exchange, before completion state.

        :param left: installed bundle fixture path.
        :param right: staged bundle fixture path.
        """
        _swap(left, right)
        raise KeyboardInterrupt("simulated process loss")

    with pytest.raises(KeyboardInterrupt):
        cleanup._macos_replace_user_bundle(str(target), str(staged), str(transaction), "1.5.1.1",
                                            cleanup._macos_tree(str(staged)), run=_native, swap=crash)
    receipt = cleanup._macos_recover_user_bundle(str(transaction), run=_native, swap=_swap)
    assert receipt["state"] == "rolled-back"
    assert cleanup._macos_tree(str(target), allow_external_links=True) == old
    assert (target / "analysis.csv").is_file()


def test_recovery_preserves_changed_content_instead_of_overwriting_it(tmp_path):
    """A journal is not authorization to replace a directory that changed afterward.

    :param tmp_path: pytest-owned transaction fixture directory.
    """
    target, staged, transaction = _transaction(tmp_path)
    cleanup._macos_replace_user_bundle(str(target), str(staged), str(transaction), "1.5.1.1",
                                        cleanup._macos_tree(str(staged)), run=_native, swap=_swap)
    sentinel = target / "post-update-result.csv"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="content changed"):
        cleanup._macos_recover_user_bundle(str(transaction), run=_native, swap=_swap)
    assert sentinel.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("interrupted_state", ["committed", "rolled-back"])
def test_restart_recovers_when_journal_publication_was_interrupted(tmp_path, monkeypatch, interrupted_state):
    """Partial journal temporaries must neither block recovery nor replace its valid state.

    This also retains the fixed temporary filename used by older candidates.

    :param tmp_path: pytest-owned transaction fixture directory.
    :param monkeypatch: pytest fixture for scoped interruption injection.
    :param interrupted_state: commit or rollback publication to interrupt.
    """
    target, staged, transaction = _transaction(tmp_path)
    old = cleanup._macos_tree(str(target), allow_external_links=True)
    original_dump = cleanup.json.dump

    def interrupted_dump(data, stream, **options):
        """Interrupt the selected journal publication after writing partial bytes.

        :param data: journal state selected for serialization.
        :param stream: open private journal temporary.
        :param options: JSON serialization options forwarded on other states.
        """
        if data["state"] == interrupted_state:
            stream.write('{"partial":')
            stream.flush()
            raise KeyboardInterrupt("interrupted journal publication")
        return original_dump(data, stream, **options)

    def verify(argv, **options):
        """Trigger verification rollback only for the rollback-publication case.

        :param argv: requested native verification command.
        :param options: runner keywords forwarded to the inert verifier.
        """
        if interrupted_state == "rolled-back" and argv[0] == "/usr/bin/codesign" and argv[-1] == str(target):
            raise RuntimeError("force rollback")
        return _native(argv, **options)

    monkeypatch.setattr(cleanup.json, "dump", interrupted_dump)
    with pytest.raises(KeyboardInterrupt):
        cleanup._macos_replace_user_bundle(str(target), str(staged), str(transaction), "1.5.1.1",
            cleanup._macos_tree(str(staged)), run=verify, swap=_swap)
    partials = list(transaction.glob("journal-*.next"))
    assert len(partials) == 1 and partials[0].read_text() == '{"partial":'
    (transaction / "journal.next").write_text("older interrupted publication")
    monkeypatch.setattr(cleanup.json, "dump", original_dump)
    for _ in range(2):
        receipt = cleanup._macos_recover_user_bundle(str(transaction), run=_native, swap=_swap)
        assert receipt["state"] == "rolled-back"
        assert cleanup._macos_tree(str(target), allow_external_links=True) == old
    assert partials[0].read_text() == '{"partial":'
    assert (transaction / "journal.next").read_text() == "older interrupted publication"


def test_commit_publication_durability_error_does_not_swap_behind_committed_journal(tmp_path, monkeypatch):
    """A failure after publishing committed state leaves identities consistent for recovery.

    :param tmp_path: pytest-owned transaction fixture directory.
    :param monkeypatch: pytest fixture for scoped durability-error injection.
    """
    target, staged, transaction = _transaction(tmp_path)
    expected = cleanup._macos_tree(str(staged))
    original_fsync = cleanup.os.fsync
    calls = 0

    def fail_committed_directory_flush(descriptor):
        """Fail only the directory flush after committed journal replacement.

        :param descriptor: file or directory descriptor passed to fsync.
        """
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("directory flush failed after committed publication")
        return original_fsync(descriptor)

    monkeypatch.setattr(cleanup.os, "fsync", fail_committed_directory_flush)
    with pytest.raises(OSError, match="directory flush"):
        cleanup._macos_replace_user_bundle(str(target), str(staged), str(transaction), "1.5.1.1",
            expected, run=_native, swap=_swap)
    assert json.loads((transaction / "journal.json").read_text())["state"] == "committed"
    assert cleanup._macos_tree(str(target)) == expected
    assert cleanup._macos_recover_user_bundle(str(transaction), run=_native, swap=_swap)["state"] == "committed"


@pytest.mark.parametrize("detach_failure,launch_failure", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("staged_change", [False, True])
def test_committed_update_reports_cleanup_and_launchservices_outcomes(tmp_path, monkeypatch, detach_failure, launch_failure, staged_change):
    """Cleanup or launcher failure retains a truthful receipt and both application trees.

    :param tmp_path: pytest-owned inert bundle and download fixture directory.
    :param monkeypatch: pytest fixture replacing network and mount inspection.
    :param detach_failure: whether the modeled detach reports a busy mount.
    :param launch_failure: whether the modeled LaunchServices request is rejected.
    :param staged_change: corrupt staged bytes before publishing readiness.
    """
    import shutil
    from types import SimpleNamespace

    if os.geteuid() == 0:
        pytest.skip("the actual replacement fixture must belong to its normal user")
    target = _bundle(tmp_path / "spaCR.app", "1.5.1.0")
    (target / "analysis.csv").write_text("preserve")
    payload = _bundle(tmp_path / "payload.app", "1.5.1.1")
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    plan = {"records": [cleanup.asdict(cleanup.InstallRecord("installer", "macos-app", "macos", str(target), running=True))],
            "ticked": [], "version": "1.5.1.1", "workdir": str(work), "pid": 123,
            "adapter": "macos-frozen-v1",
            "handshake": {"schema": 1, "token": "a" * 64,
                          "expires": cleanup.time.time() + 120}}
    handshake = cleanup._FrozenUpdateHandshake(plan)
    monkeypatch.setattr(cleanup.time, "sleep", lambda seconds: handshake.approve())
    image_bytes = b"authenticated fixture"
    digest = cleanup.hashlib.sha256(image_bytes).hexdigest()
    monkeypatch.setattr(cleanup, "_published_digest", lambda url: digest)
    monkeypatch.setattr(cleanup.os, "statvfs", lambda path: SimpleNamespace(f_flag=os.ST_RDONLY))
    original_uid = os.geteuid()
    calls = []

    def native(argv, **options):
        """Model native staging and launcher outcomes without executing a subprocess.

        :param argv: native command request made by the updater.
        :param options: runner keywords forwarded for verification commands.
        """
        calls.append((argv, os.geteuid()))
        if argv[:2] == ["/usr/bin/hdiutil", "attach"]:
            mount = argv[argv.index("-mountpoint") + 1]
            shutil.copytree(payload, Path(mount) / "spaCR.app", symlinks=True)
            return plistlib.dumps({"system-entities": [{"mount-point": mount}]})
        if argv[0] == "/usr/bin/ditto":
            shutil.copytree(argv[-2], argv[-1], symlinks=True)
            if staged_change:
                (Path(argv[-1]) / "unverified.txt").write_text("not the mounted payload")
            return ""
        if argv[:2] == ["/usr/bin/hdiutil", "detach"]:
            if detach_failure:
                raise RuntimeError("mount is busy")
            return ""
        if argv[0] == "/usr/bin/open":
            if launch_failure:
                raise RuntimeError("LaunchServices rejected the app")
            return ""
        return _native(argv, **options)

    if staged_change:
        with pytest.raises((ValueError, RuntimeError), match="staged bytes"):
            cleanup._run_macos_frozen_update(plan, run=native,
                download=lambda url, destination: Path(destination).write_bytes(image_bytes),
                wait=lambda pid: pytest.fail("unverified staging reached shutdown"), swap=_swap)
        assert handshake.status() is None
        assert not (work / "approved.json").exists()
        assert (target / "analysis.csv").read_text() == "preserve"
        assert not any(argv[0] == "/usr/bin/open" for argv, _ in calls)
        return
    receipt = cleanup._run_macos_frozen_update(plan, run=native,
        download=lambda url, destination: Path(destination).write_bytes(image_bytes),
        wait=lambda pid: handshake.status()["state"] == "ready", swap=_swap)
    assert receipt["state"] == "committed"
    assert receipt["detach"]["status"] == ("failed" if detach_failure else "detached")
    assert receipt["relaunch"]["status"] == ("failed" if launch_failure else "accepted")
    assert receipt["original_uid"] == original_uid
    assert (Path(receipt["backup"]) / "analysis.csv").read_text() == "preserve"
    assert cleanup._macos_tree(str(target)) == cleanup._macos_tree(str(payload))
    assert json.loads(Path(receipt["journal"]).read_text())["state"] == "committed"
    assert [uid for argv, uid in calls if argv[0] == "/usr/bin/open"] == [original_uid]


def _readiness_plan(tmp_path):
    """Create an owner-only, expiring plan without starting any process.

    :param tmp_path: pytest-owned private workspace.
    """
    work = tmp_path / "readiness"
    work.mkdir(mode=0o700)
    return {"adapter": "macos-frozen-v1", "version": "1.5.1.1", "pid": 123,
            "records": [], "ticked": [], "workdir": str(work),
            "handshake": {"schema": 1, "token": "a" * 64,
                          "expires": cleanup.time.time() + 120}}


@pytest.mark.parametrize("changed", ["version", "pid", "records", "ticked", "token"])
def test_readiness_cannot_be_reused_by_a_different_plan(tmp_path, changed):
    plan = _readiness_plan(tmp_path)
    cleanup._FrozenUpdateHandshake(plan).ready()
    if changed == "token":
        plan["handshake"]["token"] = "b" * 64
    else:
        plan[changed] = {"version": "1.5.1.2", "pid": 456,
                         "records": [{"root": "/another.app"}],
                         "ticked": ["/another-environment"]}[changed]
    with pytest.raises(ValueError, match="different plan"):
        cleanup._FrozenUpdateHandshake(plan).status()


def test_shutdown_approval_requires_actual_readiness(tmp_path):
    handshake = cleanup._FrozenUpdateHandshake(_readiness_plan(tmp_path))
    with pytest.raises(RuntimeError, match="not ready"):
        handshake.approve()
    assert not (Path(handshake.root) / "approved.json").exists()


def test_readiness_allows_a_symlinked_ancestor_but_preserves_plan_binding(tmp_path):
    plan = _readiness_plan(tmp_path)
    ancestor = tmp_path / "var-alias"
    ancestor.symlink_to(tmp_path, target_is_directory=True)
    plan["workdir"] = str(ancestor / "readiness")
    helper = cleanup._FrozenUpdateHandshake(plan)
    assert helper.root == str((tmp_path / "readiness").resolve())
    helper.ready()
    gui = cleanup._FrozenUpdateHandshake(plan)
    assert gui.status()["state"] == "ready"
    changed_plan = dict(plan, workdir=helper.root)
    with pytest.raises(ValueError, match="different plan"):
        cleanup._FrozenUpdateHandshake(changed_plan).status()


def test_readiness_rejects_a_symlink_as_the_helper_directory(tmp_path):
    plan = _readiness_plan(tmp_path)
    alias = tmp_path / "helper-alias"
    alias.symlink_to(plan["workdir"], target_is_directory=True)
    plan["workdir"] = str(alias)
    with pytest.raises(ValueError, match="private owner directory"):
        cleanup._FrozenUpdateHandshake(plan)
    assert not (tmp_path / "readiness" / "readiness.json").exists()


def test_long_preparation_gets_a_separate_bounded_shutdown_window(tmp_path, monkeypatch):
    now = [cleanup.time.time()]
    monkeypatch.setattr(cleanup.time, "time", lambda: now[0])
    plan = _readiness_plan(tmp_path)
    plan["handshake"]["expires"] = now[0] + cleanup._FROZEN_PREPARATION_SECONDS
    handshake = cleanup._FrozenUpdateHandshake(plan)
    now[0] += 50 * 60
    assert handshake.status() is None
    handshake.ready()
    deadline = handshake.status()["expires"]
    assert deadline == now[0] + cleanup._WAIT_SECONDS
    now[0] = deadline + 1
    with pytest.raises(RuntimeError, match="timed out"):
        handshake.approve()
    assert not (Path(handshake.root) / "approved.json").exists()


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_readiness_refuses_nonregular_messages_without_blocking(tmp_path, kind):
    handshake = cleanup._FrozenUpdateHandshake(_readiness_plan(tmp_path))
    message = Path(handshake.root) / "readiness.json"
    if kind == "symlink":
        message.symlink_to(tmp_path / "outside.json")
    else:
        os.mkfifo(message, mode=0o600)
    with pytest.raises((OSError, ValueError)):
        handshake.status()


@pytest.mark.parametrize("disarm", ["cancel", "timeout"])
def test_disarmed_plan_refuses_late_ready_and_approval(tmp_path, monkeypatch, disarm):
    handshake = cleanup._FrozenUpdateHandshake(_readiness_plan(tmp_path))
    if disarm == "cancel":
        handshake.cancel()
    else:
        monkeypatch.setattr(cleanup.time, "time", lambda: handshake.expires + 1)
    handshake._write("readiness.json", "ready")
    handshake._write("approved.json", "approved")
    with pytest.raises(RuntimeError, match="cancelled|timed out"):
        handshake.status()
    with pytest.raises(RuntimeError, match="cancelled|timed out"):
        handshake.wait_for_shutdown(lambda pid: pytest.fail("disarmed plan probed shutdown"))


def test_approved_plan_still_requires_actual_exit_and_checks_cancellation(tmp_path, monkeypatch):
    handshake = cleanup._FrozenUpdateHandshake(_readiness_plan(tmp_path))
    handshake.ready()
    handshake.approve()
    seen = []

    def wait(pid):
        seen.append(pid)
        if len(seen) == 2:
            handshake.cancel()
            return True
        return False

    monkeypatch.setattr(cleanup.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="cancelled"):
        handshake.wait_for_shutdown(wait)
    assert seen == [123, 123]


def test_helper_reports_missing_release_without_waiting_for_gui_exit(tmp_path, monkeypatch):
    from types import SimpleNamespace

    plan = _readiness_plan(tmp_path)
    target = _bundle(tmp_path / "spaCR.app", "1.5.1.0")
    (target / "analysis.csv").write_text("preserved")
    plan["records"] = [cleanup.asdict(cleanup.InstallRecord(
        "installer", "macos-app", "macos", str(target), running=True))]
    path = Path(plan["workdir"]) / "plan.json"
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(cleanup, "_published_digest", lambda url: None)
    assert cleanup._run_plan(str(path), system=SimpleNamespace(platform="macos"),
        wait=lambda pid: pytest.fail("missing release closed the GUI")) == 6
    message = cleanup._FrozenUpdateHandshake(plan).status()
    assert message["state"] == "error"
    assert "no positive published digest" in message["error"]
    assert (target / "analysis.csv").read_text() == "preserved"
    assert not (Path(plan["workdir"]) / "approved.json").exists()


def test_native_launcher_captures_rejection_in_sanitized_normal_user_environment(monkeypatch):
    """The native runner captures open failure instead of treating a child PID as launch success.

    :param monkeypatch: pytest fixture replacing subprocess execution and environment values.
    """
    from types import SimpleNamespace

    monkeypatch.setenv("PYTHONPATH", "/untrusted")
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/untrusted")
    calls = []

    def run(argv, **options):
        """Capture the native launch request and report a modeled rejection.

        :param argv: native launcher command requested by the runner.
        :param options: subprocess options, including the sanitized environment.
        """
        calls.append((argv, options))
        return SimpleNamespace(returncode=1, stderr="LaunchServices rejection", stdout="")

    monkeypatch.setattr(cleanup.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="LaunchServices rejection"):
        cleanup._macos_native(["/usr/bin/open", "-n", "/Applications/spaCR.app"])
    options = calls[0][1]
    assert options["capture_output"] and options["env"]["PYINSTALLER_RESET_ENVIRONMENT"] == "1"
    assert "PYTHONPATH" not in options["env"] and "DYLD_LIBRARY_PATH" not in options["env"]


def test_helper_preserves_committed_receipt_when_launchservices_rejects(tmp_path, monkeypatch):
    """The logged outcome distinguishes a successful replacement from a failed relaunch.

    :param tmp_path: pytest-owned plan and log directory.
    :param monkeypatch: pytest fixture supplying the completed replacement receipt.
    """
    from types import SimpleNamespace

    plan_path, log = tmp_path / "plan.json", tmp_path / "update.log"
    plan = _readiness_plan(tmp_path)
    plan["log"] = str(log)
    plan_path.write_text(json.dumps(plan))
    receipt = {"state": "committed", "backup": "/retained/previous.app", "journal": "/retained/journal.json",
               "relaunch": {"status": "failed", "error": "LaunchServices rejection"}}
    monkeypatch.setattr(cleanup, "_run_macos_frozen_update", lambda *args, **kwargs: receipt)
    assert cleanup._run_plan(str(plan_path), system=SimpleNamespace(platform="macos")) != 0
    assert json.dumps(receipt, sort_keys=True) in log.read_text()
    assert "replacement committed" in log.read_text()


def test_frozen_helper_requires_the_gui_readiness_protocol(tmp_path, monkeypatch):
    from types import SimpleNamespace

    plan = _readiness_plan(tmp_path)
    del plan["handshake"]
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(cleanup, "_run_macos_frozen_update",
                        lambda *a, **k: pytest.fail("a handshakeless plan reached replacement"))
    assert cleanup._run_plan(str(path), system=SimpleNamespace(platform="macos")) == 6


def test_native_rename_binding_requests_atomic_swap_not_replace(monkeypatch):
    """The actual Darwin backend selects RENAME_SWAP and propagates native failure.

    :param monkeypatch: pytest fixture replacing the Darwin dynamic-library loader.
    """
    import ctypes

    calls = []

    class Exchange:
        """Record native ABI use without loading a platform library."""

        def __call__(self, left, right, flags):
            """Return success after retaining exact bytes and the swap flag.

            :param self: inert native-binding spy instance.
            :param left: encoded first bundle path.
            :param right: encoded second bundle path.
            :param flags: native rename-operation flags.
            """
            calls.append((left, right, flags))
            return 0

    class Library:
        """Expose only the requested native symbol."""
        renamex_np = Exchange()

    monkeypatch.setattr(ctypes, "CDLL", lambda name, **options: Library())
    cleanup._macos_swap("/Applications/spaCR.app", "/Applications/.private/new.app")
    assert calls == [(b"/Applications/spaCR.app", b"/Applications/.private/new.app", 2)]


def test_builder_native_metadata_precedes_codesign_and_preserves_full_release(tmp_path, monkeypatch):
    """Execute only the builder's pure plist edit, never PyInstaller or codesign.

    :param tmp_path: pytest-owned plist fixture directory.
    :param monkeypatch: pytest fixture providing the builder snippet's arguments.
    """
    script = (Path(__file__).resolve().parents[1] / "packaging/build_macos.sh").read_text(encoding="utf-8")
    body = script.split("<<'BUNDLE_VERSION'\n", 1)[1].split("\nBUNDLE_VERSION", 1)[0]
    app = _bundle(tmp_path / "spaCR.app", "1.5.1.0")
    monkeypatch.setattr(sys, "argv", ["bundle-version", str(app), "1.5.1.1"])
    exec(compile(body, "build_macos.sh:BUNDLE_VERSION", "exec"), {"__name__": "__main__"})
    actual = plistlib.loads((app / "Contents/Info.plist").read_bytes())
    assert all(actual[key] == value for key, value in cleanup._macos_version_fields("1.5.1.1").items())
    assert script.index("<<'BUNDLE_VERSION'") < script.index("codesign --force")


def _protected_script(argv):
    """Decode the inert AppleScript string without invoking AppleScript or a shell.

    :param argv: protected-boundary argument vector to decode for inspection.
    """
    import ast

    assert argv[:2] == ["/usr/bin/osascript", "-e"]
    expression = argv[2]
    prefix = "do shell script "
    suffix = " with administrator privileges without altering line endings"
    assert expression.startswith(prefix) and expression.endswith(suffix)
    return ast.literal_eval(expression[len(prefix):-len(suffix)])


def test_protected_boundary_quotes_paths_as_data_and_never_executes_our_helper(tmp_path):
    """Shell metacharacters stay inside one assignment value.

    :param tmp_path: isolated download directory with adversarial spelling.
    """
    import shlex

    path = str(tmp_path / "O'Brien;$(touch SHOULD_NOT_EXIST)" / "spaCR-1.5.1.1.dmg")
    argv = cleanup._macos_protected_command("1.5.1.1", "a" * 64, image=path, original_identity=(1, 42))
    script = _protected_script(argv)
    assignments = dict(shlex.split(line)[0].split("=", 1) for line in script.splitlines()[:8])
    assert assignments["download"] == path
    assert assignments["original_identity"] == "1:42"
    assert assignments["operation"] == "replace"
    assert "spacr-update-helper" not in script
    assert "python" not in script.lower()
    assert "target=/Applications/spaCR.app" in script
    assert " password " not in argv[2]


@pytest.mark.parametrize("transaction", [
    "/Applications/.spacr-update.ABCDEF12/../../outside", "/tmp/.spacr-update.ABCDEF12",
    "/Applications/.spacr-update.ABC/EF12", "/Applications/.spacr-update.ABCDEF12;id",
])
def test_protected_recovery_rejects_unbounded_transaction_paths(transaction):
    """Administrator authorization cannot grant a caller arbitrary recovery destinations.

    :param transaction: unbounded or malformed protected transaction example.
    """
    with pytest.raises(ValueError, match="exact private"):
        cleanup._macos_protected_command("1.5.1.1", "a" * 64, transaction=transaction)


def test_protected_recovery_receipt_is_bound_to_the_requested_journal(monkeypatch):
    """A different valid-looking transaction receipt cannot be accepted.

    :param monkeypatch: pytest fixture modeling the original normal-user identity.
    """
    monkeypatch.setattr(cleanup.os, "geteuid", lambda: 501)
    with pytest.raises(ValueError, match="differs"):
        cleanup._macos_request_protected_update("1.5.1.1", "a" * 64,
            transaction="/Applications/.spacr-update.ABCDEF12",
            run=lambda argv: "rolled-back\t/Applications/.spacr-update.DIFFER12\tnot-mounted\n")


def test_protected_authorization_failure_never_launches_an_application(monkeypatch):
    """Cancellation propagates directly; no fallback shell or launcher is invoked.

    :param monkeypatch: pytest fixture modeling the original normal-user identity.
    """
    calls = []
    monkeypatch.setattr(cleanup.os, "geteuid", lambda: 501)

    def cancel(argv):
        """Model the native authentication cancellation outcome only.

        :param argv: single native authorization command request to record.
        """
        calls.append(argv)
        raise RuntimeError("user canceled (-128)")

    with pytest.raises(RuntimeError, match="canceled"):
        cleanup._macos_request_protected_update("1.5.1.1", "a" * 64,
            transaction="/Applications/.spacr-update.ABCDEF12", run=cancel)
    assert len(calls) == 1 and calls[0][0] == "/usr/bin/osascript"


def test_protected_parser_cannot_source_a_journal_as_shell_code():
    """Root recovery uses fixed-key parsing and actual inode/content validation."""
    script = _protected_script(cleanup._macos_protected_command("1.5.1.1", "a" * 64,
        transaction="/Applications/.spacr-update.ABCDEF12"))
    assert "while IFS='=' read -r key value" in script
    assert 'phase=$journal_phase' in script
    assert '. "$txn/journal"' not in script
    assert 'source "$txn/journal"' not in script
    assert '/bin/mv -n "$txn/previous.app" "$target"' in script
    assert "rm -r" not in script


@pytest.mark.parametrize("state,launch_fails", [("committed", False), ("prepared", False), ("committed", True)])
def test_helper_cli_recovers_real_user_journal_and_keeps_state_on_launch_failure(
        tmp_path, monkeypatch, capsys, state, launch_fails):
    """The reachable helper entry restores interrupted identities without losing a commit.

    :param tmp_path: private bundle and journal fixtures.
    :param monkeypatch: supplies native-platform and verification/launch boundaries.
    :param capsys: captures the helper's machine-readable receipt.
    :param state: durable commit or interrupted pre-commit journal state.
    :param launch_fails: make LaunchServices reject the request after recovery succeeds.
    """
    target, staged, transaction = _transaction(tmp_path)
    cleanup._macos_replace_user_bundle(str(target), str(staged), str(transaction), "1.5.1.1",
                                      cleanup._macos_tree(str(staged)), run=_native, swap=_swap)
    journal = transaction / "journal.json"
    data = json.loads(journal.read_text())
    data["state"] = state
    journal.write_text(json.dumps(data))
    calls = []

    def native(argv, **options):
        """Capture launch and retain native identity checks as inert tool intents.

        :param argv: requested OS command.
        :param options: native-runner binary/text options.
        """
        calls.append(argv)
        if argv[0] == "/usr/bin/open":
            if launch_fails:
                raise RuntimeError("LaunchServices refused")
            return ""
        return _native(argv, **options)

    monkeypatch.setattr(cleanup.sys, "platform", "darwin")
    monkeypatch.setattr(cleanup, "_macos_native", native)
    monkeypatch.setattr(cleanup, "_macos_swap", _swap)
    code = cleanup._main(["recover-macos-frozen", str(transaction)])
    receipt = json.loads(capsys.readouterr().out)
    expected_state = "committed" if state == "committed" else "rolled-back"
    assert receipt["state"] == expected_state
    assert json.loads(journal.read_text())["state"] == expected_state
    assert code == (6 if launch_fails else 0)
    assert receipt["relaunch"]["status"] == ("failed" if launch_fails else "accepted")
    assert calls[-1] == ["/usr/bin/open", "-n", str(target)]
    old = staged if expected_state == "committed" else target
    assert (old / "analysis.csv").read_text() == "keep my results"


def test_protected_cli_keeps_committed_state_when_detach_and_relaunch_fail(monkeypatch, capsys):
    """A verified administrator receipt survives both later cleanup and launch errors.

    :param monkeypatch: supplies the original user and captured OS transaction receipt.
    :param capsys: captures the reachable helper CLI result.
    """
    calls = []
    monkeypatch.setattr(cleanup.sys, "platform", "darwin")
    monkeypatch.setattr(cleanup.os, "geteuid", lambda: 501)
    monkeypatch.setattr(cleanup, "_macos_bundle_identity", lambda *args, **options: {})

    def native(argv, **options):
        """Model only authorization's final receipt and LaunchServices refusal.

        :param argv: exact OS request to record.
        :param options: optional native-runner output controls.
        """
        calls.append(argv)
        if argv[0] == "/usr/bin/osascript":
            return "committed\t/Applications/.spacr-update.ABCDEF12\tfailed\n"
        assert argv == ["/usr/bin/open", "-n", "/Applications/spaCR.app"]
        raise RuntimeError("LaunchServices refused")

    monkeypatch.setattr(cleanup, "_macos_native", native)
    code = cleanup._main(["recover-macos-frozen", "/Applications/.spacr-update.ABCDEF12",
                          "--protected", "--version", "1.5.1.1", "--sha256", "a" * 64])
    receipt = json.loads(capsys.readouterr().out)
    assert code == 6 and receipt["state"] == "committed"
    assert receipt["detach"]["status"] == "failed"
    assert receipt["relaunch"]["status"] == "failed"
    assert receipt["backup"] == "/Applications/.spacr-update.ABCDEF12/previous.app"
    assert receipt["journal"] == "/Applications/.spacr-update.ABCDEF12/journal"
    assert receipt["original_uid"] == 501 and len(calls) == 2


@pytest.mark.parametrize("platform,uid", [("linux", 501), ("darwin", 0)])
def test_recovery_cli_refuses_wrong_platform_or_root_before_reading_a_journal(
        monkeypatch, capsys, platform, uid):
    """Making recovery reachable does not admit another family or a privileged GUI.

    :param monkeypatch: controls only the entry guard's platform and identity.
    :param capsys: captures its refusal message.
    :param platform: unsupported OS or Darwin for the root-user case.
    :param uid: simulated effective user id.
    """
    monkeypatch.setattr(cleanup.sys, "platform", platform)
    monkeypatch.setattr(cleanup.os, "geteuid", lambda: uid)
    monkeypatch.setattr(cleanup, "_macos_recover_user_bundle",
                        lambda *a, **k: pytest.fail("read a journal before the entry guard"))
    assert cleanup._main(["recover-macos-frozen", "/does-not-exist"]) == 6
    assert "original normal user" in capsys.readouterr().out


def test_frozen_bundle_without_distribution_metadata_is_identified_by_its_versions(tmp_path):
    """Real frozen bundles ship no spaCR dist-info; Info.plist and bundled _version.py identify them.

    :param tmp_path: pytest-owned bundle fixture directory.
    """
    app = _bundle(tmp_path / "spaCR.app", "1.5.1.2")
    shutil.rmtree(app / "Contents" / "Resources" / "spacr-1.5.1.2.dist-info")
    frameworks = app / "Contents" / "Frameworks"
    (frameworks / "imageio-2.38.0.dist-info").mkdir(parents=True)
    (frameworks / "imageio-2.38.0.dist-info" / "METADATA").write_text("Name: imageio\n", encoding="utf-8")
    assert cleanup._macos_bundle_identity(str(app), "1.5.1.2", run=_native)["SPACRPackageVersion"] == "1.5.1.2"
    source = frameworks / "spacr" / "_version.py"
    source.parent.mkdir()
    source.write_text('__version__ = "1.5.1.2"\n', encoding="utf-8")
    assert cleanup._macos_bundle_identity(str(app), "1.5.1.2", run=_native)["CFBundleExecutable"] == "spacr"
    source.write_text('__version__ = "1.5.1.1"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="bundle version"):
        cleanup._macos_bundle_identity(str(app), "1.5.1.2", run=_native)
    source.unlink()
    stale = frameworks / "spacr-1.5.1.1.dist-info"
    stale.mkdir()
    (stale / "METADATA").write_text("Name: spacr\nVersion: 1.5.1.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="absent or ambiguous"):
        cleanup._macos_bundle_identity(str(app), "1.5.1.2", run=_native)
