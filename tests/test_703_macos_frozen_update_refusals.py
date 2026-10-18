"""Item 703: the macOS frozen update refuses every unproven plan before replacing anything.

The update runs only as the normal user, for exactly one running standard
bundle that is its own canonical directory, in a parent nobody else can write,
for a numeric release whose downloaded image matches its published digest
and mounts read-only. The readiness lease, its deadline and the shutdown
approval are validated message by message, and a relaunch is refused when
the updater's identity changed.

Inert bundle fixtures below ``tmp_path``; no native tool, download or mount
runs.
"""
from __future__ import annotations

import hashlib
import json
import os
import plistlib
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from spacr import install_cleanup as ic
from tests.test_frozen_macos_update import _bundle, _native, _swap
from tests.test_install_cleanup import (
    _nothing_here_reaches_this_computer as _nothing_here_reaches_this_computer,
)

pytestmark = pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0,
                                reason="normal-user POSIX bundle fixtures")

IMAGE = b"authenticated fixture"
DIGEST = hashlib.sha256(IMAGE).hexdigest()


def _plan(tmp_path, *, records=None, version="1.5.1.1", handshake=True):
    target = _bundle(tmp_path / "spaCR.app", "1.5.1.0")
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    if records is None:
        records = [ic.asdict(ic.InstallRecord("installer", "macos-app", "macos", str(target),
                                              running=True))]
    plan = {"records": records, "ticked": [], "version": version, "workdir": str(work),
            "pid": 123, "adapter": "macos-frozen-v1"}
    if handshake:
        plan["handshake"] = {"schema": 1, "token": "a" * 64, "expires": ic.time.time() + 120}
    return plan, target, work


def _payload_native(payload, *, read_only=True, mounts=1, ditto_change=None, detach=None):
    def native(argv, **options):
        if argv[:2] == ["/usr/bin/hdiutil", "attach"]:
            mount = argv[argv.index("-mountpoint") + 1]
            shutil.copytree(payload, Path(mount) / "spaCR.app", symlinks=True)
            return plistlib.dumps({"system-entities": [{"mount-point": mount}] * mounts})
        if argv[0] == "/usr/bin/ditto":
            shutil.copytree(argv[-2], argv[-1], symlinks=True)
            if ditto_change is not None:
                ditto_change()
            return ""
        if argv[:2] == ["/usr/bin/hdiutil", "detach"]:
            if detach is not None:
                raise detach
            return ""
        return _native(argv, **options)
    return native


def test_an_update_started_as_root_is_refused(tmp_path, monkeypatch):
    plan, _target, _work = _plan(tmp_path, handshake=False)
    monkeypatch.setattr(ic.os, "geteuid", lambda: 0)
    with pytest.raises(ValueError, match="normal user"):
        ic._run_macos_frozen_update(plan, run=_native)


@pytest.mark.parametrize("change", ["two-running", "ticked", "legacy-layout"])
def test_mixed_or_legacy_installations_need_a_reviewed_migration(tmp_path, change):
    plan, target, _work = _plan(tmp_path, handshake=False)
    if change == "two-running":
        plan["records"].append(dict(plan["records"][0], root=str(tmp_path / "Other.app")))
    elif change == "ticked":
        plan["ticked"] = [str(target)]
    else:
        plan["records"][0]["layout"] = "macos-package"
    with pytest.raises(ValueError, match="separately reviewed migration"):
        ic._run_macos_frozen_update(plan, run=_native)


def test_an_aliased_application_is_not_replaced(tmp_path):
    plan, target, _work = _plan(tmp_path, handshake=False)
    alias = tmp_path / "Alias.app"
    alias.symlink_to(target)
    plan["records"][0]["root"] = str(alias)
    with pytest.raises(ValueError, match="actual canonical directory"):
        ic._run_macos_frozen_update(plan, run=_native)


def test_a_protected_application_needs_the_standard_path_and_a_handoff(tmp_path, monkeypatch):
    plan, target, _work = _plan(tmp_path, handshake=False)
    real_access = os.access
    monkeypatch.setattr(ic.os, "access", lambda path, mode: False if path == str(tmp_path)
                        else real_access(path, mode))
    with pytest.raises(ValueError, match="standard application and its desktop handoff"):
        ic._run_macos_frozen_update(plan, run=_native)


def test_an_application_in_a_world_writable_folder_is_not_replaced(tmp_path):
    plan, target, _work = _plan(tmp_path, handshake=False)
    shared = tmp_path / "shared"
    shared.mkdir()
    moved = shared / "spaCR.app"
    target.rename(moved)
    shared.chmod(0o777)
    plan["records"][0]["root"] = str(moved)
    try:
        with pytest.raises(ValueError, match="world-writable parent"):
            ic._run_macos_frozen_update(plan, run=_native)
    finally:
        shared.chmod(0o755)


def test_a_non_numeric_release_is_refused(tmp_path):
    plan, _target, _work = _plan(tmp_path, handshake=False, version="1.5.1-beta")
    with pytest.raises(ValueError, match="unsupported frozen macOS version"):
        ic._run_macos_frozen_update(plan, run=_native)


def test_a_download_that_differs_from_its_published_digest_is_refused(tmp_path, monkeypatch):
    plan, target, work = _plan(tmp_path, handshake=False)
    monkeypatch.setattr(ic, "_published_digest", lambda url: DIGEST)
    with pytest.raises(ValueError, match="differs from its published digest"):
        ic._run_macos_frozen_update(plan, run=_native, download=lambda url, path:
                                    Path(path).write_bytes(b"tampered"))
    assert not list(work.glob("mounted-*"))


@pytest.mark.parametrize("problem", ["writable-mount", "two-mounts"])
def test_an_image_that_is_not_one_read_only_mount_is_detached_and_refused(
        tmp_path, monkeypatch, problem):
    plan, target, work = _plan(tmp_path, handshake=False)
    payload = _bundle(tmp_path / "payload.app", "1.5.1.1")
    monkeypatch.setattr(ic, "_published_digest", lambda url: DIGEST)
    monkeypatch.setattr(ic.os, "statvfs", lambda path: SimpleNamespace(
        f_flag=0 if problem == "writable-mount" else os.ST_RDONLY))
    native = _payload_native(payload, mounts=2 if problem == "two-mounts" else 1)
    with pytest.raises(ValueError, match="actual read-only mount"):
        ic._run_macos_frozen_update(plan, run=native,
                                    download=lambda url, path: Path(path).write_bytes(IMAGE))
    assert (target / "Contents" / "MacOS" / "spacr").read_text().endswith("1.5.1.0")


def test_an_image_changed_while_staged_and_a_failed_detach_are_both_reported(
        tmp_path, monkeypatch):
    plan, target, work = _plan(tmp_path, handshake=False)
    payload = _bundle(tmp_path / "payload.app", "1.5.1.1")
    monkeypatch.setattr(ic, "_published_digest", lambda url: DIGEST)
    monkeypatch.setattr(ic.os, "statvfs", lambda path: SimpleNamespace(f_flag=os.ST_RDONLY))
    image = work / "spaCR-1.5.1.1.dmg"
    native = _payload_native(payload, ditto_change=lambda: image.write_bytes(b"swapped"),
                             detach=RuntimeError("mount is busy"))
    with pytest.raises(RuntimeError, match="DMG changed while its payload was staged.*mount is busy"):
        ic._run_macos_frozen_update(plan, run=native,
                                    download=lambda url, path: Path(path).write_bytes(IMAGE))


def test_a_detach_failure_without_an_update_error_is_raised(tmp_path, monkeypatch):
    plan, target, work = _plan(tmp_path, handshake=False)
    payload = _bundle(tmp_path / "payload.app", "1.5.1.1")
    monkeypatch.setattr(ic, "_published_digest", lambda url: DIGEST)
    monkeypatch.setattr(ic.os, "statvfs", lambda path: SimpleNamespace(f_flag=os.ST_RDONLY))
    monkeypatch.setattr(ic, "_macos_replace_user_bundle", lambda *a, **k: None)
    native = _payload_native(payload, detach=RuntimeError("mount is busy"))
    with pytest.raises(RuntimeError, match="mount is busy"):
        ic._run_macos_frozen_update(plan, run=native, wait=lambda pid: True,
                                    download=lambda url, path: Path(path).write_bytes(IMAGE))


def test_without_a_handoff_the_desktop_must_close_before_the_exchange(tmp_path, monkeypatch):
    plan, target, work = _plan(tmp_path, handshake=False)
    payload = _bundle(tmp_path / "payload.app", "1.5.1.1")
    monkeypatch.setattr(ic, "_published_digest", lambda url: DIGEST)
    monkeypatch.setattr(ic.os, "statvfs", lambda path: SimpleNamespace(f_flag=os.ST_RDONLY))
    with pytest.raises(RuntimeError, match="did not close; the installed bundle was not changed"):
        ic._run_macos_frozen_update(plan, run=_payload_native(payload), wait=lambda pid: False,
                                    download=lambda url, path: Path(path).write_bytes(IMAGE),
                                    swap=_swap)
    assert (target / "Contents" / "MacOS" / "spacr").read_text().endswith("1.5.1.0")


def test_a_relaunch_after_the_user_changed_is_refused_but_the_receipt_kept(monkeypatch):
    monkeypatch.setattr(ic.os, "geteuid", lambda: 502)
    result = ic._macos_relaunch({"target": "/Applications/spaCR.app", "state": "committed"}, 501,
                                run=lambda argv: pytest.fail("launched"))
    assert result["state"] == "committed"
    assert result["relaunch"]["status"] == "refused"
    result = ic._macos_relaunch({"target": "/Applications/spaCR.app"}, 0,
                                run=lambda argv: pytest.fail("launched"))
    assert result["relaunch"]["status"] == "refused"


def _lease(tmp_path, **changes):
    work = tmp_path / "readiness"
    work.mkdir(mode=0o700, exist_ok=True)
    plan = {"adapter": "macos-frozen-v1", "version": "1.5.1.1", "pid": 123, "records": [],
            "ticked": [], "workdir": str(work),
            "handshake": {"schema": 1, "token": "a" * 64, "expires": ic.time.time() + 120}}
    plan["handshake"].update(changes)
    return plan, work


@pytest.mark.parametrize("change", [{"schema": 2}, {"token": "A" * 64}, {"token": "a" * 63},
                                    {"expires": float("inf")}, {"expires": 0}])
def test_a_malformed_readiness_lease_is_refused(tmp_path, change):
    plan, _work = _lease(tmp_path, **change)
    with pytest.raises(ValueError, match="invalid frozen updater readiness lease"):
        ic._FrozenUpdateHandshake(plan)


def test_a_helper_folder_swapped_during_validation_is_refused(tmp_path, monkeypatch):
    import sys

    plan, work = _lease(tmp_path)
    real_lstat = os.lstat
    calls = []

    def swapped(path, *args, **kwargs):
        details = real_lstat(path, *args, **kwargs)
        if sys._getframe(1).f_code is ic._FrozenUpdateHandshake.__init__.__code__:
            calls.append(path)
            if len(calls) == 2:
                return os.stat_result((details.st_mode, details.st_ino + 1) + tuple(details)[2:])
        return details

    monkeypatch.setattr(ic.os, "lstat", swapped)
    with pytest.raises(ValueError, match="changed during validation"):
        ic._FrozenUpdateHandshake(plan)


def test_a_readiness_deadline_beyond_the_lease_is_refused(tmp_path):
    plan, _work = _lease(tmp_path)
    handshake = ic._FrozenUpdateHandshake(plan)
    handshake._write("readiness.json", "ready", expires=handshake.expires + 10 * ic._WAIT_SECONDS)
    with pytest.raises(ValueError, match="invalid frozen updater readiness deadline"):
        handshake.check()


def test_an_approval_in_any_other_state_is_refused(tmp_path):
    plan, _work = _lease(tmp_path)
    handshake = ic._FrozenUpdateHandshake(plan)
    handshake.ready()
    handshake._write("approved.json", "maybe")
    with pytest.raises(ValueError, match="invalid frozen updater shutdown approval"):
        handshake.wait_for_shutdown(wait=lambda pid: pytest.fail("waited for exit"))
