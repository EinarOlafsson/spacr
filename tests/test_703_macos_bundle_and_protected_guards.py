"""Item 703: macOS bundle checks and the protected update refuse anything unproven.

A bundle is accepted only as its own canonical directory with the frozen
spaCR identity, an executable inside it, one distribution record of the
requested release and a supported architecture. The user-owned exchange and
its recovery refuse a transaction that is not a private sibling, a staged
copy that differs, and identities or content that changed. The protected
(administrator) transaction is built only for an exact digest and one of an
image or a recovery journal, and its receipt is checked again as the
original user.

Inert bundle-shaped fixtures below ``tmp_path``; no native tool runs.
"""
from __future__ import annotations

import ctypes
import json
import os
import plistlib
import subprocess
from types import SimpleNamespace

import pytest

from spacr import install_cleanup as ic
from tests.test_frozen_macos_update import _bundle, _native, _swap, _transaction
from tests.test_install_cleanup import (
    _nothing_here_reaches_this_computer as _nothing_here_reaches_this_computer,
)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX bundle fixtures")

DIGEST = "ab" * 32


def test_an_alias_of_a_bundle_is_not_the_bundle(tmp_path):
    bundle = _bundle(tmp_path / "spaCR.app", "1.5.1.0")
    alias = tmp_path / "alias.app"
    alias.symlink_to(bundle)
    with pytest.raises(ValueError, match="aliases are unsupported"):
        ic._macos_bundle_identity(str(alias))
    with pytest.raises(ValueError, match="actual canonical directory"):
        ic._macos_tree(str(alias))


def test_another_application_identity_is_refused(tmp_path):
    bundle = _bundle(tmp_path / "spaCR.app", "1.5.1.0")
    info = plistlib.loads((bundle / "Contents" / "Info.plist").read_bytes())
    info["CFBundleIdentifier"] = "com.example.other"
    (bundle / "Contents" / "Info.plist").write_bytes(plistlib.dumps(info))
    with pytest.raises(ValueError, match="does not match the frozen spaCR family"):
        ic._macos_bundle_identity(str(bundle))


def test_a_missing_or_escaping_executable_is_refused(tmp_path):
    bundle = _bundle(tmp_path / "spaCR.app", "1.5.1.0")
    executable = bundle / "Contents" / "MacOS" / "spacr"
    executable.chmod(0o644)
    with pytest.raises(ValueError, match="missing or not executable"):
        ic._macos_bundle_identity(str(bundle))
    outside = tmp_path / "outside-spacr"
    outside.write_text("elsewhere")
    outside.chmod(0o755)
    executable.unlink()
    executable.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes its application"):
        ic._macos_bundle_identity(str(bundle))


def test_an_unsupported_architecture_is_refused_before_verification(tmp_path):
    bundle = _bundle(tmp_path / "spaCR.app", "1.5.1.0")
    asked = []

    def run(argv, **options):
        asked.append(argv[0])
        return "ppc\n" if argv == ["/usr/bin/uname", "-m"] else ""

    with pytest.raises(ValueError, match="unsupported native macOS architecture"):
        ic._macos_bundle_identity(str(bundle), "1.5.1.0", run=run)
    assert asked == ["/usr/bin/uname"]


def test_distribution_metadata_must_name_the_release_inside_the_bundle(tmp_path):
    bundle = _bundle(tmp_path / "spaCR.app", "1.5.1.0")
    metadata = bundle / "Contents" / "Resources" / "spacr-1.5.1.0.dist-info" / "METADATA"
    metadata.write_text("Name: spacr\nVersion: 1.5.0.9\n", encoding="utf-8")
    with pytest.raises(ValueError, match="version differs from the bundle version"):
        ic._macos_frozen_version_check(str(bundle), "1.5.1.0")
    outside = tmp_path / "METADATA"
    outside.write_text("Name: spacr\nVersion: 1.5.1.0\n", encoding="utf-8")
    metadata.unlink()
    metadata.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes its application"):
        ic._macos_frozen_version_check(str(bundle), "1.5.1.0")


def test_a_bundle_holding_a_special_file_has_no_inventory(tmp_path):
    bundle = _bundle(tmp_path / "spaCR.app", "1.5.1.0")
    os.mkfifo(bundle / "Contents" / "pipe")
    with pytest.raises(ValueError, match="unsupported special file"):
        ic._macos_tree(str(bundle))


def test_a_failed_native_exchange_reports_the_system_error(monkeypatch):
    class Library:
        def __init__(self):
            def renamex_np(left, right, flags):
                assert (left, right, flags) == (b"/A/one.app", b"/A/two.app", 2)
                ctypes.set_errno(18)
                return -1

            self.renamex_np = renamex_np

    monkeypatch.setattr(ctypes, "CDLL", lambda path, use_errno: Library())
    with pytest.raises(OSError) as raised:
        ic._macos_swap("/A/one.app", "/A/two.app")
    assert raised.value.errno == 18
    assert raised.value.filename == "/A/one.app" and raised.value.filename2 == "/A/two.app"


def test_a_native_tool_returns_its_output_in_a_clean_environment(monkeypatch):
    seen = {}

    def run(argv, **options):
        seen.update(options, argv=argv)
        return SimpleNamespace(returncode=0, stdout="arm64\n", stderr="")

    monkeypatch.setenv("DYLD_INSERT_LIBRARIES", "/tmp/inject.dylib")
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    monkeypatch.setattr(subprocess, "run", run)
    assert ic._macos_native(["/usr/bin/uname", "-m"]) == "arm64\n"
    assert seen["argv"] == ["/usr/bin/uname", "-m"] and seen["text"] is True
    assert "DYLD_INSERT_LIBRARIES" not in seen["env"] and "PYTHONPATH" not in seen["env"]
    assert seen["env"]["PYINSTALLER_RESET_ENVIRONMENT"] == "1"


@pytest.mark.parametrize("change, error", [
    ("not-sibling", "canonical sibling"),
    ("staged-outside", "inside its private sibling transaction"),
    ("not-private", "private to the normal user"),
    ("other-filesystem", "same filesystem"),
    ("different-bytes", "differ from the authenticated"),
])
def test_the_user_exchange_refuses_an_unsafe_transaction(tmp_path, monkeypatch, change, error):
    target, staged, transaction = _transaction(tmp_path)
    expected = ic._macos_tree(str(staged))
    if change == "not-sibling":
        moved = tmp_path / "deeper" / transaction.name
        moved.parent.mkdir()
        transaction.rename(moved)
        transaction, staged = moved, moved / staged.name
    elif change == "staged-outside":
        outside = tmp_path / "elsewhere.app"
        staged.rename(outside)
        staged = outside
    elif change == "not-private":
        transaction.chmod(0o755)
    elif change == "other-filesystem":
        real_stat = os.stat

        def other_device(path, *args, **kwargs):
            details = real_stat(path, *args, **kwargs)
            if os.fspath(path) == str(staged):
                return os.stat_result((details.st_mode, details.st_ino, details.st_dev + 1)
                                      + tuple(details)[3:])
            return details

        monkeypatch.setattr(ic.os, "stat", other_device)
    else:
        expected = dict(expected, extra={"kind": "file"})
    with pytest.raises(ValueError, match=error):
        ic._macos_replace_user_bundle(str(target), str(staged), str(transaction), "1.5.1.1",
                                      expected, run=_native, swap=_swap)
    assert (target / "analysis.csv").read_text() == "keep my results"


@pytest.mark.parametrize("swap, error", [
    ("nothing", "identities changed"),
    ("tamper", "content changed"),
])
def test_an_exchange_that_did_not_happen_as_journaled_is_refused_and_undone(tmp_path, swap, error):
    target, staged, transaction = _transaction(tmp_path)
    expected = ic._macos_tree(str(staged))
    calls = []

    def exchange(left, right):
        calls.append((left, right))
        if swap == "tamper" or len(calls) > 1:
            _swap(left, right)
        if swap == "tamper" and len(calls) == 1:
            (target / "Contents" / "added-by-someone").write_text("x")

    with pytest.raises(ValueError, match=error):
        ic._macos_replace_user_bundle(str(target), str(staged), str(transaction), "1.5.1.1",
                                      expected, run=_native, swap=exchange)
    journal = json.loads((transaction / "journal.json").read_text())
    if swap == "nothing":
        assert journal["state"] == "prepared" and len(calls) == 1
        assert (target / "analysis.csv").read_text() == "keep my results"
    else:
        assert journal["state"] == "rolled-back" and len(calls) == 2
        assert (target / "analysis.csv").read_text() == "keep my results"


def _journaled(tmp_path):
    target, staged, transaction = _transaction(tmp_path)
    ic._macos_replace_user_bundle(str(target), str(staged), str(transaction), "1.5.1.1",
                                  ic._macos_tree(str(staged)), run=_native, swap=_swap)
    return target, transaction, transaction / "journal.json"


def test_recovery_refuses_a_shared_folder_a_linked_journal_or_moved_paths(tmp_path):
    target, transaction, journal = _journaled(tmp_path)
    transaction.chmod(0o755)
    with pytest.raises(ValueError, match="original private transaction"):
        ic._macos_recover_user_bundle(str(transaction), run=_native, swap=_swap)
    transaction.chmod(0o700)
    data = json.loads(journal.read_text())
    saved = tmp_path / "saved-journal.json"
    journal.rename(saved)
    journal.symlink_to(saved)
    with pytest.raises(ValueError, match="cannot be a symlink"):
        ic._macos_recover_user_bundle(str(transaction), run=_native, swap=_swap)
    journal.unlink()
    journal.write_text(json.dumps(dict(data, backup=str(tmp_path / "previous.app"))))
    with pytest.raises(ValueError, match="differ from the original application transaction"):
        ic._macos_recover_user_bundle(str(transaction), run=_native, swap=_swap)


def test_recovery_refuses_a_journal_whose_state_disagrees_with_the_folders(tmp_path):
    target, transaction, journal = _journaled(tmp_path)
    data = json.loads(journal.read_text())
    journal.write_text(json.dumps(dict(data, state="unknown")))
    with pytest.raises(ValueError, match="journal state and actual application identities disagree"):
        ic._macos_recover_user_bundle(str(transaction), run=_native, swap=_swap)


def test_recovery_never_replaces_a_committed_old_bundle_or_one_replaced_since(tmp_path):
    target, transaction, journal = _journaled(tmp_path)
    data = json.loads(journal.read_text())
    backup = data["backup"]
    _swap(str(target), backup)
    journal.write_text(json.dumps(dict(data, state="committed")))
    with pytest.raises(ValueError, match="cannot replace changed or committed"):
        ic._macos_recover_user_bundle(str(transaction), run=_native, swap=_swap)
    other = tmp_path / "replacement.app"
    _bundle(other, "1.5.1.1")
    os.rename(target, tmp_path / "moved-away.app")
    os.rename(other, target)
    with pytest.raises(ValueError, match="one journaled application directory was replaced"):
        ic._macos_recover_user_bundle(str(transaction), run=_native, swap=_swap)


def test_frozen_recovery_requires_matching_release_arguments(monkeypatch):
    monkeypatch.setattr(ic.sys, "platform", "darwin")
    with pytest.raises(ValueError, match="exact version and published digest"):
        ic._run_macos_frozen_recovery("/Applications/.spacr-update.abcdefgh", protected=True)
    with pytest.raises(ValueError, match="apply only to protected recovery"):
        ic._run_macos_frozen_recovery("/x", version="1.5.1.1")


@pytest.mark.parametrize("arguments, error", [
    ({"digest": "AB" * 32, "transaction": "/Applications/.spacr-update.abcdefgh"},
     "exact release digest"),
    ({"digest": DIGEST}, "exactly one replacement image"),
    ({"digest": DIGEST, "image": "/tmp/spaCR-1.5.1.0.dmg", "transaction": "/t"},
     "exactly one replacement image"),
    ({"digest": DIGEST, "image": "relative/spaCR-1.5.1.1.dmg", "original_identity": [1, 2]},
     "exact absolute DMG path"),
    ({"digest": DIGEST, "image": "/tmp/spaCR-1.5.1.1.dmg", "original_identity": [1, -2]},
     "original device and directory inode"),
])
def test_the_protected_transaction_needs_an_exact_digest_and_one_source(arguments, error):
    with pytest.raises(ValueError, match=error):
        ic._macos_protected_script("1.5.1.1", **arguments)


def test_the_protected_replacement_names_the_image_and_original_identity():
    script = ic._macos_protected_script("1.5.1.1", DIGEST, image="/tmp/w/spaCR-1.5.1.1.dmg",
                                        original_identity=[16777220, 42])
    head = dict(line.split("=", 1) for line in script.splitlines()[:8])
    assert head["operation"] == "replace" and head["download"] == "/tmp/w/spaCR-1.5.1.1.dmg"
    assert head["original_identity"] == "16777220:42" and head["expected_digest"] == DIGEST


def test_the_authorized_helper_request_is_bounded_and_never_from_root(monkeypatch):
    for command, arguments in (("apply-anything", []), ("apply-macos-root", ["bad\nline"]),
                               ("uninstall-macos-root", ["x"] * 33)):
        with pytest.raises(ValueError, match="helper request is invalid"):
            ic._macos_frozen_authorized_command(command, arguments)
    monkeypatch.setattr(ic.os, "geteuid", lambda: 0)
    with pytest.raises(ValueError, match="without administrator privileges"):
        ic._macos_frozen_authorized_command("apply-macos-root", ["--x"])


def test_a_protected_update_is_requested_only_from_a_normal_user_with_an_exact_image(
        tmp_path, monkeypatch):
    """Hermetic: on a Mac with spaCR installed the last refusal used to fall through to a real
    administrator prompt. No native authorization can run, and /Applications/spaCR.app is absent."""
    monkeypatch.setattr(ic, "_macos_native",
                        lambda *a, **k: pytest.fail("a test must never request native authorization"))
    real_isdir = os.path.isdir
    monkeypatch.setattr(ic.os.path, "isdir",
                        lambda path: False if path == "/Applications/spaCR.app" else real_isdir(path))
    monkeypatch.setattr(ic.os, "geteuid", lambda: 0)
    with pytest.raises(ValueError, match="must not run as root"):
        ic._macos_request_protected_update("1.5.1.1", DIGEST, image="/x")
    monkeypatch.setattr(ic.os, "geteuid", lambda: 501)
    image = tmp_path / "spaCR-1.5.1.1.dmg"
    image.write_bytes(b"release image")
    with pytest.raises(ValueError, match="does not match the release digest"):
        ic._macos_request_protected_update("1.5.1.1", DIGEST, image=str(image))
    digest = ic._macos_digest(str(image))
    with pytest.raises(ValueError, match="mounted payload inventory is required"):
        ic._macos_request_protected_update("1.5.1.1", digest, image=str(image))
    with pytest.raises(ValueError, match="not its actual directory"):
        ic._macos_request_protected_update("1.5.1.1", digest, image=str(image),
                                           expected_inventory={})


def test_a_protected_replacement_binds_the_installed_directory_and_checks_the_receipt(
        tmp_path, monkeypatch):
    image = tmp_path / "spaCR-1.5.1.1.dmg"
    image.write_bytes(b"release image")
    digest = ic._macos_digest(str(image))
    installed = os.stat_result((0o40755, 42, 16777220, 3, 0, 80, 96, 0, 0, 0))
    real_lstat, real_isdir, real_islink = os.lstat, os.path.isdir, os.path.islink
    target = "/Applications/spaCR.app"
    monkeypatch.setattr(ic.os, "geteuid", lambda: 501)
    monkeypatch.setattr(ic.os, "lstat",
                        lambda path, *a, **k: installed if path == target else real_lstat(path, *a, **k))
    monkeypatch.setattr(ic.os.path, "isdir", lambda path: path == target or real_isdir(path))
    monkeypatch.setattr(ic.os.path, "islink", lambda path: False if path == target else real_islink(path))
    checked = []
    monkeypatch.setattr(ic, "_macos_bundle_identity",
                        lambda bundle, version=None, run=None: checked.append((bundle, version)))
    monkeypatch.setattr(ic, "_macos_tree", lambda bundle, **k: {"Contents": {"kind": "directory"}})
    commands = []

    def run(argv):
        commands.append(argv)
        return "committed\t/Applications/.spacr-update.Ab12Cd34\tdetached\n"

    receipt = ic._macos_request_protected_update(
        "1.5.1.1", digest, image=str(image),
        expected_inventory={"Contents": {"kind": "directory"}}, run=run)
    assert "original_identity=16777220:42" in commands[0][2]
    assert receipt["state"] == "committed" and receipt["original_uid"] == 501
    assert receipt["backup"] == "/Applications/.spacr-update.Ab12Cd34/previous.app"
    assert receipt["detach"] == {"status": "detached"}
    assert checked == [(target, "1.5.1.1")]
    with pytest.raises(ValueError, match="differs from the actual mounted release"):
        ic._macos_protected_receipt("1.5.1.1", digest, "committed\t/Applications/.spacr-update"
                                    ".Ab12Cd34\tdetached", 501, expected_inventory={})


def test_a_protected_receipt_must_be_valid_unchanged_and_committed_for_an_image(monkeypatch):
    checked = []
    monkeypatch.setattr(ic, "_macos_bundle_identity",
                        lambda bundle, version=None, run=None: checked.append((bundle, version)))
    monkeypatch.setattr(ic.os, "geteuid", lambda: 501)
    rolled = "rolled-back\t/Applications/.spacr-update.Ab12Cd34\tnot-mounted"
    with pytest.raises(ValueError, match="no valid transaction receipt"):
        ic._macos_protected_receipt("1.5.1.1", DIGEST, "ok", 501)
    with pytest.raises(RuntimeError, match="identity changed"):
        ic._macos_protected_receipt("1.5.1.1", DIGEST, rolled, 502)
    with pytest.raises(ValueError, match="was rolled back"):
        ic._macos_protected_receipt("1.5.1.1", DIGEST, rolled, 501, image="/x/spaCR-1.5.1.1.dmg")
    receipt = ic._macos_protected_receipt("1.5.1.1", DIGEST, rolled, 501,
                                          transaction="/Applications/.spacr-update.Ab12Cd34")
    assert receipt["state"] == "rolled-back" and receipt["detach"] == {"status": "not-mounted"}
    assert checked == [("/Applications/spaCR.app", None)]
