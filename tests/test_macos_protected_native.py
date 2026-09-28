"""Opt-in native filesystem/AppleScript semantics; no elevation or app installation."""

import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from spacr import install_cleanup as cleanup

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or os.environ.get("SPACR_TEST_MACOS_UPDATE_SEMANTICS") != "1",
    reason="requires an explicitly assigned native macOS acceptance scope")


def test_native_apple_script_string_encoding_cannot_execute_the_embedded_transaction(tmp_path):
    """Ask AppleScript to return the exact string without authorization.

    :param tmp_path: isolated download directory with adversarial spelling.
    """
    path = str(tmp_path / "O'Brien;$(touch SHOULD_NOT_EXIST)" / "spaCR-1.5.1.1.dmg")
    argv = cleanup._macos_protected_command("1.5.1.1", "a" * 64, image=path, original_identity=(1, 42))
    expression = argv[2]
    literal = expression[len("do shell script "):-len(" with administrator privileges without altering line endings")]
    result = subprocess.run(["/usr/bin/osascript", "-e", "return " + literal], check=True, capture_output=True, text=True)
    assert "download=" in result.stdout and "SHOULD_NOT_EXIST" in result.stdout
    assert "target=/Applications/spaCR.app" in result.stdout
    assert "with administrator privileges" not in result.stdout


def test_native_copy_recreates_symlink_instead_of_reading_its_replaced_target(tmp_path):
    """The root-private-copy recipe must not follow an attacker-replaced source symlink.

    :param tmp_path: pytest-owned inert source and copy directory.
    """
    secret = tmp_path / "private-fixture"
    secret.write_bytes(b"fixture only; no privileged file is read")
    source = tmp_path / "download.dmg"
    source.symlink_to(secret)
    destination = tmp_path / "copy.dmg"
    subprocess.run(["/bin/cp", "-R", "-P", "-X", str(source), str(destination)], check=True)
    assert destination.is_symlink()
    assert not (not destination.is_symlink() and destination.is_file())


def test_native_copy_recreates_fifo_without_blocking_for_input(tmp_path):
    """A swapped special source is recreated and then rejected by the regular-file guard.

    :param tmp_path: pytest-owned FIFO and copy directory.
    """
    source = tmp_path / "download.dmg"
    os.mkfifo(source)
    destination = tmp_path / "copy.dmg"
    subprocess.run(["/bin/cp", "-R", "-P", "-X", str(source), str(destination)], check=True, timeout=10)
    assert not destination.is_file()


def test_native_find_propagates_batch_hash_failure(tmp_path):
    """The inventory's find-exec-plus form must fail when its hash subprocess fails.

    :param tmp_path: pytest-owned traversal fixture directory.
    """
    (tmp_path / "payload").write_bytes(b"fixture")
    result = subprocess.run(["/usr/bin/find", "-s", str(tmp_path), "-type", "f", "-exec", "/usr/bin/false", "{}", "+"])
    assert result.returncode != 0


def test_native_atomic_swap_preserves_both_real_directory_identities(tmp_path):
    """Exercise the actual Darwin renamex_np backend on two disposable directories.

    :param tmp_path: pytest-owned bundle exchange fixture directory.
    """
    old, new = tmp_path / "old.app", tmp_path / "new.app"
    old.mkdir()
    new.mkdir()
    (old / "unknown.csv").write_text("preserve", encoding="utf-8")
    (new / "new.txt").write_text("new", encoding="utf-8")
    old_inode, new_inode = old.stat().st_ino, new.stat().st_ino
    cleanup._macos_swap(str(old), str(new))
    assert old.stat().st_ino == new_inode and new.stat().st_ino == old_inode
    assert (new / "unknown.csv").read_text(encoding="utf-8") == "preserve"
    cleanup._macos_swap(str(old), str(new))
    assert old.stat().st_ino == old_inode and new.stat().st_ino == new_inode


def _function(name):
    """Extract the exact candidate shell function for unprivileged native semantic testing.

    :param name: literal shell-function name within the embedded candidate transaction.
    :returns: that function's unmodified shell definition.
    """
    match = re.search(r"(?ms)^" + re.escape(name) + r"\(\) \{\n.*?^\}\n", cleanup._MACOS_PROTECTED_TRANSACTION)
    assert match is not None
    return match.group()


def _inventory_script(tree, destination):
    """Prepare the candidate inventory command with its actual byte-counting locale.

    :param tree: fixture tree to inventory.
    :param destination: inventory output path outside that tree.
    :returns: quoted shell script; constructing it executes nothing.
    """
    import shlex

    return "set -eu\nLC_ALL=C\nexport LC_ALL\n" + _function("inventory") + \
        "\ninventory " + shlex.quote(str(tree)) + " " + shlex.quote(str(destination)) + "\n"


def test_native_inventory_rejects_unknown_fifo_in_actual_conditional_context(tmp_path):
    """A guard failure must propagate through matches_inventory's OR-list call context.

    :param tmp_path: pytest-owned inventory and unknown-FIFO fixture directory.
    """
    import shlex

    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "known.txt").write_text("known")
    expected = tmp_path / "expected.inventory"
    subprocess.run(["/bin/sh", "-c", _inventory_script(tree, expected)], check=True, timeout=10)
    unknown = tree / "unknown.fifo"
    os.mkfifo(unknown)
    script = "set -eu\nLC_ALL=C\nexport LC_ALL\n" + _function("inventory") + _function("matches_inventory")
    script += "\ntxn=" + shlex.quote(str(tmp_path)) + "\n"
    script += "if matches_inventory " + shlex.quote(str(tree)) + " " + shlex.quote(str(expected)) + \
        "; then exit 91; fi\n"
    subprocess.run(["/bin/sh", "-c", script], check=True, timeout=10)
    assert unknown.exists() and (tree / "known.txt").read_text() == "known"


@pytest.mark.parametrize("change", ["file-mode", "directory-mode", "link-space-boundary", "link-trailing-newline"])
def test_native_inventory_distinguishes_modes_and_exact_link_bytes(tmp_path, change):
    """Different permissions or path/target boundaries cannot have the same inventory.

    :param tmp_path: pytest-owned tree and inventory output directory.
    :param change: permission or symlink-byte change applied after the first inventory.
    """
    tree = tmp_path / "tree"
    tree.mkdir()
    regular = tree / "payload"
    regular.write_text("unchanged bytes")
    regular.chmod(0o644)
    folder = tree / "directory"
    folder.mkdir(mode=0o755)
    folder.chmod(0o755)
    assert folder.stat().st_mode & 0o777 == 0o755
    link = tree / "a b"
    link.symlink_to("c")
    before, after = tmp_path / "before.inventory", tmp_path / "after.inventory"
    subprocess.run(["/bin/sh", "-c", _inventory_script(tree, before)], check=True, timeout=10)
    if change == "file-mode":
        regular.chmod(0o600)
    elif change == "directory-mode":
        folder.chmod(0o700)
    else:
        link.unlink()
        if change == "link-space-boundary":
            (tree / "a").symlink_to("b c")
        else:
            link.symlink_to("c\n")
    subprocess.run(["/bin/sh", "-c", _inventory_script(tree, after)], check=True, timeout=10)
    before_bytes, after_bytes = before.read_bytes(), after.read_bytes()
    assert before_bytes.count(b"\0") == after_bytes.count(b"\0") == 4
    assert before_bytes.endswith(b"\0") and after_bytes.endswith(b"\0")
    assert before_bytes != after_bytes
    if change == "link-trailing-newline":
        assert b" 2:c\n\0" in after_bytes
        link.unlink()
        link.symlink_to("c\n\n")
        multiple = tmp_path / "multiple-newlines.inventory"
        subprocess.run(["/bin/sh", "-c", _inventory_script(tree, multiple)], check=True, timeout=10)
        multiple_bytes = multiple.read_bytes()
        assert b" 3:c\n\n\0" in multiple_bytes
        assert multiple_bytes != after_bytes


@pytest.mark.parametrize("interruption", ["prepared", "old-renamed", "new-renamed", "recovery-halfway"])
def test_native_restart_recovery_preserves_unknown_old_files(tmp_path, interruption):
    """Run the exact recovery core on scratch trees; root/ACL/bootstrap checks remain separate.

    :param tmp_path: pytest-owned transaction and bundle fixture directory.
    :param interruption: rename boundary whose filesystem state is reconstructed.
    """
    import shlex

    target = tmp_path / "spaCR.app"
    transaction = tmp_path / ".spacr-update.ABCDEF12"
    transaction.mkdir(mode=0o700)
    target.mkdir()
    (target / "unknown.csv").write_text("preserve", encoding="utf-8")
    new = transaction / "new.app"
    new.mkdir()
    (new / "new.txt").write_text("new", encoding="utf-8")
    old_identity = f"{target.stat().st_dev}:{target.stat().st_ino}"
    new_identity = f"{new.stat().st_dev}:{new.stat().st_ino}"
    values = {"target": str(target), "txn": str(transaction), "old_identity": old_identity,
              "new_identity": new_identity, "package_version": "1.5.1.1", "expected_digest": "a" * 64,
              "private_stage": str(tmp_path), "phase": "prepared"}
    script = "set -eu\nLC_ALL=C\nexport LC_ALL\n" + "\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items()) + "\n"
    script += "\n".join(_function(name) for name in ("identity", "inventory", "matches_inventory", "write_journal", "restore_transaction"))
    script += '\nprivate_path() { [ ! -L "$1" ] && [ -e "$1" ]; }\n'
    script += 'inventory "$target" "$txn/old.inventory"\ninventory "$txn/new.app" "$txn/source.inventory"\n'
    if interruption != "prepared":
        script += '/bin/mv "$target" "$txn/previous.app"\n'
    if interruption in {"new-renamed", "recovery-halfway"}:
        script += '/bin/mv "$txn/new.app" "$target"\n'
    if interruption == "recovery-halfway":
        script += '/bin/mv "$target" "$txn/failed.app"\n'
    script += 'restore_transaction || exit 78\n[ "$phase" = rolled-back ]\n'
    subprocess.run(["/bin/sh", "-c", script], check=True, timeout=30)
    assert target.stat().st_ino == int(old_identity.split(":")[1])
    assert (target / "unknown.csv").read_text(encoding="utf-8") == "preserve"
    retained = transaction / ("new.app" if interruption in {"prepared", "old-renamed"} else "failed.app")
    assert (retained / "new.txt").read_text(encoding="utf-8") == "new"


def test_native_readonly_mount_assertion_matches_kernel_flags(tmp_path):
    """Mount a tiny inert DMG and compare the exact shell check with actual kernel flags.

    :param tmp_path: pytest-owned image source, DMG and mountpoint directory.
    """
    import shlex

    source = tmp_path / "payload"
    source.mkdir()
    (source / "fixture.txt").write_text("no executable payload", encoding="utf-8")
    image = tmp_path / "fixture.dmg"
    subprocess.run(["/usr/bin/hdiutil", "create", "-quiet", "-size", "32m", "-fs", "HFS+",
                    "-srcfolder", str(source), "-format", "UDZO", str(image)], check=True, timeout=60)
    mount = tmp_path / "mounted"
    mount.mkdir()
    try:
        subprocess.run(["/usr/bin/hdiutil", "attach", "-readonly", "-nobrowse", "-mountpoint", str(mount),
                        str(image)], check=True, capture_output=True, timeout=60)
        assert os.statvfs(mount).f_flag & os.ST_RDONLY
        command = _function("readonly_mount") + "\nreadonly_mount " + shlex.quote(str(mount))
        subprocess.run(["/bin/sh", "-c", command], check=True, timeout=10)
    finally:
        subprocess.run(["/usr/bin/hdiutil", "detach", str(mount)], check=True, capture_output=True, timeout=60)


@pytest.mark.parametrize("state", ["committed", "rolled-back"])
@pytest.mark.parametrize("ready", ["0", "1"])
def test_native_terminal_receipt_survives_actual_detach_failure(tmp_path, state, ready):
    """A real failing hdiutil command cannot erase a durably completed transaction.

    :param tmp_path: scratch path deliberately containing no mounted device.
    :param state: verified terminal state supplied to the real cleanup function.
    :param ready: whether that terminal journal was successfully published.
    """
    import shlex

    values = {"phase": state, "txn": "/Applications/.spacr-update.ABCDEF12",
              "private_stage": str(tmp_path), "mounted": "1", "receipt_ready": ready,
              "old_identity": "", "new_identity": ""}
    script = "set -eu\n" + "\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items())
    script += "\n" + _function("finish") + "\ntrap finish EXIT\nexit 0\n"
    result = subprocess.run(["/bin/sh", "-c", script], capture_output=True, text=True, timeout=15)
    assert result.returncode == (0 if ready == "1" else 77)
    expected = f"{state}\t/Applications/.spacr-update.ABCDEF12\tfailed" if ready == "1" else ""
    assert result.stdout.strip() == expected
    assert "Read-only mount retained" in result.stderr
    if ready == "1":
        literal = script.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        transported = subprocess.run(["/usr/bin/osascript", "-e",
            f'do shell script "{literal}" without altering line endings'],
            capture_output=True, text=True, timeout=15)
        assert transported.returncode == 0
        assert transported.stdout.strip() == expected


def test_native_journal_publication_propagates_printf_failure_in_conditional_context(tmp_path):
    """A failed journal write cannot publish partial state or consume old interrupted files.

    :param tmp_path: private transaction directory with retained journal artifacts.
    """
    import shlex

    journal = tmp_path / "journal"
    journal.write_text("previous durable state\n")
    interrupted = tmp_path / "journal.next.INTERRUPTED"
    interrupted.write_text("retain partial data\n")
    values = {"txn": str(tmp_path), "phase": "recovering", "package_version": "1.5.1.1",
              "expected_digest": "a" * 64, "target": "/Applications/spaCR.app",
              "old_identity": "1:2", "new_identity": "1:3", "private_stage": str(tmp_path)}
    assignments = "\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items())
    script = "set -eu\n" + assignments + "\n" + _function("write_journal")
    failed = subprocess.run(["/bin/sh", "-c", script +
        "\nprintf() { return 75; }\nif write_journal; then exit 91; fi\n"],
        capture_output=True, text=True, timeout=15)
    assert failed.returncode == 0
    assert journal.read_text() == "previous durable state\n"
    assert interrupted.read_text() == "retain partial data\n"
    partials = set(tmp_path.glob("journal.next.*"))
    assert len(partials) == 2
    subprocess.run(["/bin/sh", "-c", script + "\nwrite_journal\n"], check=True, timeout=15)
    assert "state=recovering\n" in journal.read_text()
    assert set(tmp_path.glob("journal.next.*")) == partials
