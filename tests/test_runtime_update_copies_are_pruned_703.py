"""Every in-app update kept the runtime it replaced, and nothing ever removed the older copies.

Found natively on a Mac 2026-10-19: each update left .spacr-runtime-update-*/runtime (about 3.2 GB) beside
the runtime. At most one committed rollback copy, the newest, is kept: older committed copies are pruned at
the start of an update and after a launch that confirms the new runtime. A copy rolling back, awaiting
launch confirmation, unknown or unreadable is never touched, a link is never followed, and Uninstall lists
the copies under Temporary files. The Windows private-pip update kept .spacr-windows-update-*/previous the
same way and follows the same rule.
"""
import json
import os

import pytest

from spacr import install_cleanup as ic

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX ownership fixtures")


@pytest.fixture(autouse=True)
def _private_home(tmp_path, monkeypatch):
    """update.log goes to the test's own home, never the developer's."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


@pytest.fixture
def runtime(tmp_path):
    root = tmp_path / "spaCR"
    prefix = root / "venv"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "python").write_text("python")
    return root, str(prefix.resolve())


def _copy(root, prefix, name, state, mtime, *, family="posix-runtime", new_identity=None, inventory=None):
    marker, schema, child = ic._RUNTIME_TRANSACTIONS[family]
    folder = root / (marker + name)
    (folder / child / "lib").mkdir(parents=True)
    (folder / child / "lib" / "payload").write_bytes(b"x" * 2048)
    details = os.stat(prefix)
    journal = folder / "transaction.json"
    journal.write_text(json.dumps({
        "schema": schema, "family": family, "prefix": prefix, "rollback_runtime": str(folder / child),
        "state": state, "new_identity": new_identity or [details.st_dev, details.st_ino],
        "new_inventory": inventory if inventory is not None else {"live": True}}))
    os.utime(journal, (mtime, mtime))
    return folder


def test_two_committed_copies_leave_the_newest(runtime, tmp_path):
    root, prefix = runtime
    older = _copy(root, prefix, "older", "committed", 1000)
    newer = _copy(root, prefix, "newer", "committed", 2000)
    removed = ic._prune_runtime_transactions(prefix, "posix-runtime")
    assert [path for path, _size in removed] == [str(older)]
    assert removed[0][1] >= 2048
    assert not older.exists() and newer.is_dir()
    assert str(older) in (tmp_path / "home" / ".spacr" / "logs" / "update.log").read_text()


@pytest.mark.parametrize("state", ["rolling_back", "launch_unconfirmed", "staging", "something-new"])
def test_a_copy_that_is_not_committed_is_kept(runtime, state):
    root, prefix = runtime
    kept = _copy(root, prefix, "kept", state, 500)
    older = _copy(root, prefix, "older", "committed", 1000)
    newer = _copy(root, prefix, "newer", "committed", 2000)
    ic._prune_runtime_transactions(prefix, "posix-runtime")
    assert kept.is_dir() and newer.is_dir() and not older.exists()


def test_an_unreadable_or_foreign_journal_is_kept(runtime):
    root, prefix = runtime
    broken = _copy(root, prefix, "broken", "committed", 100)
    (broken / "transaction.json").write_text("{not json")
    other = _copy(root, prefix, "other", "committed", 200)
    data = json.loads((other / "transaction.json").read_text())
    data["prefix"] = prefix + "-elsewhere"
    (other / "transaction.json").write_text(json.dumps(data))
    _copy(root, prefix, "older", "committed", 1000)
    _copy(root, prefix, "newer", "committed", 2000)
    ic._prune_runtime_transactions(prefix, "posix-runtime")
    assert broken.is_dir() and other.is_dir()


def test_a_link_is_never_followed(runtime, tmp_path):
    root, prefix = runtime
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "data").write_text("keep")
    real = _copy(root, prefix, "real", "committed", 1000)
    (real / "transaction.json").rename(outside / "transaction.json")
    data = json.loads((outside / "transaction.json").read_text())
    (outside / "transaction.json").write_text(json.dumps(dict(data, rollback_runtime=str(root / ".spacr-runtime-update-link" / "runtime"))))
    (root / ".spacr-runtime-update-link").symlink_to(outside, target_is_directory=True)
    (real / "transaction.json").symlink_to(outside / "transaction.json")
    _copy(root, prefix, "older", "committed", 1500)
    _copy(root, prefix, "newer", "committed", 2000)
    ic._prune_runtime_transactions(prefix, "posix-runtime")
    assert (outside / "data").read_text() == "keep" and (root / ".spacr-runtime-update-link").is_symlink()
    assert real.is_dir()


def test_after_launch_only_a_confirmed_newest_copy_prunes_the_rest(runtime):
    root, prefix = runtime
    older = _copy(root, prefix, "older", "committed", 1000)
    _copy(root, prefix, "newer", "committed", 2000, inventory={"live": True})
    assert ic._prune_runtime_transactions(prefix, "posix-runtime", live=True, inventory={"live": False}) == []
    assert older.is_dir()
    removed = ic._prune_runtime_transactions(prefix, "posix-runtime", live=True, inventory={"live": True})
    assert [path for path, _ in removed] == [str(older)]


def test_after_launch_an_identity_mismatch_keeps_every_copy(runtime):
    root, prefix = runtime
    older = _copy(root, prefix, "older", "committed", 1000)
    _copy(root, prefix, "newer", "committed", 2000, new_identity=[1, 2])
    assert ic._prune_runtime_transactions(prefix, "posix-runtime", live=True, inventory={"live": True}) == []
    assert older.is_dir()


def test_after_launch_a_newer_unconfirmed_transaction_keeps_every_copy(runtime):
    root, prefix = runtime
    older = _copy(root, prefix, "older", "committed", 1000)
    _copy(root, prefix, "newer", "committed", 2000)
    _copy(root, prefix, "newest", "launch_unconfirmed", 3000)
    assert ic._prune_runtime_transactions(prefix, "posix-runtime", live=True, inventory={"live": True}) == []
    assert older.is_dir()


def test_the_launch_prune_takes_the_runtime_lease_and_skips_when_an_update_holds_it(runtime, monkeypatch):
    root, prefix = runtime
    older = _copy(root, prefix, "older", "committed", 1000)
    _copy(root, prefix, "newer", "committed", 2000)
    monkeypatch.setattr(ic, "_runtime_recovery_inventory", lambda path: {"live": True})
    lock = root / ".venv.spacr-update-lock"
    ic._take_lease_dir(str(lock))
    try:
        assert ic._prune_runtime_transactions_after_launch(prefix) == []
        assert older.is_dir()
    finally:
        ic._release_lease_dir(str(lock))
    assert [path for path, _ in ic._prune_runtime_transactions_after_launch(prefix)] == [str(older)]
    assert not lock.exists()


def test_windows_private_pip_copies_follow_the_same_rule(runtime):
    root, prefix = runtime
    older = _copy(root, prefix, "older", "committed", 1000, family="windows-private-pip")
    newer = _copy(root, prefix, "newer", "committed", 2000, family="windows-private-pip")
    posix = _copy(root, prefix, "posix", "committed", 500)
    removed = ic._prune_runtime_transactions(prefix, "windows-private-pip")
    assert [path for path, _ in removed] == [str(older)] and newer.is_dir() and posix.is_dir()


def test_both_update_paths_prune_before_they_stage():
    source = open(ic.__file__, encoding="utf-8").read()
    posix = source[source.index("def _transactional_runtime_update"):]
    assert posix.index('_prune_runtime_transactions(prefix, "posix-runtime")') < posix.index(
        'tempfile.mkdtemp(prefix=".spacr-runtime-update-"')
    windows = source[source.index("def _windows_pip_transaction"):]
    assert windows.index('_prune_runtime_transactions(prefix, "windows-online-uv" if uv_tool else "windows-private-pip")') < windows.index(
        'tempfile.mkdtemp(prefix=".spacr-windows-update-"')


def test_uninstall_lists_retained_copies_under_temporary_files(tmp_path):
    from tests.test_install_cleanup import Box, linux_online

    box = Box(tmp_path)
    root = linux_online(box, "1.5.1.4")
    copy = root / ".spacr-runtime-update-abc123"
    (copy / "runtime" / "bin").mkdir(parents=True)
    copy.chmod(0o700)
    preview = ic._uninstall_preview("lists-copies", system=box.machine("linux"))
    rows = [row for row in preview["cleanup"] if row["path"] == str(copy)]
    assert rows and rows[0]["category"] == "temporary" and rows[0]["selectable"]


@pytest.mark.parametrize("family", ["windows-private-pip", "windows-online-uv"])
@pytest.mark.parametrize("state", ["staging_failed", "cancelled"])
def test_a_failed_or_cancelled_stage_is_removed_and_the_live_runtime_kept(runtime, family, state):
    root, prefix = runtime
    failed = _copy(root, prefix, "failed", state, 3000, family=family)
    (failed / "previous").rename(failed / "staged")
    committed = _copy(root, prefix, "kept", "committed", 1000, family=family)
    removed = ic._prune_runtime_transactions(prefix, family)
    assert [path for path, _ in removed] == [str(failed)]
    assert committed.is_dir() and os.path.isfile(os.path.join(prefix, "bin", "python"))


def test_a_failed_stage_that_still_holds_the_previous_runtime_is_kept(runtime):
    root, prefix = runtime
    held = _copy(root, prefix, "held", "staging_failed", 3000, family="windows-online-uv")
    assert ic._prune_runtime_transactions(prefix, "windows-online-uv") == []
    assert (held / "previous").is_dir()


def test_the_launch_prune_removes_a_failed_stage_without_an_identity_match(runtime):
    root, prefix = runtime
    failed = _copy(root, prefix, "failed", "cancelled", 3000)
    assert [path for path, _ in ic._prune_runtime_transactions(prefix, "posix-runtime", live=True,
                                                              inventory={"live": False})] == [str(failed)]
