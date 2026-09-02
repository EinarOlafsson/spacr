"""Issue 115: spaCR could not name the filesystem on a Mac.

`filesystem_type` read `/proc/mounts` and returned None everywhere else, and
`wal_is_safe_here` turns None into False. That fails SAFE, so nothing was
corrupted -- but it cost two things:

  * every macOS and Windows user ran without WAL even on a local disk;
  * `doctor` could not tell a user on an SMB share that they WERE on one,
    which is precisely what issue 115's reporter needed: Apple Silicon, a
    measurement.db on an SMB server, and no diagnosis available.

The fall-back must keep the failing-safe property: an unknown filesystem is
still unsafe. Being able to NAME smbfs is the gain, not permission to trust it.
"""
from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest

from spacr import database_concurrency as dc


class _Part(SimpleNamespace):
    pass


def _fake_psutil(monkeypatch, partitions):
    import psutil

    monkeypatch.setattr(psutil, "disk_partitions",
                        lambda all=False: list(partitions))


def test_the_longest_mount_wins_so_a_share_is_not_read_as_the_root_disk(
        monkeypatch):
    """The bug this would have if it matched the first mount instead.

    "/" matches every path, so an SMB share under /Volumes has to beat it or
    the share is reported as APFS -- and APFS is on WAL_SAFE_FILESYSTEMS, so a
    wrong answer here is the one that enables WAL on a network share.
    """
    _fake_psutil(monkeypatch, [
        _Part(mountpoint="/", fstype="apfs"),
        _Part(mountpoint="/Volumes/lab-share", fstype="smbfs"),
    ])
    got = dc._filesystem_type_via_psutil(Path("/Volumes/lab-share/data"))
    assert got == "smbfs", "the nested share must beat the root mount"


def test_an_smb_share_is_not_wal_safe(monkeypatch):
    """The property that protects the database."""
    _fake_psutil(monkeypatch, [
        _Part(mountpoint="/", fstype="apfs"),
        _Part(mountpoint="/Volumes/lab-share", fstype="smbfs"),
    ])
    monkeypatch.setattr(dc, "filesystem_type",
                        lambda p: dc._filesystem_type_via_psutil(Path(p)))
    assert dc.wal_is_safe_here("/Volumes/lab-share/measurements.db") is False


def test_a_local_mac_disk_is_now_recognised(monkeypatch):
    """The gain. Before this, every Mac answered None and lost WAL."""
    _fake_psutil(monkeypatch, [_Part(mountpoint="/", fstype="apfs")])
    assert dc._filesystem_type_via_psutil(Path("/local/project/data")) == "apfs"
    assert "apfs" in dc.WAL_SAFE_FILESYSTEMS


def test_an_unenumerable_platform_stays_unknown_rather_than_guessing(
        monkeypatch):
    """Failing safe is preserved: unknown is still unsafe."""
    import psutil

    def boom(all=False):
        raise OSError("cannot enumerate mounts")

    monkeypatch.setattr(psutil, "disk_partitions", boom)
    assert dc._filesystem_type_via_psutil(Path("/anywhere")) is None


def test_a_platform_without_psutil_stays_unknown(monkeypatch):
    real_import = builtins.__import__

    def no_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("psutil is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_psutil)

    assert dc._filesystem_type_via_psutil(Path("/anywhere")) is None


def test_a_partition_with_no_fstype_is_skipped(monkeypatch):
    """Windows reports empty fstype for an unmounted drive letter, and an
    empty string would otherwise win the longest-mount comparison."""
    _fake_psutil(monkeypatch, [
        _Part(mountpoint="/", fstype="apfs"),
        _Part(mountpoint="/Volumes/empty", fstype=""),
    ])
    assert dc._filesystem_type_via_psutil(Path("/Volumes/empty/x")) == "apfs"


def test_a_shorter_partition_found_later_does_not_replace_the_best_match(
        monkeypatch):
    _fake_psutil(monkeypatch, [
        _Part(mountpoint="/Volumes/lab-share", fstype="smbfs"),
        _Part(mountpoint="/", fstype="apfs"),
    ])

    assert dc._filesystem_type_via_psutil(
        Path("/Volumes/lab-share/data")) == "smbfs"


def test_linux_still_uses_proc_mounts(monkeypatch):
    """The Linux path is unchanged, and is still preferred where it exists."""
    if not Path("/proc/mounts").is_file():
        pytest.skip("not Linux")
    assert dc.filesystem_type("/tmp") == dc._filesystem_type_via_psutil(
        Path("/tmp"))
