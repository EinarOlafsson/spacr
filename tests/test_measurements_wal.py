"""Issue #15: measurements hangs at completion, "database is locked".

Measure runs one worker per field, and every append goes through pandas'
``DataFrame.to_sql``, which issues a ``has_table`` probe -- a READ -- before
writing. So each worker alternates read, write, read, write against one file.

Under the rollback journal that starves the WRITER: a reader holds SHARED for
the length of its statement and a writer cannot COMMIT until every SHARED
lock is gone, so with enough workers someone is always reading and the
committing worker waits out its busy timeout. "database is locked" then
surfaces on whichever ``has_table`` loses, which is the reporter's traceback.

THE FIRST VERSION OF THIS FILE HAD THE DIRECTION BACKWARDS -- it timed a read
under a held write, expecting it to block. It measured 0.000 s and refuted
itself: in rollback-journal mode a writer holding RESERVED does not block
readers, only its brief EXCLUSIVE commit does. The measurement below is the
direction that is real, and it is why these tests time behaviour rather than
assert a pragma. A test that only checked ``PRAGMA journal_mode == 'wal'``
would have shipped the wrong explanation with a green tick.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time

import pytest

from spacr.database_concurrency import (
    NETWORK_FILESYSTEMS,
    WAL_SAFE_FILESYSTEMS,
    enable_wal_where_safe,
    wal_is_safe_here,
)


def _seed(path):
    con = sqlite3.connect(path)
    try:
        con.execute("CREATE TABLE IF NOT EXISTS cell (a INTEGER)")
        con.execute("INSERT INTO cell VALUES (1)")
        con.commit()
    finally:
        con.close()


def _time_a_commit_under_a_held_read(path, hold_seconds=1.0):
    """Seconds a writer needs to COMMIT while one reader holds SHARED.

    The reader is the shape pandas issues before every append: a query
    against sqlite_master inside an open transaction.
    """
    started = threading.Event()
    stop = threading.Event()

    def _reader():
        con = sqlite3.connect(path, timeout=10, isolation_level=None)
        try:
            con.execute("BEGIN")
            con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='cell'").fetchall()
            started.set()
            stop.wait(hold_seconds)
            con.execute("COMMIT")
        finally:
            con.close()

    thread = threading.Thread(target=_reader)
    thread.start()
    try:
        assert started.wait(5), "the reader never took its lock"
        writer = sqlite3.connect(path, timeout=10, isolation_level=None)
        try:
            begin = time.perf_counter()
            writer.execute("BEGIN IMMEDIATE")
            writer.execute("INSERT INTO cell VALUES (2)")
            writer.execute("COMMIT")
            return time.perf_counter() - begin
        finally:
            writer.close()
    finally:
        stop.set()
        thread.join(10)


@pytest.mark.skipif(
    not wal_is_safe_here(os.getcwd()),
    reason="this machine's filesystem is not one WAL is enabled on")
def test_a_writer_no_longer_waits_for_the_readers(tmp_path):
    """The actual complaint, measured on both sides of the fix."""
    before = tmp_path / "delete.db"
    _seed(before)
    blocked = _time_a_commit_under_a_held_read(before, hold_seconds=1.0)

    after = tmp_path / "wal.db"
    _seed(after)
    assert enable_wal_where_safe(after) == "WAL"
    free = _time_a_commit_under_a_held_read(after, hold_seconds=1.0)

    assert blocked > 0.5, (
        f"the DELETE-mode commit was expected to wait out the held read; "
        f"it took {blocked:.3f}s, so this test is not measuring what it "
        f"claims")
    assert free < 0.25, (
        f"the WAL commit still waited {free:.3f}s for the reader")


def test_the_mode_persists_in_the_file(tmp_path):
    """It is set once per run, so it has to outlive the connection."""
    db = tmp_path / "measurements.db"
    _seed(db)
    if enable_wal_where_safe(db) is None:
        pytest.skip("WAL is not enabled on this filesystem")

    con = sqlite3.connect(db)
    try:
        assert str(con.execute(
            "PRAGMA journal_mode").fetchone()[0]).upper() == "WAL"
    finally:
        con.close()


def test_an_unknown_filesystem_is_left_alone(tmp_path, monkeypatch):
    """Silence is the safe answer, and it is the shipped behaviour.

    An unrecognised type is NOT treated as local. "Not a type I recognise as
    networked" is a weaker claim than "local", and the cost of getting it
    wrong is WAL shared memory on storage that cannot hold it.
    """
    monkeypatch.setattr("spacr.database_concurrency.filesystem_type",
                        lambda _p: "some-future-fs")
    db = tmp_path / "measurements.db"
    _seed(db)

    assert enable_wal_where_safe(db) is None
    con = sqlite3.connect(db)
    try:
        assert str(con.execute(
            "PRAGMA journal_mode").fetchone()[0]).upper() == "DELETE"
    finally:
        con.close()


@pytest.mark.parametrize("fs", sorted(NETWORK_FILESYSTEMS))
def test_no_network_filesystem_gets_wal(tmp_path, monkeypatch, fs):
    """The reason the module made WAL opt-in in the first place."""
    monkeypatch.setattr("spacr.database_concurrency.filesystem_type",
                        lambda _p, _fs=fs: _fs)
    db = tmp_path / "measurements.db"
    _seed(db)
    assert enable_wal_where_safe(db) is None


def test_the_two_lists_do_not_overlap():
    """A filesystem cannot be both, and a typo in either would make one."""
    assert not (WAL_SAFE_FILESYSTEMS & NETWORK_FILESYSTEMS)


def test_a_platform_that_cannot_tell_says_no(tmp_path, monkeypatch):
    """`filesystem_type` reads /proc/mounts, so macOS and Windows get None."""
    monkeypatch.setattr("spacr.database_concurrency.filesystem_type",
                        lambda _p: None)
    assert wal_is_safe_here(tmp_path) is False
    assert enable_wal_where_safe(tmp_path / "measurements.db") is None


def test_an_unopenable_database_is_not_an_error(tmp_path, monkeypatch):
    """A run must not die because a journal mode could not be set."""
    monkeypatch.setattr("spacr.database_concurrency.filesystem_type",
                        lambda _p: "ext4")
    missing = tmp_path / "no-such-dir" / "measurements.db"
    assert enable_wal_where_safe(missing) is None


def test_measure_switches_the_database_before_the_workers_start(monkeypatch):
    """The call belongs on the run's path, not in a helper nobody invokes.

    Asserted on the source: reaching the line needs a full plate. What is
    pinned is that `measure_crop` calls it, and does so with the path the
    writers actually use -- `<src>/../measurements/measurements.db`.
    """
    import inspect
    from spacr import measure

    source = inspect.getsource(measure.measure_crop)
    assert "enable_wal_where_safe(" in source
    assert "'measurements'" in source or '"measurements"' in source
