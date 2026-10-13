"""What the SQLite helpers do when the database or the disk misbehaves.

The value of this module is entirely in its failure paths: a busy timeout
that gets restored even when restoring it fails, a rollback that itself
raises, a mount table that cannot be read. None of those can be produced by
asking SQLite nicely, so each is injected at the exact call that would
produce it in the wild, and the assertion is on what the caller is left
holding afterwards.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import sqlite3
import threading
import time
import types

import pytest

from spacr import database_concurrency as dc


@pytest.fixture
def db(tmp_path):
    """A small database with one table, written through the module's own connect."""
    path = str(tmp_path / "measurements.db")
    connection = dc.connect(path)
    try:
        connection.execute("CREATE TABLE t (a INTEGER PRIMARY KEY, b TEXT)")
        connection.execute("INSERT INTO t VALUES (1, 'one')")
    finally:
        connection.close()
    return path


class Fragile(sqlite3.Connection):
    """A real connection whose named statement or method fails on demand.

    Subclassed rather than wrapped so everything the module does to a
    connection — PRAGMAs, ``in_transaction``, the transaction body — is a real
    SQLite call, and only the one failure being modelled is invented.
    """

    fail_sql = ""
    fail_after = 0
    fail_commit = False
    fail_rollback = False

    def execute(self, sql, *args):
        if self.fail_sql and self.fail_sql in sql:
            if self.fail_after <= 0:
                raise sqlite3.OperationalError(f"disk I/O error on {sql!r}")
            self.fail_after -= 1
        return super().execute(sql, *args)

    def commit(self):
        if self.fail_commit:
            raise sqlite3.OperationalError("disk I/O error on commit")
        return super().commit()

    def rollback(self):
        if self.fail_rollback:
            raise sqlite3.OperationalError("disk I/O error on rollback")
        return super().rollback()


def _fragile(path, **flags):
    connection = sqlite3.connect(path, isolation_level=None, factory=Fragile)
    connection.execute("PRAGMA busy_timeout = 100")
    for name, value in flags.items():
        setattr(connection, name, value)
    return connection


# ---------------------------------------------------------------------------
# connect
# ---------------------------------------------------------------------------

def test_a_read_only_connection_cannot_change_the_journal_mode(db):
    """Asking for both is a contradiction, and it is named as one."""
    with pytest.raises(dc.DatabaseConfigurationError) as caught:
        dc.connect(db, readonly=True, journal_mode="WAL")

    assert "read-only connection" in str(caught.value)


def test_an_unsupported_journal_mode_lists_the_supported_ones(db):
    """``MEMORY`` and ``TRUNCATE`` are refused, with the allowed set shown."""
    with pytest.raises(dc.DatabaseConfigurationError) as caught:
        dc.connect(db, journal_mode="MEMORY")

    assert "DELETE" in str(caught.value) and "WAL" in str(caught.value)


def test_a_mode_sqlite_declines_to_adopt_is_a_refusal_not_a_lie(db,
                                                                monkeypatch):
    """A filesystem that cannot hold WAL leaves the caller with no connection.

    SQLite reports the mode it kept rather than raising, so the check is on
    the value it returns; nothing here can make a real filesystem do that.
    """
    real_connect = sqlite3.connect
    closed = []

    class Stubborn(sqlite3.Connection):
        def execute(self, sql, *args):
            if sql.startswith("PRAGMA journal_mode ="):
                # The mode it kept, not the one it was told.
                return super().execute("SELECT 'delete'")
            return super().execute(sql, *args)

        def close(self):
            closed.append(True)
            return super().close()

    def stubborn_connect(*args, **kwargs):
        kwargs["factory"] = Stubborn
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(dc.sqlite3, "connect", stubborn_connect)

    with pytest.raises(dc.DatabaseConfigurationError) as caught:
        dc.connect(db, journal_mode="WAL")

    assert "SQLite kept journal_mode=DELETE" in str(caught.value)
    assert "filesystem may not support it" in str(caught.value)
    assert closed == [True], "the half-configured connection is not leaked"


def test_wal_turns_synchronous_down_to_normal(tmp_path):
    """WAL plus NORMAL is the pairing this project ships."""
    path = str(tmp_path / "wal.db")
    connection = dc.connect(path, journal_mode="WAL")
    try:
        assert str(connection.execute(
            "PRAGMA journal_mode").fetchone()[0]).upper() == "WAL"
        assert int(connection.execute(
            "PRAGMA synchronous").fetchone()[0]) == 1
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# transaction
# ---------------------------------------------------------------------------

def test_an_unknown_transaction_mode_is_refused(db):
    """Only the three SQLite modes are accepted."""
    connection = dc.connect(db)
    try:
        with pytest.raises(ValueError) as caught:
            with dc.transaction(connection, mode="SORT OF"):
                pass
        assert "DEFERRED" in str(caught.value)
    finally:
        connection.close()


def test_a_begin_that_fails_for_a_reason_other_than_a_lock_is_not_retried(db):
    """A read-only connection cannot BEGIN IMMEDIATE, and that is not busy."""
    connection = dc.connect(db, readonly=True)
    try:
        with pytest.raises(sqlite3.OperationalError) as caught:
            with dc.transaction(connection, attempts=5, initial_delay=0.0):
                pass
        assert not isinstance(caught.value, dc.DatabaseBusy)
        assert "readonly" in str(caught.value)
    finally:
        connection.close()


def test_a_failure_restoring_the_busy_timeout_rolls_the_transaction_back(db):
    """The BEGIN succeeded, so an unwind here must not leave it open."""
    connection = _fragile(db, fail_sql="PRAGMA busy_timeout =", fail_after=1)
    try:
        with pytest.raises(sqlite3.OperationalError):
            with dc.transaction(connection, attempts=1, busy_timeout=5.0):
                pass
        assert not connection.in_transaction
    finally:
        connection.fail_sql = ""
        connection.close()


def test_a_rollback_that_itself_fails_does_not_hide_the_real_error(db):
    """The body's exception is what the caller sees, not the rollback's."""
    connection = _fragile(db, fail_rollback=True)
    try:
        with pytest.raises(ValueError, match="the real problem"):
            with dc.transaction(connection):
                connection.execute("INSERT INTO t VALUES (2, 'two')")
                raise ValueError("the real problem")
    finally:
        connection.fail_rollback = False
        connection.rollback()
        connection.close()


def test_a_commit_that_fails_is_raised_after_the_rollback_is_tried(db):
    """A write that cannot be committed is an error, not a silent loss."""
    connection = _fragile(db, fail_commit=True)
    try:
        with pytest.raises(sqlite3.OperationalError, match="on commit"):
            with dc.transaction(connection):
                connection.execute("INSERT INTO t VALUES (2, 'two')")
    finally:
        connection.fail_commit = False
        connection.close()

    check = dc.connect(db, readonly=True)
    try:
        assert check.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1
    finally:
        check.close()


def test_a_commit_failure_survives_a_rollback_failure_too(db):
    """Both halves failing still raises, and still raises the commit's error."""
    connection = _fragile(db, fail_commit=True, fail_rollback=True)
    try:
        with pytest.raises(sqlite3.OperationalError, match="on commit"):
            with dc.transaction(connection):
                connection.execute("INSERT INTO t VALUES (3, 'three')")
    finally:
        connection.fail_commit = False
        connection.fail_rollback = False
        connection.rollback()
        connection.close()


# ---------------------------------------------------------------------------
# the mount table
# ---------------------------------------------------------------------------

def _fake_mounts(monkeypatch, *, is_file=True, text="", error=None):
    class Mounts:
        def is_file(self):
            return is_file

        def read_text(self, **kwargs):
            if error is not None:
                raise error
            return text

    def fake_path(argument):
        if str(argument) == "/proc/mounts":
            return Mounts()
        return pathlib.Path(argument)

    monkeypatch.setattr(dc, "Path", fake_path)


def test_a_platform_with_no_proc_mounts_falls_back_rather_than_giving_up(
        monkeypatch, tmp_path):
    """UPDATED for issue 115. The INTENT is unchanged: a platform that cannot
    tell must not guess. What changed is that macOS and Windows CAN now tell.

    This used to assert None off Linux, which was accurate and was also the
    defect: `wal_is_safe_here` turns None into False, so every Mac ran without
    WAL even on a local disk, and `doctor` could not tell a user on an SMB
    share that they were on one. That reporter is issue 115.

    The failing-safe property is asserted where it still belongs -- when the
    fall-back ALSO cannot tell, below."""
    _fake_mounts(monkeypatch, is_file=False)

    import psutil
    from types import SimpleNamespace
    monkeypatch.setattr(psutil, "disk_partitions",
                        lambda all=False: [SimpleNamespace(mountpoint="/",
                                                           fstype="apfs")])
    assert dc.filesystem_type(tmp_path) == "apfs"
    assert dc.wal_is_safe_here(tmp_path) is True


def test_no_proc_mounts_and_no_partition_table_still_cannot_tell(
        monkeypatch, tmp_path):
    """The failing-safe half, kept: unknown is still unsafe."""
    _fake_mounts(monkeypatch, is_file=False)

    import psutil
    monkeypatch.setattr(psutil, "disk_partitions",
                        lambda all=False: [])
    assert dc.filesystem_type(tmp_path) is None
    assert dc.wal_is_safe_here(tmp_path) is False


def test_a_mount_table_that_cannot_be_read_cannot_tell(monkeypatch, tmp_path):
    """An OSError reading ``/proc/mounts`` is "unknown", not a crash."""
    _fake_mounts(monkeypatch, error=OSError("permission denied"))

    assert dc.filesystem_type(tmp_path) is None


def test_a_short_line_in_the_mount_table_is_skipped(monkeypatch, tmp_path):
    """A malformed row must not stop the real one below it being found."""
    _fake_mounts(monkeypatch, text=(
        "garbage\n"
        "two fields\n"
        f"/dev/sda1 / ext4 rw 0 0\n"
        f"tmpfs {tmp_path} tmpfs rw 0 0\n"))

    assert dc.filesystem_type(tmp_path) == "tmpfs"
    assert dc.wal_is_safe_here(tmp_path) is True


def test_the_longest_matching_mount_point_wins(monkeypatch, tmp_path):
    """A nested mount describes the path better than the root does."""
    _fake_mounts(monkeypatch, text=(
        "/dev/sda1 / ext4 rw 0 0\n"
        f"server:/export {tmp_path} nfs4 rw 0 0\n"))

    assert dc.filesystem_type(tmp_path) == "nfs4"
    assert dc.wal_is_safe_here(tmp_path) is False


def test_a_mount_point_with_a_space_in_it_is_decoded(monkeypatch, tmp_path):
    """``/proc/mounts`` escapes spaces as ``\\040``."""
    spaced = tmp_path / "my data"
    spaced.mkdir()
    escaped = str(spaced).replace(" ", "\\040")
    _fake_mounts(monkeypatch, text=(
        "/dev/sda1 / ext4 rw 0 0\n"
        f"/dev/sdb1 {escaped} xfs rw 0 0\n"))

    assert dc.filesystem_type(spaced) == "xfs"


# ---------------------------------------------------------------------------
# turning WAL on
# ---------------------------------------------------------------------------

def test_wal_is_not_switched_on_where_it_is_not_known_to_be_safe(monkeypatch,
                                                                 tmp_path):
    """An unrecognised filesystem stays on DELETE, which is what shipped."""
    monkeypatch.setattr(dc, "wal_is_safe_here", lambda path: False)

    assert dc.enable_wal_where_safe(tmp_path / "new.db") is None
    assert not (tmp_path / "new.db").exists()


def test_a_connection_that_refuses_wal_is_informative_not_fatal(monkeypatch,
                                                                tmp_path):
    """SQLite declining WAL leaves the run on DELETE and returns None."""
    monkeypatch.setattr(dc, "wal_is_safe_here", lambda path: True)

    def refuse(path, **kwargs):
        raise dc.DatabaseConfigurationError("no WAL on this storage")

    monkeypatch.setattr(dc, "connect", refuse)

    assert dc.enable_wal_where_safe(tmp_path / "new.db") is None


def test_a_mode_that_cannot_be_read_back_is_none_and_the_connection_closes(
        monkeypatch, db):
    """Even the read-back is allowed to fail, and the handle is still released."""
    monkeypatch.setattr(dc, "wal_is_safe_here", lambda path: True)
    made = {}

    def fragile_connect(path, **kwargs):
        connection = _fragile(path, fail_sql="PRAGMA journal_mode")
        made["connection"] = connection
        return connection

    monkeypatch.setattr(dc, "connect", fragile_connect)

    assert dc.enable_wal_where_safe(db) is None
    with pytest.raises(sqlite3.ProgrammingError):
        made["connection"].execute("SELECT 1")


def test_wal_is_switched_on_where_it_is_safe(monkeypatch, tmp_path):
    """On a local filesystem the mode is changed and reported back."""
    monkeypatch.setattr(dc, "wal_is_safe_here", lambda path: True)

    assert dc.enable_wal_where_safe(tmp_path / "fresh.db") == "WAL"


# ---------------------------------------------------------------------------
# inspecting
# ---------------------------------------------------------------------------

def test_a_corrupt_database_is_reported_by_the_quick_check(tmp_path):
    """The health snapshot carries what SQLite said, verbatim, as a warning."""
    path = str(tmp_path / "corrupt.db")
    connection = dc.connect(path)
    try:
        connection.execute("CREATE TABLE t (a INTEGER, b TEXT)")
        connection.execute("CREATE INDEX ix ON t (b)")
        for index in range(3000):
            connection.execute("INSERT INTO t VALUES (?, ?)",
                               (index, f"row{index:06d}"))
    finally:
        connection.close()
    with open(path, "r+b") as handle:
        handle.seek(12288 + 100)
        handle.write(b"\x00" * 200)

    health = dc.inspect_database(path, quick_check=True)

    assert health.quick_check != "ok"
    assert health.warnings
    assert health.warnings[0].startswith("SQLite quick_check reported:")
    assert health.to_dict()["quick_check"] == health.quick_check
    assert health.file_bytes == os.path.getsize(path)


def test_a_healthy_database_has_no_warnings(db):
    """Nothing wrong is an empty warning list, not a reassuring sentence."""
    health = dc.inspect_database(db, quick_check=True)

    assert health.quick_check == "ok"
    assert health.warnings == ()
    assert health.to_dict()["warnings"] == ()
    assert health.sqlite_version == sqlite3.sqlite_version


def test_wal_on_a_network_filesystem_is_warned_about(monkeypatch, tmp_path):
    """Shared-memory coordination does not survive NFS, and the report says so."""
    path = str(tmp_path / "net.db")
    connection = dc.connect(path, journal_mode="WAL")
    connection.execute("CREATE TABLE t (a)")
    connection.close()
    monkeypatch.setattr(dc, "filesystem_type", lambda where: "nfs4")

    health = dc.inspect_database(path)

    assert health.network_filesystem is True
    assert any("network filesystem" in line for line in health.warnings)


def test_inspecting_something_that_is_not_there_says_so(tmp_path):
    """A missing database is a FileNotFoundError naming the path."""
    with pytest.raises(FileNotFoundError):
        dc.inspect_database(tmp_path / "absent.db")


# ---------------------------------------------------------------------------
# the concurrency probe
# ---------------------------------------------------------------------------

def test_a_worker_that_cannot_connect_is_collected_not_raised(monkeypatch,
                                                              tmp_path):
    """Each thread's failure is reported per thread; the probe still returns."""
    real_connect = dc.connect

    def fussy(path, **kwargs):
        if kwargs.get("timeout") in (0.05, 0.25):
            raise sqlite3.OperationalError("no connection for this worker")
        return real_connect(path, **kwargs)

    monkeypatch.setattr(dc, "connect", fussy)

    result = dc.run_concurrency_probe(tmp_path / "probe.sqlite", writers=1,
                                      readers=1, writes_per_writer=1,
                                      journal_mode="DELETE")

    assert result.ok is False
    assert any(line.startswith("writer 0:") for line in result.errors)
    assert any(line.startswith("reader 0:") for line in result.errors)
    assert result.actual_rows == 0
    assert result.expected_rows == 1


def test_a_worker_that_outlives_the_join_deadline_is_reported(monkeypatch,
                                                              tmp_path):
    """A stalled thread is named, and its scratch database is kept to look at."""
    gate = threading.Event()
    real_connect = dc.connect

    class Blocking:
        """A reader connection whose queries do not come back."""

        def __init__(self, inner):
            self._inner = inner

        def execute(self, *args, **kwargs):
            gate.wait(timeout=20)
            return self._inner.execute(*args, **kwargs)

        def close(self):
            self._inner.close()

    def blocking_connect(path, **kwargs):
        inner = real_connect(path, **kwargs)
        if kwargs.get("readonly") and kwargs.get("timeout") == 0.25:
            return Blocking(inner)
        return inner

    monkeypatch.setattr(dc, "connect", blocking_connect)

    calls = {"n": 0}

    def monotonic():
        calls["n"] += 1
        # The first two readings set `started` and the 30-second deadline; the
        # join loop then finds the deadline already past.
        return 0.0 if calls["n"] <= 2 else 100.0

    monkeypatch.setattr(dc, "time",
                        types.SimpleNamespace(monotonic=monotonic,
                                              sleep=time.sleep))
    try:
        result = dc.run_concurrency_probe(writers=1, readers=1,
                                          writes_per_writer=1,
                                          journal_mode="DELETE")
    finally:
        gate.set()

    assert any("did not finish within 30 seconds" in line
               for line in result.errors)
    assert result.ok is False
    assert os.path.exists(result.path), "a stalled probe keeps its database"
    time.sleep(0.2)
    shutil.rmtree(os.path.dirname(result.path), ignore_errors=True)


def test_a_clean_probe_writes_every_row_and_removes_its_scratch_database():
    """The happy path: no errors, every row committed, nothing left behind."""
    result = dc.run_concurrency_probe(writers=2, readers=1,
                                      writes_per_writer=3,
                                      journal_mode="DELETE")

    assert result.ok is True
    assert result.actual_rows == result.expected_rows == 6
    assert result.reader_queries >= 1
    assert not os.path.exists(os.path.dirname(result.path))


def test_only_a_lock_error_is_worth_retrying():
    """A ValueError is never a busy database, whatever it says."""
    assert dc.is_busy_error(sqlite3.OperationalError("database is locked"))
    assert dc.is_busy_error(sqlite3.OperationalError("database table is busy"))
    assert not dc.is_busy_error(sqlite3.OperationalError("no such table: t"))
    assert not dc.is_busy_error(ValueError("database is locked"))
    assert not dc.is_busy_error(sqlite3.IntegrityError("locked"))


def test_a_database_that_does_not_exist_yet_is_placed_by_its_parent(
        monkeypatch, tmp_path):
    """A path being created gets the filesystem of the folder it will live in."""
    _fake_mounts(monkeypatch, text=(
        "/dev/sda1 / ext4 rw 0 0\n"
        f"tmpfs {tmp_path} tmpfs rw 0 0\n"))

    absent = tmp_path / "not" / "made" / "yet" / "measurements.db"

    assert dc.filesystem_type(absent) == "tmpfs"
