"""The corners of the SQLite helpers that only a misbehaving disk reaches.

Every test here drives one branch that a healthy database never takes: a
``PRAGMA busy_timeout`` restore that fails while no transaction is open, a
transaction body that ended its own transaction before failing, a ``COMMIT``
that reports an error after it has already committed, a mount table whose
best match is not its last match, and a probe worker that finishes inside the
window in which it was still being counted as stalled.

None of them can be produced by asking SQLite nicely and none of them may be
produced by racing threads either -- a test that waits for a race is a test
that passes by luck. So each failure is injected at exactly the call that
would produce it in the wild, through a real :class:`sqlite3.Connection`
subclass, and the assertion is on what the caller and the database are left
holding afterwards.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import sqlite3
import threading
import types

import pytest

from spacr import database_concurrency as dc


@pytest.fixture
def db(tmp_path):
    """A database with one table, created through the module's own connect."""
    path = str(tmp_path / "measurements.db")
    connection = dc.connect(path)
    try:
        connection.execute("CREATE TABLE t (a INTEGER PRIMARY KEY)")
    finally:
        connection.close()
    return path


class _RefusesOneStatement(sqlite3.Connection):
    """A real connection that refuses one exact statement once armed.

    Subclassed rather than mocked so ``in_transaction``, the PRAGMAs and the
    transaction body remain real SQLite behaviour, and only the single
    failure being modelled is invented.
    """

    armed_sql = ""

    def execute(self, sql, *args):
        if self.armed_sql and self.armed_sql in sql:
            raise sqlite3.OperationalError(f"disk I/O error on {sql!r}")
        return super().execute(sql, *args)


class _CommitFailsAfterCommitting(sqlite3.Connection):
    """A connection whose ``COMMIT`` lands and then reports an error.

    This is the shape a commit failure takes when the write reached the file
    and the acknowledgement did not -- a full disk on ``fsync``, an NFS
    server that dropped the reply. The transaction is over; the caller is
    told it is not.
    """

    def commit(self):
        super().commit()
        raise sqlite3.OperationalError("disk I/O error on commit")


def _fake_mounts(monkeypatch, text):
    """Make :func:`filesystem_type` read ``text`` as ``/proc/mounts``."""

    class Mounts:
        def is_file(self):
            return True

        def read_text(self, **kwargs):
            return text

    def fake_path(argument):
        if str(argument) == "/proc/mounts":
            return Mounts()
        return pathlib.Path(argument)

    monkeypatch.setattr(dc, "Path", fake_path)


# ---------------------------------------------------------------------------
# transaction: restoring the connection's own busy timeout
# ---------------------------------------------------------------------------

def test_a_busy_timeout_restore_that_fails_with_no_transaction_open_propagates(
        db):
    """The restore error replaces the lock error, and nothing is rolled back.

    ``transaction`` lowers the connection's ``busy_timeout`` for its ``BEGIN``
    retries and puts the connection's own value back in a ``finally``. When
    every ``BEGIN`` was refused there is no transaction to roll back, so the
    restore failure is simply raised -- and it is what the caller sees, with
    the :class:`DatabaseBusy` it displaced kept as its context.
    """
    holder = dc.connect(db)
    writer = sqlite3.connect(db, factory=_RefusesOneStatement,
                             isolation_level=None, timeout=0)
    try:
        writer.execute("PRAGMA busy_timeout = 30000")
        holder.execute("BEGIN EXCLUSIVE")
        writer.armed_sql = "PRAGMA busy_timeout = 30000"

        with pytest.raises(sqlite3.OperationalError) as excinfo:
            with dc.transaction(writer, attempts=1, busy_timeout=0):
                writer.execute("INSERT INTO t VALUES (1)")

        assert not isinstance(excinfo.value, dc.DatabaseBusy)
        assert "PRAGMA busy_timeout = 30000" in str(excinfo.value)
        assert isinstance(excinfo.value.__context__, dc.DatabaseBusy)
        assert "remained locked after 1" in str(excinfo.value.__context__)
        assert writer.in_transaction is False
        # The restore did not happen, so the lowered value is what is left.
        writer.armed_sql = ""
        assert writer.execute("PRAGMA busy_timeout").fetchone()[0] == 0
    finally:
        holder.rollback()
        holder.close()
        writer.close()

    verify = dc.connect(db, readonly=True)
    try:
        assert verify.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 0
    finally:
        verify.close()


# ---------------------------------------------------------------------------
# transaction: a body that ended the transaction itself
# ---------------------------------------------------------------------------

def test_a_body_that_committed_before_failing_keeps_what_it_committed(db):
    """A body's own ``COMMIT`` stands even when the body then raises.

    Nothing is rolled back, because there is no longer a transaction to roll
    back: the rows the body chose to commit are still there, and its
    exception reaches the caller unchanged.
    """
    connection = dc.connect(db)
    try:
        with pytest.raises(ValueError, match="the batch ran out of input"):
            with dc.transaction(connection):
                connection.execute("INSERT INTO t VALUES (11)")
                connection.commit()
                raise ValueError("the batch ran out of input")

        assert connection.in_transaction is False
        assert connection.execute(
            "SELECT a FROM t").fetchall() == [(11,)]
    finally:
        connection.close()

    verify = dc.connect(db, readonly=True)
    try:
        assert verify.execute("SELECT a FROM t").fetchall() == [(11,)]
    finally:
        verify.close()


def test_a_commit_that_fails_after_it_landed_does_not_lose_the_rows(db):
    """A post-commit error is raised, and the committed rows stay committed.

    ``transaction`` rolls back on a commit failure only while a transaction
    is still open. When the commit reached the file before it reported the
    error there is nothing open, so the data is left alone and the caller is
    told -- which is the only honest answer, since the write did happen.
    """
    connection = sqlite3.connect(db, factory=_CommitFailsAfterCommitting,
                                 isolation_level=None, timeout=5)
    try:
        connection.execute("PRAGMA busy_timeout = 5000")

        with pytest.raises(sqlite3.OperationalError,
                           match="disk I/O error on commit"):
            with dc.transaction(connection):
                connection.execute("INSERT INTO t VALUES (7)")

        assert connection.in_transaction is False
    finally:
        connection.close()

    verify = dc.connect(db, readonly=True)
    try:
        assert verify.execute("SELECT a FROM t").fetchall() == [(7,)]
    finally:
        verify.close()


# ---------------------------------------------------------------------------
# filesystem_type: the longest match wins whatever order it is found in
# ---------------------------------------------------------------------------

def test_a_shorter_mount_point_found_later_does_not_displace_the_best_match(
        monkeypatch, tmp_path):
    """The most specific mount describes the path, not the most recent line.

    ``/proc/mounts`` is in mount order, not path order, so the nested
    filesystem a database actually sits on can be listed above the two
    filesystems that merely contain it. Taking the last match would answer
    ``nfs4`` here and put WAL off a local disk that supports it.
    """
    nested = tmp_path / "scratch"
    nested.mkdir()
    _fake_mounts(monkeypatch, (
        f"/dev/sdb1 {nested} btrfs rw 0 0\n"
        "/dev/sda1 / ext4 rw 0 0\n"
        f"server:/export {tmp_path} nfs4 rw 0 0\n"))

    assert dc.filesystem_type(nested) == "btrfs"
    assert dc.wal_is_safe_here(nested) is True


# ---------------------------------------------------------------------------
# run_concurrency_probe: the last look at a worker that has just finished
# ---------------------------------------------------------------------------

class _ReportsItselfAliveAfterFinishing(threading.Thread):
    """A worker that keeps answering ``is_alive()`` for two checks too long.

    This is the join deadline's race made deterministic: a thread that
    finishes between the moment the probe lists it as still running and the
    moment it looks again. Only the answer is delayed; the thread itself does
    real work and really exits.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stale_reports = 2

    def is_alive(self):
        really_alive = super().is_alive()
        if not really_alive and self._stale_reports:
            self._stale_reports -= 1
            return True
        return really_alive


def test_a_worker_that_finished_in_the_join_window_is_not_reported_as_stalled(
        monkeypatch):
    """The probe looks once more before blaming a thread for not finishing.

    Every worker here has really exited by the time the deadline is checked,
    and the probe's final ``is_alive()`` says so -- so no thread is recorded
    as having missed the 30-second deadline and the run is clean. The scratch
    database is nonetheless kept, because the list of stragglers was built
    before that final look.
    """
    monkeypatch.setattr(dc, "threading", types.SimpleNamespace(
        Thread=_ReportsItselfAliveAfterFinishing,
        Barrier=threading.Barrier,
        Lock=threading.Lock,
        Event=threading.Event))

    result = dc.run_concurrency_probe(
        writers=1, readers=1, writes_per_writer=3, journal_mode="DELETE")
    try:
        assert result.errors == ()
        assert result.actual_rows == result.expected_rows == 3
        assert result.ok is True
        assert os.path.isfile(result.path), (
            "the straggler list was still truthy, so the scratch database "
            "was kept for inspection")
    finally:
        shutil.rmtree(os.path.dirname(result.path), ignore_errors=True)


# ---------------------------------------------------------------------------
# _discard_scratch: a cleanup failure never replaces the real reason
# ---------------------------------------------------------------------------

def test_a_scratch_path_the_os_rejects_still_reports_why_the_probe_failed(
        tmp_path):
    """A cleanup the OS refuses below ``OSError`` is swallowed, not raised.

    ``_discard_scratch`` removes the database a probe created but never used,
    and it runs while the real failure is already unwinding. A path with an
    embedded null byte is refused by ``sqlite3.connect`` with a
    :exc:`ValueError` and then refused again by ``os.remove`` with another
    one -- and it is the first that has to reach the caller, because the
    second says nothing about why the probe did not run.
    """
    scratch = tmp_path / "probe\x00.sqlite"

    with pytest.raises(ValueError, match="embedded null byte") as excinfo:
        dc.run_concurrency_probe(str(scratch), writers=1, readers=1,
                                 writes_per_writer=1)

    # Nothing was raised while the first error was being handled: the cleanup
    # failure was logged and dropped, so this is the connect failure itself.
    assert excinfo.value.__context__ is None
    assert os.listdir(tmp_path) == []
