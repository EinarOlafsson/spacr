"""SQLite concurrency, rollback, health, and fail-loud audit.

The tests use real files and thread-owned connections. They deliberately
exercise the rules relied on by Measure workers, run-status stamping, and the
database browser rather than mocking SQLite's lock state.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time

import pytest

import spacr.database_concurrency as db_concurrency
from spacr.cli_database import main as database_cli
from spacr.database_concurrency import (
    MINIMUM_ATTEMPT_BUSY_TIMEOUT_MS,
    DatabaseBusy,
    DatabaseConfigurationError,
    connect,
    inspect_database,
    run_concurrency_probe,
    transaction,
)
from spacr.errors import RUN_STATUS_TABLE, RunLedger


def _create_database(path, *, journal_mode=None):
    connection = connect(path, journal_mode=journal_mode)
    try:
        with transaction(connection):
            connection.execute(
                "CREATE TABLE events (id INTEGER PRIMARY KEY, value TEXT)")
    finally:
        connection.close()


def test_transaction_commits_all_statements_together(tmp_path):
    path = tmp_path / "commit.sqlite"
    _create_database(path)
    connection = connect(path)
    try:
        with transaction(connection):
            connection.execute("INSERT INTO events(value) VALUES ('one')")
            connection.execute("INSERT INTO events(value) VALUES ('two')")
    finally:
        connection.close()
    reader = connect(path, readonly=True)
    try:
        assert reader.execute(
            "SELECT value FROM events ORDER BY id").fetchall() == [
                ("one",), ("two",)]
    finally:
        reader.close()


def test_transaction_retry_preserves_connection_busy_timeout(tmp_path):
    path = tmp_path / "timeout.sqlite"
    _create_database(path)
    connection = connect(path, timeout=1.234)
    try:
        before = connection.execute("PRAGMA busy_timeout").fetchone()[0]
        with transaction(connection, attempts=7):
            connection.execute("INSERT INTO events(value) VALUES ('ok')")
        after = connection.execute("PRAGMA busy_timeout").fetchone()[0]
    finally:
        connection.close()
    assert before == after == 1234


@pytest.mark.parametrize(
    "total_ms, attempts, expected",
    [
        # The concurrency probe's own numbers: writers connect with
        # timeout=0.05 and ask for 40 attempts. Plain division gave each
        # BEGIN 1 ms, which is below the point where SQLite's busy handler
        # can take a contended lock at all.
        (50, 40, MINIMUM_ATTEMPT_BUSY_TIMEOUT_MS),
        # Large budgets must still be shared, or eight retries on a
        # 30-second connection block for four minutes.
        (30_000, 8, 3_750),
        (1_234, 7, 176),
        # The floor never inflates a budget past what the caller allowed:
        # a 10 ms connection waits at most 10 ms inside one BEGIN.
        (10, 2, 10),
        (1, 40, 1),
        (MINIMUM_ATTEMPT_BUSY_TIMEOUT_MS, 1, MINIMUM_ATTEMPT_BUSY_TIMEOUT_MS),
        # busy_timeout = 0 means "do not block"; that is a choice, not a
        # budget to be topped up.
        (0, 8, 0),
        # Degenerate inputs must not produce a negative or zero PRAGMA.
        (-5, 8, 0),
        (100, 0, 100),
    ],
)
def test_per_attempt_lock_budget_is_floored_and_capped(
        total_ms, attempts, expected):
    assert db_concurrency._attempt_busy_timeout_ms(total_ms, attempts) == expected


def test_tight_connection_still_gets_a_usable_per_attempt_budget(tmp_path):
    """The probe's writer settings must not degrade BEGIN to a single try.

    ``run_concurrency_probe`` opens writers with ``timeout=0.05`` and calls
    ``transaction(attempts=40)``. Dividing the busy timeout by the attempt
    count handed each BEGIN 1 ms. This asserts the PRAGMA actually executed
    on the connection, because that is what SQLite acts on.
    """
    path = tmp_path / "tight.sqlite"
    _create_database(path)
    connection = connect(path, timeout=0.05)
    statements = []
    try:
        connection.set_trace_callback(statements.append)
        with transaction(connection, attempts=40):
            connection.execute("INSERT INTO events(value) VALUES ('tight')")
    finally:
        connection.set_trace_callback(None)
        connection.close()
    applied = [int(sql.split("=")[1]) for sql in statements
               if sql.startswith("PRAGMA busy_timeout =")]
    # One narrowed budget for the BEGIN, then the connection's own value back.
    assert applied == [MINIMUM_ATTEMPT_BUSY_TIMEOUT_MS, 50]


def test_explicit_busy_timeout_overrides_the_connection_budget(tmp_path):
    """A write can state its own lock tolerance instead of inheriting one."""
    path = tmp_path / "explicit.sqlite"
    _create_database(path)
    connection = connect(path, timeout=0.05)
    statements = []
    try:
        connection.set_trace_callback(statements.append)
        with transaction(connection, attempts=4, busy_timeout=2.0):
            connection.execute("INSERT INTO events(value) VALUES ('explicit')")
        connection.set_trace_callback(None)
        # The connection is left exactly as the caller opened it.
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 50
    finally:
        connection.set_trace_callback(None)
        connection.close()
    applied = [int(sql.split("=")[1]) for sql in statements
               if sql.startswith("PRAGMA busy_timeout =")]
    assert applied == [500, 50]


def test_transaction_rolls_back_the_complete_body_on_error(tmp_path):
    path = tmp_path / "rollback.sqlite"
    _create_database(path)
    connection = connect(path)
    try:
        with pytest.raises(RuntimeError, match="abort the unit"):
            with transaction(connection):
                connection.execute(
                    "INSERT INTO events(value) VALUES ('must disappear')")
                raise RuntimeError("abort the unit")
        assert connection.in_transaction is False
        assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    finally:
        connection.close()


def test_nested_transaction_is_refused_instead_of_partly_committing(tmp_path):
    path = tmp_path / "nested.sqlite"
    _create_database(path)
    connection = connect(path)
    try:
        with transaction(connection):
            with pytest.raises(RuntimeError, match="Nested SQLite"):
                with transaction(connection):
                    pass
    finally:
        connection.close()


def test_busy_writer_retries_then_commits_after_lock_release(tmp_path):
    path = tmp_path / "retry.sqlite"
    _create_database(path)
    holder = connect(path, timeout=0.01)
    holder.execute("BEGIN IMMEDIATE")
    started = threading.Event()
    finished = threading.Event()
    errors = []

    def write_after_release():
        connection = connect(path, timeout=0.01)
        try:
            started.set()
            with transaction(
                connection, attempts=20, initial_delay=0.005,
                maximum_delay=0.02,
            ):
                connection.execute(
                    "INSERT INTO events(value) VALUES ('after lock')")
        except BaseException as exc:
            errors.append(exc)
        finally:
            connection.close()
            finished.set()

    worker = threading.Thread(target=write_after_release)
    worker.start()
    assert started.wait(1)
    time.sleep(0.04)
    holder.rollback()
    holder.close()
    assert finished.wait(2)
    worker.join()
    assert errors == []
    reader = connect(path, readonly=True)
    try:
        assert reader.execute(
            "SELECT value FROM events").fetchall() == [("after lock",)]
    finally:
        reader.close()


def test_exhausted_lock_budget_raises_database_busy(tmp_path):
    path = tmp_path / "busy.sqlite"
    _create_database(path)
    holder = connect(path, timeout=0.01)
    contender = connect(path, timeout=0.01)
    try:
        holder.execute("BEGIN IMMEDIATE")
        with pytest.raises(DatabaseBusy, match="remained locked"):
            with transaction(
                contender, attempts=2, initial_delay=0.001,
                maximum_delay=0.001,
            ):
                pytest.fail("a locked transaction body must never start")
        assert contender.in_transaction is False
    finally:
        holder.rollback()
        holder.close()
        contender.close()


def test_read_only_connection_is_enforced_by_sqlite(tmp_path):
    path = tmp_path / "readonly.sqlite"
    _create_database(path)
    connection = connect(path, readonly=True)
    try:
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("INSERT INTO events(value) VALUES ('no')")
    finally:
        connection.close()


def test_wal_reader_sees_committed_snapshot_during_writer_transaction(tmp_path):
    path = tmp_path / "wal.sqlite"
    _create_database(path, journal_mode="WAL")
    writer = connect(path)
    reader = connect(path, readonly=True)
    try:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("INSERT INTO events(value) VALUES ('uncommitted')")
        assert reader.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
        writer.commit()
        assert reader.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
    finally:
        writer.close()
        reader.close()


def test_disposable_probe_has_exact_rows_and_concurrent_reads():
    result = run_concurrency_probe(
        writers=4, readers=3, writes_per_writer=25, journal_mode="WAL")
    assert result.ok
    assert result.actual_rows == result.expected_rows == 100
    assert result.reader_queries > 0
    assert result.journal_mode == "WAL"
    # A default probe never leaves its stress database behind.
    assert not os.path.exists(result.path)


def test_a_probe_that_never_starts_leaves_nothing_in_the_temp_directory():
    """A journal mode the connection refuses is rejected AFTER the scratch
    database has been created, and the cleanup was the last statement of the
    function -- so every rejected run left an empty scratch database in the
    system temp directory, for good. Nothing has been written by then, so
    there is nothing to keep: the deliberate survivor is the STALLED one,
    whose database is worth inspecting."""
    import glob
    import tempfile

    pattern = os.path.join(tempfile.gettempdir(), "spacr-db-concurrency-*")
    before = set(glob.glob(pattern))

    with pytest.raises(DatabaseConfigurationError):
        run_concurrency_probe(writers=1, readers=1, writes_per_writer=1,
                              journal_mode="MEMORY")

    assert set(glob.glob(pattern)) == before


def test_a_probe_whose_setup_fails_leaves_nothing_behind(monkeypatch):
    """Any failure between making the directory and reading the metrics back
    is the same leak, and there are several -- the table create, the barrier,
    the read back. One handler rather than one per raise site."""
    import glob
    import tempfile

    import spacr.database_concurrency as module

    pattern = os.path.join(tempfile.gettempdir(), "spacr-db-concurrency-*")
    before = set(glob.glob(pattern))
    monkeypatch.setattr(module.threading, "Barrier",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("no threads today")))

    with pytest.raises(RuntimeError, match="no threads today"):
        run_concurrency_probe(writers=1, readers=1, writes_per_writer=1)

    assert set(glob.glob(pattern)) == before


def test_a_probe_at_an_explicit_path_that_fails_still_clears_that_path(
        tmp_path):
    """The scratch file is created before the journal mode is rejected, and
    it stays -- which makes the NEXT run on that path fail with
    FileExistsError against a database the user never got a probe out of."""
    path = tmp_path / "scratch.sqlite"

    with pytest.raises(DatabaseConfigurationError):
        run_concurrency_probe(path, writers=1, readers=1,
                              writes_per_writer=1, journal_mode="MEMORY")

    assert not path.exists(), (
        "the next run on this path now fails with FileExistsError")


def test_probe_refuses_to_touch_an_existing_database(tmp_path):
    path = tmp_path / "scientific-results.sqlite"
    _create_database(path)
    with pytest.raises(FileExistsError, match="refuses existing"):
        run_concurrency_probe(path)
    reader = connect(path, readonly=True)
    try:
        assert reader.execute(
            "SELECT name FROM sqlite_master WHERE name='probe_events'"
        ).fetchone() is None
    finally:
        reader.close()


@pytest.mark.parametrize(
    "keyword", ["writers", "readers", "writes_per_writer"])
def test_probe_rejects_nonpositive_work_sizes(keyword):
    options = {"writers": 1, "readers": 1, "writes_per_writer": 1}
    options[keyword] = 0
    with pytest.raises(ValueError, match=rf"{keyword} must be at least 1"):
        run_concurrency_probe(**options)


def test_health_check_is_read_only_and_reports_integrity(tmp_path):
    path = tmp_path / "health.sqlite"
    _create_database(path, journal_mode="WAL")
    before = path.read_bytes()
    health = inspect_database(path, quick_check=True)
    assert health.path == str(path)
    assert health.journal_mode == "WAL"
    assert health.quick_check == "ok"
    assert health.busy_timeout_ms == 5000
    assert path.read_bytes() == before


def test_health_check_warns_about_wal_on_network_storage(tmp_path, monkeypatch):
    path = tmp_path / "network.sqlite"
    _create_database(path, journal_mode="WAL")
    monkeypatch.setattr(db_concurrency, "filesystem_type", lambda _path: "nfs")
    health = inspect_database(path)
    assert health.network_filesystem is True
    assert any("WAL is active on a network filesystem" in warning
               for warning in health.warnings)


def test_concurrent_run_ledgers_do_not_drop_status_rows(tmp_path):
    path = tmp_path / "ledger.sqlite"
    _create_database(path)
    count = 12
    barrier = threading.Barrier(count)
    failures = []

    def stamp(index):
        ledger = RunLedger(f"worker-{index}")
        ledger.record_success(f"field-{index}")
        try:
            barrier.wait(timeout=5)
            ledger.stamp(path)
        except BaseException as exc:
            failures.append(exc)

    threads = [threading.Thread(target=stamp, args=(index,))
               for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert all(not thread.is_alive() for thread in threads)
    assert failures == []
    reader = connect(path, readonly=True)
    try:
        rows = reader.execute(
            f"SELECT name, status FROM {RUN_STATUS_TABLE}").fetchall()
    finally:
        reader.close()
    assert len(rows) == count
    assert {name for name, _status in rows} == {
        f"worker-{index}" for index in range(count)}
    assert {status for _name, status in rows} == {"complete"}


def test_database_audit_cli_inspects_and_probes_as_json(tmp_path, capsys):
    path = tmp_path / "cli.sqlite"
    _create_database(path)
    code = database_cli([
        str(path), "--quick-check", "--probe", "--writers", "2",
        "--readers", "1", "--writes", "10", "--json",
    ])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["database"]["quick_check"] == "ok"
    assert payload["probe"]["ok"] is True
    assert payload["probe"]["actual_rows"] == 20


def test_database_audit_cli_fails_loudly_for_bad_input(tmp_path, capsys):
    code = database_cli([str(tmp_path / "missing.sqlite"), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["error"].startswith("FileNotFoundError:")
