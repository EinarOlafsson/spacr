"""SQLite connection, transaction, and concurrency-audit primitives.

spaCR uses one SQLite database as the meeting point for Measure worker
processes, the annotator writer thread, read-only GUI queries, run-status
stamps, and schema migrations.  This module provides the rules those paths
share:

* every thread/process opens and closes its own connection;
* busy timeouts are explicit and write transactions retry only lock errors;
* multi-statement writes use ``BEGIN IMMEDIATE`` with rollback on every error;
* WAL is opt-in because SQLite WAL shared memory is unsafe on many network
  filesystems;
* a real reader/writer probe can verify the local SQLite/filesystem behavior.

Only the Python standard library is imported, so image workers can use it
without pulling in pandas, Qt, torch, or Cellpose.
"""
from __future__ import annotations

import contextlib
import os
import queue
import shutil
import sqlite3
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence
from urllib.parse import quote

__all__ = [
    "ConcurrencyProbeResult",
    "DatabaseBusy",
    "DatabaseConfigurationError",
    "DatabaseHealth",
    "MINIMUM_ATTEMPT_BUSY_TIMEOUT_MS",
    "connect",
    "filesystem_type",
    "inspect_database",
    "is_busy_error",
    "run_concurrency_probe",
    "transaction",
]


SAFE_JOURNAL_MODES = {"DELETE", "WAL"}
TRANSACTION_MODES = {"DEFERRED", "IMMEDIATE", "EXCLUSIVE"}
NETWORK_FILESYSTEMS = {
    "9p", "afs", "cifs", "fuse.sshfs", "gcsfuse", "lustre", "nfs",
    "nfs4", "s3fs", "smbfs",
}


class DatabaseBusy(sqlite3.OperationalError):
    """A lock remained busy after the configured retry budget."""


class DatabaseConfigurationError(RuntimeError):
    """SQLite could not apply a requested safety configuration."""


def is_busy_error(error: BaseException) -> bool:
    """Return True only for SQLite lock/busy errors worth retrying."""
    if not isinstance(error, sqlite3.OperationalError):
        return False
    message = str(error).casefold()
    return "locked" in message or "busy" in message


def _read_only_uri(path: os.PathLike | str) -> str:
    """Return a correctly escaped SQLite ``mode=ro`` URI."""
    absolute = os.path.abspath(os.path.expanduser(os.fspath(path)))
    return f"file:{quote(absolute, safe='/:')}?mode=ro"


def connect(
    path: os.PathLike | str,
    *,
    readonly: bool = False,
    timeout: float = 30.0,
    journal_mode: Optional[str] = None,
    foreign_keys: bool = True,
) -> sqlite3.Connection:
    """Open one configured connection owned by the calling thread.

    :param path: SQLite database path.
    :param readonly: open with URI ``mode=ro`` and ``query_only=ON``.
    :param timeout: seconds SQLite waits inside a lock operation.
    :param journal_mode: optional explicit ``"WAL"`` or ``"DELETE"``.
        Omit to preserve the database's current mode. WAL must not be enabled
        blindly on shared/NFS storage; use :func:`filesystem_type` or the
        concurrency probe first.
    :param foreign_keys: enable SQLite foreign-key enforcement on this
        connection. SQLite defaults it off per connection.
    :returns: connection in autocommit mode; use :func:`transaction` for
        multi-statement writes.
    :raises DatabaseConfigurationError: for an unsafe/unsupported requested
        journal mode or when SQLite refuses to apply it.
    """
    timeout = max(0.0, float(timeout))
    if readonly:
        connection = sqlite3.connect(
            _read_only_uri(path),
            uri=True,
            timeout=timeout,
            isolation_level=None,
        )
    else:
        connection = sqlite3.connect(
            os.path.abspath(os.path.expanduser(os.fspath(path))),
            timeout=timeout,
            isolation_level=None,
        )
    try:
        connection.execute(
            f"PRAGMA busy_timeout = {max(0, int(timeout * 1000))}")
        connection.execute(
            f"PRAGMA foreign_keys = {'ON' if foreign_keys else 'OFF'}")
        if readonly:
            connection.execute("PRAGMA query_only = ON")
        if journal_mode is not None:
            if readonly:
                raise DatabaseConfigurationError(
                    "A read-only connection cannot change journal_mode.")
            requested = str(journal_mode).strip().upper()
            if requested not in SAFE_JOURNAL_MODES:
                raise DatabaseConfigurationError(
                    f"journal_mode must be one of {sorted(SAFE_JOURNAL_MODES)}, "
                    f"got {journal_mode!r}.")
            actual = str(connection.execute(
                f"PRAGMA journal_mode = {requested}").fetchone()[0]).upper()
            if actual != requested:
                raise DatabaseConfigurationError(
                    f"SQLite kept journal_mode={actual}, not the requested "
                    f"{requested}. The filesystem may not support it.")
            if actual == "WAL":
                connection.execute("PRAGMA synchronous = NORMAL")
        return connection
    except BaseException:
        connection.close()
        raise


# Smallest per-attempt busy timeout worth asking SQLite for.
#
# SQLite's default busy handler sleeps down a fixed ladder --
# 1, 2, 5, 10, 15, 20, 25, 25, 25, 50, 50, 100 ms -- and clamps the final
# sleep to whatever is left of the budget. A 1 ms budget therefore buys one
# 1 ms sleep and a single re-try of the lock: BEGIN gives up while the holder
# is still inside its commit, and the caller burns a whole retry attempt on a
# lock that was about to be released. 25 ms is where the ladder reaches its
# steady step, so it is the smallest budget in which a contended writer can
# realistically hand the lock over.
MINIMUM_ATTEMPT_BUSY_TIMEOUT_MS = 25


def _attempt_busy_timeout_ms(total_busy_timeout_ms: int, attempts: int) -> int:
    """Split a total lock-wait budget over ``attempts`` BEGIN retries.

    Plain integer division collapses at the small end: the concurrency probe
    opens its writers with ``timeout=0.05`` (a 50 ms budget) and then asks for
    40 attempts, which is 1 ms each -- below the floor at which SQLite can
    acquire a contended lock at all. The split is therefore clamped:

    * never below ``MINIMUM_ATTEMPT_BUSY_TIMEOUT_MS``, so an attempt is a
      real attempt rather than an instant SQLITE_BUSY;
    * never above the total, so a caller who asked to wait at most 10 ms in
      SQLite still waits at most 10 ms in any one BEGIN;
    * zero stays zero, because a connection with ``busy_timeout = 0`` asked
      for non-blocking locking and must keep it.

    When the floor wins, ``attempts`` BEGINs can spend more than the total
    budget inside SQLite (40 x 25 ms rather than 50 ms). That is deliberate
    and is dwarfed by the retry loop's own backoff sleeps; a per-attempt
    budget of 1 ms is not a smaller budget, it is no budget.

    :param total_busy_timeout_ms: whole-operation lock-wait budget, in ms.
    :param attempts: number of ``BEGIN`` attempts to share it between.
    :returns: milliseconds to give each individual ``BEGIN``.
    """
    total = max(0, int(total_busy_timeout_ms))
    attempts = max(1, int(attempts))
    if total == 0:
        return 0
    return min(total, max(MINIMUM_ATTEMPT_BUSY_TIMEOUT_MS, total // attempts))


@contextlib.contextmanager
def transaction(
    connection: sqlite3.Connection,
    *,
    mode: str = "IMMEDIATE",
    attempts: int = 8,
    initial_delay: float = 0.01,
    maximum_delay: float = 0.25,
    busy_timeout: Optional[float] = None,
) -> Iterator[sqlite3.Connection]:
    """Run an all-or-nothing transaction with bounded lock retry.

    Only ``BEGIN`` is retried. Once a transaction starts, retrying individual
    statements could duplicate earlier writes. Any body or commit error rolls
    the complete transaction back and propagates.

    :param connection: calling thread's open autocommit connection.
    :param mode: ``DEFERRED``, ``IMMEDIATE`` (default), or ``EXCLUSIVE``.
    :param attempts: maximum attempts to acquire the transaction.
    :param initial_delay: first backoff between lock failures.
    :param maximum_delay: backoff cap.
    :param busy_timeout: total seconds this transaction may spend waiting on
        locks inside SQLite, shared over ``attempts`` and floored at
        ``MINIMUM_ATTEMPT_BUSY_TIMEOUT_MS`` per attempt. Omit to inherit the
        connection's configured ``busy_timeout``; pass it when the write's own
        tolerance differs from whatever ``timeout`` the connection happened to
        be opened with.
    :raises DatabaseBusy: when the lock outlives the retry budget.
    :raises RuntimeError: when asked to nest inside an active transaction.
    """
    selected = str(mode).strip().upper()
    if selected not in TRANSACTION_MODES:
        raise ValueError(
            f"transaction mode must be one of {sorted(TRANSACTION_MODES)}")
    if connection.in_transaction:
        raise RuntimeError(
            "Nested SQLite transactions are not supported; finish the active "
            "transaction before starting another.")
    attempts = max(1, int(attempts))
    delay = max(0.0, float(initial_delay))
    last_error: Optional[BaseException] = None
    timeout_row = connection.execute("PRAGMA busy_timeout").fetchone()
    original_busy_timeout = int(timeout_row[0]) if timeout_row else 0
    # sqlite's busy_timeout applies to *each* BEGIN. Without dividing the
    # caller's budget, eight retries on a 30-second connection can block for
    # four minutes. Share that budget across attempts -- clamped by
    # _attempt_busy_timeout_ms, because a plain division handed a 50 ms
    # connection asking for 40 attempts 1 ms per BEGIN -- then restore the
    # connection's own value before executing the transaction body.
    total_busy_timeout = (
        original_busy_timeout if busy_timeout is None
        else max(0, int(float(busy_timeout) * 1000))
    )
    attempt_busy_timeout = _attempt_busy_timeout_ms(total_busy_timeout, attempts)
    changed_timeout = attempt_busy_timeout != original_busy_timeout
    if changed_timeout:
        connection.execute(f"PRAGMA busy_timeout = {attempt_busy_timeout}")
    try:
        for attempt in range(1, attempts + 1):
            try:
                connection.execute(f"BEGIN {selected}")
                break
            except sqlite3.OperationalError as exc:
                if not is_busy_error(exc):
                    raise
                last_error = exc
                if attempt == attempts:
                    raise DatabaseBusy(
                        f"database remained locked after {attempts} "
                        f"transaction attempts: {exc}") from exc
                time.sleep(delay)
                delay = min(maximum_delay, max(initial_delay, delay * 2))
        else:  # pragma: no cover - loop always breaks or raises
            raise DatabaseBusy(str(last_error))
    finally:
        if changed_timeout:
            try:
                connection.execute(
                    f"PRAGMA busy_timeout = {original_busy_timeout}")
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise

    try:
        yield connection
    except BaseException:
        if connection.in_transaction:
            try:
                connection.rollback()
            except sqlite3.Error:
                pass
        raise
    else:
        try:
            connection.commit()
        except BaseException:
            if connection.in_transaction:
                try:
                    connection.rollback()
                except sqlite3.Error:
                    pass
            raise


def filesystem_type(path: os.PathLike | str) -> Optional[str]:
    """Best-effort Linux filesystem type for ``path``; None elsewhere.

    The longest matching mount point in ``/proc/mounts`` wins. This is
    advisory only—containers and automounters can hide the real backing store.
    """
    mounts = Path("/proc/mounts")
    if not mounts.is_file():
        return None
    target = Path(path).expanduser().resolve()
    while not target.exists() and target != target.parent:
        target = target.parent
    best: Optional[tuple] = None
    try:
        lines = mounts.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        mount = parts[1].replace("\\040", " ")
        try:
            target.relative_to(mount)
        except ValueError:
            continue
        candidate = (len(mount), parts[2])
        if best is None or candidate[0] > best[0]:
            best = candidate
    return None if best is None else str(best[1])


@dataclass(frozen=True)
class DatabaseHealth:
    """Read-only SQLite configuration and integrity snapshot."""

    path: str
    sqlite_version: str
    sqlite_threadsafe: int
    journal_mode: str
    foreign_keys: bool
    busy_timeout_ms: int
    filesystem: Optional[str]
    network_filesystem: bool
    quick_check: Optional[str]
    file_bytes: int
    wal_bytes: int
    shm_bytes: int
    warnings: Sequence[str] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot."""
        return asdict(self)


def inspect_database(
    path: os.PathLike | str,
    *,
    quick_check: bool = False,
    timeout: float = 5.0,
) -> DatabaseHealth:
    """Inspect journal/locking configuration without changing the database."""
    absolute = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if not os.path.isfile(absolute):
        raise FileNotFoundError(absolute)
    connection = connect(absolute, readonly=True, timeout=timeout)
    try:
        mode = str(connection.execute(
            "PRAGMA journal_mode").fetchone()[0]).upper()
        foreign_keys = bool(connection.execute(
            "PRAGMA foreign_keys").fetchone()[0])
        busy_timeout = int(connection.execute(
            "PRAGMA busy_timeout").fetchone()[0])
        check = None
        if quick_check:
            rows = connection.execute("PRAGMA quick_check").fetchall()
            check = "; ".join(str(row[0]) for row in rows)
    finally:
        connection.close()
    fs_type = filesystem_type(absolute)
    network = bool(fs_type and fs_type.casefold() in NETWORK_FILESYSTEMS)
    warnings: List[str] = []
    if network and mode == "WAL":
        warnings.append(
            "WAL is active on a network filesystem. SQLite WAL requires "
            "shared-memory coordination on one host; use DELETE mode unless "
            "the storage vendor explicitly guarantees WAL semantics.")
    if quick_check and check != "ok":
        warnings.append(f"SQLite quick_check reported: {check}")
    return DatabaseHealth(
        path=absolute,
        sqlite_version=sqlite3.sqlite_version,
        sqlite_threadsafe=int(sqlite3.threadsafety),
        journal_mode=mode,
        foreign_keys=foreign_keys,
        busy_timeout_ms=busy_timeout,
        filesystem=fs_type,
        network_filesystem=network,
        quick_check=check,
        file_bytes=os.path.getsize(absolute),
        wal_bytes=os.path.getsize(f"{absolute}-wal")
        if os.path.isfile(f"{absolute}-wal") else 0,
        shm_bytes=os.path.getsize(f"{absolute}-shm")
        if os.path.isfile(f"{absolute}-shm") else 0,
        warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class ConcurrencyProbeResult:
    """Outcome of a disposable simultaneous reader/writer stress probe."""

    path: str
    journal_mode: str
    writers: int
    readers: int
    writes_per_writer: int
    expected_rows: int
    actual_rows: int
    reader_queries: int
    duration_seconds: float
    errors: Sequence[str] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """True when no thread failed and every committed row exists."""
        return not self.errors and self.actual_rows == self.expected_rows

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable result including :attr:`ok`."""
        result = asdict(self)
        result["ok"] = self.ok
        return result


def run_concurrency_probe(
    path: Optional[os.PathLike | str] = None,
    *,
    writers: int = 4,
    readers: int = 3,
    writes_per_writer: int = 50,
    journal_mode: str = "WAL",
) -> ConcurrencyProbeResult:
    """Stress a new disposable database with simultaneous readers/writers.

    An explicit ``path`` must not exist: the probe never adds audit tables to
    scientific data. When omitted, a temporary database is created and
    removed after its metrics are collected.
    """
    writers = int(writers)
    readers = int(readers)
    writes_per_writer = int(writes_per_writer)
    for name, value in (
        ("writers", writers),
        ("readers", readers),
        ("writes_per_writer", writes_per_writer),
    ):
        if value < 1:
            raise ValueError(f"{name} must be at least 1, got {value}")
    temporary_dir = None
    if path is None:
        temporary_dir = tempfile.mkdtemp(prefix="spacr-db-concurrency-")
        db_path = os.path.join(temporary_dir, "probe.sqlite")
    else:
        db_path = os.path.abspath(os.path.expanduser(os.fspath(path)))
        if os.path.exists(db_path):
            raise FileExistsError(
                f"Concurrency probe refuses existing database {db_path}; "
                "choose a new scratch path.")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    setup = connect(db_path, timeout=5, journal_mode=journal_mode)
    try:
        with transaction(setup):
            setup.execute(
                "CREATE TABLE probe_events ("
                "writer INTEGER NOT NULL, sequence INTEGER NOT NULL, "
                "payload TEXT NOT NULL, PRIMARY KEY(writer, sequence))")
    finally:
        setup.close()

    barrier = threading.Barrier(writers + readers)
    writers_left = [writers]
    writers_lock = threading.Lock()
    finished = threading.Event()
    errors: "queue.Queue[str]" = queue.Queue()
    reader_queries = [0]
    reader_lock = threading.Lock()

    def writer_task(writer_id: int) -> None:
        connection = None
        try:
            connection = connect(db_path, timeout=0.05)
            barrier.wait(timeout=10)
            for sequence in range(writes_per_writer):
                with transaction(
                    connection, attempts=40, initial_delay=0.002,
                    maximum_delay=0.05,
                ):
                    connection.execute(
                        "INSERT INTO probe_events VALUES (?, ?, ?)",
                        (writer_id, sequence, f"{writer_id}:{sequence}"),
                    )
        except BaseException as exc:
            errors.put(f"writer {writer_id}: {type(exc).__name__}: {exc}")
        finally:
            if connection is not None:
                connection.close()
            with writers_lock:
                writers_left[0] -= 1
                if writers_left[0] == 0:
                    finished.set()

    def reader_task(reader_id: int) -> None:
        connection = None
        local_queries = 0
        try:
            connection = connect(db_path, readonly=True, timeout=0.25)
            barrier.wait(timeout=10)
            while not finished.is_set():
                connection.execute(
                    "SELECT COUNT(*) FROM probe_events").fetchone()
                local_queries += 1
                time.sleep(0.001)
            connection.execute(
                "SELECT COUNT(*) FROM probe_events").fetchone()
            local_queries += 1
        except BaseException as exc:
            errors.put(f"reader {reader_id}: {type(exc).__name__}: {exc}")
        finally:
            if connection is not None:
                connection.close()
            with reader_lock:
                reader_queries[0] += local_queries

    started = time.monotonic()
    threads = [
        threading.Thread(target=writer_task, args=(index,), daemon=True)
        for index in range(writers)
    ] + [
        threading.Thread(target=reader_task, args=(index,), daemon=True)
        for index in range(readers)
    ]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 30.0
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    alive = [thread for thread in threads if thread.is_alive()]
    if alive:
        # Release reader loops even if a writer stalled, then give all workers
        # one final bounded chance to close their thread-owned connection.
        finished.set()
        for thread in alive:
            thread.join(timeout=1.0)
        alive = [thread for thread in threads if thread.is_alive()]
    for thread in alive:
        if thread.is_alive():  # explicit for type checkers and readability
            errors.put(f"thread {thread.name} did not finish within 30 seconds")
    elapsed = time.monotonic() - started

    verify = connect(db_path, readonly=True, timeout=2)
    try:
        actual = int(verify.execute(
            "SELECT COUNT(*) FROM probe_events").fetchone()[0])
        actual_mode = str(verify.execute(
            "PRAGMA journal_mode").fetchone()[0]).upper()
    finally:
        verify.close()
    collected: List[str] = []
    while True:
        try:
            collected.append(errors.get_nowait())
        except queue.Empty:
            break
    result = ConcurrencyProbeResult(
        path=db_path,
        journal_mode=actual_mode,
        writers=writers,
        readers=readers,
        writes_per_writer=writes_per_writer,
        expected_rows=writers * writes_per_writer,
        actual_rows=actual,
        reader_queries=reader_queries[0],
        duration_seconds=elapsed,
        errors=tuple(collected),
    )
    if temporary_dir is not None and not alive:
        shutil.rmtree(temporary_dir)
    return result
