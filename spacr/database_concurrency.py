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
import logging
import operator
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

LOG = logging.getLogger(__name__)

__all__ = [
    "ConcurrencyProbeResult",
    "DatabaseBusy",
    "DatabaseConfigurationError",
    "DatabaseHealth",
    "MINIMUM_ATTEMPT_BUSY_TIMEOUT_MS",
    "WAL_SAFE_FILESYSTEMS",
    "connect",
    "enable_wal_where_safe",
    "filesystem_type",
    "inspect_database",
    "is_busy_error",
    "run_concurrency_probe",
    "transaction",
    "wal_is_safe_here",
]


SAFE_JOURNAL_MODES = {"DELETE", "WAL"}
TRANSACTION_MODES = {"DEFERRED", "IMMEDIATE", "EXCLUSIVE"}
NETWORK_FILESYSTEMS = {
    "9p", "afs", "cifs", "fuse.sshfs", "gcsfuse", "lustre", "nfs",
    "nfs4", "s3fs", "smbfs",
}

#: Filesystems on which WAL is known to behave. An ALLOWLIST, not the
#: complement of :data:`NETWORK_FILESYSTEMS`: "not a type I recognise as
#: networked" is a weaker claim than "local", and the gap between them is
#: someone's corrupted database on a cluster nobody here has seen. An
#: unrecognised type stays on DELETE, which is what shipped.
#:
#: exFAT and vfat are deliberately absent: neither has the byte-range locking
#: WAL's shared-memory index depends on.
WAL_SAFE_FILESYSTEMS = {
    "apfs", "btrfs", "ext2", "ext3", "ext4", "f2fs", "hfs", "hfsplus",
    "jfs", "overlay", "reiserfs", "tmpfs", "xfs", "zfs",
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
        attempt = 0
        while True:
            attempt += 1
            try:
                connection.execute(f"BEGIN {selected}")
                break
            except sqlite3.OperationalError as exc:
                if not is_busy_error(exc):
                    raise
                if attempt >= attempts:
                    raise DatabaseBusy(
                        f"database remained locked after {attempts} "
                        f"transaction attempts: {exc}") from exc
                time.sleep(delay)
                delay = min(maximum_delay, max(initial_delay, delay * 2))
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


def _filesystem_type_via_psutil(target: Path) -> Optional[str]:
    """Filesystem type for ``target`` off psutil's partition table.

    The longest matching mount point wins, exactly as the ``/proc/mounts``
    reader does -- ``/`` matches everything, so the nested mount has to beat
    it or an SMB share under ``/Volumes`` would be reported as the root
    filesystem and treated as safe.

    ``all=True`` because the default hides network mounts on some platforms,
    which are the ones this function exists to find.
    """
    try:
        import psutil
    except Exception:                                        # noqa: BLE001
        return None
    # NO WALK-UP BEFORE MATCHING. A mount point either is a prefix of this
    # path or it is not, and that is true whether or not the leaf exists yet --
    # a measurement.db about to be created on a share is still on the share.
    # Walking up to the nearest EXISTING ancestor first sent a path under a
    # share that had no file yet all the way to "/", which matches the root
    # mount and reports the local disk. That is the one wrong answer that
    # matters here: the root is usually apfs, apfs is on WAL_SAFE_FILESYSTEMS,
    # and the result would be WAL enabled on a network share.
    best: Optional[tuple] = None
    try:
        partitions = psutil.disk_partitions(all=True)
    except Exception:                                        # noqa: BLE001
        # Advisory only: a platform that refuses to enumerate mounts leaves
        # the answer unknown, which wal_is_safe_here already treats as unsafe.
        return None
    for part in partitions:
        mount = str(getattr(part, "mountpoint", "") or "")
        fstype = str(getattr(part, "fstype", "") or "")
        if not mount or not fstype:
            continue
        try:
            target.relative_to(mount)
        except ValueError:
            continue
        if best is None or len(mount) > best[0]:
            best = (len(mount), fstype)
    return None if best is None else str(best[1])


def filesystem_type(path: os.PathLike | str) -> Optional[str]:
    """Best-effort filesystem type for ``path``, or None when unknowable.

    Reads ``/proc/mounts`` on Linux and falls back to psutil's partition table
    elsewhere, so macOS and Windows get a real answer rather than None. The
    longest matching mount point wins on both paths. Advisory only--containers
    and automounters can hide the real backing store.
    """
    target = Path(path).expanduser().resolve()
    mounts = Path("/proc/mounts")
    if not mounts.is_file():
        # NOT LINUX. Until this branch existed the answer here was None on
        # every macOS and Windows machine, and `wal_is_safe_here` turns None
        # into False -- so every Mac ran without WAL even on local APFS, and,
        # worse, `doctor` could not tell a user on an SMB share that they WERE
        # on one. Issue 115 is exactly that reporter: Apple Silicon, a
        # measurement.db on an SMB server, and nothing in spaCR able to name
        # the filesystem in its own diagnosis.
        #
        # psutil is already a declared dependency and reports fstype on every
        # platform spaCR supports, so this needs no new requirement.
        return _filesystem_type_via_psutil(target)
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


def wal_is_safe_here(path: os.PathLike | str) -> bool:
    """Is ``path`` on a filesystem where WAL is known to behave?

    ``True`` only for a POSITIVELY IDENTIFIED local filesystem. Anything
    else -- a network type, an unrecognised type, or a platform where
    :func:`filesystem_type` cannot tell (it reads ``/proc/mounts``, so macOS
    and Windows always answer ``None``) -- is ``False``.

    That asymmetry is the point. The cost of a wrong ``False`` is the
    lock contention this project already survives; the cost of a wrong
    ``True`` is WAL shared memory on storage that cannot support it, which
    is a corrupted database.
    """
    fs_type = filesystem_type(path)
    if not fs_type:
        return False
    fs_type = fs_type.casefold()
    if fs_type in NETWORK_FILESYSTEMS:
        return False
    return fs_type in WAL_SAFE_FILESYSTEMS


def enable_wal_where_safe(path: os.PathLike | str) -> Optional[str]:
    """Put ``path`` into WAL when the filesystem allows it. Never raises.

    WHY THIS EXISTS (issue #15, "measurements sometimes hangs"). Measure
    runs one worker per field and every append goes through pandas'
    ``DataFrame.to_sql``, which issues a ``has_table`` probe -- a READ --
    before writing. So each worker alternates read, write, read, write
    against one file.

    Under the shipped rollback journal that combination starves the writer.
    A reader holds SHARED for the length of its statement, and a writer
    cannot COMMIT until every SHARED lock is gone, so with enough workers
    there is almost always someone reading and the committing worker waits
    out its busy timeout and raises "database is locked" -- usually
    surfacing on the next process's ``has_table``, which is the statement in
    the reporter's traceback.

    Measured on this exact shape, a commit attempted while one reader holds
    an open SELECT:

        journal_mode=delete    writer waited 1.037 s
        journal_mode=wal       writer waited 0.002 s

    WAL readers do not block a writer at all, which removes the starvation.
    It does NOT make two WRITERS concurrent -- SQLite still serialises those
    -- so this fixes the reader-blocks-writer half, which is the half the
    traceback is in.

    (The first version of this note had the direction backwards, claiming
    reads were blocked by writes. The test written to prove it measured
    0.000 s and refuted it: in rollback-journal mode a writer holding
    RESERVED does not block readers, only its brief EXCLUSIVE commit does.)

    Called once when a database is opened for a run rather than per write:
    the mode is a property of the FILE and persists, so paying for it per
    connection would buy nothing.

    :param path: the database to switch. A file that does not exist yet is
        created by the connection, which is fine -- the mode persists.
    :returns: the journal mode in force afterwards, or ``None`` when the
        database could not be opened at all.
    """
    if not wal_is_safe_here(path):
        return None
    try:
        connection = connect(path, journal_mode="WAL")
    except (sqlite3.Error, DatabaseConfigurationError, OSError):
        # A refusal here is informative, not fatal: SQLite declines WAL on
        # storage that cannot hold it, which is exactly the outcome the
        # allowlist is guessing at. Staying on DELETE is the shipped
        # behaviour, so the run continues as it always did.
        return None
    try:
        return str(connection.execute(
            "PRAGMA journal_mode").fetchone()[0]).upper()
    except sqlite3.Error:
        return None
    finally:
        connection.close()


@dataclass(frozen=True)
class DatabaseHealth:
    """Read-only SQLite configuration and integrity snapshot.

    :param path: normalized absolute path of the inspected database.
    :param sqlite_version: SQLite runtime version exposed by Python.
    :param sqlite_threadsafe: DB-API thread-safety level reported by
        :data:`sqlite3.threadsafety`.
    :param journal_mode: actual uppercase journal mode read from the database.
    :param foreign_keys: whether enforcement is enabled on the audit
        connection, not a persistent database-wide promise.
    :param busy_timeout_ms: audit connection's effective busy timeout in
        milliseconds.
    :param filesystem: detected filesystem type, or ``None`` when unavailable.
    :param network_filesystem: whether the detected type is in the known
        network-filesystem set; false with an unknown type does not prove the
        storage is local.
    :param quick_check: joined ``PRAGMA quick_check`` result when requested,
        otherwise ``None``.
    :param file_bytes: main database-file size at inspection time.
    :param wal_bytes: ``-wal`` sidecar size at inspection time, or zero when it
        is absent.
    :param shm_bytes: ``-shm`` sidecar size at inspection time, or zero when it
        is absent.
    :param warnings: actionable integrity or unsafe network-WAL findings.
    """

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
    """Outcome of a disposable simultaneous reader/writer stress probe.

    :param path: scratch database path; a clean temporary probe removes it,
        while explicit or stalled probes retain it for inspection.
    :param journal_mode: actual uppercase journal mode read after the run.
    :param writers: validated number of writer threads launched.
    :param readers: validated number of polling reader threads launched.
    :param writes_per_writer: one-row committed transactions each writer tries.
    :param expected_rows: ``writers * writes_per_writer``, independent of any
        worker failures.
    :param actual_rows: final row count verified after the bounded joins.
    :param reader_queries: total successful ``COUNT`` queries across readers.
    :param duration_seconds: monotonic worker start-to-join elapsed time,
        excluding setup and final verification.
    :param errors: immutable worker exceptions and surviving-thread timeout
        messages collected by the probe.
    """

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


def _positive_probe_count(name: str, value: Any) -> int:
    """Return one genuine positive integer, without lossy coercion."""
    if isinstance(value, bool):
        raise TypeError(
            f"{name} must be a positive integer, got {value!r}")
    try:
        count = operator.index(value)
    except TypeError as exc:
        raise TypeError(
            f"{name} must be a positive integer, got {value!r}") from exc
    if count < 1:
        raise ValueError(f"{name} must be at least 1, got {count}")
    return int(count)


def _probe_journal_mode(value: Any) -> str:
    """Return the explicit SQLite mode a stress probe must exercise."""
    if not isinstance(value, str):
        raise DatabaseConfigurationError(
            f"journal_mode must be one of {sorted(SAFE_JOURNAL_MODES)}, "
            f"got {value!r}.")
    requested = value.strip().upper()
    if requested not in SAFE_JOURNAL_MODES:
        raise DatabaseConfigurationError(
            f"journal_mode must be one of {sorted(SAFE_JOURNAL_MODES)}, "
            f"got {value!r}.")
    return requested


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

    :param path: scratch database to create. It must not already exist
        (:exc:`FileExistsError`); missing parent directories are created, and
        a run that FINISHES leaves the file on disk along with any ``-wal``
        and ``-shm`` sidecars, so every explicit run needs a fresh path. Omit
        it to probe a temporary database instead, which is removed after a
        clean finish. A run that RAISES -- a ``journal_mode`` :func:`connect`
        refuses, most often -- removes its scratch database either way: it
        never ran, so there is nothing in it to keep, and leaving one at an
        explicit path made the next run on it fail with
        :exc:`FileExistsError`. The one deliberate survivor is a worker that
        outlives the 30-second join deadline, whose database is kept for
        inspection.
    :param writers: concurrent writer threads. Each owns a connection opened
        with a 50 ms busy timeout and commits one transaction per row, so
        ``expected_rows`` is ``writers * writes_per_writer``. Must be a
        genuine positive integer; booleans, text and floats are refused.
    :param readers: concurrent read-only threads polling ``COUNT(*)`` until
        the last writer exits. They move only ``reader_queries``, never
        ``expected_rows``; at least one is required, so a writers-only probe
        cannot be expressed. Must be a genuine positive integer.
    :param writes_per_writer: rows each writer inserts, one row per
        transaction. Must be a genuine positive integer.
    :param journal_mode: mode applied once by the setup connection and then
        inherited by every worker connection. Only ``"WAL"`` (the default)
        and ``"DELETE"`` are accepted, case-insensitively; anything else
        raises :exc:`DatabaseConfigurationError` after the scratch database
        has already been created. ``None`` is refused: a stress result must
        state which locking mode it actually intended to exercise.
    :returns: result whose ``journal_mode`` is read back from the finished
        database rather than echoed from this argument, and whose ``errors``
        carry per-thread failures instead of raising.
    :raises ValueError: when ``writers``, ``readers``, or
        ``writes_per_writer`` is below 1.
    :raises TypeError: when one of the work sizes is not an integer. In
        particular, ``2.9`` is not silently truncated and ``"2"`` is not
        accepted merely because the CLI parser would have converted it.
    :raises DatabaseConfigurationError: when ``journal_mode`` is not an
        explicit ``"WAL"`` or ``"DELETE"`` string.
    """
    writers = _positive_probe_count("writers", writers)
    readers = _positive_probe_count("readers", readers)
    writes_per_writer = _positive_probe_count(
        "writes_per_writer", writes_per_writer)
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

    # A PROBE THAT NEVER RAN LEAVES NOTHING BEHIND. Everything from here to
    # the metrics is inside one handler, because a failure anywhere in it
    # happens AFTER the scratch database has been created and the cleanup
    # used to be the last statement of the function. The commonest is a
    # journal mode `connect` refuses -- 'MEMORY', 'TRUNCATE' -- which left an
    # empty scratch database in the system temp directory for good, and at an
    # explicit path left a file that makes the NEXT run on it fail with
    # FileExistsError against a database the user never got a probe out of.
    #
    # The deliberate survivor is the STALLED one: a worker that outlives the
    # join deadline is a normal return, guarded by `not alive` at the end, and
    # its database is worth keeping to look at.
    try:
        journal_mode = _probe_journal_mode(journal_mode)
        return _run_probe(
            db_path, writers=writers, readers=readers,
            writes_per_writer=writes_per_writer, journal_mode=journal_mode,
            temporary_dir=temporary_dir)
    except BaseException:
        _discard_scratch(db_path, temporary_dir)
        raise


def _discard_scratch(db_path: str, temporary_dir: Optional[str]) -> None:
    """Remove a scratch database the probe created and never used.

    Never raises: the caller is already unwinding, and a cleanup error would
    replace the real reason with a filesystem one.
    """
    try:
        if temporary_dir is not None:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            return
        # An explicit path, with the sidecars WAL leaves beside it.
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(db_path + suffix)
            except OSError:
                pass
    except Exception:      # below OSError: a path the OS rejects outright
        LOG.debug("could not remove the probe's scratch database",
                  exc_info=True)


def _run_probe(
    db_path: str,
    *,
    writers: int,
    readers: int,
    writes_per_writer: int,
    journal_mode: Optional[str],
    temporary_dir: Optional[str],
) -> ConcurrencyProbeResult:
    """The probe itself, once the scratch database's path is settled."""
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
        """Insert this writer's rows and signal when the last writer exits.

        Worker failures are collected for the probe result, and the thread's
        connection is closed whether setup, synchronization, or writing fails.
        """
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
        """Poll during writes, then contribute this reader's query count.

        A final query observes the completed database; failures are collected
        and the thread-owned read-only connection is always closed.
        """
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
    # ``is_alive`` can change between the post-join snapshot above and this
    # final check. Keep only workers that are still alive now, so a thread
    # that exits in that small window is neither reported as stalled nor used
    # to preserve an otherwise disposable scratch database.
    survivors = []
    for thread in alive:
        if thread.is_alive():
            errors.put(f"thread {thread.name} did not finish within 30 seconds")
            survivors.append(thread)
    alive = survivors
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
