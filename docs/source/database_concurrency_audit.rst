Database concurrency audit
==========================

spaCR's measurement workers, annotation writer, run-status ledger, schema
migrations, and Database Browser can access the same SQLite file from
different processes or threads. The shared
:mod:`spacr.database_concurrency` contract makes those accesses explicit:

* every thread or process owns and closes its own connection;
* every connection has a finite ``busy_timeout`` and foreign-key enforcement;
* read-only work uses SQLite ``mode=ro`` plus ``query_only``;
* multi-statement writes use ``BEGIN IMMEDIATE`` and roll back completely on a
  body or commit failure;
* only lock/busy failures while acquiring a transaction are retried, with a
  bounded exponential backoff;
* an exhausted lock budget raises :class:`spacr.database_concurrency.DatabaseBusy`
  instead of dropping a write or continuing silently.

Measure workers enqueue SQLite results for one dedicated writer. Object
tables, crop indices, intensity provenance and confluency for a field commit
in one transaction. A failed later operation rolls back the earlier rows and
schema changes for that field. Run status records success only after commit.
Standalone measurement helpers retain their specialized recovery for
``CREATE TABLE`` and schema-widening races. Run-status table creation and row
insertion are one atomic transaction. Database Browser edit validation and its
single-row update also share one transaction, so a row address cannot change
between the check and the write. Resume's multi-table
delete-before-remeasure validates and deletes under one retried write
transaction. Annotate preserves the configured journal mode, rolls back a
failed coalesced batch, retains an unsaved/error state, and reports that state
in the module instead of marking a failed commit as saved.

Measure's bounded write queue
-----------------------------

In **Preferences → Performance**, **Database write queue RAM** controls the
serialized data waiting for the SQLite writer. The default is 1 GiB; zero
uses disk-only buffering. A Python or headless call can override the saved
preference with ``database_write_queue_gib`` in the Measure settings, from
zero to 64. The allowance accounts for queued payload and transport copies;
it does not cap worker arrays, the active write, or total process memory.

Overflow is stored in a private run folder beneath
``measurements/.write_queue``. The transport holds a bounded number of
entries, and overflow descriptors refer to disk data. Packets also retain a
durable copy until commit, so cancellation, disk exhaustion or a failed writer
cannot silently discard accepted data. Successfully committed packets are
removed. Uncommitted packets and their error reports remain on disk after
failure; do not delete them before investigating the failed field.

Each field has a commit ticket written atomically with its rows. Replaying
that ticket does not append the rows again. Exhausted SQL busy/lock failures
enter a distinct serial retry pass after the primary producers finish. This
pass gives each affected field one additional attempt. Invalid inputs and
cancellation do not enter it. Readers retain their ordinary read-only
connections and the existing filesystem-dependent journal policy.

The queued field path currently applies to the SQLite measurement backend.
Optional DuckDB, Parquet and PostgreSQL mirrors retain their existing worker
path; the RAM control does not govern those external stores.

Simulation's bounded write queue
--------------------------------

Parallel simulations use the same **Database write queue RAM** allowance,
including a ``database_write_queue_gib`` override and disk-only buffering at
zero. Pending packets live beneath the dated output directory in
``.simulation_write_queue``. Workers capture all tables for a simulation;
one writer commits them with an idempotent ticket in the destination
``simulations.db``. A failed calculation does not enqueue partial tables.
An exhausted calculation or database write fails the run and retains its
pending disk evidence. Direct standalone ``run_and_save`` calls keep their
ordinary database-writing path.

Journal mode and network storage
--------------------------------

spaCR enables WAL only where it is known to be safe. WAL lets readers proceed
without blocking a committing writer, but SQLite's WAL index requires
shared-memory coordination and is not safe on many NFS, SMB, NAS, or
distributed filesystems. Before a Measure run,
:func:`spacr.database_concurrency.enable_wal_where_safe` switches
``measurements/measurements.db`` to WAL only when its filesystem is positively
identified as a local type on an allowlist (ext4, XFS, Btrfs, ZFS, APFS, tmpfs
and similar); network, unrecognised and undetectable filesystems keep the
rollback (``DELETE``) journal. Other databases retain their journal mode unless
a caller explicitly requests ``WAL`` or ``DELETE``.

:func:`spacr.database_concurrency.inspect_database` reports the active journal
mode, filesystem type, lock timeout, SQLite threading level, sidecar sizes,
and optional ``PRAGMA quick_check`` result. It emits a warning when it detects
WAL on a known network filesystem. Filesystem detection is advisory: storage
inside a container or automounter can conceal its actual backing system, so
vendor guidance remains authoritative.

Command-line audit
------------------

Inspect an existing database without modifying it::

   spacr-db-audit /data/plate/measurements/measurements.db --quick-check

Run simultaneous readers and writers against a new disposable database::

   spacr-db-audit --probe --writers 4 --readers 3 --writes 100

Use ``--json`` for CI or monitoring. ``--scratch PATH`` is accepted only when
``PATH`` does not exist; the audit deliberately refuses to place probe tables
inside scientific results. Without ``--scratch``, its temporary database is
removed after metrics are collected. The command returns nonzero for corrupt
input, failed integrity checks, thread errors, timeouts, or a row-count
mismatch.

Transaction API
---------------

Plugin and pipeline writers should use the same primitives::

   from spacr.database_concurrency import connect, transaction

   connection = connect("measurements.db", timeout=30)
   try:
       with transaction(connection):
           connection.execute(
               "INSERT INTO audit_event(name, value) VALUES (?, ?)",
               ("complete_field", "plate1_A01_1"),
           )
   finally:
       connection.close()

Connections must never be passed between threads. Do not retry statements from
inside a transaction: earlier statements might already have run. The context
manager retries only transaction acquisition, then either commits the complete
body once or rolls it back.

Stress coverage
---------------

``tests/test_database_concurrency.py`` uses real database files to verify:

* exact row counts under simultaneous reader/writer pressure;
* lock release and bounded lock exhaustion;
* atomic success, rollback, and nested-transaction refusal;
* enforced read-only connections and WAL snapshot visibility;
* concurrent run-ledger stamps with no lost rows;
* integrity/network-storage diagnostics and CLI exit behavior;
* refusal to run a destructive probe against an existing database, and removal
  of a probe's scratch database when it fails.

Annotation-batch rollback and fail-loud status are covered in
``tests/qt/test_annotate.py``, and resume cleanup across every measure-owned
table in ``tests/test_resume.py``. The existing Measure multiprocessing, schema
migration, unreadable run-status, and Qt Database Browser suites provide
integration coverage for their respective production paths.

API reference
-------------

.. automodule:: spacr.database_concurrency
   :noindex:
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: spacr.cli_database
   :noindex:
   :members:
