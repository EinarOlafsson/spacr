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

Measurement writes retain their specialized recovery for concurrent
``CREATE TABLE`` and schema-widening races. Run-status table creation and row
insertion are now one atomic transaction. Database Browser edit validation and
its single-row update also share one transaction, closing the former gap
between checking a row address and writing it. Resume's multi-table
delete-before-remeasure validates and deletes under one retried write
transaction. Annotate preserves the configured journal mode, rolls back a
failed coalesced batch, retains an unsaved/error state, and reports that state
in the module instead of marking a failed commit as saved.

Journal mode and network storage
--------------------------------

spaCR does **not** enable WAL automatically. WAL is useful for concurrent
local readers and writers, but SQLite's WAL index requires shared-memory
coordination and is not safe on many NFS, SMB, NAS, or distributed
filesystems. Existing databases retain their journal mode unless a caller
explicitly requests ``WAL`` or ``DELETE``.

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
* annotation-batch rollback and fail-loud status;
* atomic resume cleanup across every measure-owned table;
* integrity/network-storage diagnostics and CLI exit behavior;
* refusal to run a destructive probe against an existing database.

The existing Measure multiprocessing, schema migration, unreadable run-status,
and Qt Database Browser suites provide integration coverage for their
respective production paths.

API reference
-------------

.. automodule:: spacr.database_concurrency
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: spacr.cli_database
   :members:
