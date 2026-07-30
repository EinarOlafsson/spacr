"""Versioned SQLite schema migrations for spaCR measurement databases.

SQLite's ``PRAGMA user_version`` is the on-disk schema version.  Migrations
are registered here as a contiguous, ordered sequence and are applied in one
transaction.  A database created by an older spaCR release therefore follows
the same path whether it is opened for reading or for writing, while a
database created by a newer release is rejected before any mutation.

The module deliberately uses only the Python standard library.  Measurement
workers can import it without importing pandas, plotting, or optional analysis
dependencies.
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "DB_COLUMN_RENAMES",
    "DB_COLUMN_RENAME_PATTERNS",
    "DatabaseMigrationError",
    "DatabaseSchemaTooNewError",
    "Migration",
    "MigrationReport",
    "SPACR_APPLICATION_ID",
    "canonical_column_name",
    "database_schema_version",
    "ensure_database_schema",
    "migrate_connection",
    "migrate_database",
    "repair_legacy_columns",
]


# ``SPCR`` as a four-byte big-endian integer.  SQLite reserves
# ``application_id`` for applications to identify their file format.
SPACR_APPLICATION_ID = int.from_bytes(b"SPCR", "big")

# Version 1 establishes canonical metadata/feature column spellings.
CURRENT_SCHEMA_VERSION = 1


class DatabaseMigrationError(RuntimeError):
    """A measurements database could not be migrated safely."""


class DatabaseSchemaTooNewError(DatabaseMigrationError):
    """The database was written by a newer, unsupported spaCR schema."""


ColumnRename = Tuple[str, str, str]
MigrationFunction = Callable[[sqlite3.Connection], Sequence[ColumnRename]]


@dataclass(frozen=True)
class Migration:
    """One ordered database schema transition.

    ``version`` is the schema version after ``apply`` succeeds.  Consequently
    a migration numbered ``3`` upgrades version ``2`` to version ``3``.
    """

    version: int
    name: str
    apply: MigrationFunction


@dataclass(frozen=True)
class MigrationReport:
    """Result of bringing one database to a requested schema version."""

    path: Optional[str]
    from_version: int
    to_version: int
    applied: Tuple[str, ...]
    column_renames: Tuple[ColumnRename, ...]

    @property
    def changed(self) -> bool:
        """Whether at least one schema transition was applied."""

        return (
            self.from_version != self.to_version
            or bool(self.column_renames)
        )


# Legacy column spellings and their canonical spaCR names.  These constants
# live with the migration that consumes them; ``spacr.utils`` re-exports them
# for compatibility with existing callers.
DB_COLUMN_RENAMES = {
    "row": "rowID",
    "row_name": "rowID",
    "column": "columnID",
    "col": "columnID",
    "column_name": "columnID",
    "plate": "plateID",
    "plate_name": "plateID",
    "field": "fieldID",
    "field_name": "fieldID",
    "channel": "chanID",
    "time_id": "timeID",
}

DB_COLUMN_RENAME_PATTERNS = (
    (
        re.compile(
            r"^(?P<head>.*?)(?P<ring>periphery|outside)_"
            r"(?P<p>\d+)_percentile$"
        ),
        r"\g<head>\g<ring>_percentile_\g<p>",
    ),
    (
        re.compile(
            r"^organelle_summary_organelle_ch"
            r"(?P<c>\d+)_(?P<rest>.+)$"
        ),
        r"organelle_summary_organelle_channel_\g<c>_\g<rest>",
    ),
)


def canonical_column_name(name: str) -> str:
    """Return the canonical spaCR spelling for one database column."""

    renamed = DB_COLUMN_RENAMES.get(name)
    if renamed is not None:
        return renamed
    for pattern, replacement in DB_COLUMN_RENAME_PATTERNS:
        new_name, substitutions = pattern.subn(replacement, name)
        if substitutions:
            return new_name
    return name


def _quote_identifier(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise DatabaseMigrationError(f"invalid SQLite identifier: {name!r}")
    return '"' + name.replace('"', '""') + '"'


def _rename_legacy_columns(
    connection: sqlite3.Connection,
) -> Tuple[ColumnRename, ...]:
    """Apply the version-1 non-destructive column canonicalisation."""

    renamed = []
    cursor = connection.cursor()
    try:
        cursor.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        for table in tables:
            quoted_table = _quote_identifier(table)
            cursor.execute(f"PRAGMA table_info({quoted_table})")
            columns = [row[1] for row in cursor.fetchall()]
            for old in list(columns):
                new = canonical_column_name(old)
                if new == old or new in columns:
                    continue
                cursor.execute(
                    f"ALTER TABLE {quoted_table} "
                    f"RENAME COLUMN {_quote_identifier(old)} "
                    f"TO {_quote_identifier(new)}"
                )
                columns[columns.index(old)] = new
                renamed.append((table, old, new))
    finally:
        cursor.close()
    return tuple(renamed)


MIGRATIONS: Tuple[Migration, ...] = (
    Migration(
        version=1,
        name="canonicalize measurement column names",
        apply=_rename_legacy_columns,
    ),
)


def _validated_migrations(
    migrations: Sequence[Migration],
    target_version: int,
) -> Tuple[Migration, ...]:
    ordered = tuple(sorted(migrations, key=lambda item: item.version))
    versions = tuple(item.version for item in ordered)
    expected = tuple(range(1, target_version + 1))
    if versions[:target_version] != expected:
        raise DatabaseMigrationError(
            "database migrations must be contiguous from version 1 through "
            f"{target_version}; registered versions are {versions}"
        )
    return ordered


def _pragma_int(connection: sqlite3.Connection, pragma: str) -> int:
    row = connection.execute(f"PRAGMA {pragma}").fetchone()
    return int(row[0]) if row else 0


def database_schema_version(source) -> int:
    """Return ``source``'s SQLite ``user_version``.

    ``source`` may be an open :class:`sqlite3.Connection` or a path.  A path
    must already exist; inspecting a typo must not create an empty database.
    """

    if isinstance(source, sqlite3.Connection) or hasattr(source, "execute"):
        return _pragma_int(source, "user_version")
    path = os.fspath(source)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        return _pragma_int(connection, "user_version")


def _begin_migration(connection: sqlite3.Connection) -> Tuple[str, bool]:
    """Start a transaction or a savepoint when the caller already has one."""

    if connection.in_transaction:
        name = "spacr_schema_migration"
        connection.execute(f"SAVEPOINT {name}")
        return name, True
    connection.execute("BEGIN IMMEDIATE")
    return "", False


def _commit_migration(
    connection: sqlite3.Connection,
    transaction: Tuple[str, bool],
) -> None:
    name, is_savepoint = transaction
    if is_savepoint:
        connection.execute(f"RELEASE SAVEPOINT {name}")
    else:
        connection.execute("COMMIT")


def _rollback_migration(
    connection: sqlite3.Connection,
    transaction: Tuple[str, bool],
) -> None:
    name, is_savepoint = transaction
    if is_savepoint:
        connection.execute(f"ROLLBACK TO SAVEPOINT {name}")
        connection.execute(f"RELEASE SAVEPOINT {name}")
    else:
        connection.execute("ROLLBACK")


def migrate_connection(
    connection: sqlite3.Connection,
    *,
    target_version: int = CURRENT_SCHEMA_VERSION,
    migrations: Sequence[Migration] = MIGRATIONS,
    path: Optional[str] = None,
) -> MigrationReport:
    """Migrate an open SQLite connection atomically.

    A schema newer than this spaCR installation is rejected with an actionable
    error.  Every selected migration and the final ``user_version`` update
    share one transaction, so an exception leaves both schema and version
    unchanged.
    """

    current = database_schema_version(connection)
    if current > CURRENT_SCHEMA_VERSION:
        raise DatabaseSchemaTooNewError(
            f"{path or 'database'} uses spaCR database schema {current}, but "
            f"this installation supports up to {CURRENT_SCHEMA_VERSION}. "
            "Upgrade spaCR before opening this database; do not downgrade the "
            "database file."
        )
    if target_version < current:
        raise DatabaseMigrationError(
            f"database schema downgrades are not supported: "
            f"{current} -> {target_version}"
        )
    if target_version > CURRENT_SCHEMA_VERSION:
        raise DatabaseMigrationError(
            f"target schema {target_version} exceeds this spaCR installation's "
            f"schema {CURRENT_SCHEMA_VERSION}"
        )

    ordered = _validated_migrations(migrations, target_version)
    selected = tuple(
        item for item in ordered
        if current < item.version <= target_version
    )
    if not selected:
        return MigrationReport(
            path=path,
            from_version=current,
            to_version=current,
            applied=(),
            column_renames=(),
        )

    transaction = _begin_migration(connection)
    applied = []
    column_renames = []
    try:
        for migration in selected:
            changes = migration.apply(connection)
            if changes:
                column_renames.extend(changes)
            connection.execute(
                f"PRAGMA user_version = {int(migration.version)}"
            )
            applied.append(migration.name)
        connection.execute(
            f"PRAGMA application_id = {SPACR_APPLICATION_ID}"
        )
        _commit_migration(connection, transaction)
    except BaseException:
        _rollback_migration(connection, transaction)
        raise

    return MigrationReport(
        path=path,
        from_version=current,
        to_version=target_version,
        applied=tuple(applied),
        column_renames=tuple(column_renames),
    )


def migrate_database(
    db_path,
    *,
    target_version: int = CURRENT_SCHEMA_VERSION,
    migrations: Sequence[Migration] = MIGRATIONS,
    timeout: float = 30.0,
) -> MigrationReport:
    """Migrate an existing SQLite database path and close it on every path."""

    path = os.path.abspath(os.fspath(db_path))
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    connection = sqlite3.connect(path, timeout=timeout)
    try:
        return migrate_connection(
            connection,
            target_version=target_version,
            migrations=migrations,
            path=path,
        )
    finally:
        connection.close()


def repair_legacy_columns(db_path, *, timeout: float = 30.0):
    """Re-run the non-destructive column repair without changing the version.

    This compatibility operation remains useful for a manually edited
    database that already declares the current version.  Normal opens should
    use :func:`migrate_database`, which runs each migration only once.
    """

    path = os.path.abspath(os.fspath(db_path))
    connection = sqlite3.connect(path, timeout=timeout)
    transaction = _begin_migration(connection)
    try:
        renamed = _rename_legacy_columns(connection)
        _commit_migration(connection, transaction)
        return renamed
    except BaseException:
        _rollback_migration(connection, transaction)
        raise
    finally:
        connection.close()


def ensure_database_schema(
    db_path,
    *,
    target_version: int = CURRENT_SCHEMA_VERSION,
    timeout: float = 30.0,
) -> MigrationReport:
    """Migrate a database and repair schema drift at the current version.

    Old spaCR readers performed the non-destructive column repair on every
    open.  Retaining that small safety net matters for databases manually
    edited after migration, while ordinary legacy databases still follow the
    explicit one-time migration path.
    """

    report = migrate_database(
        db_path,
        target_version=target_version,
        timeout=timeout,
    )
    if report.from_version != report.to_version:
        return report
    repaired = tuple(repair_legacy_columns(db_path, timeout=timeout))
    if not repaired:
        return report
    return MigrationReport(
        path=report.path,
        from_version=report.from_version,
        to_version=report.to_version,
        applied=report.applied,
        column_renames=repaired,
    )
