"""Refusals and rollback paths of the measurement-database migrator.

A migration that half-runs is worse than one that never started: the file on
disk then matches neither schema version and no reader can trust it. These
tests cover the four ways the migrator declines to act and the one way it
undoes work it had already begun inside somebody else's transaction.
"""
from __future__ import annotations

import sqlite3

import pytest

from spacr import database_schema as dbs


def _database(path):
    with sqlite3.connect(str(path)) as db:
        db.execute("CREATE TABLE cell (plateID TEXT, area REAL)")
    return str(path)


# ---------------------------------------------------------------------------
# Identifier quoting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["", None, 7])
def test_an_unusable_identifier_is_refused_rather_than_quoted(name):
    """Quoting must refuse anything that is not a real SQLite identifier.

    The quoted result is interpolated straight into ``ALTER TABLE``. An empty
    or non-string name would produce a statement that either fails with an
    opaque syntax error mid-migration or, worse, names a different object.
    """
    with pytest.raises(dbs.DatabaseMigrationError) as excinfo:
        dbs._quote_identifier(name)
    assert "invalid SQLite identifier" in str(excinfo.value)


def test_an_embedded_quote_is_doubled_not_dropped():
    """The refusal above must not have replaced the escaping it guards."""
    assert dbs._quote_identifier('we"ird') == '"we""ird"'


# ---------------------------------------------------------------------------
# Reading a version
# ---------------------------------------------------------------------------

def test_asking_a_missing_file_for_its_version_does_not_create_one(tmp_path):
    """Inspecting a typo must raise, not leave an empty database behind.

    SQLite creates a file on connect, so a version query written naively turns
    every mistyped path into a brand new, empty, version-0 database that then
    looks like a legitimate project.
    """
    missing = tmp_path / "not_here.db"

    with pytest.raises(FileNotFoundError):
        dbs.database_schema_version(str(missing))
    assert not missing.exists()


# ---------------------------------------------------------------------------
# Version bounds
# ---------------------------------------------------------------------------

def test_a_downgrade_is_refused(tmp_path):
    """Migrating down would have to drop data the newer schema added.

    No migration in the registry knows how to reverse itself, so a requested
    downgrade can only be refused; running the upgrade steps in reverse order
    would silently produce a third schema that is neither version.
    """
    path = _database(tmp_path / "m.db")
    dbs.migrate_database(path)

    with sqlite3.connect(path) as connection:
        assert dbs.database_schema_version(connection) == 1
        with pytest.raises(dbs.DatabaseMigrationError) as excinfo:
            dbs.migrate_connection(connection, target_version=0)
    assert "downgrades are not supported" in str(excinfo.value)


def test_a_target_beyond_this_installation_is_refused(tmp_path):
    """A target this spaCR has no migration for cannot be reached.

    Accepting it would stamp a ``user_version`` the file's contents do not
    match, and every later spaCR would then refuse the database as too new.
    """
    path = _database(tmp_path / "m.db")

    with sqlite3.connect(path) as connection:
        with pytest.raises(dbs.DatabaseMigrationError) as excinfo:
            dbs.migrate_connection(
                connection, target_version=dbs.CURRENT_SCHEMA_VERSION + 1)
    assert "exceeds this spaCR installation" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Rollback inside a caller's transaction
# ---------------------------------------------------------------------------

def test_a_failed_migration_inside_a_caller_transaction_undoes_only_itself(
        tmp_path):
    """Nested migration work is discarded without touching the caller's.

    When the caller is already in a transaction the migration nests in a
    savepoint. A failure has to roll back to that savepoint and release it, so
    the caller's own uncommitted rows survive and the connection is left
    usable; rolling the whole transaction back instead would silently destroy
    work the migrator never owned.
    """
    path = _database(tmp_path / "m.db")
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.execute("BEGIN")
        connection.execute("INSERT INTO cell VALUES ('plate1', 1.0)")

        def _explode(_connection):
            raise RuntimeError("migration failed halfway")

        failing = (dbs.Migration(version=1, name="boom", apply=_explode),)
        with pytest.raises(RuntimeError):
            dbs.migrate_connection(connection, target_version=1,
                                   migrations=failing)

        assert connection.in_transaction, "the caller's transaction survives"
        rows = connection.execute("SELECT COUNT(*) FROM cell").fetchone()[0]
        assert rows == 1, "the caller's own insert was not rolled back"
        assert dbs.database_schema_version(connection) == 0
        connection.execute("COMMIT")
    finally:
        connection.close()
