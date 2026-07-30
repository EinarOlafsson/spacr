"""Version and migration contracts for spaCR SQLite databases."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from spacr.database_schema import (
    CURRENT_SCHEMA_VERSION,
    SPACR_APPLICATION_ID,
    DatabaseMigrationError,
    DatabaseSchemaTooNewError,
    Migration,
    database_schema_version,
    migrate_connection,
    migrate_database,
)


def _pragma(path, name):
    with sqlite3.connect(path) as connection:
        return connection.execute(f"PRAGMA {name}").fetchone()[0]


def _columns(path, table):
    with sqlite3.connect(path) as connection:
        return [
            row[1]
            for row in connection.execute(
                f'PRAGMA table_info("{table}")'
            ).fetchall()
        ]


def test_database_creation_sets_version_and_application_id(tmp_path):
    from spacr.io import _create_database

    path = tmp_path / "measurements.db"
    _create_database(path)

    assert database_schema_version(path) == CURRENT_SCHEMA_VERSION
    assert _pragma(path, "application_id") == SPACR_APPLICATION_ID


def test_legacy_database_follows_ordered_migration_and_preserves_rows(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            'CREATE TABLE "png_list" '
            '("plate_name" TEXT, "time_id" TEXT, "value" REAL)'
        )
        connection.execute(
            'INSERT INTO "png_list" VALUES ("plate1", "t3", 4.5)'
        )

    report = migrate_database(path)

    assert report.from_version == 0
    assert report.to_version == CURRENT_SCHEMA_VERSION
    assert report.applied == ("canonicalize measurement column names",)
    assert set(report.column_renames) == {
        ("png_list", "plate_name", "plateID"),
        ("png_list", "time_id", "timeID"),
    }
    assert _columns(path, "png_list") == ["plateID", "timeID", "value"]
    with sqlite3.connect(path) as connection:
        row = connection.execute('SELECT * FROM "png_list"').fetchone()
    assert row == ("plate1", "t3", 4.5)


def test_current_database_open_is_idempotent(tmp_path):
    path = tmp_path / "current.db"
    sqlite3.connect(path).close()

    first = migrate_database(path)
    second = migrate_database(path)

    assert first.changed
    assert not second.changed
    assert second.applied == ()
    assert second.column_renames == ()
    assert database_schema_version(path) == CURRENT_SCHEMA_VERSION


def test_migration_never_overwrites_a_conflicting_canonical_column(tmp_path):
    path = tmp_path / "both.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            'CREATE TABLE "cell" ("time_id" TEXT, "timeID" TEXT)'
        )
        connection.execute(
            'INSERT INTO "cell" VALUES ("legacy", "canonical")'
        )

    report = migrate_database(path)

    assert report.column_renames == ()
    assert _columns(path, "cell") == ["time_id", "timeID"]
    with sqlite3.connect(path) as connection:
        assert connection.execute('SELECT * FROM "cell"').fetchone() == (
            "legacy",
            "canonical",
        )


def test_newer_database_is_rejected_without_mutation(tmp_path):
    path = tmp_path / "future.db"
    future = CURRENT_SCHEMA_VERSION + 4
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE "cell" ("time_id" TEXT)')
        connection.execute(f"PRAGMA user_version = {future}")

    with pytest.raises(
        DatabaseSchemaTooNewError,
        match=rf"schema {future}.*supports up to {CURRENT_SCHEMA_VERSION}.*Upgrade",
    ):
        migrate_database(path)

    assert database_schema_version(path) == future
    assert _columns(path, "cell") == ["time_id"]


def test_failed_migration_rolls_back_ddl_and_version(tmp_path):
    path = tmp_path / "atomic.db"
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE "cell" ("legacy" TEXT)')

    def fail_after_ddl(connection):
        connection.execute(
            'ALTER TABLE "cell" RENAME COLUMN "legacy" TO "changed"'
        )
        raise RuntimeError("interrupted migration")

    migrations = (Migration(1, "deliberate failure", fail_after_ddl),)
    with pytest.raises(RuntimeError, match="interrupted migration"):
        migrate_database(path, migrations=migrations)

    assert database_schema_version(path) == 0
    assert _columns(path, "cell") == ["legacy"]
    assert _pragma(path, "application_id") == 0


def test_migration_uses_savepoint_inside_caller_transaction(tmp_path):
    path = tmp_path / "savepoint.db"
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE "cell" ("time_id" TEXT)')
        connection.execute('INSERT INTO "cell" VALUES ("t1")')
        report = migrate_connection(connection, path=str(path))
        assert connection.in_transaction
        connection.commit()

    assert report.changed
    assert _columns(path, "cell") == ["timeID"]
    assert database_schema_version(path) == CURRENT_SCHEMA_VERSION


def test_gapped_migration_registry_is_rejected_before_changes(tmp_path):
    path = tmp_path / "gap.db"
    sqlite3.connect(path).close()

    def noop(_connection):
        return ()

    with pytest.raises(DatabaseMigrationError, match="contiguous"):
        migrate_database(path, migrations=(Migration(2, "gap", noop),))

    assert database_schema_version(path) == 0


def test_measurement_writer_versions_a_new_database(tmp_path):
    from spacr.utils import _append_to_measurements_db

    path = tmp_path / "measurements.db"
    frame = pd.DataFrame({"object_label": [1], "cell_area": [4.0]})
    _append_to_measurements_db(str(path), "cell", frame)

    assert database_schema_version(path) == CURRENT_SCHEMA_VERSION
    with sqlite3.connect(path) as connection:
        assert connection.execute('SELECT COUNT(*) FROM "cell"').fetchone() == (
            1,
        )


def test_primary_reader_migrates_legacy_database_before_query(tmp_path):
    from spacr.io import _read_db

    path = tmp_path / "measurements.db"
    with sqlite3.connect(path) as connection:
        pd.DataFrame(
            {"plate_name": ["plate1"], "cell_area": [2.0]}
        ).to_sql("cell", connection, index=False)

    frames = _read_db(str(path), ["cell"])

    assert frames[0]["plateID"].tolist() == ["plate1"]
    assert database_schema_version(path) == CURRENT_SCHEMA_VERSION
