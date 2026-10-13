"""GitHub issue #115: `attempt to write a readonly database`, from a read.

A user's measurements database lived on a read-only SMB mount. Reading it died
with `OperationalError: attempt to write a readonly database` -- an error about
a write, raised by a function whose whole job is to read.

Nothing about the read needed to write. `_read_db` called
`ensure_database_schema` first, which renames legacy columns and stamps the
schema version, and that is a write. On a database that was already current it
was a no-op and the read succeeded, which is why the failure looked
intermittent and source-dependent: it only bit when the read-only database ALSO
carried a pre-migration schema.

Migration is not what makes a legacy table readable. `correct_metadata` already
canonicalises the frame on the way out -- `plate`/`row`/`col` become
`plateID`/`rowID`/`columnID` -- so the migration exists to make that permanent,
not to make it possible. So it is skipped when the database cannot be written,
and the file is opened read-only so SQLite does not try to place a journal
beside it either.

These tests pin both halves: a read-only legacy database reads and comes back
canonicalised, and a writable one is still migrated in place.
"""

import os
import sqlite3
import stat

import pytest


LEGACY_ROWS = [("p1", "A", "1", 12.5), ("p1", "A", "2", 30.0)]


def _legacy_database(path):
    """A pre-migration table: legacy column names, no schema version stamp."""
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE cell (plate TEXT, row TEXT, col TEXT, cell_area REAL)")
    connection.executemany("INSERT INTO cell VALUES (?, ?, ?, ?)", LEGACY_ROWS)
    connection.commit()
    connection.close()


@pytest.fixture
def read_only_plate(tmp_path):
    """A legacy database in a directory nobody can write to.

    BOTH are made read-only, and the directory matters as much as the file:
    SQLite writes its journal into the directory, so a writable file in a
    read-only directory is not a writable database. An SMB share mounted
    read-only presents exactly this.
    """
    folder = tmp_path / "share"
    folder.mkdir()
    database = folder / "measurements.db"
    _legacy_database(database)
    os.chmod(database, stat.S_IRUSR)
    os.chmod(folder, stat.S_IRUSR | stat.S_IXUSR)
    try:
        yield database
    finally:
        # Restore write permission or pytest cannot clean the tmp tree up.
        os.chmod(folder, stat.S_IRWXU)
        os.chmod(database, stat.S_IRUSR | stat.S_IWUSR)


def test_a_readonly_legacy_database_reads_instead_of_raising(read_only_plate):
    """The reported failure, as the reporter met it."""
    from spacr.io import _read_db

    frames = _read_db(str(read_only_plate), tables=["cell"])

    assert len(frames) == 1
    assert len(frames[0]) == len(LEGACY_ROWS)


def test_the_frame_comes_back_canonicalised_without_a_migration(read_only_plate):
    """Legacy names are fixed on the way out, which is why no write is needed.

    This is the assertion that says the fix is a fix rather than a way of
    returning something. A caller asking for `plateID` gets it from a database
    whose column is still called `plate`.
    """
    from spacr.io import _read_db

    frame = _read_db(str(read_only_plate), tables=["cell"])[0]

    assert {"plateID", "rowID", "columnID"} <= set(frame.columns)
    assert "cell_area" in frame.columns
    assert sorted(frame["columnID"].astype(str)) == ["1", "2"]


def test_the_database_is_not_modified_by_reading_it(read_only_plate):
    """No journal, no stamp, nothing new beside it."""
    from spacr.io import _read_db

    before = read_only_plate.stat().st_mtime_ns
    siblings_before = sorted(p.name for p in read_only_plate.parent.iterdir())

    _read_db(str(read_only_plate), tables=["cell"])

    assert read_only_plate.stat().st_mtime_ns == before
    assert sorted(p.name for p in read_only_plate.parent.iterdir()) == siblings_before


def test_a_writable_database_is_still_migrated(tmp_path):
    """The skip must be conditional, or nothing would ever be migrated.

    Reading a writable legacy database still stamps it, so the next reader
    finds a current schema rather than being canonicalised again forever.
    """
    from spacr.io import _read_db

    database = tmp_path / "measurements.db"
    _legacy_database(database)

    _read_db(str(database), tables=["cell"])

    connection = sqlite3.connect(database)
    try:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(cell)")
        }
    finally:
        connection.close()
    assert "plateID" in columns, (
        "a writable database was not migrated; the read-only skip is "
        "unconditional and no database would ever be brought forward")
