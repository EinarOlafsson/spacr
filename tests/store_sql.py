"""Tests read a run's measurements the way spaCR does (unified store, step S4).

Measure and Make Masks write the spaCR store in ``measurements/`` and no
longer write ``measurements.db``. A test that opened the run's database with
``sqlite3.connect`` opens it with :func:`connect` instead: an existing
database file (a fixture, or an older project) is opened by SQLite as
before, and a store project through
:class:`spacr.measurement_store._StoreConnection`, which answers the same SQL.
"""
from __future__ import annotations

import os
import sqlite3
from urllib.parse import unquote


def _path(database):
    """The filesystem path of a ``sqlite3.connect`` argument."""
    text = os.fspath(database)
    if text.startswith('file:'):
        text = unquote(text[5:].split('?', 1)[0])
        if text.startswith('//'):
            text = text[2:]
    return text


def measured(database) -> bool:
    """Whether a run wrote measurements there: a database file or a store."""
    from spacr.measurement_store import exists

    return exists(_path(database))


def databases(root) -> list:
    """The ``measurements.db`` locators under ``root`` that hold measurements.

    A store project has no ``measurements.db`` file, only ``measurements/``
    with its catalog, so a ``rglob('measurements.db')`` finds nothing there;
    this lists the file or the store locator of every measured project.
    """
    from pathlib import Path

    found = set(Path(root).rglob('measurements.db'))
    for catalog in Path(root).rglob('spacr.duckdb'):
        if catalog.parent.name == 'measurements':
            found.add(catalog.parent / 'measurements.db')
    return sorted(p for p in found if measured(p))


def connect(database, *args, **kwargs):
    """``sqlite3.connect`` for a file, the store's SQL connection for a store."""
    path = _path(database)
    beside = os.path.join(os.path.dirname(os.path.abspath(path)), 'spacr.duckdb')
    if (os.path.basename(path) == 'measurements.db' and os.path.isfile(beside)):
        from spacr.measurement_store import _StoreConnection, legacy_database

        if legacy_database(path) is None:
            readonly = 'mode=ro' in os.fspath(database)
            return _StoreConnection(path, readonly=readonly)
    return sqlite3.connect(database, *args, **kwargs)


def to_legacy(database) -> str:
    """Turn a store project into an older project with a measurements.db.

    For tests that edit a run's rows to build a legacy history (a missing
    settings row, a changed rescale record): the store's object tables are
    read-only, so the run is written to SQLite with SQLite's own types and
    the store's catalog and parts are removed. The project then reads and
    resumes as an older project does.

    :param database: the project's ``measurements.db`` path (absent).
    :returns: the database path.
    """
    import shutil

    from spacr import measurement_store as ms

    path = _path(database)
    if os.path.isfile(path) and ms.legacy_database(path):
        return path
    folder = os.path.dirname(os.path.abspath(path))
    names = ms.tables(path)
    frames = {name: ms.read(path, name) for name in names}
    for entry in ('spacr.duckdb', 'spacr.duckdb.wal', 'tables', 'rows',
                  '_pending', 'STORE_VERSION'):
        target = os.path.join(folder, entry)
        if os.path.isdir(target):
            shutil.rmtree(target)
        elif os.path.exists(target):
            os.remove(target)
    ms.forget_sessions()
    # Written the way an older Measure wrote them: through spaCR's SQLite
    # append (schema migration, widening) and its packet ticket table.
    from spacr.database_concurrency import _WRITE_TICKETS_TABLE
    from spacr.utils import _append_to_measurements_db
    with sqlite3.connect(path) as conn:
        conn.execute(f'CREATE TABLE IF NOT EXISTS {_WRITE_TICKETS_TABLE} '
                     '(ticket TEXT PRIMARY KEY, field TEXT NOT NULL)')
    for name, frame in frames.items():
        if len(frame):
            _append_to_measurements_db(path, name, frame)
        keys = [k for k in ms.UPSERT_KEYS.get(name, ()) if k in frame.columns]
        if keys:
            with sqlite3.connect(path) as conn:
                conn.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS "{name}_key" '
                             f'ON "{name}" (' + ', '.join(f'"{k}"' for k in keys)
                             + ')')
    ms.forget_sessions()
    return path


def archived_rows(database, table):
    """The rows an earlier population kept for ``table`` (unified store, S4).

    A fresh Measure run moves an older project's measurements.db to
    ``measurements/archive/measurements-<stamp>.db`` (where SQLite renamed
    its tables to ``<table>_1`` before); the newest archive is read.
    """
    import glob

    folder = os.path.dirname(os.path.abspath(_path(database)))
    files = sorted(glob.glob(os.path.join(folder, 'archive', 'measurements-*.db')))
    if not files:
        raise FileNotFoundError(f'no archived measurements.db in {folder}')
    with sqlite3.connect(f'file:{files[-1]}?mode=ro', uri=True) as conn:
        return conn.execute(f'SELECT * FROM "{table}"').fetchall()


def fingerprint(database) -> str:
    """A digest of a project's measurements, to check that nothing changed.

    The bytes of an existing database file; for a store project, the
    catalog and every table part.
    """
    import hashlib

    path = _path(database)
    digest = hashlib.sha256()
    if os.path.isfile(path):
        with open(path, 'rb') as handle:
            digest.update(handle.read())
        return digest.hexdigest()
    folder = os.path.dirname(os.path.abspath(path))
    for root in ('spacr.duckdb', 'tables'):
        target = os.path.join(folder, root)
        if os.path.isfile(target):
            files = [target]
        else:
            files = sorted(os.path.join(d, f) for d, _s, fs in os.walk(target)
                           for f in fs)
        for name in files:
            digest.update(os.path.relpath(name, folder).encode())
            with open(name, 'rb') as handle:
                digest.update(handle.read())
    return digest.hexdigest()
