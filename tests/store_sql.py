"""Tests read a run's measurements the way spaCR does (unified store, step S4).

Measure and Make Masks write the spaCR store in ``measurements/`` and no
longer write ``measurements.db``. A test that opened the run's database with
``sqlite3.connect`` opens it with :func:`connect` instead: an existing
database file (a fixture, or an older project) is opened by SQLite as
before, and a store project through
:class:`spacr.measurement_store.StoreConnection`, which answers the same SQL.
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


def connect(database, *args, **kwargs):
    """``sqlite3.connect`` for a file, the store's SQL connection for a store."""
    path = _path(database)
    beside = os.path.join(os.path.dirname(os.path.abspath(path)), 'spacr.duckdb')
    if (os.path.basename(path) == 'measurements.db' and os.path.isfile(beside)):
        from spacr.measurement_store import StoreConnection, legacy_database

        if legacy_database(path) is None:
            readonly = 'mode=ro' in os.fspath(database)
            return StoreConnection(path, readonly=readonly)
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
    with sqlite3.connect(path) as conn:
        for name, frame in frames.items():
            frame.to_sql(name, conn, index=False)
    for entry in ('spacr.duckdb', 'spacr.duckdb.wal', 'tables', 'rows',
                  '_pending', 'STORE_VERSION'):
        target = os.path.join(folder, entry)
        if os.path.isdir(target):
            shutil.rmtree(target)
        elif os.path.exists(target):
            os.remove(target)
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
