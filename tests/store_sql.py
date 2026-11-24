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
