"""SQL over an OPS run's tables in its spaCR store, for tests (item 0an).

The OPS tables used to be in ``<run>/measurements.db`` and tests read them
with ``sqlite3``. They are in the run's spaCR store now
(:mod:`spacr.ops_store`), so :func:`ops_connect` opens it through the
store's one reader, :func:`spacr.store.open`, and names each OPS table's
view as the OPS table (``ops_barcodes`` for the store's ``barcodes``), with
``plate`` and ``well`` first as the SQLite tables had them, so a test's
``SELECT ... FROM ops_barcodes`` reads the same rows. The connection it
yields answers ``execute(...)`` (iterable, ``fetchone``, ``fetchall``) as
the sqlite3 one did, and :func:`read_sql` stands in for
``pandas.read_sql_query``.
"""
from __future__ import annotations

from contextlib import contextmanager

import pandas as pd


class _Result:
    """Fetched rows, read like a sqlite3 cursor."""

    def __init__(self, rows, description):
        self.rows = list(rows)
        self.description = description

    def __iter__(self):
        return iter(self.rows)

    def fetchall(self):
        return list(self.rows)

    def fetchone(self):
        return self.rows[0] if self.rows else None


class _Connection:
    """A DuckDB connection answering like the sqlite3 one tests used."""

    def __init__(self, conn):
        self.conn = conn

    def execute(self, sql, params=()):
        relation = self.conn.execute(sql, list(params or ()))
        return _Result(relation.fetchall(), relation.description)

    def df(self, sql, params=()) -> pd.DataFrame:
        return self.conn.execute(sql, list(params or ())).df()


@contextmanager
def ops_connect(db):
    """A connection whose views are the run's OPS tables.

    :param db: the run's OPS database path (``<run>/measurements.db``).
    """
    from spacr import store
    from spacr.ops_store import OPS_TABLES, _store_project

    opened = store.open(_store_project(str(db)))
    try:
        names = set(opened.tables())
        for table in OPS_TABLES:
            name = table[4:]
            if name not in names:
                continue
            columns = [row[0] for row in opened._conn.execute(
                f'DESCRIBE SELECT * FROM "{name}"').fetchall()]
            front = [c for c in ("plate", "well") if c in columns]
            order = ", ".join(f'"{c}"' for c in front + [
                c for c in columns if c not in front])
            opened._conn.execute(f'CREATE VIEW "{table}" AS SELECT {order} '
                                 f'FROM "{name}"')
        yield _Connection(opened._conn)
    finally:
        opened.close()


def read_sql(sql: str, connection: _Connection, params=None) -> pd.DataFrame:
    """``pandas.read_sql_query`` on an :func:`ops_connect` connection."""
    return connection.df(sql, params)


def ops_sql(db, sql: str, params=None) -> pd.DataFrame:
    """The rows of ``sql`` over the run's OPS tables (see :func:`ops_connect`)."""
    with ops_connect(db) as conn:
        return conn.df(sql, params)
