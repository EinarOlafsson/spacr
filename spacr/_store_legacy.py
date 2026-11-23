"""Older measurement formats, read for the spaCR store.

Private to :mod:`spacr.store`. :func:`spacr.store.open` reads three formats
older than the store through this module, so that every reader of spaCR
data goes through the store and none needs its own code for them:

* ``measurements.db``, or any SQLite file: whole tables are read with
  SQLite's own value types, as ``pandas.read_sql_query`` gives them, and
  DuckDB views are made only when SQL is run against them;
* a ``.duckdb`` measurement copy, attached read-only;
* a ``.parquetdb`` folder (the measurement copy that
  ``measurement_backend: parquet`` writes, one subfolder of Parquet parts
  per table, with the active parts in ``.spacr-parts.json``).

The SQLite connection helper is also the one :mod:`spacr.tabular` writes
through, so a read inside an atomic write sees that write's rows.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator, List, Optional, Tuple

#: Suffixes of a DuckDB measurement copy.
_DUCKDB_SUFFIXES: Tuple[str, ...] = (".duckdb", ".ddb")

#: Suffix of a Parquet-folder measurement copy.
_PARQUET_SUFFIX = ".parquetdb"


def kind(path: Any) -> str:
    """Which older format a path holds: ``'parquet'``, ``'duckdb'`` or
    ``'sqlite'``.

    :param path: a file or folder path.
    :returns: the format name; anything that is neither a ``.parquetdb``
        folder nor a ``.duckdb``/``.ddb`` file is SQLite.
    """
    from .store import CATALOG

    text = os.fspath(path).rstrip("/\\")
    suffix = os.path.splitext(text)[1].lower()
    if suffix == _PARQUET_SUFFIX:
        return "parquet"
    if (suffix in _DUCKDB_SUFFIXES and os.path.basename(text) != CATALOG
            and not os.path.isdir(text)):
        return "duckdb"
    return "sqlite"


def is_sqlite(store) -> bool:
    """Whether an open :class:`spacr.store.Store` reads an older SQLite file.

    :param store: the store.
    :returns: True for a SQLite file, False for a store project or another
        older format.
    """
    return store.legacy is not None and kind(store.legacy) == "sqlite"


def sqlite_connection(path: Any, *, read_only: bool = True):
    """Open a SQLite file the way spaCR reads and writes measurements.

    Inside an atomic write packet for the same file, the packet's own
    connection is returned, so the caller sees its uncommitted rows.

    :param path: the SQLite file.
    :param read_only: open through ``file:...?mode=ro``.
    :returns: a :class:`sqlite3.Connection`.
    """
    from .database_concurrency import _inside_write_packet, connect

    if read_only:
        return connect(path, readonly=True)
    if _inside_write_packet(path):
        return connect(path)
    return sqlite3.connect(path, timeout=30)


@contextmanager
def _reading(path: str):
    """A read-only connection, closed afterwards unless a write packet owns it."""
    from .database_concurrency import _inside_write_packet

    conn = sqlite_connection(path, read_only=True)
    try:
        yield conn
    finally:
        if not _inside_write_packet(path):
            conn.close()


def _quoted(name: str) -> str:
    """A SQLite identifier in double quotes."""
    return '"' + str(name).replace('"', '""') + '"'


def sqlite_tables(path: str) -> List[str]:
    """The table names of a SQLite file, without SQLite's own tables.

    :param path: the SQLite file.
    :returns: the names, sorted.
    """
    with _reading(path) as conn:
        return [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name "
            "NOT LIKE 'sqlite_%' ORDER BY name").fetchall()]


def sqlite_columns(path: str, table: str) -> List[Tuple[str, str]]:
    """The columns of one SQLite table, with their declared types.

    :param path: the SQLite file.
    :param table: the table.
    :returns: ``(name, declared type)`` pairs in stored order.
    """
    with _reading(path) as conn:
        return [(str(row[1]), str(row[2] or "")) for row in conn.execute(
            f"PRAGMA table_info({_quoted(table)})").fetchall()]


def sqlite_frames(path: str, table: str, *, limit: Optional[int] = None,
                  chunksize: int = 100_000, **kwargs) -> Iterator[Any]:
    """One SQLite table in pandas frames, with SQLite's own value types.

    :param path: the SQLite file.
    :param table: the table.
    :param limit: read at most this many rows; None reads every row.
    :param chunksize: rows per frame.
    :param kwargs: passed to :func:`pandas.read_sql_query`.
    :returns: an iterator yielding at least one frame.
    """
    import pandas as pd

    quoted = _quoted(table)
    query = f"SELECT * FROM {quoted}"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    with _reading(path) as conn:
        yielded = False
        for chunk in pd.read_sql_query(query, conn, chunksize=chunksize,
                                       **kwargs):
            yielded = True
            yield chunk
        if not yielded:
            yield pd.read_sql_query(f"SELECT * FROM {quoted} LIMIT 0", conn,
                                    **kwargs)


def parquet_tables(folder: str) -> List[str]:
    """The tables of a ``.parquetdb`` folder that hold at least one part.

    A folder is read as an escaped table name only when it is exactly the
    Windows spelling of a name Windows forbids.

    :param folder: the ``.parquetdb`` folder.
    :returns: the names, sorted.
    """
    from .tabular import (_parquet_parts, _parquet_table_name,
                          _parquet_verbatim_folder)

    if not os.path.isdir(folder):
        return []
    names = set()
    for entry in os.listdir(folder):
        table = _parquet_table_name(entry)
        if (table != entry and not _parquet_verbatim_folder(folder, table)
                and _parquet_parts(folder, table)):
            names.add(table)
        elif _parquet_parts(folder, entry):
            names.add(entry)
    return sorted(names)


def attach(store) -> bool:
    """Give an open store the views of a ``.duckdb`` file or a ``.parquetdb``
    folder.

    :param store: the :class:`spacr.store.Store` being refreshed.
    :returns: True when ``store.project`` is one of those formats (attached,
        or absent and so empty), False for SQLite, which the store attaches
        itself.
    """
    from .store import _connect, _ident, _literal

    path = store.project
    found = kind(path)
    if found == "sqlite":
        return False
    store.legacy = path
    names: List[str] = []
    if found == "parquet":
        from .tabular import _parquet_parts

        names = parquet_tables(path)
        store._conn = _connect(threads=store.threads)
        for name in names:
            files = ", ".join(_literal(part)
                              for part in _parquet_parts(path, name))
            store._conn.execute(
                f"CREATE VIEW {_ident(name)} AS SELECT * FROM "
                f"read_parquet([{files}], union_by_name = true)")
    else:
        store._conn = _connect(threads=store.threads)
    if found == "duckdb" and os.path.isfile(path):
        store._conn.execute(f"ATTACH {_literal(path)} AS legacy (READ_ONLY)")
        names = sorted(row[0] for row in store._conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_catalog = 'legacy' AND table_schema = 'main' "
            "AND table_type = 'BASE TABLE'").fetchall())
        for name in names:
            store._conn.execute(
                f"CREATE VIEW {_ident(name)} AS SELECT * FROM "
                f"legacy.{_ident(name)}")
    store._tables = {name: {"sort_by": [], "key_columns": []}
                     for name in names}
    return True
