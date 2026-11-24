"""The Database Browser's reader: one handle over a spaCR store or an older file.

:class:`StoreDb` is what the Browser (``spacr.qt.screens.db_browser``) pages,
counts, filters, sorts and exports through. It reads only through
:mod:`spacr.store`, so the GUI never opens SQLite or DuckDB itself:

* a **store project** (``spacr.duckdb`` and its Parquet tables) is paged with
  :meth:`spacr.store.Store.page` by keyset over ``plate``, ``well`` and the
  table's sort keys, so page 1,000 costs what page 1 does; only the columns
  asked for are read; the unfiltered row count comes from the registry; a
  sort or a filter runs in DuckDB, on a twin of the shared store
  (:func:`spacr.store.fork`) that :meth:`StoreDb.cancel` can interrupt;
* an **older measurements.db** is read through
  :func:`spacr.store.legacy_connection` exactly as the Browser always read
  it: ``mode=ro``, keyset on ``rowid``, SQLite's own dialect;
* an older DuckDB file or Parquet folder is paged by offset.

Every handle is read-only. Edits of a store project are written as edit
rows through ``spacr.store.edit`` once that exists; until then, and always
for an older file, :meth:`StoreDb.can_edit` is False.
"""
from __future__ import annotations

import collections
import getpass
import math
import os
import re
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import store as _store

__all__ = ["READ_ERRORS", "StoreDb", "bind_parameters"]


def _read_errors() -> tuple:
    """The exception classes a read through the store can raise."""
    found: List[type] = [_store.StoreError, OSError]
    import sqlite3

    found.append(sqlite3.Error)
    try:
        import duckdb

        found.append(duckdb.Error)
    except ImportError:
        pass
    return tuple(found)


#: What a Browser read can raise: the store's own error, SQLite's, DuckDB's
#: and the operating system's. Catch this tuple rather than one engine's.
READ_ERRORS: tuple = _read_errors()

#: Rows of the page cache, over every table and filter of one handle.
_PAGE_CACHE = 64

#: Above this many columns a store table opens with the keys and the first
#: :data:`_DEFAULT_FEATURES` features, so a page never reads 2,800 columns.
_WIDE_TABLE = 60

#: Features shown first in a wide store table.
_DEFAULT_FEATURES = 30

#: The most columns one page of a store table reads.
_FETCH_COLUMNS_MAX = 400

#: Functions a raw filter may not call on a store project: each reads a file
#: or the environment, which a filter has no reason to.
_FORBIDDEN_CALLS = re.compile(
    r"\b(read_\w+|glob|getenv|sniff_csv|parquet_\w+|query_table|query|"
    r"sqlite_scan|sqlite_attach|duckdb_\w+|pragma_\w+)\s*\(",
    re.IGNORECASE)

#: A structured filter's LIKE, which SQLite matches without case.
_LIKE = "LIKE ? ESCAPE '\\'"


def _literal(value: Any) -> str:
    """A Python value as a DuckDB SQL literal."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(int(value))
    if isinstance(value, float):
        if math.isnan(value):
            return "'NaN'::DOUBLE"
        if math.isinf(value):
            return "'Infinity'::DOUBLE" if value > 0 else "'-Infinity'::DOUBLE"
        return repr(float(value))
    return "'" + str(value).replace("'", "''") + "'"


def bind_parameters(where: Optional[str], params: Sequence[Any] = ()
                    ) -> Optional[str]:
    """A filter with its ``?`` parameters written in as DuckDB literals.

    Each ``?`` outside a quoted string or identifier takes the next value,
    quoted (a text value's quotes are doubled), so a value can never end
    the literal it sits in. A structured ``LIKE ? ESCAPE '\\'`` becomes
    ``ILIKE``, because SQLite's LIKE ignores case and the Browser's
    "contains" has always meant that.

    :param where: the filter without ``WHERE``, or None.
    :param params: one value per ``?``.
    :returns: the filter, or None.
    :raises ValueError: when the counts of ``?`` and values differ.
    """
    if not where:
        return None
    values = list(params or ())
    if not values:
        if "?" in _unquoted(where):
            raise ValueError("the filter has a ? but no value for it")
        return where
    text = where.replace(_LIKE, "I" + _LIKE)
    out: List[str] = []
    quote: Optional[str] = None
    used = 0
    for char in text:
        if quote is not None:
            out.append(char)
            if char == quote:
                quote = None
            continue
        if char in ("'", '"'):
            quote = char
            out.append(char)
        elif char == "?":
            if used >= len(values):
                raise ValueError("the filter has more ? than values")
            out.append(_literal(values[used]))
            used += 1
        else:
            out.append(char)
    if used != len(values):
        raise ValueError("the filter has fewer ? than values")
    return "".join(out)


def _unquoted(text: str) -> str:
    """``text`` with its quoted strings and identifiers removed."""
    return re.sub(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"", "", text)


def _quote(name: str) -> str:
    """A SQL identifier in double quotes."""
    return '"' + str(name).replace('"', '""') + '"'


def _plain(value: Any) -> Any:
    """A cell as the Browser shows it: NaN and NA as None, NumPy as Python."""
    if value is None:
        return None
    item = getattr(value, "item", None)
    if item is not None and not isinstance(value, (str, bytes)):
        try:
            value = item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        import pandas as pd

        if value is pd.NA or value is pd.NaT:
            return None
    except ImportError:
        pass
    return value


class StoreDb:
    """A read-only Browser handle on a spaCR store or an older database.

    It answers what the Database Browser asks: the tables, their columns
    and types, how a row is addressed, a page of rows after a key, an
    estimate and an exact count, whether a filter parses, and streamed
    exports. Every read goes through :mod:`spacr.store`.

    A store project's pages come from :meth:`spacr.store.Store.page` (keyset)
    and are kept in a small LRU cache per table, filter, sort and column
    set; an older SQLite file is paged on ``rowid`` in SQLite itself.

    :param path: a project folder, ``spacr.duckdb``, a run folder, or an
        older ``measurements.db``; see :func:`spacr.store.locate`.
    :raises FileNotFoundError: when nothing there opens.
    :raises spacr.store.StoreError: when the project does not open.
    """

    def __init__(self, path: str):
        """Open ``path`` through the shared store of its project.

        :param path: what :func:`spacr.store.locate` accepts.
        """
        self.path = _store.locate(path)
        self.store = _store.shared(self.path)
        if not self.store.legacy:
            self.kind = "store"
        elif _store._legacy_is_sqlite(self.store):
            self.kind = "sqlite"
        else:
            self.kind = "duckdb"
        #: A short description of the most recent read, for tests and reports.
        self.last_sql: str = ""
        self._tables: Optional[List[str]] = None
        self._row_keys: Dict[str, Tuple[str, List[str]]] = {}
        self._types: Dict[str, Dict[str, str]] = {}
        self._columns: Dict[str, List[str]] = {}
        self._pages: "collections.OrderedDict[tuple, tuple]" = (
            collections.OrderedDict())
        self._cursors: Dict[tuple, Any] = {}
        self._active: set = set()
        self._lock = threading.Lock()
        if self.kind == "sqlite":
            with _store.legacy_connection(self.store) as con:
                con.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()

    def is_store(self) -> bool:
        """Whether this handle reads a spaCR store project (not an older file).

        :returns: True for a store project.
        """
        return self.kind == "store"

    def sync(self) -> bool:
        """Pick up a project that changed on disk (for example, a run that
        registered more wells).

        Call it on the thread that owns the shared store, before a reload.

        :returns: True when the store was opened again and the cached pages
            and schema were dropped.
        """
        current = _store.shared(self.path)
        if current is self.store:
            return False
        self.store = current
        self._tables = None
        self._row_keys.clear()
        self._types.clear()
        self._columns.clear()
        with self._lock:
            self._pages.clear()
            self._cursors.clear()
        return True

    def _twin(self):
        """A twin of the shared store for the calling thread, held as active."""
        twin = _store.fork(self.store)
        with self._lock:
            self._active.add(twin)
        return twin

    def _done(self, twin) -> None:
        """Close a twin and stop holding it."""
        with self._lock:
            self._active.discard(twin)
        twin.close()

    def cancel(self) -> int:
        """Interrupt every DuckDB read this handle is running.

        Each interrupted read raises in its own worker, whose result the
        Browser drops anyway. An older SQLite file's reads are short and
        are not interrupted.

        :returns: how many reads were interrupted.
        """
        with self._lock:
            running = list(self._active)
        for twin in running:
            _store.interrupt(twin)
        return len(running)

    def tables(self, refresh: bool = False) -> List[str]:
        """Return the tables (and, for an older file, the views) alphabetically.

        :param refresh: read the list again rather than from the cache.
        :returns: the names.
        """
        if self._tables is None or refresh:
            if refresh:
                self.sync()
            if self.kind == "sqlite":
                with _store.legacy_connection(self.store) as con:
                    rows = self._execute(con,
                        "SELECT name FROM sqlite_master "
                        "WHERE type IN ('table', 'view') "
                        "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
                self._tables = [r[0] for r in rows]
            else:
                self._tables = list(self.store.tables())
        return list(self._tables)

    def check_table(self, table: str) -> str:
        """Return ``table`` when the project has it, else raise.

        :param table: a table name.
        :returns: ``table``.
        :raises ValueError: when the project has no such table.
        """
        if table not in self.tables():
            raise ValueError(
                f"No table named {table!r} in {os.path.basename(self.path)}. "
                f"Reload the database and choose a table from its table list.")
        return table

    def table_info(self, table: str) -> List[tuple]:
        """Rows shaped like SQLite's ``PRAGMA table_info``.

        ``(cid, name, type, notnull, default, pk)``; a store table reports
        its stored types and no primary key.

        :param table: a table name.
        :returns: one row per column.
        """
        self.check_table(table)
        if self.kind == "sqlite":
            with _store.legacy_connection(self.store) as con:
                return self._execute(
                    con, f"PRAGMA table_info({_quote(table)})").fetchall()
        frame = self.store.columns(table)
        return [(i, str(name), str(kind or ""), 0, None, 0)
                for i, (name, kind) in enumerate(zip(frame["name"],
                                                     frame["stored_type"]))]

    def columns(self, table: str) -> List[str]:
        """Return the column names of ``table`` in stored order.

        :param table: a table name.
        :returns: the names.
        """
        if table not in self._columns:
            self._columns[table] = [r[1] for r in self.table_info(table)]
        return list(self._columns[table])

    def column_types(self, table: str) -> Dict[str, str]:
        """Return ``{column: declared or stored type}``.

        :param table: a table name.
        :returns: ``''`` for an untyped column of an older file.
        """
        if table not in self._types:
            self._types[table] = {r[1]: str(r[2] or "")
                                  for r in self.table_info(table)}
        return dict(self._types[table])

    def column_families(self, table: str) -> Dict[str, str]:
        """Return ``{column: feature family}`` as the store's catalog records it.

        :param table: a table name.
        :returns: an empty mapping for an older file.
        """
        if self.kind != "store":
            return {}
        frame = self.store.columns(table)
        return {str(n): str(f) for n, f in zip(frame["name"], frame["family"])
                if f is not None}

    def check_columns(self, table: str,
                      columns: Optional[Sequence[str]]) -> List[str]:
        """Return the requested columns, checked against the table.

        :param table: a table name.
        :param columns: the names, or None or empty for every column.
        :returns: the names in the requested order.
        :raises ValueError: when a name is not a column of the table.
        """
        real = self.columns(table)
        if not columns:
            return real
        known = set(real)
        unknown = [c for c in columns if c not in known]
        if unknown:
            raise ValueError(
                f"Unknown column(s) for {table!r}: "
                f"{', '.join(map(repr, unknown))}. Reload the table and "
                f"select columns shown in its column list.")
        return list(dict.fromkeys(columns))

    def key_columns(self, table: str) -> List[str]:
        """The columns that identify a row of a store table.

        :param table: a table name.
        :returns: ``plate``, ``well`` and the table's sort keys, as present;
            empty for an older file.
        """
        if self.kind != "store":
            return []
        present = set(self.columns(table))
        info = self.store._tables.get(table, {})
        sort = [c for c in (info.get("sort_by") or []) if c in present] or [
            c for c in ("object_key",) if c in present]
        return [c for c in ("plate", "well", *sort) if c in present]

    def default_columns(self, table: str) -> Optional[List[str]]:
        """The columns a wide store table opens with, or None for all of them.

        :param table: a table name.
        :returns: the keys and the first features of a store table wider
            than 60 columns; None otherwise.
        """
        names = self.columns(table)
        if self.kind != "store" or len(names) <= _WIDE_TABLE:
            return None
        info = self.store._tables.get(table, {})
        keys = [c for c in ["plate", "well",
                            *(info.get("key_columns") or [])] if c in names]
        rest = [c for c in names if c not in keys]
        return keys + rest[:_DEFAULT_FEATURES]

    def fetch_columns(self, table: str, text: str) -> Optional[List[str]]:
        """The columns a store table reads for a column search.

        :param table: a table name.
        :param text: the search, matched without case; empty restores
            :meth:`default_columns`.
        :returns: the row keys and the matching columns (at most 400), or
            None when every column is read.
        """
        if self.kind != "store":
            return None
        if not (text or "").strip():
            return self.default_columns(table)
        names = self.columns(table)
        needle = text.strip().lower()
        keys = self.key_columns(table)
        hits = [c for c in names if needle in c.lower() and c not in keys]
        if len(names) <= _WIDE_TABLE:
            return None
        return keys + hits[:_FETCH_COLUMNS_MAX]

    def row_key(self, table: str) -> Tuple[str, List[str]]:
        """Return how one row of ``table`` is addressed.

        ``("keyset", [plate, well, sort keys])`` for a store table,
        ``("rowid", [alias])`` for an ordinary table of an older file,
        ``("pk", [...])`` for a ``WITHOUT ROWID`` one, and ``("", [])`` for
        anything else (a view, an older DuckDB file).

        :param table: a table name.
        :returns: the kind and the columns.
        """
        if table in self._row_keys:
            return self._row_keys[table]
        self.check_table(table)
        key: Tuple[str, List[str]] = ("", [])
        if self.kind == "store":
            keys = self.key_columns(table)
            key = ("keyset", keys) if len(keys) > 2 else ("", [])
        elif self.kind == "sqlite":
            key = self._legacy_row_key(table)
        self._row_keys[table] = key
        return key

    def _legacy_row_key(self, table: str) -> Tuple[str, List[str]]:
        """The row address of one table of an older SQLite file."""
        with _store.legacy_connection(self.store) as con:
            found = self._execute(
                con, "SELECT type FROM sqlite_master WHERE name = ? LIMIT 1",
                (table,)).fetchone()
        if found is None or str(found[0]).lower() != "table":
            return ("", [])
        from .predictions import _rowid_alias

        info = self.table_info(table)
        alias = _rowid_alias([str(r[1]) for r in info])
        try:
            with _store.legacy_connection(self.store) as con:
                self._execute(con, f"SELECT {alias} FROM {_quote(table)} "
                               f"LIMIT 1").fetchall()
            return ("rowid", [alias])
        except READ_ERRORS:
            pk = sorted((r for r in info if int(r[5] or 0) > 0),
                        key=lambda r: int(r[5]))
            return ("pk", [r[1] for r in pk]) if pk else ("", [])

    def chunk_sql(self, table: str, columns: Sequence[str],
                  key_columns: Sequence[str], where: Optional[str] = None,
                  after: bool = False, use_offset: bool = False,
                  order_by: Optional[Tuple[str, bool]] = None) -> str:
        """The paged SELECT an older file is read with. Exposed for tests.

        Keyset on the row key (``key > ?``) unless the table has no single
        key or a sort is active, which page by ``OFFSET``; the key column
        is the sort's tiebreak, so a row is never shown twice.

        :param table: a table name.
        :param columns: the columns after the key columns.
        :param key_columns: the row-address columns.
        :param where: a checked filter without ``WHERE``.
        :param after: whether this is not the first chunk.
        :param use_offset: page by ``OFFSET``.
        :param order_by: ``(column, descending)`` or None.
        :returns: the statement, with ``?`` for the key, limit and offset.
        """
        col_sql = ", ".join(_quote(c) for c in list(key_columns) + list(columns))
        sql = f"SELECT {col_sql} FROM {_quote(table)}"
        clauses = []
        if where:
            clauses.append(f"({where})")
        if after and not use_offset and order_by is None:
            clauses.append(f"{_quote(key_columns[0])} > ?")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        if order_by is not None:
            column, descending = order_by
            terms = [f"{_quote(column)} {'DESC' if descending else 'ASC'}"]
            if key_columns:
                terms.append(f"{_quote(key_columns[0])} ASC")
            sql += " ORDER BY " + ", ".join(terms)
        elif key_columns and not use_offset:
            sql += f" ORDER BY {_quote(key_columns[0])}"
        sql += " LIMIT ?"
        if use_offset or order_by is not None:
            sql += " OFFSET ?"
        return sql

    def chunk(self, table: str, columns: Optional[Sequence[str]] = None,
              where: Optional[str] = None, params: Sequence = (),
              limit: int = 100, after: Optional[tuple] = None,
              loaded: int = 0,
              order_by: Optional[Tuple[str, bool]] = None
              ) -> Tuple[List[str], List[tuple], List[Optional[tuple]]]:
        """Return ``(columns, rows, keys)`` for the next ``limit`` rows.

        ``after`` is the key of the last row already loaded, None for the
        first chunk; ``keys[i]`` addresses ``rows[i]`` (None when the
        table has no key). A store table reads only ``columns``.

        :param table: a table name.
        :param columns: the columns, or None or empty for every column.
        :param where: a checked filter without ``WHERE``.
        :param params: its ``?`` values.
        :param limit: the most rows.
        :param after: the last loaded row's key, or None.
        :param loaded: rows loaded so far (the offset of an unkeyed table).
        :param order_by: ``(column, descending)`` over the whole table.
        :returns: the columns, the rows and their keys.
        """
        self.check_table(table)
        cols = self.check_columns(table, columns)
        if order_by is not None:
            self.check_columns(table, [order_by[0]])
        limit = int(max(1, limit))
        if self.kind == "store" and self.row_key(table)[0] == "keyset":
            return self._store_chunk(table, cols, where, params, limit, after,
                                     order_by)
        if self.kind != "sqlite":
            return self._offset_chunk(table, cols, where, params, limit,
                                      loaded, order_by)
        _kind, key_cols = self.row_key(table)
        use_offset = len(key_cols) != 1
        sql = self.chunk_sql(table, cols, key_cols, where,
                             after=after is not None, use_offset=use_offset,
                             order_by=order_by)
        args: List[Any] = list(params)
        if after is not None and not use_offset and order_by is None:
            args.append(after[0])
        args.append(limit)
        if use_offset or order_by is not None:
            args.append(int(max(0, loaded)))
        with _store.legacy_connection(self.store) as con:
            raw = self._execute(con, sql, args).fetchall()
        n = len(key_cols)
        keys: List[Optional[tuple]] = ([tuple(r[:n]) for r in raw] if n
                                       else [None] * len(raw))
        return cols, [tuple(r[n:]) for r in raw], keys

    def _store_chunk(self, table, cols, where, params, limit, after, order_by):
        """One keyset page of a store table, through the page cache."""
        bound = bind_parameters(where, params)
        order = (None if order_by is None else
                 (order_by[0], "desc" if order_by[1] else "asc"))
        place = (table, bound, order, after)
        cache = (place, tuple(cols), limit, id(self.store))
        with self._lock:
            hit = self._pages.get(cache)
            if hit is not None:
                self._pages.move_to_end(cache)
        if hit is not None:
            self.last_sql = f"cache: {table} page after {after!r}"
            return hit
        cursor = None
        if after is not None:
            with self._lock:
                cursor = self._cursors.get(place, _MISSING)
            if cursor is _MISSING:
                cursor = self._rebuilt_cursor(after, order_by)
            if cursor is _END:
                result = (list(cols), [], [])
                return result
        keys = self.key_columns(table)
        twin = self._twin()
        try:
            frame, following = twin.page(table, columns=list(cols),
                                         where=bound, order_by=order,
                                         after=cursor, limit=limit)
        finally:
            self._done(twin)
        self.last_sql = (f"Store.page({table!r}, where={bound!r}, "
                         f"order_by={order!r}, limit={limit})")
        rows = [tuple(_plain(v) for v in row) for row in
                frame[list(cols)].itertuples(index=False, name=None)]
        row_keys: List[Optional[tuple]] = [
            tuple(_plain(v) for v in row) for row in
            frame[keys].itertuples(index=False, name=None)] if keys else [
            None] * len(rows)
        result = (list(cols), rows, row_keys)
        with self._lock:
            if row_keys and row_keys[-1] is not None:
                self._cursors[(table, bound, order, row_keys[-1])] = (
                    _END if following is None else following)
            self._pages[cache] = result
            while len(self._pages) > _PAGE_CACHE:
                self._pages.popitem(last=False)
        return result

    def _rebuilt_cursor(self, after: tuple, order_by):
        """A page cursor from a row key, when the cache no longer holds it."""
        if order_by is not None:
            raise _store.StoreError(
                "the sorted page before this one is no longer cached; "
                "reload the table")
        return {"plate": after[0], "well": after[1], "after": list(after[2:])}

    def _offset_chunk(self, table, cols, where, params, limit, loaded,
                      order_by):
        """One page of an older DuckDB file or Parquet folder, by offset."""
        sql = (f"SELECT {', '.join(_quote(c) for c in cols)} FROM "
               f"{_quote(table)}")
        bound = bind_parameters(where, params)
        if bound:
            sql += f" WHERE {bound}"
        if order_by is not None:
            sql += (f" ORDER BY {_quote(order_by[0])} "
                    f"{'DESC' if order_by[1] else 'ASC'}")
        sql += f" LIMIT {int(limit)} OFFSET {int(max(0, loaded))}"
        twin = self._twin()
        try:
            frame = twin.sql(sql)
        finally:
            self._done(twin)
        self.last_sql = sql
        rows = [tuple(_plain(v) for v in row)
                for row in frame.itertuples(index=False, name=None)]
        return list(cols), rows, [None] * len(rows)

    def registry_count(self, table: str) -> Optional[int]:
        """The exact row count when it costs no scan, else None.

        A store table's count is the sum of its registered files' rows.

        :param table: a table name.
        :returns: the count, or None for an older file.
        """
        if self.kind != "store":
            return None
        self.check_table(table)
        return int(self.store.count(table))

    def estimate_count(self, table: str) -> Optional[int]:
        """An O(1) row count: exact for a store table, ``max(rowid)`` otherwise.

        Label an older file's number as an estimate; deleted rows leave
        gaps in ``rowid``.

        :param table: a table name.
        :returns: the count or estimate, or None when there is none.
        """
        self.check_table(table)
        if self.kind == "store":
            return self.registry_count(table)
        key_kind, key_cols = self.row_key(table)
        if key_kind != "rowid":
            return None
        with _store.legacy_connection(self.store) as con:
            row = self._execute(
                con, f"SELECT max({_quote(key_cols[0])}) FROM "
                     f"{_quote(table)}").fetchone()
        if not row or row[0] is None:
            return None
        return int(row[0])

    def validate_where(self, table: str, where: str,
                       params: Sequence = ()) -> None:
        """Let the reader parse ``where`` and raise if it is malformed.

        Nothing is scanned: SQLite prepares it with ``LIMIT 0``, DuckDB
        plans it with ``EXPLAIN``. A store project also refuses a filter
        that calls a function reading files.

        :param table: a table name.
        :param where: the filter without ``WHERE``.
        :param params: its ``?`` values.
        :raises ValueError: for a refused function.
        """
        self.check_table(table)
        if self.kind == "sqlite":
            sql = f"SELECT 1 FROM {_quote(table)} WHERE {where} LIMIT 0"
            with _store.legacy_connection(self.store) as con:
                self._execute(con, sql, params).fetchall()
            return
        found = _FORBIDDEN_CALLS.search(_unquoted(where))
        if found:
            raise ValueError(
                f"{found.group(1)}() is not allowed in a filter; a filter "
                f"compares columns. Remove it, or export the table and read "
                f"the file you need separately.")
        bound = bind_parameters(where, params)
        twin = self._twin()
        try:
            self.last_sql = f"EXPLAIN SELECT 1 FROM {_quote(table)} WHERE {bound}"
            twin.sql(self.last_sql)
        finally:
            self._done(twin)

    def count(self, table: str, where: Optional[str] = None,
              params: Sequence = ()) -> int:
        """Return the exact number of rows under the filter.

        Without a filter a store table answers from its registry; with one,
        DuckDB counts (on the narrow key table when the filter names only
        key columns). An older file runs ``COUNT(*)``.

        :param table: a table name.
        :param where: a checked filter without ``WHERE``, or None.
        :param params: its ``?`` values.
        :returns: the count.
        """
        self.check_table(table)
        if self.kind == "sqlite":
            sql = f"SELECT COUNT(*) FROM {_quote(table)}"
            if where:
                sql += f" WHERE {where}"
            with _store.legacy_connection(self.store) as con:
                row = self._execute(con, sql, params).fetchone()
            return int(row[0]) if row else 0
        bound = bind_parameters(where, params)
        twin = self._twin()
        try:
            self.last_sql = f"Store.count({table!r}, where={bound!r})"
            return int(twin.count(table, bound))
        finally:
            self._done(twin)

    def distinct(self, table: str, column: str, limit: int = 1000) -> List[Any]:
        """The distinct values of a column, for a filter's value list.

        :param table: a table name.
        :param column: a column of it.
        :param limit: the most values.
        :returns: the values, sorted, without nulls.
        """
        self.check_table(table)
        self.check_columns(table, [column])
        if self.kind == "sqlite":
            return _store.distinct(self.store, table, column, limit=limit)
        twin = self._twin()
        try:
            return _store.distinct(twin, table, column, limit=limit)
        finally:
            self._done(twin)

    def export(self, out_path: str, table: str,
               columns: Optional[Sequence[str]] = None,
               where: Optional[str] = None, params: Sequence = (),
               fmt: str = "csv",
               order_by: Optional[Tuple[str, bool]] = None) -> int:
        """Stream the filtered rows to a CSV, Parquet or SQLite file.

        Through :func:`spacr.store.export`, so memory stays bounded; a
        SQLite export of a table with 2,000 columns or more is split by
        feature family.

        :param out_path: the file to write.
        :param table: a table name.
        :param columns: the columns, or None for every column.
        :param where: a checked filter without ``WHERE``.
        :param params: its ``?`` values.
        :param fmt: ``"csv"``, ``"parquet"`` or ``"sqlite"``.
        :param order_by: ``(column, descending)`` or None.
        :returns: the number of rows written.
        """
        self.check_table(table)
        cols = self.check_columns(table, columns)
        order = (None if order_by is None else
                 (order_by[0], "desc" if order_by[1] else "asc"))
        if self.kind == "sqlite":
            source, held, args = self.store, where, list(params)
        else:
            source = _store.fork(self.store)
            held, args = bind_parameters(where, params), []
        try:
            self.last_sql = f"export {fmt} {table!r} where={held!r}"
            done = _store.export(source, out_path, fmt=fmt, tables=table,
                                 where=held, params=args, features=cols,
                                 order_by=order)
        finally:
            if source is not self.store:
                source.close()
        return int(done["rows"])

    def export_csv(self, out_path: str, table: str,
                   columns: Optional[Sequence[str]] = None,
                   where: Optional[str] = None, params: Sequence = (),
                   chunk: int = 5000) -> int:
        """Stream the filtered rows to ``out_path`` as CSV.

        :param out_path: the CSV file; missing folders are made.
        :param table: a table name.
        :param columns: the columns, or None for every column.
        :param where: a checked filter without ``WHERE``.
        :param params: its ``?`` values.
        :param chunk: kept for callers of the older Browser; the store
            streams in its own batches.
        :returns: the number of data rows written.
        """
        return self.export(out_path, table, columns=columns, where=where,
                           params=params, fmt="csv")

    def can_edit(self) -> bool:
        """Whether a cell edit can be written for this project.

        Only a store project, and only once ``spacr.store.edit`` (the
        append-only edit log) exists; an older file is never written.

        :returns: True when :meth:`edit_cell` can write.
        """
        return self.kind == "store" and callable(getattr(_store, "edit", None))

    def edit_cell(self, table: str, column: str, value: Any,
                  key: Sequence[Any]) -> int:
        """Record one cell edit as an edit row (``source='browser'``).

        Nothing in a Parquet part changes: the edit is appended to the
        project's edit log through ``spacr.store.edit``, and the resolved
        views show it.

        :param table: a store table.
        :param column: the edited column.
        :param value: the new value (None clears it).
        :param key: the row's key, as :meth:`chunk` returned it.
        :returns: how many edit rows were written.
        :raises spacr.store.StoreError: when this project cannot take edits.
        """
        if not self.can_edit():
            raise _store.StoreError(
                "this project is read-only in the Database Browser: an older "
                "measurements.db is never written, and a store project takes "
                "edits once spacr.store.edit exists")
        names = self.key_columns(table)
        if len(names) != len(key):
            raise _store.StoreError("that row's key is incomplete; reload")
        row = dict(zip(names, key))
        row.update({"table": table, "column": column, "value": value,
                    "op": "clear" if value is None else "set"})
        try:
            author = getpass.getuser()
        except (KeyError, OSError):
            author = "unknown"
        return int(_store.edit(self.path, [row], kind="browser",
                               author=author, source="browser"))

    def _execute(self, con, sql: str, params: Sequence = ()):
        """Run one statement on an older file's connection, remembering it."""
        self.last_sql = sql
        return con.execute(sql, tuple(params))


#: The page cache's mark for "no cursor recorded for this key".
_MISSING = object()

#: The page cache's mark for "the page before was the last one".
_END = object()
