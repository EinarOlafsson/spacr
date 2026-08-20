"""One reader and one writer for every table spaCR opens or saves.

Why this module exists
----------------------
An audit across ``spacr/`` found 248
tabular reads and writes -- ``pd.read_csv``, ``pd.read_sql*``, ``.to_csv``,
``.to_sql`` -- and **thirteen** call sites that normalised a column name at
all. There was no funnel; there were 248 doors, and which spelling of a key a
frame ended up with depended on which door it came through.

The resulting failure has a consistent shape: ``columnID``
worked as a filter, because *some* path downstream renamed on the way to the
fit, while the CSV picker read the raw header and offered ``column_name`` and
``column`` -- the names a user must not have to know.

So: one place normalises, and every reader goes through it. Doing it at the
funnel is the point. The picker becomes correct for free, because it reads
through the same door and therefore sees ``columnID``.

What a read guarantees
----------------------
Every frame that leaves :func:`read_table` / :func:`read_database` has

* canonical metadata column names (:func:`spacr.schema.canonical_column_name`,
  which folds case *and* punctuation);
* **one** column per metadata key, with the collision reported -- printed
  when the duplicates agreed, warned with a row count when they did not
  (:func:`spacr.schema.resolve_metadata_collisions`);
* the ``pplate1`` plate-value repair applied to every plate-bearing column
  (:func:`spacr.schema.normalise_plate_columns`).

Writing
-------
**Decided, not accidental: spaCR writes canonical names.** ``write_table``
and ``write_database`` canonicalise on the way out by default, so a frame
assembled by hand cannot re-export ``column_name`` and start the cycle again.
The header of an exported file therefore changes for anyone whose downstream
script reads ``column_name`` -- that is a release note, and it is the
deliberate half of this compatibility trade-off.
``canonicalise=False`` is there for a caller who owes an external format an
exact header.

The guards that moved here rather than being left behind
--------------------------------------------------------
* **A ``~`` path is expanded, once, for every reader** -- GitHub issue #108,
  where a ``src`` beginning with ``~`` was resolved against the working
  directory and refused with ``FileNotFoundError: ~<DB>``. ``$HOME`` and
  ``%USERPROFILE%`` too: a settings CSV carried between machines routinely
  holds one.
* **The measurements schema migration runs on open**, exactly as
  ``io._read_db`` does it, so a legacy database is repaired by any reader
  rather than only by the one that remembered.

Dependencies
------------
``pandas``, ``sqlite3`` and :mod:`spacr.schema`. Nothing else at module
scope -- no ``spacr.utils``, no matplotlib, no torch. That is a requirement
rather than tidiness: the CSV picker and the SQL column list have to be able
to import this to get canonical names, and they cannot pay for a torch
import to do it.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from . import schema

__all__ = [
    'TabularFormatError',
    'CSV_SUFFIXES', 'DATABASE_SUFFIXES', 'TABLE_SUFFIXES',
    'resolve_path', 'table_format',
    'read_table', 'write_table',
    'read_database', 'write_database',
    'database_tables', 'table_columns',
]


class TabularFormatError(ValueError):
    """A path whose suffix names no format this module can read."""


#: Delimited text, and the separator each suffix implies. ``None`` lets
#: pandas sniff, which is what a ``.txt`` of unknown provenance needs.
CSV_SUFFIXES: Dict[str, Optional[str]] = {
    '.csv': ',',
    '.tsv': '\t',
    '.tab': '\t',
    '.txt': None,
}

#: SQLite, by every suffix spaCR has ever written one under.
DATABASE_SUFFIXES: Tuple[str, ...] = ('.db', '.sqlite', '.sqlite3', '.db3')

#: Every suffix :func:`read_table` understands.
TABLE_SUFFIXES: Tuple[str, ...] = (
    tuple(CSV_SUFFIXES) + DATABASE_SUFFIXES
    + ('.parquet', '.pq', '.feather', '.xlsx', '.xls', '.xlsm')
)


def resolve_path(path: Any) -> str:
    """Expand ``~`` and environment variables in a path, once, for everyone.

    GitHub issue #108: a ``src`` beginning with ``~`` produced
    ``~/.../measurements.db``, which the migration resolved against the
    working directory and refused with ``FileNotFoundError: ~<DB>``. Fixed at
    the funnel rather than at the ~99 sites that build a measurements path by
    string concatenation.

    :param path: a path, a :class:`os.PathLike`, or anything else (returned
        unchanged, so a caller may pass an open connection through).
    :returns: the expanded path, or the object it was given.
    """
    if isinstance(path, (str, os.PathLike)):
        return os.path.expanduser(os.path.expandvars(os.fspath(path)))
    return path


def table_format(path: Any) -> str:
    """Which reader a path needs: ``'csv'``, ``'sqlite'``, ``'parquet'``,
    ``'feather'`` or ``'excel'``.

    :param path: a path.
    :returns: the format name.
    :raises TabularFormatError: when the suffix names no known format.
    """
    suffix = os.path.splitext(str(resolve_path(path)))[1].lower()
    if suffix in CSV_SUFFIXES:
        return 'csv'
    if suffix in DATABASE_SUFFIXES:
        return 'sqlite'
    if suffix in ('.parquet', '.pq'):
        return 'parquet'
    if suffix == '.feather':
        return 'feather'
    if suffix in ('.xlsx', '.xls', '.xlsm'):
        return 'excel'
    raise TabularFormatError(
        f'{path!r}: {suffix or "no suffix"} is not a table format spaCR '
        f'reads. Known suffixes: {", ".join(sorted(TABLE_SUFFIXES))}.')


def _canonicalise(frame, canonicalise, report, warn, repair_plate_ids=True):
    """Apply the vocabulary, or not, in one place both readers share."""
    if not canonicalise:
        return frame
    return schema.canonicalise_frame(
        frame, report=report, warn=warn, repair_plate_ids=repair_plate_ids)


def read_table(source: Any, *, table: Optional[str] = None,
               canonicalise: bool = True,
               report: Optional[Callable[[str], None]] = print,
               warn: Optional[Callable[[str], None]] = None,
               repair_plate_ids: bool = True,
               **kwargs) -> pd.DataFrame:
    """Read one table, whatever it is stored in, with canonical column names.

    CSV, TSV, SQLite, Parquet, Feather and Excel, chosen by suffix. A SQLite
    path needs ``table``; every other format ignores it.

    :param source: path to the file. ``~`` and ``$VARS`` are expanded.
    :param table: the table name, for a database.
    :param canonicalise: apply the vocabulary. **There is no reason to turn
        this off on the ordinary path** -- it is what makes the picker and
        the run agree about what a column is called. Off is for a caller
        inspecting a file exactly as written.
    :param report: called with each agreeing-collision message; ``print`` by
        default, ``None`` to silence.
    :param warn: called with each disagreeing-collision message; ``None``
        routes to :func:`warnings.warn`.
    :param repair_plate_ids: collapse a doubled ``pp`` plate prefix.
    :param kwargs: passed to the underlying pandas reader.
    :returns: a :class:`pandas.DataFrame`.
    """
    kind = table_format(source)
    path = resolve_path(source)
    if kind == 'sqlite':
        if table is None:
            raise ValueError(
                f'{source!r} is a database; read_table needs table=<name>. '
                f'Tables present: {", ".join(database_tables(source))}.')
        frames = read_database(source, [table], canonicalise=canonicalise,
                               report=report, warn=warn,
                               repair_plate_ids=repair_plate_ids, **kwargs)
        return frames[0]
    if kind == 'csv':
        suffix = os.path.splitext(path)[1].lower()
        separator = CSV_SUFFIXES.get(suffix)
        if separator is not None:
            kwargs.setdefault('sep', separator)
        elif 'sep' not in kwargs:
            kwargs['sep'] = None
            kwargs.setdefault('engine', 'python')
        frame = pd.read_csv(path, **kwargs)
    elif kind == 'parquet':
        frame = pd.read_parquet(path, **kwargs)
    elif kind == 'feather':
        frame = pd.read_feather(path, **kwargs)
    else:
        frame = pd.read_excel(path, **kwargs)
    return _canonicalise(frame, canonicalise, report, warn, repair_plate_ids)


def write_table(frame: pd.DataFrame, path: Any, *,
                canonicalise: bool = True, index: bool = False,
                **kwargs) -> str:
    """Write one frame, format chosen by suffix, with canonical column names.

    See the module docstring for why writing canonical was chosen over
    writing back what was read.

    :param frame: the frame.
    :param path: destination. ``~`` and ``$VARS`` are expanded; the parent
        directory is created.
    :param canonicalise: rename legacy spellings on the way out.
    :param index: pandas' ``index`` argument, defaulted to ``False`` because
        every spaCR export that ever wanted the index has a real column for
        it and an unnamed ``Unnamed: 0`` on re-read is a bug in waiting.
    :param kwargs: passed to the underlying pandas writer.
    :returns: the resolved path written.
    """
    kind = table_format(path)
    target = resolve_path(path)
    parent = os.path.dirname(os.path.abspath(target))
    if parent:
        os.makedirs(parent, exist_ok=True)
    if canonicalise:
        mapping = schema.canonical_rename_plan(frame.columns)
        if mapping:
            frame = frame.rename(columns=mapping)
    if kind == 'csv':
        suffix = os.path.splitext(target)[1].lower()
        separator = CSV_SUFFIXES.get(suffix)
        if separator is not None:
            kwargs.setdefault('sep', separator)
        frame.to_csv(target, index=index, **kwargs)
    elif kind == 'parquet':
        frame.to_parquet(target, index=index, **kwargs)
    elif kind == 'feather':
        frame.reset_index(drop=not index).to_feather(target, **kwargs)
    elif kind == 'excel':
        frame.to_excel(target, index=index, **kwargs)
    else:
        raise TabularFormatError(
            f'{path!r} is a database; use write_database(frame, db, table).')
    return target


def _quote_identifier(name: Any) -> str:
    """Quote a SQLite identifier, refusing anything that is not one."""
    if not isinstance(name, str) or not name:
        raise ValueError(f'Invalid table name: {name!r}')
    return '"' + name.replace('"', '""') + '"'


def _connect(db: Any, *, migrate: bool, read_only: bool = False):
    """Open a database, running the schema migration first if asked.

    ``read_only`` opens through SQLite's ``file:...?mode=ro`` URI. A merge
    reads several of the user's measurement databases at once and must not be
    able to write to any of them, so it opens read-only -- which also means
    it cannot migrate, and ``migrate`` and ``read_only`` are refused
    together rather than one silently winning.
    """
    path = resolve_path(db)
    if read_only and migrate:
        raise ValueError(
            'read_only=True cannot migrate: a migration writes. Pass '
            'migrate=False, or open read/write.')
    if migrate:
        from .database_schema import ensure_database_schema
        ensure_database_schema(path)
    if read_only:
        return sqlite3.connect(f'file:{path}?mode=ro', uri=True, timeout=30)
    return sqlite3.connect(path, timeout=30)


def database_tables(db: Any, *, migrate: bool = False) -> Tuple[str, ...]:
    """The table names in a database, sorted.

    :param db: path to the database. ``~`` and ``$VARS`` are expanded.
    :param migrate: run the schema migration first. ``False``, because
        listing what is there must not rewrite it.
    :returns: the table names.
    """
    with _connect(db, migrate=migrate) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return tuple(sorted(row[0] for row in rows))


def table_columns(source: Any, *, table: Optional[str] = None,
                  canonicalise: bool = True) -> Tuple[str, ...]:
    """The column names a table would be read with, without reading it.

    The CSV button and the SQL column list show what the
    run will see -- ``columnID``, never ``column_name`` -- and they get that
    by asking the reader rather than by growing a second copy of the
    vocabulary.

    A collapsed duplicate is **not** listed twice: a picker that offered both
    ``well`` and ``wellID`` would let a user choose a column the run will not
    find.

    :param source: a CSV/Parquet/Excel path, or a database path with
        ``table``.
    :param table: the table name, for a database.
    :param canonicalise: apply the vocabulary.
    :returns: the column names, in order.
    """
    kind = table_format(source)
    if kind == 'sqlite':
        if table is None:
            raise ValueError(
                f'{source!r} is a database; table_columns needs table=<name>.')
        with _connect(source, migrate=False) as conn:
            frame = pd.read_sql_query(
                f'SELECT * FROM {_quote_identifier(table)} LIMIT 0', conn)
    elif kind == 'csv':
        frame = read_table(source, canonicalise=False, report=None, nrows=0)
    elif kind == 'excel':
        frame = read_table(source, canonicalise=False, report=None, nrows=0)
    else:
        frame = read_table(source, canonicalise=False, report=None).head(0)
    if canonicalise:
        frame = schema.canonicalise_frame(
            frame, report=None, warn=lambda message: None,
            repair_plate_ids=False)
    return tuple(str(name) for name in frame.columns)


def read_database(db: Any, tables: Any, *, canonicalise: bool = True,
                  report: Optional[Callable[[str], None]] = print,
                  warn: Optional[Callable[[str], None]] = None,
                  repair_plate_ids: bool = True,
                  migrate: bool = True,
                  read_only: bool = False,
                  limit: Optional[int] = None,
                  chunksize: int = 100_000,
                  **kwargs) -> List[pd.DataFrame]:
    """Read one or more tables out of a SQLite database.

    Expands ``~``, validates identifiers before opening the database, runs the
    schema migration when requested, and reads in chunks to limit peak memory.
    A missing table raises :class:`ValueError` with its name.

    :param db: path to the database.
    :param tables: a table name, or a sequence of them.
    :param canonicalise: apply the vocabulary to each frame.
    :param report: see :func:`read_table`.
    :param warn: see :func:`read_table`.
    :param repair_plate_ids: collapse a doubled ``pp`` plate prefix.
    :param migrate: run the schema migration on open.
    :param read_only: open through ``file:...?mode=ro``, so the read cannot
        write to the user's database. Incompatible with ``migrate``.
    :param limit: read at most this many rows per table. ``None`` reads all
        of them.
    :param chunksize: rows per chunk.
    :param kwargs: passed to :func:`pandas.read_sql_query`.
    :returns: one frame per requested table, in the order asked for.
    :raises ValueError: when a table is not in the database.
    """
    names = [tables] if isinstance(tables, str) else list(tables)
    for name in names:
        _quote_identifier(name)
    frames: List[pd.DataFrame] = []
    with _connect(db, migrate=migrate, read_only=read_only) as conn:
        present = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        for name in names:
            if name not in present:
                raise ValueError(f'Table not found in database: {name}')
            quoted = _quote_identifier(name)
            query = f'SELECT * FROM {quoted}'
            if limit is not None:
                query += f' LIMIT {int(limit)}'
            chunks = list(pd.read_sql_query(
                query, conn, chunksize=chunksize, **kwargs))
            if not chunks:
                frame = pd.read_sql_query(
                    f'SELECT * FROM {quoted} LIMIT 0', conn, **kwargs)
            elif len(chunks) == 1:
                frame = chunks[0]
            else:
                frame = pd.concat(chunks, ignore_index=True)
            del chunks
            frames.append(_canonicalise(frame, canonicalise, report, warn,
                                        repair_plate_ids))
    return frames


def write_database(frame: pd.DataFrame, db: Any, table: str, *,
                   if_exists: str = 'append', canonicalise: bool = True,
                   index: bool = False, migrate: bool = False,
                   **kwargs) -> str:
    """Write one frame into a SQLite table, with canonical column names.

    :param frame: the frame.
    :param db: path to the database; created if absent, ``~`` expanded.
    :param table: the table name.
    :param if_exists: ``'append'`` (spaCR's usual), ``'replace'``, ``'fail'``.
    :param canonicalise: rename legacy spellings on the way out. On by
        default so a frame assembled by hand cannot put ``column_name`` back
        into a database the reader will then have to repair.
    :param index: write the index. ``False``.
    :param migrate: run the schema migration first. ``False``: writing a
        scratch table must not migrate the user's measurements.
    :param kwargs: passed to :meth:`pandas.DataFrame.to_sql`.
    :returns: the resolved database path.
    """
    _quote_identifier(table)
    target = resolve_path(db)
    parent = os.path.dirname(os.path.abspath(target))
    if parent:
        os.makedirs(parent, exist_ok=True)
    if canonicalise:
        mapping = schema.canonical_rename_plan(frame.columns)
        if mapping:
            frame = frame.rename(columns=mapping)
    with _connect(target, migrate=migrate) as conn:
        frame.to_sql(table, conn, if_exists=if_exists, index=index, **kwargs)
    return target
