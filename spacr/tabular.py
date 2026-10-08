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
* A ``~`` path is expanded, once, for every reader -- GitHub issue #108,
  where a ``src`` beginning with ``~`` was resolved against the working
  directory and refused with ``FileNotFoundError: ~<DB>``. ``$HOME`` and
  ``%USERPROFILE%`` too: a settings CSV carried between machines routinely
  holds one.
* **The measurements schema migration runs on open**, exactly as
  ``io._read_db`` does it, so a legacy database is repaired by any reader
  rather than only by the one that remembered.

Dependencies
------------
``pandas``, ``sqlite3`` and :mod:`spacr.schema`. The alternative
measurement stores import ``duckdb`` or ``psycopg`` only when a DuckDB,
Parquet or PostgreSQL store is opened. Nothing else at module
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


def _fetched(source: Any) -> Any:
    """A local copy of a table stored in the cloud, or ``source`` unchanged.

    An ``s3://``, ``gs://``, ``az://`` or ``https://`` address is fetched
    once into spaCR's cloud cache and read from there, and fetched again
    only when the stored object changes. Credentials come from the
    standard places (see :class:`spacr.ome_zarr._CloudOptions`). Anything
    else is returned as given, without importing the cloud code.

    :param source: a path, an address, or an open connection.
    :returns: a local path for an address, ``source`` otherwise.
    """
    if isinstance(source, str) and '://' in source:
        from .ome_zarr import _cloud_local_copy, _is_cloud_url
        if _is_cloud_url(source):
            return _cloud_local_copy(source)
    return source


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


def _read_query(db: Any, sql: str, *, params: Any = None,
               canonicalise: bool = True,
               report: Optional[Callable[[str], None]] = print,
               warn: Optional[Callable[[str], None]] = None,
               repair_plate_ids: bool = True,
               **kwargs) -> pd.DataFrame:
    """Read a query off an ALREADY OPEN connection, with canonical columns.

    The entry point the funnel was missing.

    PRIVATE ON PURPOSE, and only until the localized API catalogs are next
    regenerated. A public name here joins the API surface, and the
    surface is mirrored symbol-for-symbol into
    ``docs/source/_static/i18n/api/*.json`` for nine locales -- which
    needs the translation models to rebuild. Publishing it now would put
    a public function in the package that the localized API pages do not
    carry, which is the exact failure the surface ratchet in
    tests/test_api_i18n_extractor.py exists to catch. Nothing about the
    function is internal; the underscore is a release constraint and
    should come off with the next catalog rebuild. :func:`read_table` and
    :func:`read_database` both take a PATH and open the database
    themselves, which is right for a caller that wants one table and
    wrong for one that has a connection already and reads several off
    it -- reopening per table changes the transaction each read sees and
    costs a connect apiece. Callers in that position had no canonical
    reader to use and reached for ``pandas.read_sql_query`` directly,
    which is how a frame with un-canonicalised column names gets into the
    package: it does not fail, it returns a number.

    :param db: an open DB-API connection. Not a path -- use
        :func:`read_database` when what you have is a path.
    :param sql: the query.
    :param params: bound parameters, passed through to pandas.
    :param canonicalise: apply the vocabulary. See :func:`read_table`.
    :param report: called with each agreeing-collision message.
    :param warn: called with each disagreeing-collision message.
    :param repair_plate_ids: collapse a doubled ``pp`` plate prefix.
    :param kwargs: passed to the underlying pandas reader.
    :returns: a :class:`pandas.DataFrame`.
    """
    frame = pd.read_sql_query(sql, db, params=params, **kwargs)
    return _canonicalise(frame, canonicalise, report, warn, repair_plate_ids)


def read_table(source: Any, *, table: Optional[str] = None,
               canonicalise: bool = True,
               report: Optional[Callable[[str], None]] = print,
               warn: Optional[Callable[[str], None]] = None,
               repair_plate_ids: bool = True,
               **kwargs) -> pd.DataFrame:
    """Read one table, whatever it is stored in, with canonical column names.

    CSV, TSV, SQLite, Parquet, Feather and Excel, chosen by suffix. A
    database -- SQLite, a ``.duckdb`` file, a ``.parquetdb`` store or a
    ``postgresql://`` string -- needs ``table``; every other format ignores
    it.

    :param source: path to the file. ``~`` and ``$VARS`` are expanded. A
        cloud address (``s3://``, ``gs://``, ``az://``, ``https://``) is read
        from a cached local copy.
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
    kind = ('sqlite' if _backend_of(source) != 'sqlite'
            else table_format(source))
    source = _fetched(source)
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


#: The schema-metadata key a Parquet file written by :func:`_write_parquet`
#: carries its spaCR description under, beside pandas' own ``pandas`` key.
_PARQUET_METADATA_KEY = b'spacr'

_PYARROW_MISSING_MESSAGE = """\
Writing Parquet with its schema metadata needs pyarrow, which is not
installed in this environment (missing module: {module}).

Install it with:

    python -m pip install pyarrow\
"""

_PYREADR_MISSING_MESSAGE = """\
Writing R data files (.rds) needs pyreadr, which is not installed in this
environment (missing module: {module}).

Install it with:

    python -m pip install pyreadr

The Parquet tables and the R loader script do not need it: R reads the
Parquet files with the arrow or nanoparquet package.\
"""


def _require_optional(module_name: str, message: str):
    """Import an optional module, or raise ``ImportError`` with ``message``.

    :param module_name: the module to import.
    :param message: the install instructions, with a ``{module}`` field
        naming the module that failed to import.
    :returns: the imported module.
    :raises ImportError: when the module cannot be imported.
    """
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        missing = (getattr(exc, 'name', None) or module_name).split('.')[0]
        raise ImportError(message.format(module=missing)) from exc


def _publish(target: str, write: Callable[[str], None]) -> str:
    """Write through a temporary sibling file and move it into place.

    A failed write leaves any earlier file at ``target`` untouched and no
    partial file behind.
    """
    parent = os.path.dirname(os.path.abspath(target))
    os.makedirs(parent, exist_ok=True)
    base, suffix = os.path.splitext(os.path.basename(target))
    pending = os.path.join(parent, f'.{base}.{os.getpid()}.pending{suffix}')
    try:
        write(pending)
        os.replace(pending, target)
    finally:
        if os.path.exists(pending):
            os.remove(pending)
    return target


_OPENPYXL_MISSING_MESSAGE = """\
Writing an Excel workbook with several sheets needs openpyxl, which is not
installed in this environment (missing module: {module}).

Install it with:

    python -m pip install openpyxl\
"""


def _write_workbook(sheets: Dict[str, pd.DataFrame], path: Any, *,
                    index: bool = False) -> str:
    """Write several frames as the sheets of one Excel workbook.

    Column names are written exactly as given: a workbook that follows an
    outside template has to keep that template's headings.

    :param sheets: sheet name to frame, in sheet order.
    :param path: the ``.xlsx`` destination; written through a temporary
        sibling, so a failure leaves any earlier workbook in place.
    :param index: pandas' ``index`` argument.
    :returns: the resolved path written.
    :raises ImportError: when openpyxl is not installed.
    """
    _require_optional('openpyxl', _OPENPYXL_MISSING_MESSAGE)
    target = resolve_path(path)

    def _write(pending: str) -> None:
        """Write every sheet into ``pending``."""
        with pd.ExcelWriter(pending, engine='openpyxl') as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=str(name)[:31], index=index)

    return _publish(target, _write)


def _write_parquet(frame: pd.DataFrame, path: Any, *,
                   metadata: Optional[Dict[str, Any]] = None,
                   canonicalise: bool = True, index: bool = False,
                   compression: str = 'snappy') -> str:
    """Write one frame as Parquet, keeping its dtypes and a spaCR description.

    Categorical columns are stored as Parquet dictionaries, which pandas
    reads back as categoricals and R's arrow package as factors. Integer,
    float, boolean and string columns keep their types. ``metadata`` is
    stored as JSON under the ``spacr`` key of the file's schema metadata,
    beside the ``pandas`` key pyarrow writes; :func:`_parquet_metadata`
    reads it back. The file is written beside the target and moved into
    place, so a failed write keeps any earlier file.

    :param frame: the frame.
    :param path: destination ``.parquet``. ``~`` and ``$VARS`` are expanded;
        the parent directory is created.
    :param metadata: JSON-serialisable description of the table. Values
        JSON cannot represent are stored as their string form.
    :param canonicalise: rename legacy metadata spellings on the way out.
        Off for a table whose column names are not measurement metadata,
        such as a feature dictionary with a ``channel`` column.
    :param index: store the index as a column.
    :param compression: Parquet codec. ``'snappy'`` by default, which every
        Parquet reader supports.
    :returns: the resolved path written.
    :raises ImportError: with install instructions when pyarrow is missing.
    """
    import json

    pa = _require_optional('pyarrow', _PYARROW_MISSING_MESSAGE)
    parquet = _require_optional('pyarrow.parquet', _PYARROW_MISSING_MESSAGE)
    target = resolve_path(path)
    if canonicalise:
        mapping = schema.canonical_rename_plan(frame.columns)
        if mapping:
            frame = frame.rename(columns=mapping)
    table = pa.Table.from_pandas(frame, preserve_index=index)
    stored = dict(table.schema.metadata or {})
    if metadata is not None:
        stored[_PARQUET_METADATA_KEY] = json.dumps(
            metadata, default=str, sort_keys=True).encode('utf-8')
    table = table.replace_schema_metadata(stored)
    return _publish(target, lambda pending: parquet.write_table(
        table, pending, compression=compression))


def _parquet_metadata(path: Any) -> Dict[str, Any]:
    """The spaCR description stored in a Parquet file by :func:`_write_parquet`.

    Only the schema is read, not the data.

    :param path: a ``.parquet`` file.
    :returns: the stored dict, or ``{}`` when the file carries none.
    :raises ImportError: with install instructions when pyarrow is missing.
    """
    import json

    parquet = _require_optional('pyarrow.parquet', _PYARROW_MISSING_MESSAGE)
    stored = parquet.read_schema(resolve_path(path)).metadata or {}
    raw = stored.get(_PARQUET_METADATA_KEY)
    return json.loads(raw.decode('utf-8')) if raw else {}


def _write_rds(frame: pd.DataFrame, path: Any, *,
               canonicalise: bool = True) -> str:
    """Write one frame as an R data frame in an ``.rds`` file.

    Written with pyreadr, which stores numbers as doubles, booleans as
    logicals and text and categorical columns as character vectors: R
    factor levels and integer types are not kept. The Parquet tables keep
    both. Written beside the target and moved into place.

    :param frame: the frame. The index is not written.
    :param path: destination ``.rds``; the parent directory is created.
    :param canonicalise: rename legacy metadata spellings on the way out.
    :returns: the resolved path written.
    :raises ImportError: with install instructions when pyreadr is missing.
    """
    pyreadr = _require_optional('pyreadr', _PYREADR_MISSING_MESSAGE)
    target = resolve_path(path)
    if canonicalise:
        mapping = schema.canonical_rename_plan(frame.columns)
        if mapping:
            frame = frame.rename(columns=mapping)
    frame = frame.reset_index(drop=True)
    return _publish(target, lambda pending: pyreadr.write_rds(pending, frame))


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
        from .database_concurrency import connect
        return connect(path, readonly=True)
    from .database_concurrency import _inside_write_packet, connect
    if _inside_write_packet(path):
        return connect(path)
    return sqlite3.connect(path, timeout=30)


def database_tables(db: Any, *, migrate: bool = False) -> Tuple[str, ...]:
    """The table names in a database, sorted.

    :param db: path to the database, or any store :func:`read_database`
        reads. ``~`` and ``$VARS`` are expanded.
    :param migrate: run the schema migration first. ``False``, because
        listing what is there must not rewrite it.
    :returns: the table names.
    """
    backend = _backend_of(db)
    if backend != 'sqlite':
        return _store_tables(db, backend)
    db = _fetched(db)
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
    backend = _backend_of(source)
    kind = 'sqlite' if backend != 'sqlite' else table_format(source)
    source = _fetched(source)
    if kind == 'sqlite':
        if table is None:
            raise ValueError(
                f'{source!r} is a database; table_columns needs table=<name>.')
        frame = next(_iter_chunks(source, table, limit=0))
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
    """Read one or more tables out of a measurement database.

    Expands ``~``, validates identifiers before opening the database, runs the
    schema migration when requested, and reads in chunks to limit peak memory.
    A missing table raises :class:`ValueError` with its name.

    SQLite is the default store. A ``.duckdb`` file, a ``.parquetdb`` folder
    of Parquet parts or a ``postgresql://`` connection string is read the
    same way, through duckdb or psycopg imported only then; the schema
    migration, ``read_only`` and ``kwargs`` apply to SQLite alone.

    :param db: path to the database, or a PostgreSQL connection string.
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
    backend = _backend_of(db)
    if backend != 'sqlite':
        present = set(_store_tables(db, backend))
        for name in names:
            if name not in present:
                raise ValueError(f'Table not found in database: {name}')
            chunks = list(_iter_chunks(db, name, limit=limit,
                                       chunksize=chunksize))
            frame = (chunks[0] if len(chunks) == 1
                     else pd.concat(chunks, ignore_index=True))
            del chunks
            frames.append(_canonicalise(frame, canonicalise, report, warn,
                                        repair_plate_ids))
        return frames
    db = _fetched(db)
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
    """Write one frame into a database table, with canonical column names.

    SQLite by default; a ``.duckdb`` path, a ``.parquetdb`` path or a
    ``postgresql://`` string writes to that store instead, where ``append``
    also adds columns the table lacks and ``migrate`` and ``kwargs`` are
    ignored.

    :param frame: the frame.
    :param db: path to the database, or a PostgreSQL connection string;
        created if absent, ``~`` expanded.
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
    if canonicalise:
        mapping = schema.canonical_rename_plan(frame.columns)
        if mapping:
            frame = frame.rename(columns=mapping)
    backend = _backend_of(db)
    if backend != 'sqlite':
        _store_write(frame, db, backend, table, if_exists)
        return db if backend == 'postgres' else resolve_path(db)
    target = resolve_path(db)
    parent = os.path.dirname(os.path.abspath(target))
    os.makedirs(parent, exist_ok=True)
    with _connect(target, migrate=migrate) as conn:
        frame.to_sql(table, conn, if_exists=if_exists, index=index, **kwargs)
    return target


_DUCKDB_SUFFIXES: Tuple[str, ...] = ('.duckdb', '.ddb')

_PARQUET_STORE_SUFFIX = '.parquetdb'

_POSTGRES_PREFIXES: Tuple[str, ...] = ('postgresql://', 'postgres://')

_STORE_BACKENDS: Tuple[str, ...] = ('sqlite', 'duckdb', 'parquet', 'postgres')

_DUCKDB_MISSING_MESSAGE = """\
A DuckDB measurement store needs duckdb, which is not installed in this
environment (missing module: {module}).

Install it with:

    python -m pip install "spacr[databases]"\
"""

_PSYCOPG_MISSING_MESSAGE = """\
A PostgreSQL measurement store needs psycopg 3, which is not installed in
this environment (missing module: {module}).

Install it with:

    python -m pip install "spacr[databases]"\
"""


def _backend_of(db: Any) -> str:
    """Which store a database locator names.

    ``postgresql://`` or ``postgres://`` is a PostgreSQL connection string,
    a ``.duckdb`` or ``.ddb`` path a DuckDB file, a ``.parquetdb`` path a
    folder holding one subfolder of Parquet parts per table, and anything
    else, an open connection included, SQLite.

    :param db: a path, a connection string or an open connection.
    :returns: one of ``'sqlite'``, ``'duckdb'``, ``'parquet'``,
        ``'postgres'``.
    """
    if isinstance(db, str) and db.strip().lower().startswith(
            _POSTGRES_PREFIXES):
        return 'postgres'
    if not isinstance(db, (str, os.PathLike)):
        return 'sqlite'
    path = str(resolve_path(db)).rstrip('/\\')
    suffix = os.path.splitext(path)[1].lower()
    if suffix in _DUCKDB_SUFFIXES:
        return 'duckdb'
    if suffix == _PARQUET_STORE_SUFFIX:
        return 'parquet'
    return 'sqlite'


def _duckdb_connect(db: Any, *, read_only: bool = False):
    """Open a DuckDB file, creating its folder when writing."""
    duckdb = _require_optional('duckdb', _DUCKDB_MISSING_MESSAGE)
    path = resolve_path(db)
    if not read_only:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = duckdb.connect(path, read_only=read_only)
    conn.execute('SET enable_progress_bar = false')
    return conn


def _postgres_connect(dsn: str):
    """Open a PostgreSQL connection from a libpq connection string.

    A bare ``postgresql://`` takes host, database, user and password from
    the standard ``PG*`` environment variables and ``~/.pgpass``.
    """
    psycopg = _require_optional('psycopg', _PSYCOPG_MISSING_MESSAGE)
    return psycopg.connect(dsn.strip())


def _parquet_parts(store: str, table: str) -> List[str]:
    """The Parquet part files of ``table`` in a Parquet store, oldest first."""
    folder = os.path.join(store, table)
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, name) for name in sorted(os.listdir(folder))
            if name.endswith('.parquet') and not name.startswith('.')]


def _store_tables(db: Any, backend: str) -> Tuple[str, ...]:
    """The table names in a DuckDB, Parquet or PostgreSQL store, sorted."""
    if backend == 'parquet':
        store = resolve_path(db)
        if not os.path.isdir(store):
            return ()
        return tuple(sorted(name for name in os.listdir(store)
                            if _parquet_parts(store, name)))
    query = ("SELECT table_name FROM information_schema.tables "
             "WHERE table_schema = current_schema() "
             "AND table_type = 'BASE TABLE'")
    if backend == 'duckdb':
        if not os.path.exists(resolve_path(db)):
            return ()
        with _duckdb_connect(db, read_only=True) as conn:
            rows = conn.execute(query).fetchall()
    else:
        with _postgres_connect(db) as conn:
            rows = conn.execute(query).fetchall()
    return tuple(sorted(row[0] for row in rows))


def _iter_chunks(db: Any, table: str, *, limit: Optional[int] = None,
                 chunksize: int = 100_000):
    """Yield a table of any store in frames of at most ``chunksize`` rows.

    At least one frame is yielded, empty with the table's columns when the
    table has no rows, so a caller always learns the header. Nothing is
    canonicalised; the store's column names come back as written.
    """
    backend = _backend_of(db)
    quoted = _quote_identifier(table)
    query = f'SELECT * FROM {quoted}'
    if limit is not None:
        query += f' LIMIT {int(limit)}'
    if backend == 'sqlite':
        with _connect(db, migrate=False) as conn:
            yielded = False
            for chunk in pd.read_sql_query(query, conn, chunksize=chunksize):
                yielded = True
                yield chunk
            if not yielded:
                yield pd.read_sql_query(f'SELECT * FROM {quoted} LIMIT 0',
                                        conn)
        return
    if backend == 'parquet':
        parquet = _require_optional('pyarrow.parquet',
                                    _PYARROW_MISSING_MESSAGE)
        parts = _parquet_parts(resolve_path(db), table)
        remaining = limit
        yielded = False
        for part in parts:
            for batch in parquet.ParquetFile(part).iter_batches(
                    batch_size=chunksize):
                frame = batch.to_pandas()
                if remaining is not None:
                    frame = frame.iloc[:max(remaining, 0)]
                    remaining -= len(frame)
                if len(frame) or not yielded:
                    yielded = True
                    yield frame
                if remaining is not None and remaining <= 0:
                    return
        if not yielded and parts:
            yield parquet.read_schema(parts[0]).empty_table().to_pandas()
        return
    if backend == 'duckdb':
        with _duckdb_connect(db, read_only=True) as conn:
            result = conn.execute(query)
            reader = (result.to_arrow_reader(chunksize)
                      if hasattr(result, 'to_arrow_reader')
                      else result.fetch_record_batch(chunksize))
            yielded = False
            for batch in reader:
                yielded = True
                yield batch.to_pandas()
            if not yielded:
                yield reader.schema.empty_table().to_pandas()
        return
    with _postgres_connect(db) as conn:
        with conn.cursor(name='spacr_read') as cursor:
            cursor.execute(query)
            yielded = False
            while True:
                rows = cursor.fetchmany(chunksize)
                if rows or not yielded:
                    names = [column.name for column in cursor.description]
                    yielded = True
                    yield pd.DataFrame.from_records(rows, columns=names)
                if not rows:
                    break


def _postgres_type(dtype: Any) -> str:
    """The PostgreSQL column type a pandas dtype is stored as."""
    if pd.api.types.is_bool_dtype(dtype):
        return 'BOOLEAN'
    if pd.api.types.is_integer_dtype(dtype):
        return 'BIGINT'
    if pd.api.types.is_float_dtype(dtype):
        return 'DOUBLE PRECISION'
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return 'TIMESTAMP'
    return 'TEXT'


def _store_write(frame: pd.DataFrame, db: Any, backend: str, table: str,
                 if_exists: str) -> None:
    """Write one frame into a DuckDB, Parquet or PostgreSQL store.

    ``append`` adds any column the table does not have yet, so a later run
    that measures more features still lands in the same table.
    """
    if if_exists not in ('append', 'replace', 'fail'):
        raise ValueError(f"if_exists must be 'append', 'replace' or 'fail', "
                         f"not {if_exists!r}.")
    present = table in _store_tables(db, backend)
    if present and if_exists == 'fail':
        raise ValueError(f'Table {table!r} already exists in {db!r}.')
    fresh = not present or if_exists == 'replace'
    quoted = _quote_identifier(table)
    if backend == 'parquet':
        import shutil
        import uuid
        _require_optional('pyarrow', _PYARROW_MISSING_MESSAGE)
        folder = os.path.join(resolve_path(db), table)
        if present and if_exists == 'replace':
            shutil.rmtree(folder)
        part = os.path.join(
            folder, f'part-{len(_parquet_parts(resolve_path(db), table)):06d}'
                    f'-{uuid.uuid4().hex[:8]}.parquet')
        _publish(part, lambda pending: frame.to_parquet(pending, index=False))
        return
    if backend == 'duckdb':
        with _duckdb_connect(db) as conn:
            conn.register('spacr_frame', frame)
            if present and if_exists == 'replace':
                conn.execute(f'DROP TABLE {quoted}')
            if fresh:
                conn.execute(
                    f'CREATE TABLE {quoted} AS SELECT * FROM spacr_frame')
                return
            existing = {row[0] for row in conn.execute(
                f'DESCRIBE {quoted}').fetchall()}
            for name, kind, *_ in conn.execute(
                    'DESCRIBE SELECT * FROM spacr_frame').fetchall():
                if name not in existing:
                    conn.execute(f'ALTER TABLE {quoted} ADD COLUMN '
                                 f'{_quote_identifier(name)} {kind}')
            conn.execute(
                f'INSERT INTO {quoted} BY NAME SELECT * FROM spacr_frame')
        return
    columns = [str(name) for name in frame.columns]
    with _postgres_connect(db) as conn:
        if present and if_exists == 'replace':
            conn.execute(f'DROP TABLE {quoted}')
        if fresh:
            conn.execute(f'CREATE TABLE {quoted} (' + ', '.join(
                f'{_quote_identifier(name)} {_postgres_type(frame[name].dtype)}'
                for name in columns) + ')')
        else:
            existing = {row[0] for row in conn.execute(
                'SELECT column_name FROM information_schema.columns '
                'WHERE table_schema = current_schema() AND table_name = %s',
                (table,)).fetchall()}
            for name in columns:
                if name not in existing:
                    conn.execute(
                        f'ALTER TABLE {quoted} ADD COLUMN '
                        f'{_quote_identifier(name)} '
                        f'{_postgres_type(frame[name].dtype)}')
        values = frame.astype(object).where(frame.notna(), None)
        header = ', '.join(_quote_identifier(name) for name in columns)
        with conn.cursor() as cursor:
            with cursor.copy(
                    f'COPY {quoted} ({header}) FROM STDIN') as copy:
                for row in values.itertuples(index=False, name=None):
                    copy.write_row(row)


def _migrate_database(source: Any, target: Any, *, tables: Any = None,
                      chunksize: int = 100_000,
                      report: Optional[Callable[[str], None]] = print
                      ) -> Tuple[str, ...]:
    """Copy tables from one measurement store into another.

    Any direction between SQLite, DuckDB, a Parquet store and PostgreSQL:
    each table is streamed in chunks, replacing a table of the same name
    in ``target``, with column names kept exactly as stored. SQLite's own
    bookkeeping tables are skipped.

    :param source: the store to copy from.
    :param target: the store to copy into; created if absent.
    :param tables: a table name or a sequence of them; ``None`` copies all.
    :param chunksize: rows held in memory at once.
    :param report: called with one line per copied table; ``None`` to
        silence.
    :returns: the table names copied.
    """
    if tables is None:
        names = [name for name in database_tables(source)
                 if not name.startswith('sqlite_')]
    else:
        names = [tables] if isinstance(tables, str) else list(tables)
    for name in names:
        rows = 0
        for index, chunk in enumerate(
                _iter_chunks(source, name, chunksize=chunksize)):
            write_database(chunk, target, name,
                           if_exists='replace' if index == 0 else 'append',
                           canonicalise=False)
            rows += len(chunk)
        if report is not None:
            report(f'Copied {name}: {rows} row(s) to '
                   f'{_backend_of(target)} store.')
    return tuple(names)


def _query_store(db: Any, sql: str, *, canonicalise: bool = True,
                 report: Optional[Callable[[str], None]] = print,
                 warn: Optional[Callable[[str], None]] = None) -> pd.DataFrame:
    """Run one SQL query against any store and return the canonical frame.

    A Parquet store is queried through an in-memory DuckDB whose views read
    each table's parts, so an aggregate over a very large screen scans only
    the columns it names instead of loading the table.

    :param db: a SQLite, DuckDB or Parquet path, or a PostgreSQL string.
    :param sql: the query, naming tables as they are stored.
    :param canonicalise: apply the vocabulary to the result.
    :param report: see :func:`read_table`.
    :param warn: see :func:`read_table`.
    :returns: a :class:`pandas.DataFrame`.
    """
    backend = _backend_of(db)
    if backend == 'sqlite':
        with _connect(db, migrate=False) as conn:
            frame = pd.read_sql_query(sql, conn)
    elif backend == 'duckdb':
        with _duckdb_connect(db, read_only=True) as conn:
            frame = conn.execute(sql).df()
    elif backend == 'parquet':
        duckdb = _require_optional('duckdb', _DUCKDB_MISSING_MESSAGE)
        store = resolve_path(db)
        with duckdb.connect() as conn:
            conn.execute('SET enable_progress_bar = false')
            for name in _store_tables(store, 'parquet'):
                files = ', '.join("'" + part.replace("'", "''") + "'"
                                  for part in _parquet_parts(store, name))
                conn.execute(
                    f'CREATE VIEW {_quote_identifier(name)} AS SELECT * FROM '
                    f'read_parquet([{files}], union_by_name = true)')
            frame = conn.execute(sql).df()
    else:
        with _postgres_connect(db) as conn:
            cursor = conn.execute(sql)
            names = [column.name for column in cursor.description]
            frame = pd.DataFrame.from_records(cursor.fetchall(), columns=names)
    return _canonicalise(frame, canonicalise, report, warn)
