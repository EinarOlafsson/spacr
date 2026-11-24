"""One reader and one writer for every table spaCR opens or saves.

Purpose
----------------------
spaCR reads and writes tables in many places (``pd.read_csv``,
``pd.read_sql*``, ``.to_csv``, ``.to_sql``). Without a shared reader, the
spelling of a key column in a frame depends on which code path read it.

For example, ``columnID`` could work as a filter, because *some* path
downstream renamed on the way to the fit, while the CSV picker read the raw
header and offered ``column_name`` and ``column`` -- names a user should not
need to know.

This module normalises names in one place and every reader goes through it, so
the picker also sees ``columnID``.

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
**spaCR writes canonical names.** ``write_table`` and ``write_database``
canonicalise on the way out by default, so a frame assembled by hand cannot
re-export ``column_name`` and start the cycle again. The header of an exported
file therefore changes for anyone whose downstream script reads ``column_name``
-- a documented compatibility change. ``canonicalise=False`` is there for a
caller who owes an external format an exact header.

Path and schema guards
--------------------------------------------------------
* A ``~`` path is expanded, once, for every reader -- GitHub issue #108,
  where a ``src`` beginning with ``~`` was resolved against the working
  directory and refused with ``FileNotFoundError: ~<DB>``. ``$HOME`` and
  ``%USERPROFILE%`` too: a settings CSV carried between machines routinely
  holds one.
* **The measurements schema migration runs on open**, exactly as
  ``io._read_db`` does it, so a legacy database is repaired by any reader.

Dependencies
------------
``pandas``, ``sqlite3`` and :mod:`spacr.schema`. The alternative measurement
stores import ``duckdb`` or ``psycopg`` only when a DuckDB, Parquet or
PostgreSQL store is opened. Nothing else at module scope -- no ``spacr.utils``,
no matplotlib, no torch. The CSV picker and the SQL column list import this
module to get canonical names and must not import torch.
"""

from __future__ import annotations

import os
import re
import sqlite3
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from . import schema

__all__ = [
    'TabularFormatError',
    'CSV_SUFFIXES', 'COMPRESSED_SUFFIXES', 'DATABASE_SUFFIXES', 'TABLE_SUFFIXES',
    'resolve_path', 'table_format',
    'read_table', 'read_table_columns', 'write_table',
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

#: Compression suffixes pandas undoes on its own; ``x.tsv.gz`` is read as the
#: ``.tsv`` beneath it.
COMPRESSED_SUFFIXES: Tuple[str, ...] = ('.gz', '.bz2', '.xz', '.zst')

#: SQLite, by every suffix spaCR has ever written one under.
DATABASE_SUFFIXES: Tuple[str, ...] = ('.db', '.sqlite', '.sqlite3', '.db3')

#: Every suffix :func:`read_table` understands.
TABLE_SUFFIXES: Tuple[str, ...] = (
    tuple(CSV_SUFFIXES) + DATABASE_SUFFIXES
    + ('.parquet', '.pq', '.feather', '.xlsx', '.xls', '.xlsm')
)


def resolve_path(path: Any) -> str:
    """Expand ``~`` and environment variables in a path.

    GitHub issue #108: a ``src`` beginning with ``~`` produced
    ``~/.../measurements.db``, which the migration resolved against the working
    directory and refused with ``FileNotFoundError: ~<DB>``. Expanding here
    covers every site that builds a measurements path by string concatenation.

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


def _table_suffix(path: Any) -> str:
    """The lower-case suffix that names a path's format.

    A compression suffix is looked through, so ``links.txt.gz`` gives
    ``.txt``.

    :param path: a path.
    :returns: the suffix, with its dot, or an empty string.
    """
    stem, suffix = os.path.splitext(str(resolve_path(path)))
    if suffix.lower() in COMPRESSED_SUFFIXES:
        inner = os.path.splitext(stem)[1].lower()
        if inner in CSV_SUFFIXES:
            return inner
    return suffix.lower()


def table_format(path: Any) -> str:
    """Which reader a path needs: ``'csv'``, ``'sqlite'``, ``'parquet'``,
    ``'feather'`` or ``'excel'``.

    Delimited text compressed with gzip, bzip2, xz or zstd (``.tsv.gz``) is
    ``'csv'``; pandas decompresses it.

    :param path: a path.
    :returns: the format name.
    :raises TabularFormatError: when the suffix names no known format.
    """
    suffix = _table_suffix(path)
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
    :param canonicalise: apply the vocabulary. Keep it on for ordinary reads:
        it is what makes the picker and the run agree about what a column is
        called. Off is for a caller inspecting a file exactly as written.
    :param report: called with each agreeing-collision message; ``print`` by
        default, ``None`` to silence.
    :param warn: called with each disagreeing-collision message; ``None``
        routes to :func:`warnings.warn`.
    :param repair_plate_ids: collapse a doubled ``pp`` plate prefix.
    :param kwargs: passed to the underlying pandas reader.
    :returns: a :class:`pandas.DataFrame`.
    """
    kind = ('sqlite' if _backend_of(source) != 'sqlite'
            or _is_project_folder(source) else table_format(source))
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
        suffix = _table_suffix(path)
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


def _parse_setting_value(value: Any) -> Any:
    """Turn one cell of a settings CSV back into the Python value it was.

    ``True``/``False`` (any case), a literal list, tuple or dict (whose
    values are parsed the same way), an int, a float, an empty cell
    (``None``), or the text as written.

    :param value: the cell as pandas read it.
    :returns: the parsed value.
    """
    if pd.isna(value) or value == '':
        return None
    if not isinstance(value, str):
        return value
    if value.strip().lower() == 'true':
        return True
    if value.strip().lower() == 'false':
        return False
    if value.startswith(('(', '[', '{')):
        import ast
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, dict):
                parsed = {k: _parse_setting_value(v) for k, v in parsed.items()}
            return parsed
        except (ValueError, SyntaxError):
            pass
    try:
        if '.' in value or 'e' in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        pass
    return value


def _settings_columns(frame: pd.DataFrame, setting_key: str, setting_value: str,
                      source: Any = '') -> Tuple[str, str]:
    """The key and value columns of a settings table.

    ``setting_key``/``setting_value`` when both are columns, otherwise the
    ``Key``/``Value`` pair :func:`spacr.utils.save_settings` writes.

    :param frame: the table as :func:`read_table` read it.
    :param setting_key: the key column asked for.
    :param setting_value: the value column asked for.
    :param source: the file the table came from, named in the error.
    :returns: ``(key column, value column)``.
    :raises ValueError: when neither pair is present.
    """
    if setting_key in frame.columns and setting_value in frame.columns:
        return setting_key, setting_value
    if 'Key' in frame.columns and 'Value' in frame.columns:
        return 'Key', 'Value'
    raise ValueError(
        f"CSV file must contain {setting_key} and {setting_value} "
        f"columns (or the Key/Value pair save_settings writes); "
        f"{os.path.basename(str(source))} has "
        f"{list(frame.columns)}.")


def _settings_from_table(frame: pd.DataFrame, setting_key: str = 'setting_key',
                         setting_value: str = 'setting_value',
                         source: Any = '') -> Dict[str, Any]:
    """The settings dict a two-column settings table holds.

    What :func:`spacr.utils.load_settings` returns for the same table, here
    so the GUI can read a settings CSV without importing :mod:`spacr.utils`
    (torch, torchvision, scikit-image, statsmodels: seconds the first time,
    on the GUI thread).

    :param frame: the table as :func:`read_table` read it.
    :param setting_key: the key column.
    :param setting_value: the value column.
    :param source: the file the table came from, named in the error.
    :returns: ``{key: parsed value}``, see :func:`_parse_setting_value`.
    :raises ValueError: when neither column pair is present
        (:func:`_settings_columns`).
    """
    setting_key, setting_value = _settings_columns(
        frame, setting_key, setting_value, source=source)
    return {key: _parse_setting_value(value)
            for key, value in zip(frame[setting_key], frame[setting_value])}


def read_table_columns(source: Any, columns: List[str], *, chunk_rows: int = 262144,
                       table: Optional[str] = None) -> pd.DataFrame:
    """Read only some columns of one table, a chunk of rows at a time.

    A Parquet file is decoded row group by row group, so the columns left out are never
    brought into memory: a whole-well cell table of 2,800 columns costs only the few
    columns asked for. Another format is read with its ``usecols`` or ``columns`` argument
    through :func:`read_table`. Column names are kept as written, not canonicalised.

    :param source: the table path.
    :param columns: the column names wanted; a name the table lacks is skipped.
    :param chunk_rows: rows decoded per chunk of a Parquet file.
    :param table: the table name, for a database.
    :returns: the frame with the wanted columns, in the order given.
    """
    kind = 'sqlite' if _backend_of(source) != 'sqlite' else table_format(source)
    present = set(table_columns(source, table=table, canonicalise=False))
    wanted = [c for c in dict.fromkeys(columns) if c in present]
    if kind != 'parquet':
        frame = read_table(source, table=table, canonicalise=False, report=None)
        return frame[wanted]
    parquet = _require_optional('pyarrow.parquet', _PYARROW_MISSING_MESSAGE)
    pyarrow = _require_optional('pyarrow', _PYARROW_MISSING_MESSAGE)
    reader = parquet.ParquetFile(resolve_path(_fetched(source)))
    batches = list(reader.iter_batches(batch_size=int(chunk_rows), columns=wanted))
    if not batches:
        return reader.schema_arrow.empty_table().select(wanted).to_pandas()
    return pyarrow.Table.from_batches(batches).to_pandas()


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
        every spaCR export that ever wanted the index has a real column for it
        and an unnamed index reads back as ``Unnamed: 0``.
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
        suffix = _table_suffix(target)
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
    from .store import _legacy_sqlite_connection
    return _legacy_sqlite_connection(path, read_only=read_only)


def _is_project_folder(source: Any) -> bool:
    """Whether ``source`` is a folder :func:`spacr.store.open` reads.

    That is a store project (a folder holding ``spacr.duckdb``, or a
    ``measurements/`` folder that does), a folder holding
    ``measurements.db``, or a ``.parquetdb`` folder.
    """
    if not isinstance(source, (str, os.PathLike)):
        return False
    path = resolve_path(source)
    return os.path.isdir(path) and _store_project(path) is not None


def _store_project(source: Any) -> Optional[str]:
    """The path :func:`spacr.store.open` reads for a spaCR measurement
    locator, or None.

    * a folder holding ``spacr.duckdb``, or holding ``measurements/`` that
      does, opens as that store project;
    * a ``spacr.duckdb`` path opens its project folder;
    * a folder holding ``measurements.db`` (directly or in
      ``measurements/``) and a ``.parquetdb`` folder open as older formats;
    * any other path is a database file (SQLite or ``.duckdb``) and opens as
      itself.

    A PostgreSQL string, an open connection or a folder with none of these
    gives None.

    :param source: a path or connection string.
    :returns: the path to open, or None.
    """
    from .store import _LEGACY_NAMES, CATALOG

    if not isinstance(source, (str, os.PathLike)) or (
            _backend_of(source) == 'postgres'):
        return None
    path = str(resolve_path(source)).rstrip('/\\') or str(resolve_path(source))
    if os.path.isdir(path):
        if _backend_of(path) == 'parquet':
            return path
        for folder in (path, os.path.join(path, 'measurements')):
            if os.path.isfile(os.path.join(folder, CATALOG)):
                return folder
        if any(os.path.isfile(os.path.join(path, name))
               for name in _LEGACY_NAMES):
            return path
        return None
    if os.path.basename(path) == CATALOG:
        return os.path.dirname(path)
    return path


@contextmanager
def _opened_store(source: Any):
    """The :class:`spacr.store.Store` for a measurement locator.

    Yields None for a SQLite file that does not exist, so a reader sees an
    empty database rather than creating one.

    :param source: a path :func:`_store_project` resolves.
    :raises ValueError: when ``source`` is not a spaCR measurement locator.
    """
    from . import store

    project = _store_project(source)
    if project is None:
        raise ValueError(f'{source!r} is not a spaCR measurement store, '
                         f'measurements.db or database file.')
    if not os.path.exists(project) and _backend_of(project) == 'sqlite':
        yield None
        return
    with store.open(project, verify='none') as opened:
        yield opened


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
    if migrate and os.path.isfile(resolve_path(db)):
        from .database_schema import ensure_database_schema
        ensure_database_schema(resolve_path(db))
    with _opened_store(db) as opened:
        return tuple(opened.tables()) if opened is not None else ()


def table_columns(source: Any, *, table: Optional[str] = None,
                  canonicalise: bool = True) -> Tuple[str, ...]:
    """The column names a table would be read with, using a schema or header when possible.

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
    kind = ('sqlite' if backend != 'sqlite' or _is_project_folder(source)
            else table_format(source))
    source = _fetched(source)
    if kind == 'sqlite':
        if table is None:
            raise ValueError(
                f'{source!r} is a database; table_columns needs table=<name>.')
        frame = next(_iter_chunks(source, table, limit=0))
    elif kind == 'csv':
        frame = read_table(source, canonicalise=False, report=None, nrows=0)
    elif kind == 'parquet':
        parquet = _require_optional('pyarrow.parquet', _PYARROW_MISSING_MESSAGE)
        frame = parquet.read_schema(resolve_path(source)).empty_table().to_pandas()
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
    if backend == 'postgres':
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
    if read_only and migrate and backend == 'sqlite':
        raise ValueError(
            'read_only=True cannot migrate: a migration writes. Pass '
            'migrate=False, or open read/write.')
    path = resolve_path(db)
    if migrate and backend == 'sqlite' and not os.path.isdir(path):
        from .database_schema import ensure_database_schema
        ensure_database_schema(path)
    with _opened_store(db) as opened:
        present = set(opened.tables()) if opened is not None else set()
        for name in names:
            if name not in present:
                raise ValueError(f'Table not found in database: {name}')
            chunks = list(opened._frames(name, limit=limit,
                                        chunksize=chunksize, **kwargs))
            frame = (chunks[0] if len(chunks) == 1
                     else pd.concat(chunks, ignore_index=True))
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
    ``postgresql://`` string writes to that store instead, where ``migrate``
    and ``kwargs`` are ignored. On every store, ``append`` first adds the
    columns the table lacks.

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
    if if_exists not in ('append', 'replace', 'fail'):
        raise ValueError(f"if_exists must be 'append', 'replace' or 'fail', "
                         f"not {if_exists!r}.")
    target = resolve_path(db)
    parent = os.path.dirname(os.path.abspath(target))
    os.makedirs(parent, exist_ok=True)
    with _connect(target, migrate=migrate) as conn:
        existing = _sqlite_table_columns(conn, table)
        if existing and if_exists == 'fail':
            raise ValueError(f'Table {table!r} already exists in {target!r}.')
        if existing and if_exists == 'append':
            _sqlite_add_missing_columns(conn, table, frame, existing, index)
        frame.to_sql(table, conn, if_exists=if_exists, index=index, **kwargs)
    return target


def _sqlite_table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    """The column names of ``table``, or ``[]`` when the table is absent."""
    rows = conn.execute(
        f'PRAGMA table_info({_quote_identifier(table)})').fetchall()
    return [str(row[1]) for row in rows]


def _sqlite_column_type(values: Any) -> str:
    """The SQLite type pandas' ``to_sql`` gives a column of this dtype."""
    from pandas.api import types

    if types.is_bool_dtype(values) or types.is_integer_dtype(values):
        return 'INTEGER'
    if types.is_float_dtype(values):
        return 'REAL'
    if types.is_datetime64_any_dtype(values):
        return 'TIMESTAMP'
    return 'TEXT'


def _sqlite_add_missing_columns(conn: sqlite3.Connection, table: str,
                                frame: pd.DataFrame, existing: List[str],
                                index: bool) -> None:
    """Add to ``table`` every column ``frame`` writes that it does not have.

    SQLite compares column names without regard to case, so ``Area`` is
    already present when the table has ``area``. A column another writer
    added in the meantime is taken as present.
    """
    written = frame.reset_index() if index else frame
    have = {name.lower() for name in existing}
    quoted = _quote_identifier(table)
    for name in written.columns:
        if str(name).lower() in have:
            continue
        try:
            conn.execute(f'ALTER TABLE {quoted} ADD COLUMN '
                         f'{_quote_identifier(str(name))} '
                         f'{_sqlite_column_type(written[name])}')
        except sqlite3.OperationalError as error:
            if 'duplicate column name' not in str(error).lower():
                raise
        have.add(str(name).lower())


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


_WINDOWS_FOLDER_FORBIDDEN = frozenset('<>:"|?*')


def _windows_forbids(char: str) -> bool:
    """Whether Windows refuses ``char`` anywhere in a folder name."""
    return char in _WINDOWS_FOLDER_FORBIDDEN or ord(char) < 32


def _parquet_windows_folder_name(table: str) -> str:
    """The spelling of ``table``'s folder that Windows can create.

    Windows refuses ``< > : " | ? *``, control characters and a trailing dot
    or space in a folder name; each is written as ``%XX``. Every name Windows
    already accepted, ``%`` included, is its own folder, as it always was.
    """
    encoded = ''.join(f'%{ord(char):02X}' if _windows_forbids(char) else char
                      for char in table)
    if encoded and encoded[-1] in '. ':
        encoded = encoded[:-1] + f'%{ord(encoded[-1]):02X}'
    return encoded


def _parquet_folder_name(table: str, *, windows: Optional[bool] = None) -> str:
    """The folder this system gives ``table`` in a Parquet store.

    POSIX keeps every table in the folder of its own name, exactly as stores
    have always been written; Windows uses
    :func:`_parquet_windows_folder_name`, which is the same folder for every
    name Windows could already create.
    """
    if os.name == 'nt' if windows is None else windows:
        return _parquet_windows_folder_name(table)
    return table


def _parquet_table_name(folder: str) -> str:
    """The table a store folder holds.

    A folder is read as an escaped name only when it is exactly the Windows
    spelling of a name Windows forbids; every other folder is its own table.
    """
    table = re.sub(r'%([0-9A-Fa-f]{2})',
                   lambda match: chr(int(match.group(1), 16)), folder)
    if table != folder and _parquet_windows_folder_name(table) == folder:
        return table
    return folder


def _parquet_table_folder(store: str, table: str) -> str:
    """The folder of ``table``: this system's spelling, or the other system's.

    A store copied between Windows and POSIX keeps its tables: when this
    system's folder is absent and the other spelling exists, that is used.
    """
    native = os.path.join(store, _parquet_folder_name(table))
    if os.path.isdir(native):
        return native
    for other in (table, _parquet_windows_folder_name(table)):
        candidate = os.path.join(store, other)
        if candidate == native:
            continue
        try:
            if os.path.isdir(candidate):
                return candidate
        except (OSError, ValueError):
            pass
    return native


def _parquet_verbatim_folder(store: str, table: str) -> bool:
    """Whether ``table`` also has a folder of its own unescaped name."""
    try:
        return os.path.isdir(os.path.join(store, table))
    except (OSError, ValueError):
        return False


def _parquet_parts(store: str, table: str) -> List[str]:
    """The active Parquet parts, using a published snapshot when one exists."""
    if table in ('.', '..') or '/' in table or '\\' in table:
        raise ValueError(f'Invalid Parquet table name: {table!r}')
    folder = _parquet_table_folder(store, table)
    if not os.path.isdir(folder):
        return []
    manifest = _parquet_manifest(folder)
    if manifest is not None:
        return [os.path.join(folder, part['name'])
                for part in manifest['active']]
    return [os.path.join(folder, name) for name in sorted(os.listdir(folder))
            if name.endswith('.parquet') and not name.startswith('.')]


@contextmanager
def _parquet_store_lock(store: str):
    """Serialize spaCR writers of one Parquet store across processes."""
    os.makedirs(store, exist_ok=True)
    with open(os.path.join(store, '.spacr-write.lock'), 'a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b'\0')
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == 'nt':
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _parquet_part_record(path: str) -> dict:
    """Record an immutable part's basename, size and content digest."""
    import hashlib

    before = os.stat(path)
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    after = os.stat(path)
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
         before.st_ctime_ns) !=
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
             after.st_ctime_ns)):
        raise RuntimeError(f'Parquet part changed while being read: {path}')
    return {'name': os.path.basename(path), 'size': after.st_size,
            'sha256': digest.hexdigest()}


def _parquet_manifest(folder: str):
    """Validate and return a complete active snapshot, or None for legacy parts."""
    import json

    path = os.path.join(folder, '.spacr-parts.json')
    if not os.path.lexists(path):
        return None
    with open(path, 'r', encoding='utf-8') as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or set(manifest) != {
            'version', 'active', 'retired'} or type(manifest['version']) is not int \
            or manifest['version'] != 1:
        raise ValueError(f'Invalid Parquet part manifest: {path}')
    if not manifest['active']:
        raise ValueError(f'Invalid Parquet part manifest: {path}')
    names = set()
    for category in ('active', 'retired'):
        if not isinstance(manifest[category], list):
            raise ValueError(f'Invalid Parquet part manifest: {path}')
        for part in manifest[category]:
            if not isinstance(part, dict) or set(part) != {
                    'name', 'size', 'sha256'}:
                raise ValueError(f'Invalid Parquet part manifest: {path}')
            name = part['name']
            digest = part['sha256']
            if (not isinstance(name, str) or not name.endswith('.parquet')
                    or name in ('', '.', '..') or '/' in name or '\\' in name
                    or name in names or type(part['size']) is not int
                    or part['size'] < 0 or not isinstance(digest, str)
                    or len(digest) != 64
                    or any(c not in '0123456789abcdef' for c in digest)):
                raise ValueError(f'Invalid Parquet part manifest: {path}')
            names.add(name)
            if category == 'active':
                actual = _parquet_part_record(os.path.join(folder, name))
                if actual != part:
                    raise ValueError(f'Parquet part changed outside spaCR: {name}')
    visible = {name for name in os.listdir(folder)
               if name.endswith('.parquet') and not name.startswith('.')}
    if not visible.issubset(names):
        raise ValueError(f'Untracked Parquet part in {folder}')
    return manifest


def _write_parquet_manifest_payload(pending: str, payload: dict):
    """Flush one staged active-snapshot manifest before publication."""
    import json

    with open(pending, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, sort_keys=True, separators=(',', ':'))
        handle.flush()
        os.fsync(handle.fileno())


def _publish_parquet_manifest(folder: str, active: list, retired: list):
    """Publish one active part set with a single atomic file replacement."""
    path = os.path.join(folder, '.spacr-parts.json')
    payload = {'version': 1, 'active': active, 'retired': retired}
    _publish(path, lambda pending: _write_parquet_manifest_payload(pending, payload))


def _store_tables(db: Any, backend: str) -> Tuple[str, ...]:
    """The table names in a DuckDB, Parquet or PostgreSQL store, sorted.

    A DuckDB file and a Parquet folder are read through
    :func:`spacr.store.open`; PostgreSQL, an export target, is asked here.
    """
    if backend in ('parquet', 'duckdb'):
        with _opened_store(db) as opened:
            return tuple(opened.tables())
    query = ("SELECT table_name FROM information_schema.tables "
             "WHERE table_schema = current_schema() "
             "AND table_type = 'BASE TABLE'")
    with _postgres_connect(db) as conn:
        rows = conn.execute(query).fetchall()
    return tuple(sorted(row[0] for row in rows))


def _iter_chunks(db: Any, table: str, *, limit: Optional[int] = None,
                 chunksize: int = 100_000):
    """Yield a table of any store in frames of at most ``chunksize`` rows.

    At least one frame is yielded, empty with the table's columns when the
    table has no rows, so a caller always learns the header. Nothing is
    canonicalised; the store's column names come back as written.

    Every spaCR store (a store project, ``measurements.db``, a ``.duckdb``
    copy or a ``.parquetdb`` folder) is read by :meth:`spacr.store.Store._frames`;
    only PostgreSQL, an export target, is read here.
    """
    if _backend_of(db) != 'postgres':
        with _opened_store(db) as opened:
            if opened is None or table not in opened.tables():
                raise ValueError(f'Table not found in database: {table}')
            yield from opened._frames(table, limit=limit, chunksize=chunksize)
        return
    quoted = _quote_identifier(table)
    query = f'SELECT * FROM {quoted}'
    if limit is not None:
        query += f' LIMIT {int(limit)}'
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
    quoted = _quote_identifier(table)
    if backend == 'parquet':
        import uuid
        _require_optional('pyarrow', _PYARROW_MISSING_MESSAGE)
        if table in ('.', '..') or '/' in table or '\\' in table:
            raise ValueError(f'Invalid Parquet table name: {table!r}')
        store = resolve_path(db)
        with _parquet_store_lock(store):
            folder = _parquet_table_folder(store, table)
            manifest = _parquet_manifest(folder) if os.path.isdir(folder) else None
            previous = _parquet_parts(store, table)
            if previous and if_exists == 'fail':
                raise ValueError(f'Table {table!r} already exists in {db!r}.')
            active = (manifest['active'] if manifest is not None else
                      [_parquet_part_record(part) for part in previous])
            retired = manifest['retired'] if manifest is not None else []
            part = os.path.join(folder, f'.spacr-part-{uuid.uuid4().hex}.parquet')
            published = False
            try:
                _publish(part, lambda pending: frame.to_parquet(pending, index=False))
                new = _parquet_part_record(part)
                if if_exists == 'replace':
                    retired = retired + active
                    active = [new]
                else:
                    active = active + [new]
                _publish_parquet_manifest(folder, active, retired)
                published = True
            finally:
                if not published and os.path.exists(part):
                    os.remove(part)
        return
    present = table in _store_tables(db, backend)
    if present and if_exists == 'fail':
        raise ValueError(f'Table {table!r} already exists in {db!r}.')
    fresh = not present or if_exists == 'replace'
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

    A SQLite file is queried in SQLite's dialect, through the store's SQLite
    connection. A store project, a ``.duckdb`` copy and a ``.parquetdb``
    folder are queried with :meth:`spacr.store.Store.sql` in DuckDB's
    dialect, over views that read only the columns the query names.

    :param db: a SQLite, DuckDB or Parquet path, or a PostgreSQL string.
    :param sql: the query, naming tables as they are stored.
    :param canonicalise: apply the vocabulary to the result.
    :param report: see :func:`read_table`.
    :param warn: see :func:`read_table`.
    :returns: a :class:`pandas.DataFrame`.
    """
    backend = _backend_of(db)
    if backend == 'sqlite' and not os.path.isdir(resolve_path(db)):
        with _connect(db, migrate=False, read_only=True) as conn:
            frame = pd.read_sql_query(sql, conn)
    elif backend != 'postgres':
        with _opened_store(db) as opened:
            frame = opened.sql(sql)
    else:
        with _postgres_connect(db) as conn:
            cursor = conn.execute(sql)
            names = [column.name for column in cursor.description]
            frame = pd.DataFrame.from_records(cursor.fetchall(), columns=names)
    return _canonicalise(frame, canonicalise, report, warn)
