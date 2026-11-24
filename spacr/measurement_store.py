"""Measure's and Make Masks' tables in the spaCR store (unified store, step S4).

Measure, Make Masks and the steps after them save their tables in the
project's spaCR store, ``<project>/measurements/``, which is the folder that
held ``measurements.db`` before. Nothing here is a second writer or reader:
every table is written by :func:`spacr.store.write_table` and
:class:`spacr.store.WellWriter` and read by :func:`spacr.store.open`. This
module only says which partition a row belongs to and how the tables that
used to be upserted in SQLite are merged.

How a Measure run is written:

* each field's rows go to the run's spool store, ``measurements/_spool/
  measure/``, one part per table and field (the partition ``well`` is
  ``<well>@<field>``), with a ``measure_fields`` row written last, which is
  the field's commit mark for a resume;
* when the run ends, :func:`compact` writes every well of every table once
  into the project store with the streamed writer, in ``object_key`` order,
  with the recorded per-column dtype map, registers it with the run's
  settings and removes the spool.

Tables keyed by field rather than by object (``image``, ``intensity_rescale``,
``confluency``, ``object_counts``, ``settings``) keep the upsert meaning they
had in SQLite: a later row for the same key replaces the earlier one column
by column, and the well's part is rewritten (the store replaces a well's
earlier part when the new one is registered).

A project that still holds a legacy ``measurements.db`` keeps reading it
through the store; :func:`retire_legacy_database` copies its tables into the
store and moves the file to ``measurements/archive/`` when a new run starts
there.
"""
from __future__ import annotations

import contextlib
import contextvars
import datetime
import os
import re
import shutil
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

#: Nothing here is public API: it is spaCR's internal vocabulary over
#: :mod:`spacr.store` (whose writer and reader are the public ones) until the
#: edit log and the Database Browser's store backend (steps S5 and S6).
__all__: List[str] = []

#: The run spool's folder inside the measurements folder.
SPOOL = os.path.join('_spool', 'measure')

#: The table whose row marks a field's spool parts complete.
FIELDS_TABLE = 'measure_fields'

#: The legacy SQLite file name.
LEGACY_NAME = 'measurements.db'

#: The partition of rows that belong to no plate (settings, run records).
RUN_PARTITION = ('_run', '_run')

#: Tables merged by key (an SQLite upsert before): table -> key columns.
UPSERT_KEYS: Dict[str, tuple] = {
    'image': ('file_name',),
    'fov': ('file_name',),
    'intensity_rescale': ('prcf',),
    'confluency': ('prcf',),
    'object_counts': ('file_name', 'count_type'),
}

#: Tables replaced whole by each write.
REPLACE_TABLES = frozenset({'settings'})

#: Columns every object table returns with a feature query.
KEY_COLUMNS = ('prcfo', 'prcf', 'plateID', 'rowID', 'columnID', 'fieldID',
               'timeID', 'object_label', 'object_key', 'file_name',
               'condition', 'crop_format')

#: The open field packet of this thread: table -> list of frames.
_PACKET: contextvars.ContextVar = contextvars.ContextVar(
    'spacr_measurement_store_packet', default=None)


def project_of(db_path: Any) -> str:
    """The store project of a measurements locator.

    :param db_path: a ``measurements.db`` path (it need not exist), the
        measurements folder, or a run folder holding ``measurements/``.
    :returns: the measurements folder.
    """
    path = os.path.abspath(os.path.expanduser(os.fspath(db_path)))
    if os.path.basename(path) == LEGACY_NAME or path.endswith('.db'):
        return os.path.dirname(path)
    if os.path.basename(path) == 'measurements':
        return path
    if os.path.isdir(os.path.join(path, 'measurements')):
        return os.path.join(path, 'measurements')
    return path


def _file(db_path: Any) -> Optional[str]:
    """``db_path`` itself when it is an existing database file (any name)."""
    try:
        path = os.path.abspath(os.path.expanduser(os.fspath(db_path)))
    except TypeError:
        return None
    if os.path.basename(path) == 'spacr.duckdb':
        return None
    if os.path.basename(path) == LEGACY_NAME:
        return legacy_database(path)
    return path if os.path.isfile(path) else None


def legacy_database(db_path: Any) -> Optional[str]:
    """The legacy ``measurements.db`` of a locator, when it exists.

    :param db_path: as for :func:`project_of`.
    :returns: the file's path, or None.
    """
    path = os.path.join(project_of(db_path), LEGACY_NAME)
    if not os.path.isfile(path):
        return None
    if (os.path.getsize(path) == 0 or (os.path.isfile(os.path.join(
            os.path.dirname(path), 'spacr.duckdb')) and _empty_sqlite(path))):
        return None
    return path


def _empty_sqlite(path: str) -> bool:
    """Whether a SQLite file holds no table (a file a stray connect created)."""
    import sqlite3

    try:
        with contextlib.closing(sqlite3.connect(
                f'file:{path}?mode=ro', uri=True, timeout=5)) as conn:
            return conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE "
                                "type = 'table'").fetchone()[0] == 0
    except sqlite3.Error:
        return False


def store_native(db_path: Any) -> bool:
    """Whether a project's tables are written to the store.

    A project is a store project when its measurements folder holds the
    catalog (``spacr.duckdb``) and no legacy ``measurements.db``. Measure
    and Make Masks make every project they write one
    (:func:`ensure_store`); a project that still holds a legacy file keeps
    it until :func:`retire_legacy_database` moves it aside.

    :param db_path: as for :func:`project_of`.
    """
    from . import store

    return (legacy_database(db_path) is None and os.path.isfile(
        os.path.join(project_of(db_path), store.CATALOG)))


def ensure_store(db_path: Any, *, retire: bool = True) -> str:
    """Make a project a store project: the catalog, and no legacy file.

    A legacy ``measurements.db`` is copied into the store and moved to
    ``archive/`` first (when ``retire``); then the catalog is created if it
    is missing. Measure and Make Masks call this before their first write.

    :param db_path: the project's ``measurements.db`` path or folder.
    :param retire: retire a legacy file; False leaves it (a resumed legacy
        run keeps writing it).
    :returns: the measurements folder.
    """
    from . import store

    project = project_of(db_path)
    if legacy_database(db_path) is not None:
        if not retire:
            return project
        retire_legacy_database(db_path)
    os.makedirs(project, exist_ok=True)
    if not os.path.isfile(os.path.join(project, store.CATALOG)):
        store.register(project)
    return project


def spool_of(db_path: Any) -> str:
    """The run spool store of a project."""
    return os.path.join(project_of(db_path), SPOOL)


def _text(value: Any) -> str:
    """A partition value as text; a missing one becomes ``_``."""
    if value is None:
        return '_'
    try:
        import pandas as pd

        if pd.isna(value):
            return '_'
    except (TypeError, ValueError):
        pass
    text = str(value)
    return text if text else '_'


def _partitions(frame) -> List[tuple]:
    """``(plate, well, rows)`` of a frame, in sorted partition order.

    The plate is ``plateID`` and the well ``rowID`` + ``columnID``; a frame
    without them is read from ``prcfo``/``prcf``/``file_name``
    (``plate_row_column_...``), and a frame with none of these is one run
    partition.
    """
    import numpy as np

    columns = set(frame.columns)
    if {'plateID', 'rowID', 'columnID'} <= columns:
        plates = (frame['plate'] if 'plate' in columns
                  else frame['plateID']).map(_text)
        wells = (frame['well'].map(_text) if 'well' in columns else
                 frame['rowID'].map(_text) + frame['columnID'].map(_text))
    elif {'plate', 'well'} <= columns:
        plates = frame['plate'].map(_text)
        wells = frame['well'].map(_text)
    else:
        source = next((c for c in ('prcfo', 'prcf', 'file_name')
                       if c in columns), None)
        if source is None and not ({'plate', 'well'} & columns):
            return [(RUN_PARTITION[0], RUN_PARTITION[1],
                     np.arange(len(frame)))]
        from . import schema

        plates, wells = [], []
        for value in (frame[source] if source is not None
                      else [None] * len(frame)):
            if value is None:
                plates.append(RUN_PARTITION[0])
                wells.append(RUN_PARTITION[1])
                continue
            try:
                field = schema.parse_field_stem(str(value).split('_o')[0]
                                                if source == 'prcfo'
                                                else value)
                plates.append(_text(field.plateID))
                wells.append(_text(field.rowID) + _text(field.columnID))
            except Exception:
                parts = str(value).split('_')
                plates.append(_text(parts[0] if parts else None))
                wells.append(''.join(parts[1:3]) or '_')
        import pandas as pd

        plates = pd.Series(plates, index=frame.index)
        wells = pd.Series(wells, index=frame.index)
    if 'plate' in columns:
        plates = frame['plate'].map(_text)
    if 'well' in columns:
        wells = frame['well'].map(_text)
    keys = list(zip(plates.astype(str), wells.astype(str)))
    groups: Dict[tuple, List[int]] = {}
    for position, key in enumerate(keys):
        groups.setdefault(key, []).append(position)
    return [(plate, well, np.asarray(groups[(plate, well)], dtype=np.int64))
            for plate, well in sorted(groups)]


def object_keys(prcfo: Sequence[Any]):
    """The ``object_key`` of each ``prcfo``: one integer per object.

    :param prcfo: the objects' text keys.
    :returns: an int64 array (``ops_run_infra.integer_cell_key``), with
        ``2**62 + position`` where a key does not parse.
    """
    from .measure import _measure_object_keys

    return _measure_object_keys(list(prcfo))


def _with_keys(table: str, frame):
    """``frame`` with an ``object_key`` when it has ``prcfo`` and lacks one."""
    if 'prcfo' in frame.columns and 'object_key' not in frame.columns:
        frame = frame.copy()
        frame['object_key'] = object_keys(frame['prcfo'].astype(str))
    return frame


#: Columns a table may carry that the store keeps for its crop pointer;
#: they are stored under these names and given back under their own.
POINTER_RENAMES = {'crop_store': 'pool_store', 'crop_index': 'pool_index'}

#: Tables whose own ``crop_store``/``crop_index`` columns are kept renamed.
POINTER_TABLES = frozenset({'ssCell_Crops'})


def _stored_names(table: str, frame):
    """``frame`` with a crop table's own pointer columns renamed for the store."""
    if table in POINTER_TABLES:
        present = {k: v for k, v in POINTER_RENAMES.items() if k in frame.columns}
        if present:
            frame = frame.rename(columns=present)
    return frame


def _given_names(table: str, frame):
    """A crop table read from the store with its pointer columns' own names."""
    if table in POINTER_TABLES:
        back = {v: k for k, v in POINTER_RENAMES.items() if v in frame.columns}
        if back:
            frame = frame.drop(columns=[k for k in back.values()
                                        if k in frame.columns]).rename(columns=back)
    return frame


def _clean(frame):
    """A frame whose object columns hold one Python type each.

    SQLite stored whatever a column held; Parquet needs one type per
    column, so a column mixing numbers and text is stored as text.
    """
    import pandas as pd

    frame = frame.reset_index(drop=True)
    for name in frame.columns:
        column = frame[name]
        if column.dtype != object:
            continue
        kinds = {type(v) for v in column if v is not None
                 and not (isinstance(v, float) and v != v)}
        if len(kinds) > 1 and not kinds <= {int, float, bool}:
            frame[name] = column.map(
                lambda v: None if v is None or (isinstance(v, float)
                                                and v != v) else str(v))
        elif kinds and kinds <= {bytes}:
            continue
        elif not kinds:
            frame[name] = pd.Series([None] * len(frame), dtype=object)
    return frame


@contextlib.contextmanager
def field_packet():
    """Collect one field's table writes, to spool them together.

    Inside the block :func:`append` adds frames to the packet instead of
    writing them; :func:`spool_field` writes the packet.

    :yields: the packet, ``{table: [frames]}``.
    """
    packet: Dict[str, List[Any]] = {}
    token = _PACKET.set(packet)
    try:
        yield packet
    finally:
        _PACKET.reset(token)


def in_packet() -> bool:
    """Whether a field packet is being collected in this thread."""
    return _PACKET.get() is not None


def append(db_path: Any, table: str, frame, *, run_id: Optional[str] = None):
    """Add rows to a project table.

    In a field packet the rows wait for :func:`spool_field`. Otherwise each
    well's part is rewritten with the rows added (an upsert table merges
    them by key, a replace table is replaced) and registered.

    :param db_path: the project's ``measurements.db`` path or folder.
    :param table: the table.
    :param frame: the rows (a DataFrame).
    :param run_id: the run the rows belong to.
    :returns: the number of rows given.
    """
    if frame is None or len(frame) == 0:
        return 0
    packet = _PACKET.get()
    if packet is not None:
        packet.setdefault(table, []).append(frame.copy())
        return len(frame)
    write_rows(project_of(db_path), table, frame, run_id=run_id)
    return len(frame)


def _merge_upsert(old, new, keys: Sequence[str]):
    """``old`` with the rows of ``new`` merged in by ``keys``.

    A key present in both keeps one row; each column takes the new value
    where the new row has one and the old value otherwise, as the SQLite
    upserts did. Row order is the old order, then new keys in their order.
    """
    import pandas as pd

    if old is None or len(old) == 0:
        frame = new
    else:
        frame = pd.concat([old, new], ignore_index=True, sort=False)
    keys = [k for k in keys if k in frame.columns]
    if not keys:
        return frame.reset_index(drop=True)
    ident = frame[keys].astype(str).agg('\x1f'.join, axis=1)
    frame = frame.assign(_upsert_key=ident.to_numpy())
    merged = frame.groupby('_upsert_key', sort=False, dropna=False).last()
    first = frame.drop_duplicates('_upsert_key')['_upsert_key']
    merged = merged.loc[first.to_numpy()].reset_index(drop=True)
    return merged[[c for c in frame.columns if c != '_upsert_key']]


def _sort_by(frame) -> tuple:
    """The store sort order for a frame: ``object_key`` then ``timeID``."""
    return tuple(c for c in ('object_key', 'timeID') if c in frame.columns)


def _key_columns(frame) -> tuple:
    """The key columns a frame holds."""
    return tuple(c for c in KEY_COLUMNS if c in frame.columns)


def _registered(project: str, table: str):
    """The registered wells of a table: ``[(plate, well)]``, or None when absent."""
    from . import store

    if not os.path.isfile(os.path.join(project, store.CATALOG)):
        return None
    with store.open(project, verify='none') as opened:
        if table not in opened.tables():
            return None
        found = opened.sql(f'SELECT DISTINCT plate, well FROM "{table}__keys"'
                           if f'{table}__keys' in opened.tables() else
                           f'SELECT DISTINCT plate, well FROM "{table}"')
    return sorted(set(zip(found['plate'].astype(str), found['well'].astype(str))))


def _existing_wells(project: str, table: str, wells=None) -> Dict[tuple, Any]:
    """Registered rows of a table per well, as Arrow tables (stored names).

    :param wells: the ``(plate, well)`` pairs to read; None reads every well.
    """
    from . import store

    out: Dict[tuple, Any] = {}
    if not os.path.isfile(os.path.join(project, store.CATALOG)):
        return out
    with store.open(project, verify='none') as opened:
        if table not in opened.tables():
            return out
        if wells is None:
            found = opened.sql(f'SELECT DISTINCT plate, well FROM "{table}"')
            wells = sorted(set(zip(found['plate'].astype(str),
                                   found['well'].astype(str))))
        for plate, well in wells:
            data = opened.sql(f'SELECT * FROM "{table}" WHERE plate = ? AND '
                              f'well = ?', [plate, well], as_='arrow')
            if not data.num_rows:
                continue
            drop = [c for c in ('crop_store', 'crop_index')
                    if c in data.column_names]
            if table not in POINTER_TABLES:
                drop += [c for c in ('plate', 'well') if c in data.column_names]
            out[(plate, well)] = data.drop_columns(drop)
    return out


def _existing_well(project: str, table: str, plate: str, well: str):
    """The registered rows of one well of a table as a DataFrame, or None."""
    found = _existing_wells(project, table, [(plate, well)])
    data = found.get((plate, well))
    return None if data is None else data.to_pandas()


def _arrow(frame):
    """A DataFrame (or Arrow table) as an Arrow table without its index."""
    import pyarrow as pa

    if isinstance(frame, pa.Table):
        return frame
    return pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(None)


def _ranges(parts) -> Dict[str, tuple]:
    """``column -> (min, max)`` over several Arrow tables of one schema."""
    import pyarrow as pa
    import pyarrow.compute as pc

    ranges: Dict[str, tuple] = {}
    if not parts:
        return ranges
    schema = parts[0].schema
    for field in schema:
        kind = field.type
        if not (pa.types.is_floating(kind) or pa.types.is_integer(kind)):
            continue
        low = high = None
        for part in parts:
            column = part.column(field.name)
            if pa.types.is_floating(kind):
                column = pc.if_else(pc.is_nan(column), pa.scalar(None, kind),
                                    column)
            stats = pc.min_max(column).as_py()
            if stats['min'] is None:
                continue
            low = stats['min'] if low is None else min(low, stats['min'])
            high = stats['max'] if high is None else max(high, stats['max'])
        ranges[field.name] = ((float('nan'), float('nan')) if low is None
                              else (low, high))
    return ranges


def _unified(parts):
    """One schema for Arrow tables: columns in first-seen order, types promoted."""
    import pyarrow as pa

    schema = pa.unify_schemas([p.schema for p in parts],
                              promote_options='permissive')
    return pa.schema([pa.field(f.name, pa.string() if pa.types.is_null(f.type)
                               else f.type) for f in schema])


def rewrite_table(project: str, table: str, wells: Mapping[tuple, Any], *,
                  run_id: Optional[str] = None, narrow: bool = False,
                  channels: Sequence[str] = (), register: bool = True,
                  sort: bool = True) -> int:
    """Write wells of a table with ONE schema, through the store's writer.

    The store holds one schema per table (step S0), so every well written
    together gets the same columns and types: the union of the parts in
    first-seen order, promoted where they differ, and with ``narrow`` the
    recorded per-column dtypes guarded by the ranges over all of them. A
    table whose columns changed must be rewritten whole (every well).

    :param project: the measurements folder.
    :param table: the table.
    :param wells: ``{(plate, well): DataFrame or Arrow table or [them]}``.
    :param run_id: the run.
    :param narrow: apply the recorded dtype map.
    :param channels: the run's channel names, for the dtype families.
    :param register: register afterwards.
    :param sort: sort each well's rows by ``object_key`` (and ``timeID``).
    :returns: the rows written.
    """
    from . import store

    batches = {}
    for key in sorted(wells):
        value = wells[key]
        items = value if isinstance(value, (list, tuple)) else [value]
        arrows = [_arrow(_clean(v) if not hasattr(v, 'schema') else v)
                  for v in items if v is not None and len(v)]
        batches[key] = [a.drop_columns([c for c in ('plate', 'well')
                                        if c in a.column_names])
                        for a in arrows]
    flat = [b for items in batches.values() for b in items]
    if not flat:
        return 0
    schema = _unified(flat)
    batches = {k: [_conform(b, schema) for b in v] for k, v in batches.items()}
    ranges = _ranges([b for v in batches.values() for b in v]) if narrow else None
    names = set(schema.names)
    sort_by = tuple(c for c in ('object_key', 'timeID') if c in names)
    keys = tuple(c for c in KEY_COLUMNS if c in names)
    rows = 0
    for (plate, well), items in batches.items():
        if not items:
            continue
        if sort and sort_by:
            import pyarrow as pa
            import pyarrow.compute as pc

            data = pa.concat_tables(items)
            order = pc.sort_indices(data, sort_keys=[(c, 'ascending')
                                                     for c in sort_by])
            items = [data.take(order)]
        with store.writer(project, table, plate=plate, well=well,
                          run_id=run_id, sort_by=sort_by, key_columns=keys,
                          schema=schema, ranges=ranges, narrow=narrow,
                          channels=channels) as part:
            for batch in items:
                part.append(batch)
        rows += part.rows
    if register:
        store.register(project)
    return rows


def _schema_fits(project: str, table: str, parts) -> bool:
    """Whether new parts can join a registered table without a rewrite.

    True when the table is absent, or when every column of the parts is a
    column of the table with the same stored type.
    """
    from . import store

    if not os.path.isfile(os.path.join(project, store.CATALOG)):
        return True
    with store.open(project, verify='none') as opened:
        if table not in opened.tables():
            return True
        columns = opened.columns(table)
    stored = dict(zip(columns['name'], columns['stored_type'].astype(str)))
    for part in parts:
        for field in part.schema:
            if field.name in ('plate', 'well'):
                continue
            if field.name not in stored or str(field.type) != stored[field.name]:
                return False
    return True


def write_rows(project: str, table: str, frame, *, run_id: Optional[str] = None,
               narrow: bool = False, register: bool = True) -> Dict[str, Any]:
    """Write a frame into a project table, well by well, and register it.

    Each touched well's part is rewritten with its earlier rows plus these
    (merged by key for an upsert table; a replace table is replaced). When
    the rows bring a column or a type the table does not hold, every well is
    rewritten with the widened schema (one schema per table).

    :param project: the measurements folder.
    :param table: the table.
    :param frame: the rows.
    :param run_id: the run they belong to.
    :param narrow: apply the recorded dtype map.
    :param register: register the parts afterwards.
    :returns: ``{"wells": n, "rows": n}``.
    """
    import pandas as pd

    from . import store

    frame = _clean(_stored_names(table, _with_keys(table, frame)))
    touched = [(p, w, frame.iloc[r]) for p, w, r in _partitions(frame)]
    if table in REPLACE_TABLES:
        drop_tables(project, [table])
        old = {}
    else:
        old = _existing_wells(project, table, [(p, w) for p, w, _f in touched])
    new = {}
    for plate, well, part in touched:
        before = old.get((plate, well))
        before = None if before is None else before.to_pandas()
        if table in UPSERT_KEYS:
            part = _merge_upsert(before, part, UPSERT_KEYS[table])
        elif before is not None:
            part = pd.concat([before, part], ignore_index=True, sort=False)
        new[(plate, well)] = _arrow(_clean(part))
    if not _schema_fits(project, table, list(new.values())):
        everything = _existing_wells(project, table)
        everything.update(new)
        new = everything
    rewrite_table(project, table, new, run_id=run_id, narrow=narrow,
                  register=register)
    return {'wells': len(touched), 'rows': int(len(frame))}


def _spool_table(table: str, stem: str) -> str:
    """The spool's table of one field's rows of ``table``."""
    return f'{table}~{stem}'


def spool_field(db_path: Any, field: str, packet: Mapping[str, List[Any]], *,
                ticket: str, run_id: Optional[str] = None,
                timelapse: bool = False) -> Dict[str, int]:
    """Write one field's packet to the run spool, its commit mark last.

    Each table's rows of the field are their own spool table
    (``<table>~<field>``), since fields may differ in their columns and a
    store table has one schema; :func:`compact` unifies them.

    :param db_path: the project's ``measurements.db`` path or folder.
    :param field: the field's file name.
    :param packet: ``{table: [frames]}`` from :func:`field_packet`.
    :param ticket: the packet's deterministic ticket.
    :param run_id: the run.
    :param timelapse: parse a timepoint from the field name.
    :returns: rows written per table.
    """
    import pandas as pd

    from . import schema, store

    spool = spool_of(db_path)
    os.makedirs(spool, exist_ok=True)
    stem = os.path.splitext(os.path.basename(str(field)))[0]
    try:
        identity = schema.parse_field_stem(stem, timelapse=timelapse)
        plate = _text(identity.plateID)
        well = _text(identity.rowID) + _text(identity.columnID)
    except Exception:
        plate, well = '_', '_'
    written: Dict[str, int] = {}
    for table in sorted(packet):
        frames = [f for f in packet[table] if f is not None and len(f)]
        if not frames:
            continue
        frame = (frames[0] if len(frames) == 1
                 else pd.concat(frames, ignore_index=True, sort=False))
        if table in UPSERT_KEYS:
            frame = _merge_upsert(None, frame, UPSERT_KEYS[table])
        frame = _clean(_with_keys(table, frame))
        groups = _partitions(frame)
        if len(groups) == 1 and groups[0][:2] == RUN_PARTITION:
            groups = [(plate, well, groups[0][2])]
        for part_plate, part_well, rows in groups:
            part = frame.iloc[rows]
            store.write_table(spool, _spool_table(table, stem), part,
                              plate=part_plate, well=part_well, run_id=run_id,
                              sort_by=_sort_by(part),
                              key_columns=_key_columns(part))
        written[table] = int(len(frame))
    mark = pd.DataFrame([{'ticket': str(ticket), 'field': str(field),
                          'stem': stem, 'tables': ','.join(sorted(written)),
                          'run_id': str(run_id)}])
    store.write_table(spool, FIELDS_TABLE, mark, plate=plate,
                      well=f'{well}@{stem}', run_id=run_id, sort_by=())
    return written


def spooled_fields(db_path: Any) -> List[str]:
    """The fields whose spool parts are complete (their mark is written).

    The spool's waiting manifests are registered first, so a run killed
    after a field's mark keeps that field.

    :param db_path: the project's ``measurements.db`` path or folder.
    :returns: the field file names.
    """
    from . import store

    spool = spool_of(db_path)
    if not os.path.isdir(spool):
        return []
    store.register(spool)
    if not os.path.isfile(os.path.join(spool, store.CATALOG)):
        return []
    with store.open(spool, verify='none') as opened:
        if FIELDS_TABLE not in opened.tables():
            return []
        frame = opened.query(table=FIELDS_TABLE, features=['field'])
    return sorted(set(frame['field'].astype(str)))


def measured_fields(db_path: Any) -> List[str]:
    """The fields a Measure run has committed, compacted or still spooled.

    :param db_path: the project's ``measurements.db`` path or folder.
    :returns: the field file names.
    """
    from . import store

    done = set(spooled_fields(db_path))
    project = project_of(db_path)
    if os.path.isfile(os.path.join(project, store.CATALOG)):
        with store.open(project, verify='none') as opened:
            if FIELDS_TABLE in opened.tables():
                frame = opened.query(table=FIELDS_TABLE, features=['field'])
                done |= set(frame['field'].astype(str))
    return sorted(done)


def _field_order(stem: str):
    """Sort key of a field stem: field number, then time, then the stem."""
    from . import schema

    stem = stem.split('@', 1)[-1]
    try:
        identity = schema.parse_field_stem(stem, timelapse=stem.count('_') >= 3)
        field = int(str(identity.fieldID).lstrip('f') or 0)
        time = identity.timeID
        time = int(str(time).lstrip('t')) if time not in (None, '') else -1
        return (field, time, stem)
    except Exception:
        return (1 << 62, -1, stem)


def compact(db_path: Any, *, run_id: Optional[str] = None,
            settings: Optional[Mapping[str, Any]] = None,
            channels: Sequence[str] = ()) -> Dict[str, Any]:
    """Write the run spool into the project store, one part per well.

    Each table is written once, all its wells with one schema (the union of
    its fields' columns) and, for the object tables, the recorded per-column
    dtypes guarded by the ranges over the whole table; rows stay in
    ``object_key`` order. Wells the project already holds keep their earlier
    rows, and an upsert table is merged by key. The project is registered
    with the run and the spool removed.

    :param db_path: the project's ``measurements.db`` path or folder.
    :param run_id: the run.
    :param settings: the run settings, recorded with the run.
    :param channels: the run's channel names, for the dtype families.
    :returns: ``{"tables": {table: rows}, "registered": ...}``.
    """
    import pandas as pd

    from . import store

    project = project_of(db_path)
    spool = spool_of(db_path)
    report: Dict[str, Any] = {'project': project, 'tables': {}}
    if not os.path.isdir(spool):
        return report
    store.register(spool)
    if not os.path.isfile(os.path.join(spool, store.CATALOG)):
        discard_spool(db_path)
        return report
    with store.open(spool, verify='none') as opened:
        grouped: Dict[str, List[str]] = {}
        for name in opened.tables():
            if name.endswith('__keys') or name.endswith('__rows') or '~' not in name:
                continue
            table, stem = name.split('~', 1)
            grouped.setdefault(table, []).append(stem)
        if FIELDS_TABLE in opened.tables():
            marks = opened.query(table=FIELDS_TABLE).drop(
                columns=['plate', 'well'], errors='ignore')
            write_rows(project, FIELDS_TABLE, marks, run_id=run_id,
                       register=False)
            report['tables'][FIELDS_TABLE] = int(len(marks))
        for table in sorted(grouped):
            stems = sorted(grouped[table], key=_field_order)
            wells: Dict[tuple, List[Any]] = {}
            for stem in stems:
                data = opened.sql(f'SELECT * FROM "{_spool_table(table, stem)}"',
                                  as_='arrow')
                pairs = sorted(set(zip(map(str, data.column('plate').to_pylist()),
                                       map(str, data.column('well').to_pylist()))))
                for key in pairs:
                    if len(pairs) == 1:
                        chunk = data
                    else:
                        import pyarrow.compute as pc
                        chunk = data.filter(pc.and_(
                            pc.equal(data.column('plate'), key[0]),
                            pc.equal(data.column('well'), key[1])))
                    wells.setdefault(key, []).append(chunk.drop_columns(
                        [c for c in ('plate', 'well') if c in chunk.column_names]))
            if table in UPSERT_KEYS or table in REPLACE_TABLES:
                frame = pd.concat([b.to_pandas() for items in wells.values()
                                   for b in items], ignore_index=True, sort=False)
                write_rows(project, table, frame, run_id=run_id,
                           register=False)
                report['tables'][table] = int(len(frame))
                continue
            old = _existing_wells(project, table)
            for key, data in old.items():
                wells.setdefault(key, []).insert(0, data)
            report['tables'][table] = rewrite_table(
                project, table, wells, run_id=run_id, narrow=True,
                channels=channels, register=False, sort=False)
    run = None
    if run_id is not None:
        run = {'run_id': run_id, 'settings': dict(settings or {})}
    report['registered'] = store.register(project, run=run)
    discard_spool(db_path)
    return report


def _conform(data, schema):
    """An Arrow table cast to ``schema``, a missing column as nulls."""
    import pyarrow as pa

    arrays = []
    for field in schema:
        if field.name in data.column_names:
            column = data.column(field.name)
            if column.type != field.type:
                column = column.cast(field.type)
            arrays.append(column)
        else:
            arrays.append(pa.nulls(data.num_rows, field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def discard_spool(db_path: Any) -> None:
    """Remove a run spool (written, or one a fresh run will not continue)."""
    spool = spool_of(db_path)
    shutil.rmtree(spool, ignore_errors=True)
    parent = os.path.dirname(spool)
    try:
        if os.path.isdir(parent) and not os.listdir(parent):
            os.rmdir(parent)
    except OSError:
        pass


def read(db_path: Any, table: str, *, where=None, features=None):
    """One project table as a DataFrame, through the store's reader.

    The store's partition columns ``plate`` and ``well`` (and an empty crop
    pointer) are dropped when the table carries its own ``plateID``, so the
    frame has the columns its writer gave it.

    :param db_path: the project's ``measurements.db`` path or folder.
    :param table: the table.
    :param where: as for :meth:`spacr.store.Store.query`.
    :param features: as for :meth:`spacr.store.Store.query`.
    :returns: the rows, or an empty frame when the table is absent.
    """
    import pandas as pd

    from . import store

    project = project_of(db_path)
    source = _file(db_path) or legacy_database(db_path) or project
    try:
        opened = store.open(source, verify='none')
    except store.StoreError:
        return pd.DataFrame()
    with opened:
        if table not in opened.tables():
            return pd.DataFrame()
        if opened.legacy:
            frame = next(iter([pd.concat(list(opened._frames(table)),
                                         ignore_index=True)]))
            return frame
        frame = opened.query(where, table=table, features=features)
    return _given_names(table, legacy_shape(frame, table))


def legacy_shape(frame, table=None):
    """A store frame with the columns its writer gave it.

    Drops ``plate``/``well`` when ``plateID`` is present (or when they are
    the run partition), an all-empty crop pointer and ``object_key`` (the
    store's integer key, which a reader written for measurements.db would
    take for a feature).
    """
    if table in POINTER_TABLES:
        return frame
    drop = [c for c in ('object_key',) if c in frame.columns]
    drop += _partition_only(frame)
    for column in ('crop_store', 'crop_index'):
        if column in frame.columns and frame[column].isna().all():
            drop.append(column)
    return frame.drop(columns=drop) if drop else frame


def _partition_only(frame) -> List[str]:
    """The ``plate``/``well`` columns of a store frame that only repeat its keys.

    A table that carried its own ``plate`` or ``well`` (a value other than
    ``plateID``, or ``rowID`` + ``columnID``) keeps it; the store's partition
    columns that merely repeat those keys, or name the run partition, go.
    """
    out = []
    if not len(frame):
        return [c for c in ('plate', 'well') if c in frame.columns]
    if 'plate' in frame.columns:
        plate = frame['plate'].astype(str)
        if ((plate == RUN_PARTITION[0]).all() or ('plateID' in frame.columns
                and (plate == frame['plateID'].map(_text)).all())):
            out.append('plate')
    if 'well' in frame.columns:
        well = frame['well'].astype(str)
        if ((well == RUN_PARTITION[1]).all() or ({'rowID', 'columnID'}
                <= set(frame.columns) and (well == frame['rowID'].map(_text)
                                           + frame['columnID'].map(_text)).all())):
            out.append('well')
    return out


def tables(db_path: Any) -> List[str]:
    """The tables a project holds (store or legacy), without key tables."""
    from . import store

    source = _file(db_path) or legacy_database(db_path) or project_of(db_path)
    try:
        with store.open(source, verify='none') as opened:
            return [t for t in opened.tables() if not t.endswith('__keys')
                    and not t.endswith('__rows')]
    except store.StoreError:
        return []


def drop_tables(db_path: Any, names: Iterable[str]) -> List[str]:
    """Remove tables from a project store (a rebuilt table's old rows).

    :param db_path: the project's ``measurements.db`` path or folder.
    :param names: the tables.
    :returns: the removed paths.
    """
    from . import store

    return store._drop_tables(project_of(db_path), list(names))


def archive_tables(db_path: Any, names: Iterable[str]) -> Dict[str, str]:
    """Keep an earlier run's tables under ``archive/`` and drop them.

    The parts are copied to ``measurements/archive/<stamp>/tables/`` before
    the tables leave the catalog, so a fresh run never mixes populations
    and the earlier one stays readable as Parquet.

    :param db_path: the project's ``measurements.db`` path or folder.
    :param names: the tables to archive.
    :returns: ``{table: archive folder}``.
    """
    project = project_of(db_path)
    present = set(tables(db_path))
    names = sorted(set(names) & present)
    if not names:
        return {}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        '%Y%m%dT%H%M%S%fZ')
    target = os.path.join(project, 'archive', stamp)
    moved = {}
    for name in names:
        for kind in ('tables', 'rows'):
            for suffix in ('', '__keys', '__rows'):
                source = os.path.join(project, kind, name + suffix)
                if os.path.isdir(source):
                    destination = os.path.join(target, kind, name + suffix)
                    os.makedirs(os.path.dirname(destination), exist_ok=True)
                    shutil.copytree(source, destination)
        moved[name] = os.path.join(target, 'tables', name)
    drop_tables(db_path, names)
    return moved


def retire_legacy_database(db_path: Any, *, keep: Iterable[str] = ()) -> Optional[str]:
    """Copy a legacy ``measurements.db`` into the store and move it aside.

    Every table but those a fresh run rebuilds is copied into the project
    store through :func:`write_rows`; the file then moves to
    ``measurements/archive/measurements-<stamp>.db``, where the store no
    longer reads it.

    :param db_path: the project's ``measurements.db`` path or folder.
    :param keep: the tables not to copy (a fresh run rebuilds them; the
        archived file still holds them).
    :returns: the archived file, or None when there was none.
    """
    legacy = legacy_database(db_path)
    if legacy is None:
        return None
    from . import store
    from .tabular import read_table

    project = project_of(db_path)
    skip = set(keep)
    with store.open(legacy, verify='none') as opened:
        names = [t for t in opened.tables() if t not in skip
                 and not t.startswith('sqlite_')
                 and not t.startswith('_spacr_write_queue')]
    for name in names:
        frame = read_table(legacy, table=name, canonicalise=False, report=None)
        if len(frame):
            write_rows(project, name, frame, register=False)
    store.register(project)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        '%Y%m%dT%H%M%S%fZ')
    archive = os.path.join(project, 'archive')
    os.makedirs(archive, exist_ok=True)
    target = os.path.join(archive, f'measurements-{stamp}.db')
    for suffix in ('', '-wal', '-shm', '-journal'):
        if os.path.exists(legacy + suffix):
            os.replace(legacy + suffix, target + suffix)
    print(f'[store] {LEGACY_NAME} copied into the spaCR store ({len(names)} '
          f'tables) and kept as {target}.')
    return target


# ---------------------------------------------------------------------------
# SQL over a store project, for the readers and the annotation writers that
# were written against measurements.db.
# ---------------------------------------------------------------------------

#: Tables a connection holds in memory, so the annotation writers can add
#: and update their columns; a commit rewrites the edited wells' parts.
#: (A minimal store-backed path until the edit log of step S5.)
EDITABLE_PREFIXES = ('png_list', 'ssCell_Crops')

#: The most values (rows x columns) a view is held in memory for, when SQL
#: addresses its rows by ``rowid``.
ROWID_VALUES = 400_000_000

_ROWID = __import__('re').compile(r'\b(rowid|_rowid_|oid)\b', __import__('re').I)

_WRITE_TARGET = __import__('re').compile(
    r'\s*(?:UPDATE|INSERT\s+(?:OR\s+\w+\s+)?INTO|REPLACE\s+INTO|DELETE\s+FROM|'
    r'ALTER\s+TABLE|DROP\s+TABLE(?:\s+IF\s+EXISTS)?)\s+'
    r'(?P<name>"(?:[^"]|"")+"|[\w.]+)', __import__('re').I)

_SESSIONS: Dict[tuple, Any] = {}
_SESSION_LOCK = __import__('threading').RLock()


def exists(db_path: Any) -> bool:
    """Whether a project holds measurements: a legacy file or a store catalog.

    :param db_path: as for :func:`project_of`.
    """
    from . import store

    if _file(db_path) is not None or legacy_database(db_path) is not None:
        return True
    return os.path.isfile(os.path.join(project_of(db_path), store.CATALOG))


def _editable(name: str) -> bool:
    """Whether a table is one the annotation writers edit."""
    return any(name == p or name.startswith(p + '_') for p in EDITABLE_PREFIXES)


class _Session:
    """One in-memory DuckDB database over a store project's views.

    Shared by every connection of the process to that project (each gets
    its own cursor), rebuilt when the catalog changes on disk.
    """

    def __init__(self, project: str):
        """Open the store and make the SQL objects the old readers expect."""
        from . import store

        self.project = project
        self.store = store.open(project, verify='none')
        self.db = self.store._conn
        self.lock = __import__('threading').RLock()
        self.tables = [t for t in self.store.tables()
                       if not t.endswith('__keys') and not t.endswith('__rows')]
        self.editable = [t for t in self.tables if _editable(t)]
        self.materialized = set()
        for name in self.tables:
            quoted = '"' + name.replace('"', '""') + '"'
            source = '"' + ('__store_' + name).replace('"', '""') + '"'
            self.db.execute(f'ALTER VIEW {quoted} RENAME TO {source}')
            columns = [r[0] for r in self.db.execute(
                f'DESCRIBE SELECT * FROM {source}').fetchall()]
            hidden = ['object_key']
            if name not in self.editable and 'plate' in columns:
                sample = self.db.execute(
                    f'SELECT * FROM {source} LIMIT 256').df()
                hidden += _partition_only(sample)
            if name not in POINTER_TABLES:
                hidden += ['crop_store', 'crop_index']
            hidden = [c for c in hidden if c in columns]
            select = ('SELECT * EXCLUDE (' + ', '.join(
                f'"{c}"' for c in hidden) + ')') if hidden else 'SELECT *'
            kind = 'TABLE' if name in self.editable else 'VIEW'
            self.db.execute(f'CREATE {kind} {quoted} AS {select} FROM {source}')
        for name in self.editable:
            quoted = '"' + name.replace('"', '""') + '"'
            back = {v: k for k, v in POINTER_RENAMES.items()}
            if name in POINTER_TABLES:
                for stored, given in back.items():
                    try:
                        self.db.execute(f'ALTER TABLE {quoted} RENAME COLUMN '
                                        f'"{stored}" TO "{given}"')
                    except Exception:
                        pass
            self.db.execute(f'CREATE TABLE "__shadow_{name}" AS '
                            f'SELECT * FROM {quoted}')
        self._master()

    def materialize(self, name: str) -> None:
        """Hold a view's rows in memory, so SQL may address them by ``rowid``.

        :raises sqlite3.OperationalError: when the table holds more values
            than :data:`ROWID_VALUES` (the Database Browser's paged store
            reader, step S6, serves those).
        """
        import sqlite3

        with self.lock:
            if name in self.materialized or name in self.editable:
                return
            quoted = '"' + name.replace('"', '""') + '"'
            rows = self.db.execute(f'SELECT COUNT(*) FROM {quoted}').fetchone()[0]
            width = len(self.db.execute(
                f'DESCRIBE SELECT * FROM {quoted}').fetchall())
            if rows * max(width, 1) > ROWID_VALUES:
                raise sqlite3.OperationalError(
                    f'{name} has {rows} rows x {width} columns, too many to '
                    f'address by rowid in memory; read it through '
                    f'spacr.store.open instead.')
            self.db.execute(f'CREATE TABLE "__rowid_{name}" AS SELECT * FROM '
                            f'{quoted}')
            self.db.execute(f'DROP VIEW {quoted}')
            self.db.execute(f'ALTER TABLE "__rowid_{name}" RENAME TO {quoted}')
            self.materialized.add(name)

    def _master(self) -> None:
        """A ``sqlite_master`` listing the project's tables as tables."""
        names = sorted(set(self.tables) | set(self._created()))
        rows = ' UNION ALL '.join(
            f"SELECT 'table' AS type, {_sql_text(n)} AS name, "
            f"{_sql_text(n)} AS tbl_name, 0 AS rootpage, "
            f"NULL::VARCHAR AS sql" for n in names) or (
            "SELECT 'table' AS type, NULL::VARCHAR AS name, NULL::VARCHAR AS "
            "tbl_name, 0 AS rootpage, NULL::VARCHAR AS sql WHERE false")
        self.db.execute(f'CREATE OR REPLACE VIEW sqlite_master AS {rows}')

    def _created(self) -> List[str]:
        """Tables a connection created in this session (not yet stored)."""
        found = self.db.execute(
            "SELECT table_name FROM information_schema.tables WHERE "
            "table_schema = 'main' AND table_type = 'BASE TABLE'").fetchall()
        return [r[0] for r in found if not r[0].startswith('__shadow_')
                and not r[0].startswith('__store_')
                and not r[0].startswith('__rowid_')
                and r[0] not in self.materialized]

    def persist(self) -> int:
        """Rewrite the wells whose rows a connection changed; register them.

        :returns: the number of wells rewritten.
        """
        from . import store

        written = 0
        with self.lock:
            for name in list(self._created()):
                quoted = '"' + name.replace('"', '""') + '"'
                shadow = '"__shadow_' + name.replace('"', '""') + '"'
                if name not in self.tables:
                    written += self._persist_new(name, quoted, shadow)
                    continue
                columns = [r[1] for r in self.db.execute(
                    f'PRAGMA table_info({quoted})').fetchall()]
                have = {r[1] for r in self.db.execute(
                    f'PRAGMA table_info({shadow})').fetchall()}
                listed = ', '.join('"' + c.replace('"', '""') + '"'
                                   for c in columns)
                old = ', '.join(('"' + c.replace('"', '""') + '"') if c in have
                                else f'NULL AS "{c}"' for c in columns)
                if 'plate' not in columns or 'well' not in columns:
                    changed = self.db.execute(
                        f'SELECT COUNT(*) FROM (SELECT {listed} FROM {quoted} '
                        f'EXCEPT ALL SELECT {old} FROM {shadow})').fetchone()[0]
                    changed += self.db.execute(
                        f'SELECT COUNT(*) FROM (SELECT {old} FROM {shadow} '
                        f'EXCEPT ALL SELECT {listed} FROM {quoted})').fetchone()[0]
                    if changed:
                        written += self._persist_new(name, quoted, shadow,
                                                     new=False)
                    continue
                dirty = self.db.execute(
                    f'SELECT DISTINCT plate, well FROM (SELECT {listed} FROM '
                    f'{quoted} EXCEPT ALL SELECT {old} FROM {shadow})'
                ).fetchall()
                dirty += self.db.execute(
                    f'SELECT DISTINCT plate, well FROM (SELECT {old} FROM '
                    f'{shadow} EXCEPT ALL SELECT {listed} FROM {quoted})'
                ).fetchall()
                if dirty and set(columns) != have:
                    dirty = [tuple(r) for r in self.db.execute(
                        f'SELECT DISTINCT plate, well FROM {quoted}').fetchall()]
                parts = {}
                for plate, well in sorted(set(dirty), key=str):
                    parts[(str(plate), str(well))] = self._part(
                        name, quoted, plate, well)
                if parts and not _schema_fits(self.project, name,
                                              list(parts.values())):
                    for plate, well in self.db.execute(
                            f'SELECT DISTINCT plate, well FROM {quoted}'
                    ).fetchall():
                        key = (str(plate), str(well))
                        if key not in parts:
                            parts[key] = self._part(name, quoted, plate, well)
                if parts:
                    rewrite_table(self.project, name, parts, register=False)
                    written += len(parts)
                if dirty:
                    self.db.execute(f'DROP TABLE {shadow}')
                    self.db.execute(f'CREATE TABLE {shadow} AS SELECT * FROM '
                                    f'{quoted}')
            if written:
                store.register(self.project)
                _SESSIONS.pop(self.key, None)
                _SESSIONS[self._key()] = self
        return written

    def _persist_new(self, name: str, quoted: str, shadow: str, *,
                     new: bool = True) -> int:
        """Store a table a connection created (a generated crop set, say).

        Its wells are its ``plate``/``well`` columns, or derived from its
        keys as for :func:`write_rows`; it becomes editable from then on.
        """
        frame = self.store.sql(f'SELECT * FROM {quoted}')
        if frame.empty:
            return 0
        if {'plate', 'well'} <= set(frame.columns):
            groups = {(str(p), str(w)): g for (p, w), g in frame.groupby(
                [frame['plate'].astype(str), frame['well'].astype(str)])}
        else:
            groups = {(p, w): frame.iloc[r] for p, w, r in _partitions(frame)}
        parts = {k: _stored_names(name, _with_keys(name, g))
                 for k, g in groups.items()}
        if not new:
            drop_tables(self.project, [name])
        rewrite_table(self.project, name, parts, register=False)
        if new:
            self.tables.append(name)
            self.editable.append(name)
        self.db.execute(f'CREATE OR REPLACE TABLE {shadow} AS SELECT * FROM '
                        f'{quoted}')
        return len(parts)

    def _part(self, name: str, quoted: str, plate, well):
        """One well of an in-memory table as the Arrow table the store keeps."""
        import pyarrow as pa

        data = self.store.sql(
            f'SELECT * FROM {quoted} WHERE plate IS NOT DISTINCT FROM ? AND '
            f'well IS NOT DISTINCT FROM ?', [plate, well], as_='arrow')
        if 'prcfo' in data.column_names and 'object_key' not in data.column_names:
            data = data.append_column('object_key', pa.array(
                object_keys([str(v) for v in data.column('prcfo').to_pylist()]),
                pa.int64()))
        if name in POINTER_TABLES:
            data = data.rename_columns([POINTER_RENAMES.get(c, c)
                                        for c in data.column_names])
        else:
            data = data.drop_columns([c for c in ('crop_store', 'crop_index')
                                      if c in data.column_names])
        return data

    def _key(self) -> tuple:
        """This session's cache key: the project and its catalog's stamp."""
        self.key = _catalog_key(self.project)
        return self.key


def _sql_text(value: str) -> str:
    """A SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def _catalog_key(project: str) -> tuple:
    """``(project, catalog mtime, size)``."""
    from . import store

    path = os.path.join(project, store.CATALOG)
    stat = os.stat(path)
    return (project, stat.st_mtime_ns, stat.st_size)


def _session(project: str) -> _Session:
    """The process's session over ``project``, rebuilt when its catalog changed."""
    with _SESSION_LOCK:
        key = _catalog_key(project)
        session = _SESSIONS.get(key)
        if session is None:
            for old in [k for k in _SESSIONS if k[0] == project]:
                _SESSIONS.pop(old, None)
            session = _Session(project)
            session.key = key
            _SESSIONS[key] = session
        return session


def forget_sessions() -> None:
    """Close and drop every cached SQL session.

    Measure calls this before it forks its workers: a forked child must not
    inherit an open DuckDB database (its threads do not survive the fork).
    """
    with _SESSION_LOCK:
        for session in list(_SESSIONS.values()):
            try:
                session.store.close()
            except Exception:
                pass
        _SESSIONS.clear()


class StoreCursor:
    """A DB-API cursor over a :class:`StoreConnection`, SQLite-shaped."""

    def __init__(self, connection: 'StoreConnection'):
        """Bind to the connection's DuckDB cursor."""
        self.connection = connection
        self._cursor = connection._db
        self.rowcount = -1
        self.lastrowid = None
        self.description = None

    def execute(self, sql: str, parameters=()):
        """Run one statement; a change marks the connection dirty."""
        self.connection._run(self, sql, parameters)
        return self

    def executemany(self, sql: str, sequence):
        """Run one statement per parameter row."""
        rows = [tuple(r) for r in sequence]
        if not rows:
            return self
        self.connection._run(self, sql, rows, many=True)
        return self

    def fetchone(self):
        """The next row, or None."""
        return self.connection._fetch(self._cursor.fetchone())

    def fetchall(self):
        """Every remaining row."""
        return [self.connection._fetch(r) for r in self._cursor.fetchall()]

    def fetchmany(self, size: int = 1):
        """Up to ``size`` rows."""
        return [self.connection._fetch(r) for r in self._cursor.fetchmany(size)]

    def __iter__(self):
        """Iterate over the remaining rows."""
        return iter(self.fetchall())

    def close(self) -> None:
        """Nothing to release; the connection owns the cursor."""


class StoreConnection(__import__('sqlite3').Connection):
    """A SQLite-shaped connection to a store project (read, and annotation edits).

    What it is for: the readers and annotation writers that were written
    against ``measurements.db`` keep their SQL while the tables live in the
    spaCR store. Object tables are the store's views (read-only); the crop
    tables (``png_list*``, ``ssCell_Crops``) are held in memory, so
    ``ALTER TABLE ... ADD COLUMN`` and ``UPDATE`` work, and :meth:`commit`
    rewrites the edited wells through :func:`spacr.store.write_table`.

    :param db_path: the project's ``measurements.db`` path (absent) or folder.
    :param readonly: refuse writes.
    """

    in_transaction = False

    def __init__(self, db_path: Any, *, readonly: bool = False):
        """Open (or reuse) the project's session and take a cursor.

        The class is a :class:`sqlite3.Connection` (over an empty in-memory
        database it never uses) so pandas and spaCR's own checks treat it
        as one; every statement goes to the store's DuckDB views.
        """
        super().__init__(':memory:', isolation_level=None,
                         check_same_thread=False)
        self.project = project_of(db_path)
        self.readonly = readonly
        self._session = _session(self.project)
        self._db = self._session.db.cursor()
        self._row_factory = None
        self._changes = 0
        self._dirty = False
        self._closed = False

    @property
    def row_factory(self):
        """The row factory, as on :class:`sqlite3.Connection`."""
        return self._row_factory

    @row_factory.setter
    def row_factory(self, value):
        """Set the row factory."""
        self._row_factory = value

    @property
    def total_changes(self):
        """Rows changed through this connection."""
        return self._changes

    def cursor(self) -> StoreCursor:
        """A cursor."""
        return StoreCursor(self)

    def _fetch(self, row):
        """Apply ``row_factory`` to one row."""
        import sqlite3

        if row is None or self._row_factory is None:
            return row
        names = [d[0] for d in (self._db.description or ())]
        if self._row_factory is sqlite3.Row:
            return _Row(names, row)
        return self._row_factory(_RowCursor(self._db.description), row)

    def _run(self, cursor, sql: str, parameters, many: bool = False):
        """Run SQL through DuckDB, translating what SQLite spells differently."""
        import sqlite3

        import duckdb

        text = str(sql).strip()
        head = text.split(None, 1)[0].upper() if text else ''
        if head in ('BEGIN', 'COMMIT', 'END', 'ROLLBACK', 'SAVEPOINT',
                    'RELEASE'):
            if head in ('COMMIT', 'END'):
                self.commit()
            return
        if head == 'PRAGMA':
            lowered = text.lower()
            if 'table_info' not in lowered and 'table_xinfo' not in lowered:
                cursor.description = None
                self._db.execute('SELECT 0 WHERE false')
                return
        if head == 'VACUUM' or (head == 'CREATE' and ' INDEX ' in
                                f' {text.upper()} '):
            return
        if _ROWID.search(text):
            for name in self._session.tables:
                if name in self._session.editable:
                    continue
                if re.search(r'(?<![\w"])"?' + re.escape(name) + r'"?(?![\w"])',
                             text):
                    self._session.materialize(name)
        writes = head in ('INSERT', 'UPDATE', 'DELETE', 'ALTER', 'CREATE',
                          'DROP', 'REPLACE')
        if writes and self.readonly:
            raise sqlite3.OperationalError(
                'attempt to write a readonly database')
        target = _WRITE_TARGET.match(text)
        if target is not None:
            name = target.group('name').strip('"')
            if (name in self._session.tables
                    and name not in self._session.editable):
                raise sqlite3.OperationalError(
                    f'{name} is a measurement table of the spaCR store and '
                    f'is read-only here; only the crop tables (png_list, '
                    f'ssCell_Crops) take edits until the edit log (step S5).')
        try:
            with self._session.lock:
                if many:
                    self._db.executemany(text, parameters)
                    count = len(parameters)
                else:
                    self._db.execute(text, list(parameters or ()))
                    count = -1
        except duckdb.Error as error:
            raise sqlite3.OperationalError(str(error)) from error
        cursor.description = self._db.description
        cursor.rowcount = count
        if writes:
            self._dirty = True
            self._changes += max(count, 1)
            if head in ('CREATE', 'DROP', 'ALTER'):
                self._session._master()
        return cursor

    def execute(self, sql: str, parameters=()):
        """Run one statement and return its cursor."""
        cursor = StoreCursor(self)
        return cursor.execute(sql, parameters)

    def executemany(self, sql: str, sequence):
        """Run one statement per parameter row."""
        cursor = StoreCursor(self)
        return cursor.executemany(sql, sequence)

    def commit(self) -> None:
        """Write the edited wells into the store."""
        if self._dirty:
            self._session.persist()
            self._dirty = False

    def rollback(self) -> None:
        """Nothing is undone in memory; the store keeps its last commit."""
        self._dirty = False

    def close(self) -> None:
        """Commit pending edits and release the cursor."""
        if self._closed:
            return
        try:
            self.commit()
        finally:
            self._closed = True
            try:
                self._db.close()
            except Exception:
                pass
            super().close()

    def __enter__(self):
        """The connection, for ``with``."""
        return self

    def __exit__(self, kind, value, trace):
        """Commit on success, as ``sqlite3.Connection`` does; never close."""
        if kind is None:
            self.commit()
        return False


class _Row(tuple):
    """A row read by index or by column name, as ``sqlite3.Row`` is."""

    def __new__(cls, names, values):
        """Keep the values and their column names."""
        row = super().__new__(cls, values)
        row._names = list(names)
        return row

    def __getitem__(self, key):
        """A value by position or by column name."""
        if isinstance(key, str):
            return tuple.__getitem__(self, self._names.index(key))
        return tuple.__getitem__(self, key)

    def keys(self):
        """The column names."""
        return list(self._names)


class _RowCursor:
    """The ``description`` a ``row_factory`` such as ``sqlite3.Row`` reads."""

    def __init__(self, description):
        """Keep the column description."""
        self.description = description


def connect(db_path: Any, *, readonly: bool = False):
    """A connection to a project's measurements: SQLite for a legacy file,
    :class:`StoreConnection` for a store project.

    :param db_path: the project's ``measurements.db`` path or folder.
    :param readonly: open read-only.
    :returns: the connection.
    """
    legacy = _file(db_path) or legacy_database(db_path)
    if legacy is not None:
        from .database_concurrency import connect as _sqlite

        return _sqlite(legacy, readonly=readonly)
    return StoreConnection(db_path, readonly=readonly)
