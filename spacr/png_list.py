"""Resolve crop records stored in the ``png_list`` database table.

The public helper joins ``png_list`` rows to measurement-table locations so
the corresponding object can be cut from ``merged/*.npy``. This lightweight
module avoids importing segmentation or model dependencies; :mod:`spacr.io`
re-exports the helper for compatibility.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd

from .object_roles import ORGANELLE_ROLES

__all__ = ["PNG_LIST_ID_COLUMNS", "crop_rows_from_png_list"]

#: Which ``png_list`` column carries the object id, per crop mode.
PNG_LIST_ID_COLUMNS = {
    'cell': 'cell_id', 'nucleus': 'nucleus_id', 'pathogen': 'pathogen_id',
    'cytoplasm': 'cytoplasm_id',
    **{role: f'{role}_id' for role in ORGANELLE_ROLES},
}


def _object_id_int(value):
    """Return the integer in a ``png_list`` object id (``'o12'`` -> ``12``).

    ``'omulti'`` / ``'onone'`` -- a crop that overlaps several objects or none
    -- have no single label to cut, and come back as None.

    :param value: stored object identifier, optionally prefixed by ``"o"``.
    :returns: exact integer label, or ``None`` for missing, non-integral,
        boolean, non-finite, or non-numeric values.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return (int(number)
                if np.isfinite(number) and number.is_integer() else None)
    text = str(value).strip()
    if text[:1] in ('o', 'O'):
        text = text[1:]
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _crop_join_token(value, *, time=False):
    """Normalize a scalar identity token; missing values never match.

    :param value: stored plate, well, field or time identifier.
    :param time: also accept the canonical ``t`` prefix for timepoints.
    :returns: text key or None for missing/invalid identifiers.
    """
    if value is None or pd.isna(value) or isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return str(int(value)) if np.isfinite(value) and value.is_integer() else None
    text = str(value).strip()
    if time:
        digits = text[1:] if text.startswith('t') else text
        if digits.isascii() and digits.isdigit():
            text = str(int(digits))
    return text or None


def _attach_object_crop_paths(db_path, frame, object_type):
    """Add uniquely identified PNG paths to physical object rows without writes.

    :param db_path: source SQLite database, opened read-only.
    :param frame: measurement rows; their columns, order and index are preserved.
    :param object_type: physical object table with a known crop-mode ID column.
    :returns: frame with missing png_path values filled where the complete
        field/object/time identity has exactly one path; ambiguous or incomplete
        identities remain unmatched. Existing nonempty paths are retained.
    """
    import logging
    from pathlib import Path

    log = logging.getLogger(__name__)
    identity = ['plateID', 'rowID', 'columnID', 'fieldID']
    id_column = PNG_LIST_ID_COLUMNS.get(object_type)
    if (id_column is None or frame.empty
            or not set(identity + ['object_label']).issubset(frame.columns)
            or frame.columns.duplicated().any()):
        return frame
    if ('png_path' in frame
            and frame['png_path'].fillna('').astype(str).str.strip().ne('').all()):
        return frame
    uri = Path(db_path).resolve().as_uri() + '?mode=ro'
    with sqlite3.connect(uri, uri=True) as db:
        columns = {row[1] for row in db.execute('PRAGMA table_info("png_list")')}
        if not columns:
            return frame
        required = identity + [id_column, 'png_path']
        if not set(required).issubset(columns):
            log.info('Crop review: png_list lacks the complete %s crop identity',
                     object_type)
            return frame
        times = [name for name in ('timeID', 'time_id') if name in columns]
        selected = required + times
        crops = pd.read_sql_query(
            'SELECT ' + ', '.join('"' + name + '"' for name in selected)
            + ' FROM "png_list"', db)
    left_times = [name for name in ('timeID', 'time_id') if name in frame]
    if bool(left_times) != bool(times):
        log.warning('Crop review: measurement and crop timepoint identities '
                    'differ; paths not attached')
        return frame
    left = pd.DataFrame({key: frame[key].map(_crop_join_token).to_numpy()
                         for key in identity})
    right = pd.DataFrame({key: crops[key].map(_crop_join_token).to_numpy()
                          for key in identity})
    left['object_label'] = frame['object_label'].map(_object_id_int).to_numpy()
    right['object_label'] = crops[id_column].map(_object_id_int).to_numpy()
    left['object_label'] = left['object_label'].where(left['object_label'] > 0)
    right['object_label'] = right['object_label'].where(right['object_label'] > 0)
    if times:
        for original, names, key_frame in (
                (frame, left_times, left), (crops, times, right)):
            normalized = [original[name].map(
                lambda value: _crop_join_token(value, time=True)) for name in names]
            if len(normalized) == 2 and not normalized[0].equals(normalized[1]):
                log.warning('Crop review: conflicting timepoint aliases; '
                            'paths not attached')
                return frame
            key_frame['timeID'] = normalized[0].to_numpy()
    keys = list(left.columns)
    right['png_path'] = crops['png_path'].to_numpy()
    right = right.dropna(subset=keys + ['png_path'])
    right = right[right['png_path'].astype(str).str.strip().ne('')]
    right = right.drop_duplicates(subset=keys + ['png_path'])
    ambiguous = right.duplicated(subset=keys, keep=False)
    if ambiguous.any():
        log.warning('Crop review: %d crop rows have conflicting paths; '
                    'ambiguous objects remain unmatched',
                    int(ambiguous.sum()))
    unique = right.loc[~ambiguous]
    mapping = dict(zip(unique[keys].itertuples(index=False, name=None),
                       unique['png_path']))
    paths = [mapping.get(key) if all(pd.notna(value) for value in key) else None
             for key in left.itertuples(index=False, name=None)]
    if not any(path is not None for path in paths):
        return frame
    result = frame.copy(deep=False)
    if 'png_path' in frame:
        old = frame['png_path'].tolist()
        paths = [previous if pd.notna(previous) and str(previous).strip() else path
                 for previous, path in zip(old, paths)]
    result['png_path'] = paths
    return result


def _merged_field_paths(db_path, object_type='cell'):
    """Return ``{(plateID, rowID, columnID, fieldID): (path_name, file_name)}``.

    Read off a measurement table, which is where
    :func:`spacr.utils._merge_and_save_to_database` records the merged array
    each object came from. ``png_list`` records neither, so this is the join
    that lets a ``png_list`` row be cut on demand.

    The requested object's own table is preferred and the other object tables
    are tried in turn, because every one of them names the same field.

    :param db_path: measurement database to inspect without creating it.
    :param object_type: preferred measurement table for resolving field paths.
    :returns: field identifiers mapped to merged-array directory and filename.
    """
    out = {}
    if not os.path.isfile(db_path):
        return out
    order = [object_type] + [t for t in ('cell', 'cytoplasm', 'nucleus',
                                         'pathogen', 'organelle')
                             if t != object_type]
    from .database_concurrency import connect as _connect_database

    conn = _connect_database(db_path)
    try:
        for table in order:
            try:
                rows = conn.execute(
                    f'SELECT DISTINCT plateID, rowID, columnID, fieldID, '
                    f'path_name, file_name FROM "{table}"').fetchall()
            except sqlite3.Error:
                continue
            for plate, row, col, field, path_name, file_name in rows:
                out.setdefault((plate, row, col, field), (path_name, file_name))
            if out:
                break
    finally:
        conn.close()
    return out


def crop_rows_from_png_list(db_path, png_df, object_type='cell', verbose=True):
    """Add the locations and labels required to cut ``png_list`` objects.

    ``png_list`` records where a crop was *written* and which object it came
    from (``<object>_id``), but not which merged array produced it. This joins
    the object table on plate/row/column/field to recover ``path_name``, and
    turns ``'o12'`` into ``12``.

    Rows whose object id is ``'omulti'`` / ``'onone'`` (a crop overlapping
    several objects or none) cannot be cut from a single label and are
    dropped, with a count, rather than silently producing the wrong object.

    :param db_path: path to the ``measurements.db`` that contains ``png_df``.
    :param png_df: rows read from ``png_list`` or a compatible object table.
    :param object_type: crop mode used to select the object-id column. The
        default is ``'cell'``; supported names are the keys of
        :data:`PNG_LIST_ID_COLUMNS`.
    :param verbose: print the number of unusable rows when ``True``.
    :returns: a copy of ``png_df`` with ``path_name``, ``object_label``,
        ``object_type`` and ``object_label_type`` columns, minus the rows
        that cannot be cut. ``object_type`` is what was ASKED for and is what
        the crop cutter reads to choose a mask plane; ``object_label_type``
        is which object's labels were actually available, and the two differ
        when a png_list written for one crop mode is read for another.
    :raises ValueError: if ``object_type`` is unsupported, or if its ID column
        is absent while multiple other object-ID columns make fallback
        ambiguous.
    """
    df = png_df.copy()
    try:
        id_col = PNG_LIST_ID_COLUMNS[object_type]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"object_type must be one of {sorted(PNG_LIST_ID_COLUMNS)}; "
            f"got {object_type!r}") from exc
    effective_object_type = object_type
    if id_col not in df.columns:
        alternatives = [
            (mode, candidate)
            for mode, candidate in PNG_LIST_ID_COLUMNS.items()
            if candidate in df.columns
        ]
        if len(alternatives) > 1:
            raise ValueError(
                f"{object_type!r} needs {id_col!r}, but this frame carries "
                f"multiple alternate object ID columns: "
                f"{sorted(column for _mode, column in alternatives)}")
        if alternatives:
            effective_object_type, id_col = alternatives[0]
    if id_col in df.columns:
        labels = df[id_col].map(_object_id_int)
    elif 'object_label' in df.columns:
        labels = df['object_label'].map(_object_id_int)
    else:
        labels = pd.Series([None] * len(df), index=df.index)

    key_cols = ['plateID', 'rowID', 'columnID', 'fieldID']
    if 'path_name' in df.columns and df['path_name'].notna().any():
        pass
    elif all(c in df.columns for c in key_cols):
        fields = _merged_field_paths(db_path, effective_object_type)
        keys = list(zip(*(df[c] for c in key_cols)))
        df['path_name'] = [fields.get(k, (None, None))[0] for k in keys]
    else:
        df['path_name'] = None
    df['object_label'] = labels
    df['object_type'] = object_type
    df['object_label_type'] = effective_object_type

    usable = df['object_label'].notna() & df['path_name'].notna()
    dropped = int((~usable).sum())
    if dropped and verbose:
        print(f"crop_rows_from_png_list: {dropped} of {len(df)} png_list rows "
              f"cannot be cut from merged/ (no single object label, or no "
              f"matching row in the '{effective_object_type}' table); they "
              f"are skipped.")
    return df[usable].copy()
