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
    """
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return None if np.isnan(value) else int(value)
    text = str(value).strip()
    if text[:1] in ('o', 'O'):
        text = text[1:]
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _merged_field_paths(db_path, object_type='cell'):
    """Return ``{(plateID, rowID, columnID, fieldID): (path_name, file_name)}``.

    Read off a measurement table, which is where
    :func:`spacr.utils._merge_and_save_to_database` records the merged array
    each object came from. ``png_list`` records neither, so this is the join
    that lets a ``png_list`` row be cut on demand.

    The requested object's own table is preferred and the other object tables
    are tried in turn, because every one of them names the same field.
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
    :returns: a copy of ``png_df`` with ``path_name``, ``object_label`` and
        ``object_type`` columns, minus the rows that cannot be cut.
    """
    df = png_df.copy()
    id_col = PNG_LIST_ID_COLUMNS.get(object_type, 'cell_id')
    if id_col not in df.columns:
        # A png_list written for one crop mode carries only that mode's id
        # column; fall back to whichever object column it does have.
        for candidate in PNG_LIST_ID_COLUMNS.values():
            if candidate in df.columns:
                id_col = candidate
                break
    if id_col in df.columns:
        labels = df[id_col].map(_object_id_int)
    elif 'object_label' in df.columns:
        # Not a png_list at all: a frame that already came off the object
        # table (crop_rows_from_object_table) carries the integer label
        # directly. Looking up a column that is not there would drop every row.
        labels = df['object_label'].map(_object_id_int)
    else:
        labels = pd.Series([None] * len(df), index=df.index)

    key_cols = ['plateID', 'rowID', 'columnID', 'fieldID']
    if 'path_name' in df.columns and df['path_name'].notna().any():
        pass                    # the frame already names its merged array
    elif all(c in df.columns for c in key_cols):
        # png_list records where a crop was written, never which merged array
        # produced it; the object table is the only place that link exists.
        fields = _merged_field_paths(db_path, object_type)
        keys = list(zip(*(df[c] for c in key_cols)))
        df['path_name'] = [fields.get(k, (None, None))[0] for k in keys]
    else:
        df['path_name'] = None
    df['object_label'] = labels
    df['object_type'] = object_type

    usable = df['object_label'].notna() & df['path_name'].notna()
    dropped = int((~usable).sum())
    if dropped and verbose:
        print(f"crop_rows_from_png_list: {dropped} of {len(df)} png_list rows "
              f"cannot be cut from merged/ (no single object label, or no "
              f"matching row in the '{object_type}' table); they are skipped.")
    return df[usable].copy()
