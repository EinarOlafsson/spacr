"""Transactional database writes for interactive Image UMAP selections."""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Iterable, Mapping, Sequence, Tuple


def _quoted_identifier(name: str) -> str:
    """Return a safely quoted SQLite identifier.

    :param name: annotation-column name to quote.
    :returns: identifier with embedded quotes escaped for SQLite.
    :raises ValueError: when the stripped name is empty or contains a NUL.
    """
    text = str(name or "").strip()
    if not text or "\x00" in text:
        raise ValueError("Annotation column must be a non-empty SQLite name.")
    return '"' + text.replace('"', '""') + '"'


def write_umap_annotations(
    records: Sequence[Mapping],
    values: Iterable[int],
    column: str,
) -> Tuple[int, int]:
    """Write one integer value per UMAP record into ``png_list``.

    :param records: rows carrying ``db_path`` and original ``db_png_path``.
    :param values: integer annotations aligned one-to-one with ``records``.
    :param column: annotation column to add or update in ``png_list``.

    Records are grouped by database so a multi-plate embedding commits once
    per file.  The original ``png_path`` value from the database is the update
    key; corrected/display paths are deliberately not used.

    :returns: ``(rows_updated, records_skipped)``.
    :raises ValueError: when records and values differ in length or ``column``
        is not a valid SQLite identifier.
    """
    records = list(records)
    values = list(values)
    if len(records) != len(values):
        raise ValueError("records and values must have the same length")
    quoted = _quoted_identifier(column)
    grouped = defaultdict(list)
    skipped = 0
    for record, value in zip(records, values):
        db_path = record.get("db_path")
        png_path = record.get("db_png_path")
        try:
            db_path = os.fspath(db_path)
            png_path = os.fspath(png_path)
        except TypeError:
            skipped += 1
            continue
        if not db_path or not png_path or not os.path.isfile(db_path):
            skipped += 1
            continue
        grouped[db_path].append((int(value), png_path))

    from . import store

    name = quoted[1:-1].replace('""', '"')
    updated = 0
    for db_path, pairs in grouped.items():
        changed = store.edit(
            db_path, [{"key": path, "value": value} for value, path in pairs],
            kind="umap_lasso", source="umap", table="png_list", column=name,
            key_column="png_path", types={name: "INTEGER"}, timeout=30)
        updated += changed
        skipped += max(0, len(pairs) - changed)
    return updated, skipped
