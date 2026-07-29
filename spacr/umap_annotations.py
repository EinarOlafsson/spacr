"""Transactional database writes for interactive Image UMAP selections."""
from __future__ import annotations

import os
import sqlite3
from collections import defaultdict
from typing import Iterable, Mapping, Sequence, Tuple


def _quoted_identifier(name: str) -> str:
    """Return a safely quoted SQLite identifier."""
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

    Records are grouped by database so a multi-plate embedding commits once
    per file.  The original ``png_path`` value from the database is the update
    key; corrected/display paths are deliberately not used.

    :returns: ``(rows_updated, records_skipped)``.
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

    updated = 0
    for db_path, pairs in grouped.items():
        with sqlite3.connect(db_path, timeout=30) as connection:
            present = {
                row[1] for row in
                connection.execute('PRAGMA table_info("png_list")')}
            if not present:
                skipped += len(pairs)
                continue
            if column not in present:
                connection.execute(
                    f'ALTER TABLE "png_list" ADD COLUMN {quoted} INTEGER')
            before = connection.total_changes
            connection.executemany(
                f'UPDATE "png_list" SET {quoted} = ? WHERE png_path = ?',
                pairs,
            )
            changed = connection.total_changes - before
            updated += changed
            skipped += max(0, len(pairs) - changed)
    return updated, skipped
