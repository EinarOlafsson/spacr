"""Copy real annotation data and relocate only its image-path references.

The original project is read-only. The prepared database keeps every existing
label and row identity; image bytes are checked before a recording may use it.
Run this, like every Python tutorial tool, through tools/run_capped.sh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

from capture_policy import _PRIVATE_PATH


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def prepare(source: Path, destination: Path, visible: Path):
    source = source.resolve()
    destination = destination.absolute()
    visible = visible.absolute()
    if _PRIVATE_PATH.search(str(visible)):
        raise ValueError('The visible dataset path must be neutral')
    if destination.exists():
        raise FileExistsError('Use a fresh destination to preserve previous evidence')
    if source == destination or source in destination.parents:
        raise ValueError('The copy must be outside the source project')
    database = source / 'measurements/measurements.db'
    with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as original:
        original.execute('BEGIN')
        columns = [r[1] for r in original.execute('PRAGMA table_info(png_list)')]
        if 'png_path' not in columns:
            raise ValueError('The source must contain a real png_list table')
        # SQLite identifiers are case-insensitive: the biological rowID
        # column shadows "rowid". Use the unshadowed internal row identifier.
        if '_rowid_' in {name.lower() for name in columns}:
            raise ValueError('The internal row identifier is shadowed')
        rows = original.execute('SELECT _rowid_, * FROM png_list ORDER BY _rowid_').fetchall()
        if not rows:
            raise ValueError('The annotation source has no crop rows')
        path_index = columns.index('png_path') + 1
        changes, images = [], {}
        for row in rows:
            old = Path(row[path_index])
            relative = old.relative_to(source) if old.is_absolute() else old
            image = (source / relative).resolve()
            if source not in image.parents or image.suffix.lower() != '.png':
                raise ValueError('A crop reference escapes the source project')
            images[relative.as_posix()] = digest(image)
            changes.append((str(visible / relative), row[0]))
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns(
            'measurements.db', 'measurements.db-wal', 'measurements.db-shm'))
        target_database = destination / 'measurements/measurements.db'
        with sqlite3.connect(target_database) as copied:
            original.backup(copied)
            before = copied.execute('SELECT _rowid_, * FROM png_list ORDER BY _rowid_').fetchall()
            if before != rows:
                raise RuntimeError('The database snapshot differs from the source')
            copied.executemany('UPDATE png_list SET png_path=? WHERE _rowid_=?', changes)
            after = copied.execute('SELECT _rowid_, * FROM png_list ORDER BY _rowid_').fetchall()
            unchanged = lambda data: [r[:path_index] + r[path_index + 1:] for r in data]
            if unchanged(before) != unchanged(after):
                raise RuntimeError('Relocating paths changed labels or row identities')
            if [(r[path_index], r[0]) for r in after] != changes:
                raise RuntimeError('Relocating paths did not preserve each crop identity')
        for relative, expected in images.items():
            if digest(destination / relative) != expected or digest(source / relative) != expected:
                raise RuntimeError('Source or copied crop bytes changed')
    receipt = dict(schema=1, accepted=True, source=str(source), destination=str(destination),
                   visible_source=str(visible), rows=len(rows), unique_images=len(images),
                   changed_columns=['png_path'], existing_labels_preserved=True,
                   pixels_preserved=True, source_written=False,
                   image_sha256=images,
                   existing_values_sha256=hashlib.sha256(json.dumps(
                       unchanged(rows), ensure_ascii=False).encode()).hexdigest())
    (destination / 'annotation_capture_copy.json').write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--visible-source', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source, args.destination, args.visible_source)
    print(f"Verified {result['rows']} rows and {result['unique_images']} unchanged images")


if __name__ == '__main__':
    main()
