"""Verify the actual batch/watch databases with exact metadata exclusions."""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifact-root', type=Path, required=True,
                    help='Original private pytest database fixture directory')
ROOT = parser.parse_args().artifact_root.resolve()
FROZEN = json.loads(Path(__file__).with_name(
    'scientific-audit-scientific-column-audit.json').read_text())
METADATA = ('file_name', 'path_name', 'png_path')
PAIRS = (
    ('per_field', 'pytest-01/test_files_copied_in_one_by_on0', ('cell', 'nucleus')),
    ('projected_z', 'pytest-01/test_projected_z_watch_matches0', ('cell', 'nucleus')),
    ('native_T1', 'pytest-01/test_native_z_t1_matches_raw_v0', ('cell', 'nucleus')),
    ('mapped_timelapse', 'pytest-01/test_mapped_series_waits_for_f0', ('cell', 'nucleus')),
    ('saved_CV', 'pytest-02/test_real_mask_and_measure_pre0', ('png_list',)),
)


def rows(path, table):
    with sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True) as connection:
        columns = [row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')]
        science = [name for name in columns if name not in METADATA]
        listed = ', '.join(f'"{name}"' for name in science)
        records = connection.execute(f'SELECT {listed} FROM "{table}"').fetchall()
    return science, sorted(records, key=repr), [name for name in columns if name in METADATA]


results = []
for mode, relative, tables in PAIRS:
    case = ROOT / relative
    batch = case / 'batch/measurements/measurements.db'
    watch = case / 'watched/spacr_watch/measurements/measurements.db'
    for table in tables:
        left, right = rows(batch, table), rows(watch, table)
        assert left == right, (mode, table)
        results.append({
            'mode': mode, 'artifact_root': relative, 'table': table,
            'rows': len(left[1]), 'scientific_columns': len(left[0]),
            'pathogen_columns_checked': [name for name in left[0] if 'pathogen' in name],
            'excluded_metadata': left[2], 'all_scientific_columns_and_rows_equal': True,
            'batch_db_sha256': hashlib.sha256(batch.read_bytes()).hexdigest(),
            'watch_db_sha256': hashlib.sha256(watch.read_bytes()).hexdigest(),
        })
        expected = next(pair for pair in FROZEN['pairs']
                        if pair['mode'] == mode and pair['table'] == table)
        assert results[-1] == expected, (mode, table, 'frozen receipt mismatch')

assert list(METADATA) == FROZEN['metadata_exclusions']
assert len(results) == len(FROZEN['pairs']) == 9
print(json.dumps({'frozen_test_commit': FROZEN['test_commit'], 'pairs': [
    {'mode': row['mode'], 'table': row['table'], 'rows': row['rows'],
     'scientific_columns': row['scientific_columns'],
     'pathogen_columns': len(row['pathogen_columns_checked'])} for row in results]}, indent=2))
