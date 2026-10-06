"""Verify the actual batch/watch databases with exact metadata exclusions."""

import hashlib
import json
import sqlite3
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
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

changed_apps = subprocess.check_output(
    ['git', 'diff', '25031e230', 'HEAD', '--name-only', '--', 'spacr'], text=True).strip()
assert not changed_apps
report = {
    'base': '25031e230fe21eb12f80bdf3ebae49e080b1eadc',
    'test_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'test_source_sha256': hashlib.sha256(
        Path('tests/test_watch_folder_and_analyse.py').read_bytes()).hexdigest(),
    'app_source_changes': [], 'cuda_visible_devices': '', 'test_memory_cap_gib': 4,
    'initial_broad_tests': {'passed': 97, 'failed_expected_png_directory_metadata': 1},
    'affected_final_tests': {'passed': 2}, 'true_scientific_mismatches': [],
    'metadata_exclusions': METADATA, 'pairs': results,
    'inference_acceptance': 'CPU fixture doubles; actual normalization, Measure and collection',
}
(ROOT / 'scientific-column-audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'test_commit': report['test_commit'], 'pairs': [
    {'mode': row['mode'], 'table': row['table'], 'rows': row['rows'],
     'scientific_columns': row['scientific_columns'],
     'pathogen_columns': len(row['pathogen_columns_checked'])} for row in results]}, indent=2))
