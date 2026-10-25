"""Native NTFS sharing and console encoding around saved pipeline settings."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(os.name != 'nt', reason='native Windows sharing/console behavior')


def _run_tree(args, folder, timeout, **options):
    """Run ``args`` with output in files; on timeout kill it and every process it started.

    With pipes, a pool worker that outlives a timed-out child keeps them open
    and ``subprocess.run`` then waits for it forever on Windows.
    """
    import psutil

    out_path, err_path = folder / 'child-stdout.bin', folder / 'child-stderr.bin'
    with out_path.open('wb') as out, err_path.open('wb') as err:
        process = subprocess.Popen(args, stdout=out, stderr=err, **options)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            tree = [process.pid]
            try:
                tree += [child.pid for child in psutil.Process(process.pid).children(recursive=True)]
            except psutil.Error:
                pass
            for pid in reversed(tree):
                try:
                    psutil.Process(pid).kill()
                except psutil.Error:
                    pass
            process.wait()
            pytest.fail(f'child did not finish within {timeout} s:\n'
                        + err_path.read_bytes().decode('cp1252', errors='replace')[-6000:])
    return subprocess.CompletedProcess(args, process.returncode,
                                       out_path.read_bytes(), err_path.read_bytes())


def test_native_locked_csv_preserves_both_snapshots_and_recovers(tmp_path, capsys):
    from spacr.utils import load_settings, save_settings

    folder = tmp_path / 'settings with spaces'
    original = {'src': str(folder), 'value': 1, 'channels': [0, 1]}
    replacement = {**original, 'value': 2}
    save_settings(original, name='sample')
    target = folder / 'settings/sample.csv'
    sibling = target.with_suffix('.json')
    before = target.read_bytes(), sibling.read_bytes()
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateFileW(str(target), 0x80000000, 1, None, 3, 0x80, None)
    assert handle not in (None, ctypes.c_void_p(-1).value), ctypes.WinError(ctypes.get_last_error())
    capsys.readouterr()
    try:
        save_settings(replacement, name='sample')
        assert 'Warning: could not save settings' in capsys.readouterr().out
        assert (target.read_bytes(), sibling.read_bytes()) == before
        assert sorted(path.name for path in target.parent.iterdir()) == ['sample.csv', 'sample.json']
    finally:
        assert kernel.CloseHandle(handle)
    save_settings(replacement, name='sample')
    assert load_settings(target)['value'] == 2
    assert json.loads(sibling.read_text(encoding='utf-8'))['value'] == 2
    assert sorted(path.name for path in target.parent.iterdir()) == ['sample.csv', 'sample.json']


def test_native_cp1252_console_preserves_unicode_settings_and_path(tmp_path):
    folder = tmp_path / 'settings space \u0394ku80 \u00b5m \u7a7a\u767d'
    profile = tmp_path / 'profile'
    code = r'''
import json, sys
import ctypes
from ctypes import wintypes
from pathlib import Path
from spacr.utils import load_settings, save_settings
assert sys.stdout.encoding.lower() == 'cp1252' and sys.stdout.errors == 'strict'
root = Path(sys.argv[1])
settings = {'src': str(root), 'strain': '\u0394ku80', 'units': '\u00b5m',
            'notes': '\u7a7a\u767d with spaces', 'channels': [0, 1], 'value': 3}
save_settings(settings, name='sample', show=True)
target = root / 'settings/sample.csv'
assert load_settings(target) == settings
assert json.loads(target.with_suffix('.json').read_text(encoding='utf-8')) == settings
assert sorted(path.name for path in target.parent.iterdir()) == ['sample.csv', 'sample.json']
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                              wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
kernel.CreateFileW.restype = wintypes.HANDLE
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
changed = {**settings, 'value': 4}
before = target.read_bytes(), target.with_suffix('.json').read_bytes()
handle = kernel.CreateFileW(str(target), 0x80000000, 1, None, 3, 0x80, None)
assert handle not in (None, ctypes.c_void_p(-1).value)
try:
    save_settings(changed, name='sample', show=True)
    assert (target.read_bytes(), target.with_suffix('.json').read_bytes()) == before
finally:
    assert kernel.CloseHandle(handle)
handle = kernel.CreateFileW(str(target.with_suffix('.json')), 0x80000000, 1, None, 3, 0x80, None)
assert handle not in (None, ctypes.c_void_p(-1).value)
try:
    save_settings(changed, name='sample', show=True)
    assert load_settings(target) == changed
    assert target.with_suffix('.json').read_bytes() == before[1]
finally:
    assert kernel.CloseHandle(handle)
save_settings(changed, name='sample', show=True)
assert load_settings(target) == changed
assert json.loads(target.with_suffix('.json').read_text(encoding='utf-8')) == changed
assert sorted(path.name for path in target.parent.iterdir()) == ['sample.csv', 'sample.json']
print('N693_NATIVE_CP1252_UNICODE_PASS')
'''
    environment = os.environ.copy()
    environment.update(PYTHONIOENCODING='cp1252:strict', PYTHONUTF8='0',
                       SPACR_HOME=str(profile / '.spacr'), SPACR_ISOLATED_PROFILE='1',
                       USERPROFILE=str(profile), HOME=str(profile),
                       APPDATA=str(profile / 'roaming'), LOCALAPPDATA=str(profile / 'local'))
    process = _run_tree([sys.executable, '-c', code, str(folder)], tmp_path, 60,
                        cwd=Path(__file__).resolve().parents[1], env=environment)
    assert process.returncode == 0, process.stderr.decode('cp1252', errors='replace')
    assert b'N693_NATIVE_CP1252_UNICODE_PASS' in process.stdout
    assert b'Warning: could not save settings' in process.stdout
    assert b'Warning: could not write' in process.stdout
    assert 'UnicodeEncodeError' not in process.stderr.decode('cp1252', errors='replace')

def _assert_native_report_lock_recovery(db_path):
    import csv
    from spacr.measure import _emit_infection_report

    assert sys.stdout.encoding.lower() == 'cp1252' and sys.stdout.errors == 'strict'
    target = Path(db_path).with_name('infection_report.csv')
    before = target.read_bytes()
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateFileW(str(target), 0x80000000, 1, None, 3, 0x80, None)
    assert handle not in (None, ctypes.c_void_p(-1).value), ctypes.WinError(ctypes.get_last_error())
    try:
        _emit_infection_report(str(db_path))
        assert target.read_bytes() == before
    finally:
        assert kernel.CloseHandle(handle)
    _emit_infection_report(str(db_path))
    assert target.read_bytes() == before
    with target.open(encoding='utf-8', newline='') as stream:
        assert list(csv.DictReader(stream))
    assert not target.with_name('.infection_report.csv.tmp').exists()
    print('N693_NATIVE_CP1252_REPORT_LOCK_RECOVERY_PASS')


def test_native_cp1252_measure_unicode_source_writes_expected_objects(tmp_path, synth_masks_multi, rng):
    import numpy as np
    from tests.test_measure_crop_core_synth import _build_merged_stack, _settings_for, _write_stack

    folder = tmp_path / 'measure space \u0394ku80 \u00b5m \u7a7a\u767d'
    data = _build_merged_stack(synth_masks_multi, rng)
    merged, _ = _write_stack(folder, data)
    settings = _settings_for(merged, n_jobs=1, save_png=False, save_arrays=False,
                             plot=False, measure_gpu=False)
    settings_file = tmp_path / 'measure-settings.json'
    settings_file.write_text(json.dumps(settings), encoding='utf-8')
    expected_areas = {name: sorted(int((labels == label).sum()) for label in np.unique(labels) if label)
                      for name, labels in synth_masks_multi.items() if name in ('cell', 'nucleus', 'pathogen')}
    code = r'''
import json, math, sqlite3, sys
from pathlib import Path
from spacr.measure import measure_crop
assert sys.stdout.encoding.lower() == 'cp1252' and sys.stdout.errors == 'strict'
settings = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
measure_crop(settings)
root = Path(settings['src']).parent
expected_areas = json.loads(sys.argv[2])
with sqlite3.connect(root / 'measurements/measurements.db') as connection:
    connection.row_factory = sqlite3.Row
    for table, count in {'cell': 4, 'nucleus': 4, 'pathogen': 2, 'cytoplasm': 4}.items():
        rows = list(connection.execute('SELECT * FROM "' + table + '"'))
        assert len(rows) == count, (table, len(rows))
        if table in expected_areas:
            assert sorted(row[table + '_area'] for row in rows) == expected_areas[table]
        assert len({(row['prcf'], row['object_label']) for row in rows}) == count
        for row in rows:
            assert row[table + '_area'] > 0
            assert Path(row['path_name']).parent == root
            assert Path(row['path_name']).stem == row['file_name']
            intensities = [row[key] for key in row.keys() if 'mean_intensity' in key]
            assert intensities and all(value is not None and math.isfinite(value) for value in intensities)
from tests.test_n693_native_windows_settings import _assert_native_report_lock_recovery
_assert_native_report_lock_recovery(root / 'measurements/measurements.db')
print('N693_NATIVE_CP1252_MEASURE_UNICODE_PASS')
'''
    profile = tmp_path / 'profile'
    environment = os.environ.copy()
    environment.update(PYTHONIOENCODING='cp1252:strict', PYTHONUTF8='0',
                       SPACR_HOME=str(profile / '.spacr'), SPACR_ISOLATED_PROFILE='1',
                       USERPROFILE=str(profile), HOME=str(profile),
                       APPDATA=str(profile / 'roaming'), LOCALAPPDATA=str(profile / 'local'))
    process = _run_tree([sys.executable, '-c', code, str(settings_file), json.dumps(expected_areas)],
                        tmp_path, 120, cwd=Path(__file__).resolve().parents[1], env=environment)
    assert process.returncode == 0, (process.stdout.decode('cp1252', errors='replace')[-3000:] + '\n' +
                                     process.stderr.decode('cp1252', errors='replace')[-6000:])
    assert b'N693_NATIVE_CP1252_MEASURE_UNICODE_PASS' in process.stdout
    assert b'N693_NATIVE_CP1252_REPORT_LOCK_RECOVERY_PASS' in process.stdout
    assert b'The infection report could not be written:' in process.stdout
    assert b'Processing folder:' in process.stdout
    assert b'Infection report:' in process.stdout
    assert b'Successfully completed run' in process.stdout
    assert np.array_equal(np.load(merged / 'plate1_A01_F001.npy'), data)
