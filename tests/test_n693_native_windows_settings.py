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
    process = subprocess.run([sys.executable, '-c', code, str(folder)],
                             cwd=Path(__file__).resolve().parents[1], env=environment,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    assert process.returncode == 0, process.stderr.decode('cp1252', errors='replace')
    assert b'N693_NATIVE_CP1252_UNICODE_PASS' in process.stdout
    assert b'Warning: could not save settings' in process.stdout
    assert b'Warning: could not write' in process.stdout
    assert 'UnicodeEncodeError' not in process.stderr.decode('cp1252', errors='replace')
