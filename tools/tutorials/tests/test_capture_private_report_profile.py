"""The capture profile never changes the real anonymous-report preferences."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]


def child(tmp_path, mode):
    stage = tmp_path / 'capture'
    for name in ('profile', 'app-state', 'config/home'):
        (stage / name).mkdir(parents=True, exist_ok=True)
    sentinel = tmp_path / 'outside' / 'anonymous_reports' / 'state.json'
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    original = b'{"mode":"all","notice_shown":false,"retained":"untouched"}'
    sentinel.write_bytes(original)
    env = {**os.environ, 'HOME': str(stage / 'profile'),
           'SPACR_HOME': str(stage / 'app-state'),
           'XDG_CONFIG_HOME': str(stage / 'config/home'),
           'SPACR_TUTORIAL_CACHE_ISOLATED': '1', 'SPACR_NO_NETWORK': '1',
           'CUDA_VISIBLE_DEVICES': '', 'QT_QPA_PLATFORM': 'offscreen'}
    if mode == 'outside_home':
        env['SPACR_HOME'] = str(sentinel.parent.parent)
    if mode == 'missing_isolation':
        env.pop('SPACR_TUTORIAL_CACHE_ISOLATED')
    if mode == 'escaping_state':
        (stage / 'app-state/anonymous_reports').symlink_to(sentinel.parent, target_is_directory=True)
    code = """
import os,sys,json
from pathlib import Path
sys.path[:0] = [os.getcwd(),str(Path.cwd()/'tools/tutorials')]
import capture_host_mode as host
host.use_this_checkout()
from spacr import crashreport
if sys.argv[1] == 'ineffective':
    crashreport.mark_anonymous_report_notice_shown = lambda: None
try:
    actual = host.prepare_report_notice()
except RuntimeError:
    if sys.argv[1] == 'valid':
        raise
else:
    assert sys.argv[1] == 'valid', 'Unsafe or ineffective profile was accepted'
    assert actual['anonymous_reports_enabled'] is False
    assert actual['anonymous_report_notice_due'] is False
    state = Path(actual['anonymous_report_state'])
    assert state.is_relative_to(Path(os.environ['SPACR_HOME']))
    saved = json.loads(state.read_text())
    assert saved['mode'] == 'off' and saved['notice_shown'] is True
"""
    result = subprocess.run([sys.executable, '-c', code, mode], cwd=ROOT,
                            env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert sentinel.read_bytes() == original
    if mode in ('outside_home', 'missing_isolation'):
        assert not (stage / 'app-state/anonymous_reports').exists()


@pytest.mark.parametrize('mode', ['valid', 'outside_home', 'missing_isolation',
                                'escaping_state', 'ineffective'])
def test_report_notice_preparation_is_private_and_verified(tmp_path, mode):
    child(tmp_path, mode)
