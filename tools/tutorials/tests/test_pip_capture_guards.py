"""Require real installation evidence and select the visible Qt window."""
import json
from pathlib import Path

import pytest


@pytest.fixture
def recorder(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    import capture_pip_installation
    return capture_pip_installation


def test_visible_window_is_not_the_last_matching_one_pixel_helper(recorder):
    tree = ('0x400007 "spaCR": ("spacr" "spaCR") 2560x720+0+0 +0+0\n'
            '0x400003 "spaCR": ("spacr" "spaCR") 1x1+0+0 +0+0\n'
            '0x700004 "Another spaCR task": () 3840x2160+0+0 +0+0')
    assert recorder.visible_app_window(tree) == 0x400007
    with pytest.raises(ValueError, match='No visible'):
        recorder.visible_app_window(tree.split('\n', 1)[1])


def test_larger_real_window_wins_regardless_of_order(recorder):
    first = '0x400007 "spaCR": () 3840x2160+0+0 +0+0'
    second = '0x400009 "spaCR": () 640x480-2+0 -2+0'
    assert recorder.visible_app_window(first + '\n' + second) == 0x400007
    assert recorder.visible_app_window(second + '\n' + first) == 0x400007


def test_only_a_complete_private_installation_is_accepted(recorder, tmp_path):
    root = tmp_path / 'installation_runs' / 'verified'
    root.mkdir(parents=True)
    receipt = dict(accepted=True, steps=[dict(returncode=0, completed=True) for _ in range(6)])
    (root / 'receipt.json').write_text(json.dumps(receipt))
    assert recorder.installation(tmp_path, root)[1] == root / 'venv'
    outside = tmp_path / 'other_environment'
    outside.mkdir()
    (outside / 'receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='Only a private'):
        recorder.installation(tmp_path, outside)
    for modified in [dict(receipt, accepted=False), dict(receipt, steps=receipt['steps'][:-1]),
                     dict(receipt, steps=receipt['steps'][:-1] + [dict(returncode=1, completed=True)]),
                     dict(receipt, steps=receipt['steps'][:-1] + [dict(returncode=0, completed=False)])]:
        (root / 'receipt.json').write_text(json.dumps(modified))
        with pytest.raises(ValueError, match='six successful'):
            recorder.installation(tmp_path, root)


def test_conda_uses_its_own_five_step_receipt_and_prefix(recorder, tmp_path):
    root = tmp_path / 'installation_runs' / 'conda'
    root.mkdir(parents=True)
    receipt = dict(accepted=True, steps=[dict(returncode=0, completed=True) for _ in range(5)])
    (root / 'receipt.json').write_text(json.dumps(receipt))
    assert recorder.installation(tmp_path, root, 'conda')[1] == root / 'env'
    with pytest.raises(ValueError, match='six successful'):
        recorder.installation(tmp_path, root, 'pip')
    receipt['steps'].append(dict(returncode=0, completed=True))
    (root / 'receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='five successful'):
        recorder.installation(tmp_path, root, 'conda')


def test_conda_commands_do_not_pretend_to_be_the_pip_environment(recorder, tmp_path):
    rows = recorder.commands('conda', '1.5.0.4', tmp_path / 'env')
    assert len(rows) == 5
    assert rows[1][1][-4:] == ['list', '--prefix', str(tmp_path / 'env'), 'spacr']
    assert rows[2][2] == 'spaCR: 1.5.0.4'
    assert rows[3][2] == str(tmp_path / 'env')
    assert rows[4][1] == ['spacr-doctor']
    pip = recorder.commands()
    assert pip[1][1] == ['python', '-m', 'pip', '--version']
    assert pip[2][2] == 'spaCR: 1.5.0.5'
    assert pip[3][1] == ['python', '-m', 'pip', 'check']
    assert pip[4][1] == ['spacr-doctor', '--no-gpu-probe']


def test_linux_installer_requires_its_three_steps_and_actual_runtime(recorder, tmp_path):
    root = tmp_path / 'installation_runs' / 'linux'
    root.mkdir(parents=True)
    receipt = dict(accepted=True, installer_tag='v1.5.0.5',
                   steps=[dict(returncode=0, completed=True) for _ in range(3)])
    (root / 'receipt.json').write_text(json.dumps(receipt))
    assert recorder.installation(tmp_path, root, 'linux_installer')[1] == root / 'runtime/venv'
    assert recorder.installed_version(receipt, 'linux_installer') == '1.5.0.5'
    with pytest.raises(ValueError, match='five successful'):
        recorder.installation(tmp_path, root, 'conda')
    for modified in [dict(receipt, accepted=False), dict(receipt, steps=receipt['steps'][:-1]),
                     dict(receipt, steps=receipt['steps'][:-1] + [dict(returncode=1, completed=True)]),
                     dict(receipt, steps=receipt['steps'][:-1] + [dict(returncode=0, completed=False)])]:
        (root / 'receipt.json').write_text(json.dumps(modified))
        with pytest.raises(ValueError, match='three successful'):
            recorder.installation(tmp_path, root, 'linux_installer')


def test_linux_commands_read_the_actual_profile_and_launcher(recorder, tmp_path):
    import subprocess
    import sys
    prefix = tmp_path / 'runtime/venv'
    prefix.mkdir(parents=True)
    (prefix.parent / 'install-profile.json').write_text(json.dumps(
        dict(requested_backend='auto', active_backend='cuda')))
    launcher = tmp_path / 'bin/spacr'
    launcher.parent.mkdir()
    launcher.write_text(f'#!/bin/sh\nexec "{prefix}/bin/python" -m spacr.qt "$@"\n')
    rows = recorder.commands('linux_installer', '1.5.0.5', prefix)
    assert len(rows) == 5
    assert rows[1][0] == '02_installer_backend'
    assert rows[3][0] == '04_installed_launcher'
    for row in [rows[1], rows[3]]:
        assert row[2] in subprocess.check_output([sys.executable, *row[1][1:]], text=True)
    assert rows[2][2] == 'spaCR: 1.5.0.5'
    assert rows[4][1] == ['spacr-doctor', '--no-gpu-probe']


def test_privacy_dialog_is_real_exact_title_and_our_process(recorder):
    name = 'spaCR privacy and optional account setup'
    properties = '_NET_WM_PID(CARDINAL) = 1234\nWM_TRANSIENT_FOR:  not found.\n'
    assert recorder.privacy_window_owned(name, properties, 1234)
    assert recorder.privacy_window_owned(name + ' — spaCR', properties, 1234)
    assert not recorder.privacy_window_owned('Unrelated ' + name, properties, 1234)
    assert not recorder.privacy_window_owned(name, properties, 123)
    assert not recorder.privacy_window_owned(name, properties, 12345)
    assert not recorder.privacy_window_owned(name, 'WM_TRANSIENT_FOR = 1234\n', 1234)


def test_privacy_dismissal_requires_hidden_or_actually_destroyed_window(recorder):
    tree = '  0x80000e "spaCR privacy and optional account setup — spaCR": () 960x342+0+0\n'
    remaining = '  0x800007 "spaCR": () 3840x2160+0+0\n'
    assert recorder.dialog_dismissed(0, 'Map State: IsUnMapped', tree, 0x80000e)
    assert recorder.dialog_dismissed(1, '', remaining, 0x80000e)
    assert not recorder.dialog_dismissed(0, 'Map State: IsViewable', tree, 0x80000e)
    assert not recorder.dialog_dismissed(1, '', tree, 0x80000e)
    assert not recorder.dialog_dismissed(2, '', remaining, 0x80000e)
