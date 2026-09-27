"""Verifier-only synthetic receipts; these tests do not establish native acceptance."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / 'tools/accept_historical_upgrade.py'
SOURCE_BINDINGS = ROOT / 'tools/historical_upgrade_sources.json'
TARGET = '1.5.1.0'


def _fixture(tmp_path):
    """Create labeled receipt fixtures without installing or launching any spaCR."""
    evidence = tmp_path / 'evidence'
    evidence.mkdir()
    prefix = tmp_path / 'installer/venv'
    known = json.loads(SOURCE_BINDINGS.read_text())
    sources = ('spacr/updater.py', 'spacr/qt/app.py', 'spacr/qt/__init__.py')
    executable = str(prefix / 'bin/python')
    states = []
    for version in ('1.5.0.1', '1.5.0.5', TARGET):
        hashes = {row['path']: row['sha256'] for row in known.get(version, {}).get('files', [])}
        states.append(dict(version=version, prefix=str(prefix), executable=executable,
            installed_sources={name: {'path': str(prefix / 'site-packages' / name),
                'sha256': hashes.get(name, '7' * 64)} for name in sources},
            pip_present=False, torch_version='2.8.0+cpu', torch_cuda_build=None,
            cuda_available=False, torch_origin=str(prefix / 'site-packages/torch/__init__.py'),
            platform='synthetic verifier fixture', transport_scope='no network or native execution'))
    records = {}
    for name, state in zip(('before', 'repaired', 'after'), states):
        records[name] = dict(passed=True, state=state, fixture_kind='verifier-only; not native evidence')
    commands = [
        [executable, '-m', 'pip', 'install', '--upgrade', 'spacr'],
        [str(prefix.parent / 'bootstrap/uv'), 'pip', 'install', '--upgrade', '--python', executable, 'spacr'],
    ]
    for index, name in enumerate(('broken-gui', 'fixed-gui')):
        records[name] = dict(passed=True, state=copy.deepcopy(states[index]),
            fixture_kind='verifier-only; not native evidence', target_version=TARGET,
            action_text='Check for updates…', actual_subprocess_commands=[commands[index]],
            terminal_dialog=('pip returned exit code 1. Check the terminal for details.' if index == 0 else
                             'Upgrade finished. Restart spaCR to use it.'))
    (evidence / 'broken-gui.log').write_text('Verifier-only fixture: No module named pip\n')
    return evidence, records


def _verify(evidence, records):
    """Run only the stdlib final-verifier branch, never its installed-state or GUI paths."""
    for name, value in records.items():
        (evidence / f'{name}.json').write_text(json.dumps(value))
    output = evidence / 'acceptance.json'
    completed = subprocess.run([sys.executable, '-I', str(DRIVER), 'verify',
        '--target-version', TARGET, '--evidence', str(evidence), '--output', str(output)],
        cwd=evidence, capture_output=True, text=True, timeout=20)
    return completed, json.loads(output.read_text()) if output.exists() else None


def test_verifier_accepts_complete_synthetic_contract_only(tmp_path):
    """A consistent fixture exercises the verifier, without claiming a real installation."""
    evidence, records = _fixture(tmp_path)
    completed, accepted = _verify(evidence, records)
    assert completed.returncode == 0, completed.stderr
    assert accepted['passed'] is True and accepted['errors'] == []
    assert all(row['fixture_kind'].startswith('verifier-only') for row in accepted['stages'].values())


@pytest.mark.parametrize(('fault', 'message'), [
    ('source_escape', 'source origin escapes'),
    ('source_digest', 'exact historical PyPI wheel'),
    ('source_missing', 'exact installed source provenance'),
    ('source_bad_digest', 'source digest is malformed'),
    ('executable_escape', 'executable location escapes'),
    ('torch_escape', 'torch origin escapes'),
    ('wrong_version', 'Version transition'),
    ('changed_prefix', 'private installation environment changed'),
    ('pip_seeded', 'pip was installed'),
    ('cuda_wheel', 'CPU backend'),
    ('cuda_available', 'CPU backend'),
    ('gui_target', 'same public upgrade target'),
    ('gui_prefix', 'same version-bound private installation'),
    ('gui_version', 'same version-bound private installation'),
    ('gui_provenance', 'GUI provenance differs'),
    ('gui_action', 'Help action witness'),
    ('gui_command', 'actual expected updater subprocess'),
    ('gui_terminal', 'required actual terminal dialog'),
    ('failed_stage', 'One or more stages failed'),
    ('missing_failure_text', 'missing-pip failure is absent'),
])
def test_verifier_rejects_incomplete_or_contradictory_receipts(tmp_path, fault, message):
    """Each deliberately corrupted verifier fixture must fail independently."""
    evidence, records = _fixture(tmp_path)
    before = records['before']['state']
    after = records['after']['state']
    gui = records['fixed-gui']
    source = before['installed_sources']['spacr/updater.py']
    if fault == 'source_escape':
        source['path'] = str(tmp_path / 'checkout/spacr/updater.py')
    elif fault == 'source_digest':
        source['sha256'] = '0' * 64
    elif fault == 'source_missing':
        del before['installed_sources']['spacr/updater.py']
    elif fault == 'source_bad_digest':
        source['sha256'] = 'not-a-digest'
    elif fault == 'executable_escape':
        before['executable'] = str(tmp_path / 'host-python/python')
    elif fault == 'torch_escape':
        after['torch_origin'] = str(tmp_path / 'global/torch/__init__.py')
    elif fault == 'wrong_version':
        after['version'] = '1.5.0.5'
    elif fault == 'changed_prefix':
        after['prefix'] = str(tmp_path / 'different-venv')
    elif fault == 'pip_seeded':
        after['pip_present'] = True
    elif fault == 'cuda_wheel':
        after['torch_cuda_build'] = '12.8'
    elif fault == 'cuda_available':
        after['cuda_available'] = True
    elif fault == 'gui_target':
        gui['target_version'] = '9.0.0'
    elif fault == 'gui_prefix':
        gui['state']['prefix'] = str(tmp_path / 'other-venv')
    elif fault == 'gui_version':
        gui['state']['version'] = '1.5.0.1'
    elif fault == 'gui_provenance':
        gui['state']['installed_sources']['spacr/updater.py']['sha256'] = '0' * 64
    elif fault == 'gui_action':
        gui['action_text'] = 'About'
    elif fault == 'gui_command':
        gui['actual_subprocess_commands'] = []
    elif fault == 'gui_terminal':
        gui['terminal_dialog'] = 'No updates.'
    elif fault == 'failed_stage':
        records['broken-gui']['passed'] = False
    else:
        (evidence / 'broken-gui.log').write_text('Verifier-only unrelated failure\n')
    completed, accepted = _verify(evidence, records)
    assert completed.returncode != 0
    assert accepted['passed'] is False
    assert any(message in error for error in accepted['errors'])


def test_verifier_rejects_absent_failure_log(tmp_path):
    """Missing old-process output cannot be replaced by a success boolean."""
    evidence, records = _fixture(tmp_path)
    (evidence / 'broken-gui.log').unlink()
    completed, accepted = _verify(evidence, records)
    assert completed.returncode != 0
    assert accepted is None or accepted['passed'] is False


def test_verifier_allows_normal_venv_interpreter_symlink(tmp_path):
    """A managed-Python symlink is valid when its venv executable directory is retained."""
    evidence, records = _fixture(tmp_path)
    executable = Path(records['before']['state']['executable'])
    executable.parent.mkdir(parents=True)
    runtime = tmp_path / 'managed-python/python'
    runtime.parent.mkdir()
    runtime.write_bytes(b'fixture, never executed')
    try:
        executable.symlink_to(runtime)
    except OSError as error:
        pytest.skip(f'Platform cannot create the verifier-only interpreter symlink: {error}')
    completed, accepted = _verify(evidence, records)
    assert completed.returncode == 0, completed.stderr
    assert accepted['passed'] is True
