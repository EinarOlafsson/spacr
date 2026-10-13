"""Verifier-only synthetic receipts; these tests never establish native acceptance.

Only two clearly marked fixture payloads substitute the pinned artifact/bootstrap
digests inside the test process. Every fixture file exists; receipt and helper
hashes use real SHA256. No installed Python, GUI, network or child process runs.
"""
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'tools/accept_historical_upgrade.py'
ARTIFACT_SHA = '2b6b07ad12926288f693c01945f2fc4555b0a37022fd73dce932e2d8b6e4ee76'
BOOTSTRAP_SHA = '94f05a5b8150f7491c7319eb3beeb62e8405b0fdf6a2d498f58313103ac7b28e'
ARTIFACT_BYTES = b'VERIFIER-ONLY FIXTURE: not a released Windows installer'
BOOTSTRAP_BYTES = b'VERIFIER-ONLY FIXTURE: not an executable bootstrap'
PINNED_FIXTURE_DIGESTS = {ARTIFACT_BYTES: ARTIFACT_SHA, BOOTSTRAP_BYTES: BOOTSTRAP_SHA}
REAL_SHA256 = hashlib.sha256
BINDING_ERROR = 'Explicit bootstrap recovery lacks exact original/source/receipt bindings'
BOOTSTRAP_ERROR = 'Explicit bootstrap recovery did not execute the unchanged original bootstrap'
ACCEPTANCE_ERROR = 'Explicit bootstrap recovery cannot replace failed wrapper or GUI acceptance'
WRAPPER_ERROR = 'Original Windows wrapper success contradicts retained failure or recovery evidence'
BINDING_KEYS = ('original_artifact_sha256', 'bootstrap_sha256',
    'original_failure_receipt_sha256', 'original_child_receipt_sha256',
    'uv_replay_receipt_sha256', 'driver_sha256')


def _write_json(path, payload):
    """Write one synthetic receipt at path from the supplied payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def _real_sha(path):
    """Return the actual SHA256 of the file at path."""
    return REAL_SHA256(path.read_bytes()).hexdigest()


def _run_verifier(case, monkeypatch):
    """Run the real verify CLI over case and return its receipt and exit code."""
    def fixture_sha256(data=b'', *args, **kwargs):
        if isinstance(data, bytes) and data in PINNED_FIXTURE_DIGESTS:
            return SimpleNamespace(hexdigest=lambda: PINNED_FIXTURE_DIGESTS[data])
        return REAL_SHA256(data, *args, **kwargs)

    output = case.evidence / 'verification.json'
    with monkeypatch.context() as context:
        context.setattr(hashlib, 'sha256', fixture_sha256)
        context.setattr(sys, 'argv', [str(TOOL), 'verify', '--evidence', str(case.evidence),
            '--repair-version', '1.5.0.6', '--target-version', '1.5.0.7', '--output', str(output)])
        with pytest.raises(SystemExit) as stopped:
            runpy.run_path(str(TOOL), run_name='__main__')
    return json.loads(output.read_text()), stopped.value.code


@pytest.fixture
def verifier_only_case(tmp_path, monkeypatch):
    """Build complete synthetic evidence and prove its positive verifier control."""
    evidence = tmp_path / 'verifier-only-evidence'
    evidence.mkdir()
    (evidence / 'SpaCR-1.5.0.1-Windows-Online-Setup.exe').write_bytes(ARTIFACT_BYTES)
    (evidence / 'original-bootstrap.ps1').write_bytes(BOOTSTRAP_BYTES)
    prefix = tmp_path / 'SpaCR/venv'
    python = prefix / 'Scripts/python.exe'
    known = json.loads((ROOT / 'tools/historical_upgrade_sources.json').read_text())
    states = []
    for version in ('1.5.0.1', '1.5.0.6', '1.5.0.7'):
        source_hashes = {row['path']: row['sha256'] for row in known.get(version, {}).get('files', [])}
        states.append(dict(version=version, prefix=str(prefix), executable=str(python),
            installed_sources={name: dict(path=str(prefix / 'Lib/site-packages' / name),
                sha256=source_hashes.get(name, 'e' * 64)) for name in
                ('spacr/updater.py', 'spacr/qt/app.py', 'spacr/qt/__init__.py')},
            pip_present=False, torch_version='verifier-only-cpu', torch_cuda_build=None,
            cuda_available=False, torch_origin=str(prefix / 'Lib/site-packages/torch/__init__.py'),
            platform='Windows-verifier-only-fixture'))
    for name, state in zip(('before', 'repaired', 'after'), states):
        _write_json(evidence / f'{name}.json', dict(passed=True, state=state))
    for name, state, command, terminal in (
            ('broken-gui', states[0], [str(python), '-m', 'pip', 'install', '--upgrade', 'spacr'],
                'pip returned exit code 1'),
            ('fixed-gui', states[1], [str(prefix.parent / 'bootstrap/uv.exe'), 'pip', 'install',
                '--upgrade', '--python', str(python), 'spacr'], 'Upgrade finished. Restart spaCR to use it.')):
        _write_json(evidence / f'{name}.json', dict(passed=True, state=state, target_version='1.5.0.7',
            action_text='Check for updates', actual_subprocess_commands=[command], terminal_dialog=terminal))
    (evidence / 'broken-gui.log').write_text('VERIFIER-ONLY FIXTURE: No module named pip\n')
    # These observation/capture fields are emitted by the existing native workflow.
    observation = dict(status='failed', artifact_sha256=ARTIFACT_SHA, observed_tree_exited=True,
        deadline_minutes=40, command=[str(evidence / 'SpaCR-1.5.0.1-Windows-Online-Setup.exe'), '/S'],
        started_utc='2026-09-28T00:00:00Z', completed_utc='2026-09-28T00:01:00Z',
        dialogs_dismissed=False, pid=4242, bootstrap_gone=True, transcript_ended=True,
        error='Exact original installer error dialog and ended bootstrap prove failure; no dialog was dismissed.',
        failure_dialog=dict(text='spaCR installation failed with exit code 1. The existing installation, if any, was preserved.'))
    capture = dict(available=True, original_child_exited=True, original_child_exit_code=1,
        expected_uv_present_before_diagnostic=False, bootstrap_sha256=BOOTSTRAP_SHA,
        engine_sha256='a' * 64, owner_sid='S-1-5-21-verifier-only',
        executable_path=str(tmp_path / 'SysWOW64/WindowsPowerShell/v1.0/powershell.exe'),
        working_directory=str(tmp_path / 'spaCR-online-installer'), script_sha256='b' * 64)
    replay = dict(status='captured', child_exit_code=0, timed_out=False, stdout_complete=True,
        stderr_complete=True, expected_uv_present=True, uv_sha256='c' * 64,
        engine_sha256=capture['engine_sha256'], account_sid=capture['owner_sid'],
        bootstrap_sha256=BOOTSTRAP_SHA, script_sha256=capture['script_sha256'])
    _write_json(evidence / 'installer-observation.json', observation)
    _write_json(evidence / 'original-uv-child.json', capture)
    _write_json(evidence / 'uv-child-diagnostic/receipt.json', replay)
    recovery = dict(status='passed', child_exit_code=0, timed_out=False, stdout_complete=True,
        stderr_complete=True, original_wrapper_accepted=False, gui_update_accepted=False,
        frozen_update_accepted=False, original_artifact_sha256=ARTIFACT_SHA, bootstrap_sha256=BOOTSTRAP_SHA,
        original_failure_receipt_sha256=_real_sha(evidence / 'installer-observation.json'),
        original_child_receipt_sha256=_real_sha(evidence / 'original-uv-child.json'),
        uv_replay_receipt_sha256=_real_sha(evidence / 'uv-child-diagnostic/receipt.json'),
        driver_sha256=_real_sha(ROOT / 'tools/recover_historical_windows_bootstrap.ps1'),
        engine_sha256=capture['engine_sha256'], account_sid=capture['owner_sid'],
        cwd=capture['working_directory'], private_prefix=str(prefix), private_python=str(python),
        uv_sha256=replay['uv_sha256'], bootstrap_argv=[capture['executable_path'], '-NoProfile',
            '-ExecutionPolicy', 'Bypass', '-File', str((evidence / 'original-bootstrap.ps1').resolve()),
            '-InstallRoot', str(prefix.parent), '-Version', '1.5.0.1', '-TorchBackend', 'cpu'])
    _write_json(evidence / 'bootstrap-recovery/receipt.json', recovery)
    for name in ('stdout.txt', 'stderr.txt'):
        (evidence / 'bootstrap-recovery' / name).write_text('VERIFIER-ONLY FIXTURE OUTPUT\n')
    case = SimpleNamespace(evidence=evidence, observation=observation, capture=capture,
        replay=replay, recovery=recovery, states=states)
    control, code = _run_verifier(case, monkeypatch)
    assert code == 0 and control['passed'] and control['errors'] == []
    assert control['windows_installation']['original_wrapper_accepted'] is False
    return case


def test_verifier_only_complete_recovery_fixture_reaches_every_gate(verifier_only_case, monkeypatch):
    receipt, code = _run_verifier(verifier_only_case, monkeypatch)
    assert code == 0 and receipt['passed'] and receipt['errors'] == []


@pytest.mark.parametrize('binding', BINDING_KEYS)
@pytest.mark.parametrize('change', ('missing', 'contradictory'))
def test_verifier_rejects_missing_or_changed_recovery_binding(verifier_only_case, monkeypatch, binding, change):
    case = verifier_only_case
    if change == 'missing':
        del case.recovery[binding]
    else:
        case.recovery[binding] = '0' * 64
    _write_json(case.evidence / 'bootstrap-recovery/receipt.json', case.recovery)
    receipt, code = _run_verifier(case, monkeypatch)
    assert code == 1 and not receipt['passed']
    assert receipt['errors'] == [BINDING_ERROR]


@pytest.mark.parametrize(('index', 'replacement'), ((0, 'other-powershell.exe'), (5, 'other-bootstrap.ps1'),
    (7, 'other-install-root'), (9, '1.5.0.6'), (11, 'cu124')))
def test_verifier_rejects_changed_original_bootstrap_argv(verifier_only_case, monkeypatch, index, replacement):
    case = verifier_only_case
    case.recovery['bootstrap_argv'][index] = replacement
    _write_json(case.evidence / 'bootstrap-recovery/receipt.json', case.recovery)
    receipt, code = _run_verifier(case, monkeypatch)
    assert code == 1 and not receipt['passed']
    assert receipt['errors'] == [BOOTSTRAP_ERROR]


def test_verifier_rejects_uv_only_replay_without_bootstrap_receipt(verifier_only_case, monkeypatch):
    case = verifier_only_case
    (case.evidence / 'bootstrap-recovery/receipt.json').unlink()
    receipt, code = _run_verifier(case, monkeypatch)
    assert code == 1 and not receipt['passed']
    assert len(receipt['errors']) == 1
    assert receipt['errors'][0].startswith('Windows installation/recovery provenance is incomplete:')
    assert 'bootstrap-recovery' in receipt['errors'][0] and 'receipt.json' in receipt['errors'][0]


def test_verifier_rejects_uv_only_replay_labeled_bootstrap_completion(verifier_only_case, monkeypatch):
    case = verifier_only_case
    case.recovery['status'] = 'captured'
    _write_json(case.evidence / 'bootstrap-recovery/receipt.json', case.recovery)
    receipt, code = _run_verifier(case, monkeypatch)
    assert code == 1 and not receipt['passed']
    assert receipt['errors'] == [ACCEPTANCE_ERROR]


def test_verifier_rejects_recovery_accepting_original_wrapper(verifier_only_case, monkeypatch):
    case = verifier_only_case
    case.recovery['original_wrapper_accepted'] = True
    _write_json(case.evidence / 'bootstrap-recovery/receipt.json', case.recovery)
    receipt, code = _run_verifier(case, monkeypatch)
    assert code == 1 and not receipt['passed']
    assert receipt['errors'] == [ACCEPTANCE_ERROR]
    assert receipt['windows_installation']['original_wrapper_accepted'] is False


def test_verifier_rejects_failed_original_wrapper_relabeled_passed(verifier_only_case, monkeypatch):
    case = verifier_only_case
    case.observation.update(status='passed', exit_code=0)
    _write_json(case.evidence / 'installer-observation.json', case.observation)
    receipt, code = _run_verifier(case, monkeypatch)
    assert code == 1 and not receipt['passed']
    assert receipt['errors'] == [WRAPPER_ERROR]
    assert receipt['windows_installation']['original_wrapper_accepted'] is False


def _set_successful_wrapper_schema(case):
    """Replace case's observation with the actual workflow success schema."""
    _write_json(case.evidence / 'installer-observation.json',
        dict(status='passed', exit_code=0, artifact_sha256=ARTIFACT_SHA, deadline_minutes=40,
            command=[str(case.evidence / 'SpaCR-1.5.0.1-Windows-Online-Setup.exe'), '/S'],
            started_utc='2026-09-28T00:00:00Z', completed_utc='2026-09-28T00:01:00Z',
            dialogs_dismissed=False, pid=4242))
    shutil.rmtree(case.evidence / 'bootstrap-recovery')
    shutil.rmtree(case.evidence / 'uv-child-diagnostic')


def test_verifier_rejects_success_label_with_failed_original_uv_child(verifier_only_case, monkeypatch):
    case = verifier_only_case
    _set_successful_wrapper_schema(case)
    receipt, code = _run_verifier(case, monkeypatch)
    assert code == 1 and not receipt['passed']
    assert receipt['errors'] == [WRAPPER_ERROR]
    assert receipt['windows_installation']['original_wrapper_accepted'] is False


def test_verifier_only_genuine_success_schema_does_not_require_recovery(verifier_only_case, monkeypatch):
    case = verifier_only_case
    _set_successful_wrapper_schema(case)
    case.capture.update(original_child_exit_code=0, expected_uv_present_before_diagnostic=True)
    _write_json(case.evidence / 'original-uv-child.json', case.capture)
    receipt, code = _run_verifier(case, monkeypatch)
    assert code == 0 and receipt['passed'] and receipt['errors'] == []
    assert receipt['windows_installation']['original_wrapper_accepted'] is True


def test_verifier_still_rejects_unpinned_artifact(verifier_only_case, monkeypatch):
    case = verifier_only_case
    (case.evidence / 'SpaCR-1.5.0.1-Windows-Online-Setup.exe').write_bytes(b'unpinned fixture bytes')
    receipt, code = _run_verifier(case, monkeypatch)
    assert code == 1 and not receipt['passed']
    assert receipt['errors'] == ['Original Windows artifact identity changed']


def test_verifier_still_requires_real_help_update_command(verifier_only_case, monkeypatch):
    case = verifier_only_case
    path = case.evidence / 'fixed-gui.json'
    gui = json.loads(path.read_text())
    gui['actual_subprocess_commands'] = []
    _write_json(path, gui)
    receipt, code = _run_verifier(case, monkeypatch)
    assert code == 1 and not receipt['passed']
    assert receipt['errors'] == ['fixed-gui lacks the actual expected updater subprocess']
