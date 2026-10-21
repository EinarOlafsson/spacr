"""A retained private runtime can be restored without trusting changed paths."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from spacr import install_cleanup as ic
from tests.test_pip_runtime_transaction_703 import installation
from tests.test_macos_runtime_transaction_703 import _snapshot, _wheel

pytestmark = pytest.mark.posix_only


@pytest.fixture
def updated(installation, tmp_path):
    record, machine, prefix = installation
    prefix.chmod(0o700)
    prefix.parent.chmod(0o700)
    original = _snapshot(prefix)
    wheel = _wheel(tmp_path / 'new-wheels', '1.5.1.5')
    def run(argv):
        if argv[1:6] == ['-I', '-m', 'pip', 'install', '--upgrade']:
            argv = [*argv[:-1], '--isolated', '--no-cache-dir', '--no-index', '--no-deps', str(wheel)]
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return result.returncode, result.stdout + result.stderr
    ic._transactional_runtime_update(
        record, '1.5.1.5', [record.python, '-I', '-m', 'pip', 'install', '--upgrade', 'spacr==1.5.1.5'],
        machine=machine, runner=run)
    transaction, = prefix.parent.glob('.spacr-runtime-update-*')
    return prefix, transaction, machine, original, run


def test_real_exchange_restores_old_runtime_and_preserves_new_one(updated):
    prefix, transaction, machine, original, run = updated
    new = _snapshot(prefix)
    waited = []
    result = ic._recover_runtime_update(str(transaction), 43210, machine=machine,
                                       runner=run, wait=lambda pid: waited.append(pid) or True)
    assert result['state'] == 'rolled_back' and waited == [43210]
    assert _snapshot(prefix) == original
    assert _snapshot(transaction / 'runtime') == new
    assert ic._recover_runtime_update(str(transaction), 43210, machine=machine,
                                     runner=run, wait=lambda pid: True)['state'] == 'rolled_back'
    assert _snapshot(prefix) == original


@pytest.mark.parametrize('change', ['current-source', 'retained-source', 'directory-identity',
                                    'journal-symlink', 'transaction-symlink', 'shared-parent',
                                    'legacy-journal', 'live-process', 'existing-lock'])
def test_recovery_refuses_changed_or_busy_state_without_mutation(updated, tmp_path, change):
    prefix, transaction, machine, _original, run = updated
    path = transaction
    if change == 'current-source':
        (prefix / 'unexpected-source.py').write_text('new source', encoding="utf-8")
    elif change == 'retained-source':
        (transaction / 'runtime' / 'unexpected-source.py').write_text('new source', encoding="utf-8")
    elif change == 'directory-identity':
        retained = transaction / 'runtime'
        retained.rename(transaction / 'previous-runtime')
        import shutil
        shutil.copytree(transaction / 'previous-runtime', retained, symlinks=True)
    elif change == 'journal-symlink':
        journal = transaction / 'transaction.json'
        journal.rename(transaction / 'saved.json')
        journal.symlink_to('saved.json')
    elif change == 'transaction-symlink':
        path = tmp_path / 'linked-transaction'
        path.symlink_to(transaction, target_is_directory=True)
    elif change == 'shared-parent':
        prefix.parent.chmod(0o777)
    elif change == 'legacy-journal':
        journal = transaction / 'transaction.json'
        data = json.loads(journal.read_text(encoding="utf-8"))
        data['schema'] = 1
        journal.write_text(json.dumps(data), encoding="utf-8")
    elif change == 'existing-lock':
        (prefix.parent / ('.' + prefix.name + '.spacr-update-lock')).mkdir()
    before = _snapshot(prefix.parent)
    with pytest.raises((ValueError, OSError, RuntimeError)):
        ic._recover_runtime_update(str(path), 43210, machine=machine, runner=run,
                                   wait=lambda pid: change != 'live-process')
    assert _snapshot(prefix.parent) == before


def test_failed_restored_identity_exchanges_back_without_losing_either_runtime(updated):
    prefix, transaction, machine, _original, run = updated
    before = _snapshot(prefix), _snapshot(transaction / 'runtime')
    probes = []
    def fail_final(argv):
        probes.append(argv)
        if len(probes) == 2:
            return 1, 'deliberately refused restored interpreter'
        return run(argv)
    with pytest.raises(RuntimeError, match='restored runtime'):
        ic._recover_runtime_update(str(transaction), 43210, machine=machine,
                                   runner=fail_final, wait=lambda pid: True)
    assert (_snapshot(prefix), _snapshot(transaction / 'runtime')) == before
    assert json.loads((transaction / 'transaction.json').read_text(encoding="utf-8"))['state'] == 'committed'
    assert all('-B' in argv for argv in probes)


def test_bytecode_cache_changes_do_not_disable_source_verified_recovery(updated):
    prefix, transaction, machine, original, run = updated
    cache = prefix / 'lib' / '__pycache__'
    cache.mkdir()
    (cache / 'volatile.pyc').write_bytes(b'cache only')
    result = ic._recover_runtime_update(str(transaction), 43210, machine=machine,
                                       runner=run, wait=lambda pid: True)
    assert result['state'] == 'rolled_back'
    assert _snapshot(prefix) == original
    assert (transaction / 'runtime' / 'lib' / '__pycache__' / 'volatile.pyc').read_bytes() == b'cache only'


def test_interrupted_post_exchange_recovery_completes_from_actual_orientation(updated):
    prefix, transaction, machine, original, run = updated
    journal = transaction / 'transaction.json'
    data = json.loads(journal.read_text(encoding="utf-8"))
    data['state'] = 'rolling_back'
    ic._runtime_recovery_journal(str(transaction), data)
    ic._runtime_exchange(str(prefix), str(transaction / 'runtime'))
    assert ic._recover_runtime_update(str(transaction), 43210, machine=machine,
                                     runner=run, wait=lambda pid: True)['state'] == 'rolled_back'
    assert _snapshot(prefix) == original


def test_command_reports_invalid_recovery_without_traceback_or_install(tmp_path):
    result = subprocess.run([sys.executable, '-I', ic.__file__, 'recover-runtime',
                             str(tmp_path / 'absent'), '--pid', '43210'],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 6
    assert result.stdout.startswith('Runtime recovery stopped:')
    assert not result.stderr
