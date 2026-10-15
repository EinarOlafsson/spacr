"""Real offline pip wheels stage and roll back only temporary private venvs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import venv

import pytest

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX runtime atomic exchange requires native Windows replacement separately")

from spacr import install_cleanup as ic
from tests.test_macos_runtime_transaction_703 import _snapshot, _wheel


@pytest.fixture
def installation(tmp_path):
    prefix = tmp_path / 'private runtime' / 'venv'
    venv.EnvBuilder(with_pip=True).create(prefix)
    python = prefix / 'bin/python'
    old = _wheel(tmp_path / 'old-wheels', '1.5.1.4')
    result = subprocess.run([str(python), '-I', '-m', 'pip', '--isolated', '--no-cache-dir',
                             'install', '--no-index', '--no-deps', str(old)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    import packaging
    site = Path(subprocess.check_output([str(python), '-I', '-c',
        'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip())
    import shutil
    shutil.copytree(Path(packaging.__file__).parent, site / 'packaging', ignore=shutil.ignore_patterns('__pycache__'))
    kept = prefix / 'unknown-user-folder' / 'keep.bin'
    kept.parent.mkdir()
    kept.write_bytes(b'not installer-owned data')
    record = ic.InstallRecord(kind='environment', layout='venv', platform='linux', root=str(prefix),
                              python=str(python), version='1.5.1.4', running=True)
    machine = ic._Machine(platform='linux', environ={'HOME': str(tmp_path / 'user')},
                          fs_root=str(tmp_path), executable=str(python), running_prefix=str(prefix))
    return record, machine, prefix


@pytest.mark.parametrize('failure', [None, 'install', 'dependencies', 'import', 'identity', 'restart', 'wrong-version'])
def test_real_pip_transaction_keeps_the_original_until_verified(installation, tmp_path, failure):
    record, machine, prefix = installation
    before = _snapshot(prefix)
    wheel = _wheel(tmp_path / 'new-wheels', '1.5.1.6' if failure == 'wrong-version' else '1.5.1.5', broken=failure == 'import')
    command = [record.python, '-I', '-m', 'pip', 'install', '--upgrade', '--isolated',
               '--no-cache-dir', '--no-index', '--no-deps', str(wheel)]
    calls = []
    def run(argv):
        calls.append(argv)
        assert argv[0] != sys.executable
        if argv[4:6] == ['install', '--upgrade']:
            assert Path(argv[0]).parent.parent != prefix
            assert _snapshot(prefix) == before
            if failure == 'install':
                return 9, 'deliberately failed temporary package install'
        if argv[1:] == ['-I', '-m', 'pip', 'check'] and failure == 'dependencies':
            return 8, 'deliberately unsatisfied dependency'
        if argv[:3] == [record.python, '-I', '-c'] and failure == 'identity':
            return 7, 'deliberately refused activated identity'
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return completed.returncode, completed.stdout + completed.stderr
    def restart(version):
        assert version == '1.5.1.5'
        if failure == 'restart':
            raise OSError('deliberately refused temporary relaunch')
    if failure:
        with pytest.raises((RuntimeError, OSError)):
            ic._transactional_runtime_update(record, '1.5.1.5', command, machine=machine,
                                             runner=run, activate=restart)
        assert _snapshot(prefix) == before
        assert not list(prefix.parent.glob('.spacr-runtime-update-*'))
    else:
        version, _ = ic._transactional_runtime_update(record, '1.5.1.5', command,
                                                      machine=machine, runner=run, activate=restart)
        assert version == '1.5.1.5'
        transaction, = prefix.parent.glob('.spacr-runtime-update-*')
        assert _snapshot(transaction / 'runtime') == before
        assert json.loads((transaction / 'transaction.json').read_text())['state'] == 'committed'
        output = subprocess.check_output([str(prefix / 'bin/spacr-probe')], text=True)
        assert output.strip() == '1.5.1.5'
        assert (prefix / 'unknown-user-folder/keep.bin').read_bytes() == b'not installer-owned data'
    assert all('/.spacr-runtime-update-' in argv[0] for argv in calls if argv[1:4] == ['-I', '-m', 'pip'])
    assert not (prefix.parent / '.venv.spacr-update-lock').exists()


@pytest.mark.parametrize('change', ['other-python', 'wrong-operation'])
def test_pip_runtime_refuses_unapproved_command_before_running_it(installation, change):
    record, machine, prefix = installation
    before = _snapshot(prefix)
    command = [record.python, '-I', '-m', 'pip', 'install', '--upgrade', 'spacr==1.5.1.5']
    if change == 'other-python':
        command[0] = sys.executable
    else:
        command[4] = 'uninstall'
    with pytest.raises(ValueError, match='does not target this private runtime'):
        ic._transactional_runtime_update(record, '1.5.1.5', command, machine=machine,
                                        runner=lambda argv: pytest.fail('No package command may run'))
    assert _snapshot(prefix) == before
