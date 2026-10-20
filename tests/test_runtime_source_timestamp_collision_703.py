"""An atomic runtime replacement invalidates same-second old Python bytecode."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from spacr import install_cleanup as cleanup
from tests.test_macos_runtime_transaction_703 import installation, _installer, _snapshot, _wheel

pytestmark = pytest.mark.skipif(os.name == 'nt', reason='Exercises POSIX runtime replacement')


@pytest.mark.parametrize('external_cache', [False, True])
def test_changed_same_size_source_invalidates_old_cache_without_deleting_it(installation, tmp_path, external_cache):
    record, machine, prefix, site = installation
    environment = os.environ.copy()
    environment.pop('PYTHONPYCACHEPREFIX', None)
    cache = tmp_path / 'user owned external cache'
    if external_cache:
        environment['PYTHONPYCACHEPREFIX'] = str(cache)
    old_sources = {name: (site / 'spacr' / name).stat() for name in ('qt.py', '_version.py', '__init__.py')}
    prime = subprocess.run([str(prefix / 'bin/spacr-probe')],
                           env=environment, capture_output=True, text=True, timeout=15)
    assert prime.returncode == 0 and prime.stdout.strip() == '1.5.1.4'
    cache_before = _snapshot(cache) if external_cache else None
    old_runtime = _snapshot(prefix)
    wheel = _wheel(tmp_path / 'new wheels', '1.5.1.5')
    uv = _installer(tmp_path, wheel)
    staged_details = {}

    def run(argv):
        code, output = cleanup._run(argv, timeout=30)
        if argv[1:3] == ['pip', 'install'] and code == 0:
            staged = Path(argv[argv.index('--python') + 1]).parent.parent
            relative = site.relative_to(prefix)
            for name, old in old_sources.items():
                path = staged / relative / 'spacr' / name
                os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
                assert path.stat().st_size == old.st_size
                staged_details[name] = (path.stat().st_mtime_ns, path.stat().st_atime_ns, path.stat().st_mode)
        return code, output

    version, _ = cleanup._transactional_runtime_update(
        record, '1.5.1.5', [str(uv), 'pip', 'install', '--upgrade', '--python', record.python, 'spacr'],
        machine=machine, runner=run)
    assert version == '1.5.1.5'
    for name in ('qt.py', '_version.py'):
        path = site / 'spacr' / name
        assert int(path.stat().st_mtime) != int(old_sources[name].st_mtime)
        assert path.stat().st_mode == staged_details[name][2]
    assert (site / 'spacr/__init__.py').stat().st_mtime_ns == staged_details['__init__.py'][0]
    if external_cache:
        assert _snapshot(cache) == cache_before
    reopened = subprocess.run([str(prefix / 'bin/spacr-probe')], env=environment,
                              capture_output=True, text=True, timeout=15)
    assert reopened.returncode == 0 and reopened.stdout.strip() == '1.5.1.5', reopened.stderr
    transaction = next(prefix.parent.glob('.spacr-runtime-update-*/transaction.json'))
    state = json.loads(transaction.read_text(encoding="utf-8"))
    assert _snapshot(Path(state['rollback_runtime'])) == old_runtime
