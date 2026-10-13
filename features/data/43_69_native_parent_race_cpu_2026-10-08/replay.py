import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

root = Path('/mnt/wd4tb/spacr-worktrees/codex-fresh-home-contracts-20261008')
scratch = Path('/mnt/wd4tb/scratch/native-69-parent-race-20261008')
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
nodes = [
    'tests/qt/test_live_preview_remove_background.py',
    'tests/qt/test_make_masks_parent_source_race_guards.py::test_puncta_requires_an_explicit_valid_parent',
    'tests/qt/test_make_masks_parent_source_race_guards.py::test_native_sources_changed_during_detection_cannot_publish_a_result[image]',
    'tests/qt/test_make_masks_parent_source_race_guards.py::test_native_sources_changed_during_detection_cannot_publish_a_result[parent]',
]
paths = ['spacr/qt/screens/make_masks.py', 'spacr/qt/widgets/primary_mask_selector.py', 'spacr/qt/bridge.py', 'tests/qt/test_make_masks_parent_source_race_guards.py', 'tests/qt/test_live_preview_remove_background.py', 'tests/conftest.py', 'tests/qt/conftest.py']
identity = {}
for path in paths:
    raw = (root / path).read_bytes()
    hashes = {}
    for source in ('69ba4e462f2ea60924a977b192d006ce9b70f42c', '0b8c2c4120a0ede4de568d484ea3fc9d517f1de3'):
        frozen = subprocess.check_output(['git', 'show', source + ':' + path], cwd=root)
        assert frozen == raw, path
        hashes[source] = hashlib.sha256(frozen).hexdigest()
    identity[path] = hashes
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', SPACR_DEVICE='cpu', QT_QPA_PLATFORM='offscreen', MPLBACKEND='Agg', PYTHONPATH='.:/mnt/wd4tb/scratch/ci-55bad-20261008/pytest842-overlay', COVERAGE_FILE=str(scratch / '.coverage'))
command = ['/usr/bin/gdb', '-nx', '-nh', '-batch', '-x', str(scratch / 'gdb.commands'), '--args', sys.executable, '-m', 'pytest', '-vv', '--tb=short', '-p', 'no:randomly', *nodes, '--cov=spacr', '--cov-branch', '--cov-report=json:' + str(scratch / 'coverage.json')]
record = {'argv': command, 'cwd': str(root), 'source_sha256': identity, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'launcher_pid': os.getpid(), 'core_limit': list(resource.getrlimit(resource.RLIMIT_CORE)), 'scope': 'One original-order serial two-file/13-node debug replay with branch tracing. Hosted gw1 came from two-worker xdist; this bounded replay does not reproduce that complete long-order/parallel context.'}
(scratch / 'replay-provenance.json').write_text(json.dumps(record, indent=2) + '\n')
with (scratch / 'gdb.log').open('wb') as output:
    process = subprocess.Popen(command, cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT)
    record['gdb_pid'] = process.pid
    (scratch / 'replay-provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    try:
        record['gdb_exit_code'] = process.wait(timeout=300)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=15)
        record['timed_out'] = True
    record['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (scratch / 'replay-provenance.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
