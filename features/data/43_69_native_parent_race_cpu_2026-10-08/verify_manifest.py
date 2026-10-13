import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--git')
parser.add_argument('--bind-current', action='store_true')
args = parser.parse_args()
here = Path(__file__).resolve().parent
root = next(parent for parent in here.parents if (parent / '.git').exists())
prefix = here.relative_to(root).as_posix()

def read(path):
    if args.git:
        return subprocess.check_output(['git', 'show', args.git + ':' + path], cwd=root)
    return (root / path).read_bytes()

def payload(name):
    return read(prefix + '/' + name)

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

manifest = json.loads(payload('MANIFEST.json'))
for name, expected in manifest['payloads'].items():
    raw = payload(name)
    assert len(raw) == expected['bytes'] and digest(raw) == expected['sha256'], name
receipt = json.loads(payload('receipt.json'))
job = json.loads(payload('coverage0-job.json'))
assert job['id'] == receipt['job_id'] and job['run_id'] == receipt['run_id']
assert job['head_sha'] == receipt['head_sha'] and job['conclusion'] == 'cancelled'
steps = {row['name']: row['conclusion'] for row in job['steps']}
assert steps['Recover a coverage worker native backtrace'] == 'skipped'
assert steps['Upload coverage native failure evidence'] == 'skipped'
artifacts = json.loads(payload('hosted-artifacts.json'))['artifacts']
assert not any(row['name'].startswith('coverage-native-') for row in artifacts)
assert any(row['id'] == receipt['capture']['coverage0_process_artifact_id'] for row in artifacts)
for name, expected in receipt['raw_logs'].items():
    raw = gzip.decompress(payload(name))
    assert len(raw) == expected['bytes'] and digest(raw) == expected['sha256'], name
hosted_log = gzip.decompress(payload('coverage0.log.gz'))
assert b'Fatal Python error: Segmentation fault' in hosted_log
assert gzip.decompress(payload('hosted-fault-excerpt.txt.gz')) == b''.join(hosted_log.splitlines(keepends=True)[5409:5570])
gdb_log = gzip.decompress(payload('gdb.log.gz'))
assert b'12 passed in 77.70s' in gdb_log
assert b'[Inferior 1 (process 3815218) exited normally]' in gdb_log
assert b'No stack.' in gdb_log
assert b'received signal SIGSEGV' not in gdb_log
assert receipt['local_replay']['actual_nodes'] == 12
assert not receipt['local_replay']['native_signal']
for path, name in receipt['frozen_sources'].items():
    assert digest(gzip.decompress(payload(name))) == receipt['current_sha256'][path], path
    if args.bind_current:
        assert digest(read(path)) == receipt['current_sha256'][path], path
assert digest(gzip.decompress(payload('frozen-tests-workflow.yml.gz'))) == receipt['workflow_sha256']
coverage = json.loads(gzip.decompress(payload('focused-three-source-coverage.json.gz')))
assert len(coverage['files']) == 3
print('Verified', len(manifest['payloads']), 'payloads, actual hosted fault and 12-pass negative replay; bind_current=', args.bind_current)
