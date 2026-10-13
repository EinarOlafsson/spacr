"""Check immutable six-job evidence, raw bytes and current production bindings."""
import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--git')
parser.add_argument('--bind-current', action='store_true')
args = parser.parse_args()
here = Path(__file__).resolve().parent
root = next(p for p in here.parents if (p / '.git').exists())
prefix = here.relative_to(root).as_posix()

def read(path):
    if args.git:
        return subprocess.check_output(['git', 'show', f'{args.git}:{path}'], cwd=root)
    return (root / path).read_bytes()

def digest(data):
    return hashlib.sha256(data).hexdigest()

manifest = json.loads(read(prefix + '/MANIFEST.json'))
for name, record in manifest['payloads'].items():
    data = read(prefix + '/' + name)
    assert len(data) == record['bytes'] and digest(data) == record['sha256'], name
receipt = json.loads(read(prefix + '/receipt.json'))
assert len(receipt['jobs']) == 6
assert receipt['verdict'] == {'success': 4, 'failure': 2}
for job in receipt['jobs']:
    raw = gzip.decompress(read(prefix + '/' + job['phase'] + '.log.gz'))
    assert len(raw) == job['raw_log_bytes'] and digest(raw) == job['raw_log_sha256']
    metadata = json.loads(read(prefix + '/' + job['phase'] + '-job.json'))
    assert metadata['status'] == 'completed' and metadata['conclusion'] == job['conclusion']
    assert metadata['id'] == job['id'] and metadata['run_id'] == receipt['run_id']
    assert metadata['head_sha'] == receipt['head_sha'] and metadata['run_attempt'] == 1
    if job['phase'] == 'fast0':
        partial = receipt['partial_fast0_observation']
        assert digest(raw[:partial['bytes']]) == partial['sha256']
if args.bind_current:
    for path in receipt['unchanged_production_files']:
        record = receipt['source_bindings'][path]
        assert record['4133_sha256'] == record['86d_sha256'] == digest(read(path)), path
print(f"Verified {len(manifest['payloads'])} payloads and six full raw logs; bind_current={args.bind_current}")
