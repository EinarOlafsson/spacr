import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
p = argparse.ArgumentParser()
p.add_argument('--git')
p.add_argument('--bind-current', action='store_true')
a = p.parse_args()
here = Path(__file__).resolve().parent
root = next(path for path in here.parents if (path / '.git').exists())
prefix = here.relative_to(root).as_posix()
def read(path):
    return subprocess.check_output(['git', 'show', a.git + ':' + path], cwd=root) if a.git else (root / path).read_bytes()
def digest(raw):
    return hashlib.sha256(raw).hexdigest()
m = json.loads(read(prefix + '/MANIFEST.json'))
for name, record in m['payloads'].items():
    raw = read(prefix + '/' + name)
    assert len(raw) == record['bytes'] and digest(raw) == record['sha256'], name
r = json.loads(read(prefix + '/receipt.json'))
assert len(r['jobs']) == 6 and r['verdict'] == {'cancelled': 6, 'success': 0, 'failure': 0}
for row in r['jobs']:
    metadata = json.loads(read(prefix + '/' + row['phase'] + '-job.json'))
    assert metadata['status'] == 'completed' and metadata['conclusion'] == 'cancelled'
    assert metadata['id'] == row['id'] and metadata['run_id'] == r['run_id']
    assert metadata['head_sha'] == r['head_sha'] and metadata['run_attempt'] == 1
    raw = gzip.decompress(read(prefix + '/' + row['phase'] + '.log.gz'))
    assert len(raw) == row['raw_log_bytes'] and digest(raw) == row['raw_log_sha256']
run = json.loads(read(prefix + '/run-terminal.json'))
assert run['id'] == r['run_id'] and run['head_sha'] == r['head_sha'] and run['conclusion'] == 'cancelled'
if a.bind_current:
    for path, expected in r['unchanged_production_sha256'].items():
        assert digest(read(path)) == expected, path
print('Verified', len(m['payloads']), 'payloads and six cancelled/partial full raw logs; bind_current=', a.bind_current)
