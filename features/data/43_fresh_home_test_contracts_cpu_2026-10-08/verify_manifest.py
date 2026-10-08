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
for record in r['test_transition']:
    name = Path(record['path']).name
    before = gzip.decompress(read(prefix + '/before-' + name + '.gz'))
    after = gzip.decompress(read(prefix + '/after-' + name + '.gz'))
    assert digest(before) == record['before_sha256']
    assert digest(after) == record['after_sha256']
    assert before.count(b'MainWindow()') == record['replacements']
    assert before.replace(b'MainWindow()', b'MainWindow(initial_app="__home__")') == after
    if a.bind_current:
        assert digest(read(record['path'])) == record['after_sha256'], record['path']
for phase, marker in [('before', b'6 failed, 1 passed'), ('after', b'7 passed in 28.10s')]:
    assert marker in gzip.decompress(read(prefix + '/' + phase + '.log.gz'))
observations = json.loads(read(prefix + '/seeded-after.json'))
assert len(observations) == 10
assert all(row['initial_app'] == '__home__' and row['starts_at_home'] and not row['screens_after_constructor'] for row in observations)
if a.bind_current:
    for path, expected in r['production_sha256'].items():
        assert digest(read(path)) == expected, path
print('Verified', len(m['payloads']), 'payloads, six exact test-only transitions and ten real Home constructors; bind_current=', a.bind_current)
