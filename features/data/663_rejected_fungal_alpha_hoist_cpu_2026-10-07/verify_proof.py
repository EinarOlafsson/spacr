"""Verify rejected alpha-hoist payloads and all eight native worker bindings."""
import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--git', metavar='REV')
args = parser.parse_args()

def read(name):
    """Read archive bytes from the filesystem or selected committed tree."""
    if args.git:
        return subprocess.check_output(['git', 'show', args.git + ':features/data/' + root.name + '/' + name], cwd=root)
    return (root / name).read_bytes()

manifest = json.loads(read('manifest.json'))
for name, entry in manifest['files'].items():
    data = read(name)
    assert len(data) == entry['bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256'], name
decision = json.loads(read('decision.json'))
assert not decision['app_changed'] and not decision['hard24FPS_accepted']
for name, digest in decision['source_sha256'].items():
    assert hashlib.sha256(gzip.decompress(read(name + '_ambient.py.gz'))).hexdigest() == digest
parity = json.loads(read('receipt.json'))
assert parity['source_sha256'] == decision['source_sha256']
assert sum(row['geometry_equal_cases'] for row in parity['rows']) == 28
assert sum(len(row['native_pairs']) for row in parity['rows']) == 12
assert all(pair['different_pixels'] == 0 and pair['hashes']['before'] == pair['hashes']['candidate']
           for row in parity['rows'] for pair in row['native_pairs'])
live = json.loads(read('live_summary.json'))
assert sum(len(row['runs']) for row in live['rows']) == 8
for recipe in live['rows']:
    assert [run['phase'] for run in recipe['runs']] == ['A1', 'B1', 'B2', 'A2']
    for run in recipe['runs']:
        raw = json.loads(read(run['file']))
        source = 'before' if run['variant'] == 'before' else 'candidate'
        assert raw['source_sha256'] == decision['source_sha256'][source] == run['source_sha256']
        assert raw['viewport'] == raw['buffer'] == [3840, 2160]
        assert raw['controls']['fps'] == 24 and raw['controls']['density'] == 3
        assert raw['worker_stopped_200ms_after_hide'] and run['workers_stopped']
        assert not raw['profile_enabled']
print(f"Verified {len(manifest['files'])} payloads, 12 native pairs and eight worker bindings; rejected.")
