"""Verify compressed terminal logs and their recorded raw hashes."""
import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

folder = Path(__file__).resolve().parent
root = folder.parents[2]
prefix = folder.relative_to(root).as_posix()
parser = argparse.ArgumentParser()
parser.add_argument('--git', action='store_true')
args = parser.parse_args()

def read(name):
    """Read a payload from disk or the current committed Git tree."""
    if args.git:
        return subprocess.check_output(['git', '-C', str(root), 'show', f'HEAD:{prefix}/{name}'])
    return (folder / name).read_bytes()

manifest = json.loads(read('MANIFEST.json'))
for name, expected in manifest.items():
    payload = read(name)
    assert len(payload) == expected['bytes'], name
    assert hashlib.sha256(payload).hexdigest() == expected['sha256'], name
receipt = json.loads(read('receipt.json'))
assert len(receipt['jobs']) == 6
assert len({job['job_id'] for job in receipt['jobs']}) == 6
for job in receipt['jobs']:
    raw = gzip.decompress(read(job['archive']))
    assert len(raw) == job['raw_bytes'], job['lane']
    assert hashlib.sha256(raw).hexdigest() == job['sha256'], job['lane']
print('Verified', len(manifest), 'payloads and six exact raw terminal logs')
