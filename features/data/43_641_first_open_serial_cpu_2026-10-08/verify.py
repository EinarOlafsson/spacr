"""Verify immutable evidence and optionally bind the accepted files to a Git ref."""
import argparse
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

manifest = json.loads(read(prefix + '/MANIFEST.json'))
for name, record in manifest['payloads'].items():
    data = read(prefix + '/' + name)
    assert len(data) == record['bytes'], name
    assert hashlib.sha256(data).hexdigest() == record['sha256'], name
if args.bind_current:
    receipt = json.loads(read(prefix + '/receipt.json'))
    for path, expected in {**receipt['after_bindings'], **receipt['production_bindings']}.items():
        assert hashlib.sha256(read(path)).hexdigest() == expected, path
print(f"Verified {len(manifest['payloads'])} payloads; bind_current={args.bind_current}")
