"""Verify all payload bytes from disk or a named immutable Git revision."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--git', dest='revision')
args = parser.parse_args()
base = Path(__file__).resolve().parent
root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
relative = base.relative_to(root).as_posix()


def read(name):
    """Read one archive-relative payload from the selected evidence source."""
    if args.revision:
        return subprocess.check_output(['git', 'show', f'{args.revision}:{relative}/{name}'])
    return (base / name).read_bytes()


manifest = json.loads(read('manifest.json'))
for entry in manifest['payloads']:
    payload = read(entry['path'])
    assert len(payload) == entry['bytes'], entry['path']
    assert hashlib.sha256(payload).hexdigest() == entry['sha256'], entry['path']
print(f"Verified {len(manifest['payloads'])} payloads from {args.revision or 'filesystem'}")
