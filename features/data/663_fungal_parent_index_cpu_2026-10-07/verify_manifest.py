"""Verify every frozen proof payload without rerunning native benchmarks."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

folder = Path(__file__).resolve().parent
root = folder.parents[2]
prefix = folder.relative_to(root).as_posix()
parser = argparse.ArgumentParser()
parser.add_argument('--git', action='store_true')
arguments = parser.parse_args()


def read(name):
    """Read a payload from filesystem or committed Git blobs."""
    if arguments.git:
        return subprocess.check_output(['git', '-C', str(root), 'show', f'HEAD:{prefix}/{name}'])
    return (folder / name).read_bytes()


manifest = json.loads(read('MANIFEST.json'))
for name, expected in manifest.items():
    data = read(name)
    assert len(data) == expected['bytes'], name
    assert hashlib.sha256(data).hexdigest() == expected['sha256'], name
print('Verified', len(manifest), 'source-bound payloads')
